"""validity.py -- the chart contract shared by fitting, packaging and runtime.

Physical validity (joint-closure residual on the declared joints), the runtime chart topology, the
physical handoff ``u_j = E_j(D_i(u_i))`` and the ``patch_atlas.pt`` artifact constants.  Extracted
unchanged from the retired ``patch_atlas.py`` / ``decoder.py`` (now under ``atlas_rom/legacy``)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch

CONSTRAINT_TOL = 0.1             # ceiling on sqrt(closure residual), physical units
ARTIFACT = "patch_atlas.pt"
ARTIFACT_FORMAT = "patch_atlas_v3"


# --- loop closure on the declared joints ------------------------------------------------------

def closure_attachments(spec, include_grounded=False):
    """Body-local attachment points for every two-body joint with a pivot.

    A joint's world pivot at the rest pose maps into each body's local frame
    through that body's `initial_transform`; the decoder must then keep the two
    images coincident at every coordinate.  Bodies whose vertices are rest-baked
    (identity transform) simply carry the world pivot itself.

    ``include_grounded`` adds the joints between a driven body and the fixed
    ground.  The ground does not move, but the body's pivot image can still
    drift off the fixed world pivot, so these joints can tear exactly like
    two-body ones.  They are returned with ``ib == -1`` and ``pb`` equal to the
    fixed world pivot (see :func:`closure_residual`).  Off by default so that
    existing callers keep their historical two-body-only residual.
    """
    order = spec.mechanism.body_order
    rows = []
    for joint in spec.mechanism.joints:
        # A prismatic joint deliberately permits the two attachment points to
        # separate along the rail.  Point-coincidence closure is only valid for
        # revolute/spherical/fixed pivots; the oracle still enforces the full prismatic
        # axis constraint during sampling.
        if joint.kind == "prismatic":
            continue
        if joint.pivot is None or joint.body_b is None:
            continue
        a_driven, b_driven = joint.body_a in order, joint.body_b in order
        if a_driven and b_driven:
            pa = np.linalg.inv(spec.mechanism.body(joint.body_a).initial_transform) @ \
                np.append(joint.pivot, 1.0)
            pb = np.linalg.inv(spec.mechanism.body(joint.body_b).initial_transform) @ \
                np.append(joint.pivot, 1.0)
            rows.append((order.index(joint.body_a), pa[:3],
                         order.index(joint.body_b), pb[:3]))
        elif include_grounded and (a_driven != b_driven):
            driven = joint.body_a if a_driven else joint.body_b
            local = np.linalg.inv(spec.mechanism.body(driven).initial_transform) @ \
                np.append(joint.pivot, 1.0)
            rows.append((order.index(driven), local[:3], -1, np.asarray(joint.pivot, float)))
    if not rows:
        return None
    return (np.array([r[0] for r in rows], int), np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows], int), np.array([r[3] for r in rows], float))


def closure_residual(transforms, attachments):
    """Squared distance between the two images of each joint pivot (torch or numpy).

    A row with ``ib < 0`` is a grounded joint: its second image is the fixed
    world point ``pb`` rather than a body-local point carried by body ``ib``.
    """
    if attachments is None:
        return None
    ia, pa, ib, pb = attachments
    ground = ib < 0
    ib_safe = np.where(ground, 0, ib)
    if isinstance(transforms, torch.Tensor):
        dtype, device = transforms.dtype, transforms.device
        pa_t = torch.as_tensor(pa, dtype=dtype, device=device)
        pb_t = torch.as_tensor(pb, dtype=dtype, device=device)
        Ta, Tb = transforms[..., ia, :, :], transforms[..., ib_safe, :, :]
        wa = (Ta[..., :3] @ pa_t.unsqueeze(-1)).squeeze(-1) + Ta[..., 3]
        wb = (Tb[..., :3] @ pb_t.unsqueeze(-1)).squeeze(-1) + Tb[..., 3]
        if ground.any():
            wb = torch.where(torch.as_tensor(ground, device=device).unsqueeze(-1), pb_t, wb)
        return ((wa - wb) ** 2).sum(-1)
    T = np.asarray(transforms, float)
    wa = np.einsum("...ij,...j->...i", T[..., ia, :, :3], pa) + T[..., ia, :, 3]
    wb = np.einsum("...ij,...j->...i", T[..., ib_safe, :, :3], pb) + T[..., ib_safe, :, 3]
    if ground.any():
        wb = np.where(ground[:, None], pb, wb)
    return ((wa - wb) ** 2).sum(-1)


# --- physical validity ------------------------------------------------------------------------

class PhysicalValidity(Protocol):
    """Optional mechanism validity hook.

    ``residual`` returns one non-negative squared residual per pose (or
    ``None`` when the check is unavailable).  It must remain differentiable
    when used by :func:`train_patch`.
    """

    def residual(self, packed_q, transforms): ...


@dataclass(frozen=True)
class ClosureValidity:
    attachment: object
    case_spec: object

    def residual(self, packed_q, transforms):
        return closure_residual(transforms, self.attachment)

    def joint_limit_invalid(self, transforms):
        """Return hard negative-witness labels for declared physical stops.

        This is used only while fitting the learned domain/SDF fields.  The
        decoder training loss remains differentiable and continues to use the
        closure residual above.
        """
        from atlas_rom.spec.coordinates import outside_joint_limits
        T = (transforms.detach().cpu().numpy()
             if hasattr(transforms, "detach") else np.asarray(transforms))
        return np.asarray([outside_joint_limits(self.case_spec, t) is not None
                           for t in T], dtype=bool)


class PenetrationValidity:
    """Contact validity from the body meshes alone: how far bodies overlap.

    Each driven body gets a signed-distance grid of its rest-baked mesh
    (negative inside), cached on disk.  For a pose, surface points of every body
    are carried into every other body's frame and the grid is sampled
    trilinearly, so the residual is differentiable.  One value per ordered body
    pair: the mean squared penetration depth over that body's surface points.
    No joint, contact pair or tolerance is declared; a settled simulator pose
    scores near zero and a pose that drives one body through another does not.
    """

    def __init__(self, case_spec, *, cache_dir=None, voxel=0.0625, points_per_body=256,
                 pad=0.25, max_cells_per_body=2 ** 22, seed=0, sdf_backend="trimesh"):
        import hashlib
        import os
        import trimesh
        from atlas_rom.spec import geometry
        if sdf_backend not in ("trimesh", "superdex"):
            raise ValueError(f"unknown sdf_backend {sdf_backend!r}")
        self.case_spec = case_spec
        self.sdf_backend = sdf_backend
        order = list(case_spec.mechanism.body_order)
        self.n_bodies = len(order)
        grids, lows, highs, points = [], [], [], []
        for name in order:
            V, F = geometry.build_mesh(case_spec.mechanism.body(name).mesh)
            mesh = trimesh.Trimesh(V, F, process=False)
            key = hashlib.sha256(np.ascontiguousarray(V).tobytes() + np.ascontiguousarray(F).tobytes()
                                 + np.float64(voxel).tobytes() + np.float64(pad).tobytes()
                                 + sdf_backend.encode()).hexdigest()[:16]
            if sdf_backend == "superdex":
                # The oracle's own collider grid: training sees the field the data was sampled with.
                path = None if cache_dir is None else os.path.join(cache_dir, f"superdex_sdf_{name}_{key}.npz")
                if path is not None and os.path.isfile(path):
                    with np.load(path) as cached:
                        sdf, low, high = cached["values"], cached["low"], cached["high"]
                else:
                    from atlas_rom.simulate.superdex_oracle import baked_sdf_grid
                    sdf, low, high = baked_sdf_grid(V, F, voxel=voxel, max_cells=max_cells_per_body)
                    sdf = sdf.astype(np.float32)
                    if path is not None:
                        os.makedirs(cache_dir, exist_ok=True)
                        np.savez(path, values=sdf, low=low, high=high)
                grids.append(sdf); lows.append(low); highs.append(high)
                points.append(self._surface_points(mesh, V, F, points_per_body, seed))
                continue
            low, high = V.min(0) - pad, V.max(0) + pad
            shape = np.ceil((high - low) / voxel).astype(int) + 1
            if int(np.prod(shape)) > int(max_cells_per_body):
                raise MemoryError(f"{name}: SDF grid {shape.tolist()} exceeds {max_cells_per_body} cells")
            high = low + (shape - 1) * voxel
            key = hashlib.sha256(np.ascontiguousarray(V).tobytes() + np.ascontiguousarray(F).tobytes()
                                 + np.float64(voxel).tobytes() + np.float64(pad).tobytes()).hexdigest()[:16]
            path = None if cache_dir is None else os.path.join(cache_dir, f"contact_sdf_{name}_{key}.npy")
            if path is not None and os.path.isfile(path):
                sdf = np.load(path)
            else:
                axes = [np.linspace(low[i], high[i], shape[i]) for i in range(3)]
                grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
                inside = np.concatenate([trimesh.proximity.signed_distance(mesh, chunk)
                                         for chunk in np.array_split(grid, max(1, len(grid) // 50000))])
                sdf = (-inside).reshape(shape).astype(np.float32)        # negative inside
                if path is not None:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(path, sdf)
            grids.append(sdf); lows.append(low); highs.append(high)
            points.append(self._surface_points(mesh, V, F, points_per_body, seed))
        self._grids = [torch.as_tensor(g)[None, None] for g in grids]       # [1,1,X,Y,Z]
        self._low = np.asarray(lows); self._high = np.asarray(highs)
        self._points = np.asarray(points)                                   # [B,P,3]
        self.pairs = [(a, b) for a in range(self.n_bodies) for b in range(self.n_bodies) if a != b]

    @staticmethod
    def _surface_points(mesh, V, F, count, seed):
        rng = np.random.default_rng(seed)
        area = mesh.area_faces / mesh.area_faces.sum()
        face = rng.choice(len(F), size=int(count), p=area)
        w = rng.random((int(count), 2)); flip = w.sum(1) > 1; w[flip] = 1 - w[flip]
        tri = V[F[face]]
        return tri[:, 0] + w[:, :1] * (tri[:, 1] - tri[:, 0]) + w[:, 1:] * (tri[:, 2] - tri[:, 0])

    def residual(self, packed_q, transforms):
        numpy_in = not isinstance(transforms, torch.Tensor)
        T = torch.as_tensor(np.asarray(transforms, float)) if numpy_in else transforms
        lead = T.shape[:-3]
        T = T.reshape(-1, self.n_bodies, 3, 4)
        pts = torch.as_tensor(self._points, dtype=T.dtype, device=T.device)
        world = torch.einsum("nbij,bpj->nbpi", T[..., :3], pts) + T[..., None, :, 3]
        out = []
        for a, b in self.pairs:
            local = torch.einsum("nji,npj->npi", T[:, b, :, :3], world[:, a] - T[:, b, None, :, 3])
            low = torch.as_tensor(self._low[b], dtype=T.dtype, device=T.device)
            high = torch.as_tensor(self._high[b], dtype=T.dtype, device=T.device)
            unit = 2.0 * (local - low) / (high - low) - 1.0                  # [-1,1] over the grid
            # grid_sample indexes (x,y,z) as (W,H,D): reverse to match the [X,Y,Z] array.
            coords = unit.flip(-1).reshape(1, -1, 1, 1, 3)
            grid = self._grids[b].to(dtype=T.dtype, device=T.device)
            sdf = torch.nn.functional.grid_sample(grid, coords, mode="bilinear", padding_mode="border",
                                                  align_corners=True).reshape(len(T), -1)
            out.append(torch.relu(-sdf).square().mean(-1))
        r = torch.stack(out, -1).reshape(*lead, len(self.pairs))
        return r.detach().cpu().numpy() if numpy_in else r


class CombinedValidity:
    """Concatenated residuals of several validity checks (closure, contact, ...)."""

    def __init__(self, parts, case_spec):
        self.parts = [p for p in parts if p is not None]
        self.case_spec = case_spec
        closure = [p for p in self.parts if hasattr(p, "joint_limit_invalid")]
        if closure:
            self.joint_limit_invalid = closure[0].joint_limit_invalid

    def residual(self, packed_q, transforms):
        values = [p.residual(packed_q, transforms) for p in self.parts]
        values = [v for v in values if v is not None]
        if not values:
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.cat(values, -1)
        return np.concatenate(values, -1)


def validity_from_case(case_spec, include_grounded=True, include_contact=False,
                       contact_cache_dir=None, **contact_options) -> Optional[PhysicalValidity]:
    """Closure validity for a case.  ``include_grounded`` adds the joints to the
    fixed ground (training default: an unconstrained grounded pivot drifts, and
    on rotate those two joints dominated the worst-joint gap).  The ROM runtime
    passes ``False`` so its closure readout keeps its historical meaning.

    ``include_contact`` (off by default) adds :class:`PenetrationValidity` for a
    case whose bodies interact through contact, so decoded poses are also held
    to non-penetration rather than only to the declared joints."""
    attachment = closure_attachments(case_spec, include_grounded=include_grounded)
    closure = ClosureValidity(attachment, case_spec) if attachment is not None else None
    if not include_contact or case_spec.contact_model == "none":
        return closure
    sampling = getattr(getattr(case_spec, "build", None), "sampling", None)
    if getattr(sampling, "oracle_backend", None) == "superdex":
        # Use the oracle's collider SDF, at the oracle's resolution, unless overridden.
        from atlas_rom.simulate.superdex_oracle import MAX_SDF_CELLS
        oracle = dict(getattr(sampling, "oracle_options", None) or {})
        contact_options.setdefault("sdf_backend", "superdex")
        if contact_options["sdf_backend"] == "superdex":
            contact_options.setdefault("voxel", float(oracle.get("sdf_voxel", 0.0625)))
            contact_options.setdefault("max_cells_per_body", int(oracle.get("max_sdf_cells", MAX_SDF_CELLS)))
    contact = PenetrationValidity(case_spec, cache_dir=contact_cache_dir, **contact_options)
    return CombinedValidity([closure, contact], case_spec)


def _validity_residual(validity, packed_q, transforms):
    if validity is None:
        return None
    if hasattr(validity, "residual"):
        return validity.residual(packed_q, transforms)
    return validity(packed_q, transforms)


# --- runtime topology + handoff ---------------------------------------------------------------

class LearnedSupportTopology:
    """Route only through learned support fields and physical AE transfers.

    Unlike the retired anchor-cloud topology, this object holds no sampled
    coordinates, frames, kNN tree, declared chart geometry, or membership
    labels.  Candidate destination coordinates are the learned physical map
    ``E_j(D_i(u_i))`` and authority is the destination's learned support logit.
    """
    def __init__(self, models, support_fields, *, stay_margin=0.0):
        if len(models) != len(support_fields) or not models:
            raise ValueError("learned routing needs one support field per chart")
        self.models = list(models)
        self.support_fields = list(support_fields)
        self.stay_margin = float(stay_margin)
        self._margins, self._coordinates = {}, {}

    def initial_u(self, chart):
        # Initial conditions must be supplied by an encoded physical reset in
        # real runtime use.  A zero coordinate is a neutral API fallback, not
        # an anchor or sample lookup.
        return np.zeros(self.models[int(chart)].d, float), -1

    def locate(self, chart, u):
        import torch
        chart = int(chart)
        source_u = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
        with torch.no_grad():
            source_pose = self.models[chart].decode(source_u)
            self._margins, self._coordinates = {}, {}
            for target, (model, field) in enumerate(zip(self.models, self.support_fields)):
                candidate = source_u if target == chart else model.encode(source_pose)
                self._coordinates[target] = candidate.squeeze(0).numpy()
                self._margins[target] = float(field(candidate).squeeze())
        return self._margins[chart], -1, None

    def routing_target(self, chart, _cell):
        chart = int(chart)
        if self._margins.get(chart, -np.inf) >= self.stay_margin:
            return chart
        target = max(self._margins, key=self._margins.get)
        return target if self._margins[target] >= 0.0 else chart

    def wrap(self, _chart, u):
        return np.asarray(u, float).copy()


class RouterTopology:
    """Topology for the advantage router: no support fields, no routing here.

    The runtime only needs an initial coordinate and the (identity) coordinate wrap; chart choice is made by
    ``rom.router`` (``patch_rom.adv_router``)."""

    def __init__(self, models):
        self.models = list(models)

    def initial_u(self, chart):
        return np.zeros(self.models[int(chart)].d, float), -1

    def wrap(self, _chart, u):
        return np.asarray(u, float).copy()


def build_topology(models, support_fields=None):
    """Create runtime topology.  With support fields: the legacy learned-support routing; without: ``RouterTopology``."""
    if support_fields is None:
        return RouterTopology(models)
    return LearnedSupportTopology(models, support_fields)


def transition(models, i, j, u_i):
    """u_j = E_j(D_i(u_i)) -- the physical handoff between overlapping
    charts."""
    import torch
    with torch.no_grad():
        u_i_t = torch.as_tensor(np.asarray(u_i, float)).unsqueeze(0)
        q = models[i].decode(u_i_t)
        u_j = models[j].encode(q)
    return u_j.squeeze(0).numpy()


def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
