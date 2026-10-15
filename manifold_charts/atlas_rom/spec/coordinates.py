"""coordinates.py — read the declared intrinsic coordinates off a configuration.

`CoordinateDef.source` names a rule; this module implements each rule exactly once,
for both a numpy config cloud (the sampling stage, building `samples.npz`) and a
torch batch of decoded transforms (the dynamics stage, where actuator Jacobians are
differentiated through the decoder).  Together they replace
`train_chart_decoder.build_latent(case, …)`'s per-case `if case == …` ladder.

Config packing throughout is the pipeline's existing one: per body, the flattened
`T[:3,:4]` (12 numbers), bodies in `mechanism.body_order`.

The `reference="first"` option on `body_angle` measures each angle relative to the
first sample's orientation, so a seed configuration reads ~0 — the convention the
rotate/gear latents already used.
"""
from __future__ import annotations

import numpy as np
import torch


# ─── low-level readers (numpy, on a config cloud) ─────────────────────────────

def unpack(configs, n_bodies):
    """(N, 12*n) -> (N, n, 3, 4)."""
    C = np.asarray(configs, float)
    return C.reshape(len(C), n_bodies, 3, 4)


def _axis_angle(R, axis, R0=None):
    """Signed rotation angle (rad) of R about `axis`, relative to R0.

    The full 3x3 formula (not `atan2(R10,R00)`) so it is correct for a body that
    rotates about an arbitrary world axis, not only +z.
    """
    R = np.asarray(R, float)
    if R0 is not None:
        R = R @ np.asarray(R0, float).T
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    cos = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sign = np.sign(a @ w)
    return float((sign if sign != 0 else 1.0) * theta)


def _relative_rotation(T, ia, ib):
    """R_a^T R_b for one sample's (n,3,4) stack; a `None` index means GROUND.

    A joint to a fixed body (a grounded pivot) is the common case, and its
    'relative' angle is just the moving body's own orientation — the fixed body
    contributes identity, so the same formula covers both without a branch at the
    call site.
    """
    Ra = np.eye(3) if ia is None else T[ia, :, :3]
    Rb = np.eye(3) if ib is None else T[ib, :, :3]
    return Ra.T @ Rb


def joint_travel(transforms, joint, body_order):
    """A single joint's own travel at one configuration, measured from rest.

    Revolute  -> signed angle (rad) about the joint axis.
    Prismatic -> signed displacement (m) along it.

    `transforms` is one sample's (n_bodies, 3, 4) stack.  Every body's mesh is
    baked at its rest world pose and starts at transform = identity, so the
    joint's declared world pivot IS its local pivot in both bodies and the rest
    travel is exactly 0 — which is what makes `limit_lo <= 0 <= limit_hi` the
    right thing for a spec to assert.  A `None` body index is ground.

    This is the reader behind `JointDef.limit_lo/limit_hi`, and it is the same
    quantity `CoordinateDef(source="joint_angle")` reports for a revolute.
    """
    T = np.asarray(transforms, float)
    ia = body_order.index(joint.body_a) if joint.body_a in body_order else None
    ib = body_order.index(joint.body_b) if joint.body_b in body_order else None
    if ia is None and ib is None:
        raise ValueError(f"joint {joint.name!r} connects two non-driven bodies; "
                         f"it has no travel to read")
    if joint.kind == "revolute":
        return _axis_angle(_relative_rotation(T, ia, ib), joint.axis)

    def anchor(i):
        if i is None:                       # ground: the pivot never moves
            return np.asarray(joint.pivot, float)
        return T[i, :, :3] @ np.asarray(joint.pivot, float) + T[i, :, 3]

    # The rail belongs to body_a and turns with it, exactly as in
    # `ideal_oracle._transverse`.
    Ra = np.eye(3) if ia is None else T[ia, :, :3]
    return float((anchor(ib) - anchor(ia)) @ (Ra @ np.asarray(joint.axis, float)))


def outside_joint_limits(spec, transforms):
    """The first declared joint limit this configuration violates, or None."""
    order = spec.mechanism.body_order
    for joint in spec.mechanism.joints:
        if not joint.limited:
            continue
        value = joint_travel(transforms, joint, order)
        if value < joint.limit_lo or value > joint.limit_hi:
            return joint.name, value
    return None


# ─── the coordinate map (numpy) ───────────────────────────────────────────────

class CoordinateMap:
    """Evaluate a case's declared coordinates on recorded configurations.

    Built once from the spec; `__call__(configs, fields)` returns `U (N, d)`.
    Stateful only in the sense that `reference="first"` and `isomap` need the
    cloud they were fitted on — those are captured at fit time and reused, so the
    coordinate of a configuration never depends on which batch it arrived in.
    """

    def __init__(self, spec):
        self.spec = spec
        self.coords = list(spec.coordinates)
        self.body_order = spec.mechanism.body_order
        self.n_bodies = len(self.body_order)
        self._reference = {}          # coordinate name -> R0 (3,3)
        self._isomap = None
        self._fitted = False

    # -- fit -------------------------------------------------------------------
    def fit(self, configs, fields=None):
        """Capture cloud-dependent state (angle references, the isomap embedding)."""
        T = unpack(configs, self.n_bodies)
        for c in self.coords:
            if c.source == "body_angle" and c.args.get("reference", "first") == "first":
                ib = self.body_order.index(c.args["body"])
                self._reference[c.name] = T[0, ib, :, :3].copy()
        isomap_dims = [c for c in self.coords if c.source == "isomap"]
        if isomap_dims:
            self._isomap = self._fit_isomap(configs, isomap_dims)
        self._fitted = True
        return self

    def _fit_isomap(self, configs, dims):
        """Nonlinear intrinsic coordinates of the config cloud (geodesic MDS).

        Used when no body angle or translation is a global coordinate — the
        five-bar case.  Transductive: the embedding is stored with the cloud it
        was fitted on, and new configurations are placed by kNN interpolation.
        """
        from scipy.sparse.csgraph import shortest_path
        from scipy.spatial import cKDTree

        X = np.asarray(configs, float)
        k = int(dims[0].args.get("n_neighbors", 12))
        tree = cKDTree(X)
        dist, idx = tree.query(X, k=min(k + 1, len(X)))
        n = len(X)
        graph = np.full((n, n), np.inf)
        np.fill_diagonal(graph, 0.0)
        for i in range(n):
            for d, j in zip(dist[i][1:], idx[i][1:]):
                graph[i, int(j)] = graph[int(j), i] = d
        geo = shortest_path(graph, method="D", directed=False)
        finite = np.isfinite(geo)
        if not finite.all():                      # disconnected: cap at the diameter
            geo[~finite] = geo[finite].max() * 2.0
        G2 = geo ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ G2 @ J
        w, V = np.linalg.eigh(0.5 * (B + B.T))
        order = np.argsort(w)[::-1]
        n_out = max(int(d.args.get("component", i)) for i, d in enumerate(dims)) + 1
        keep = order[:n_out]
        emb = V[:, keep] * np.sqrt(np.clip(w[keep], 0.0, None))
        return {"points": X, "embedding": emb, "tree": tree,
                "k": min(k, len(X))}

    def _isomap_transform(self, configs):
        state = self._isomap
        X = np.asarray(configs, float)
        if np.array_equal(X, state["points"]):
            return state["embedding"]
        dist, idx = state["tree"].query(X, k=state["k"])
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        w = 1.0 / np.maximum(dist, 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkd->nd", w, state["embedding"][idx])

    # -- evaluate --------------------------------------------------------------
    def __call__(self, configs, fields=None):
        if not self._fitted:
            self.fit(configs, fields)
        fields = fields or {}
        C = np.asarray(configs, float)
        T = unpack(C, self.n_bodies)
        iso = self._isomap_transform(C) if self._isomap is not None else None
        columns = []
        for c in self.coords:
            columns.append(self._column(c, C, T, fields, iso))
        if not columns:
            # A residual-discovery case (`coordinates=[]`) has nothing to stack;
            # callers like `sample.inside_domains` still index this by column,
            # so the shape has to stay (N, 0) rather than raising.
            return np.empty((C.shape[0], 0))
        return np.column_stack(columns)

    def _column(self, c, C, T, fields, iso):
        a = c.args
        if c.source == "body_translation":
            ib = self.body_order.index(a["body"])
            ax = "xyz".index(a.get("axis", "x"))
            return T[:, ib, ax, 3].copy()
        if c.source == "body_angle":
            ib = self.body_order.index(a["body"])
            axis = np.asarray(a.get("axis", (0, 0, 1)), float)
            R0 = self._reference.get(c.name)
            return np.array([_axis_angle(T[i, ib, :, :3], axis, R0)
                             for i in range(len(T))])
        if c.source == "joint_angle":
            joint = self.spec.mechanism.joint(a["joint"])
            ia = (self.body_order.index(joint.body_a)
                  if joint.body_a in self.body_order else None)
            ib = (self.body_order.index(joint.body_b)
                  if joint.body_b in self.body_order else None)
            if ia is None and ib is None:
                raise ValueError(f"coordinate {c.name!r}: joint {joint.name!r} "
                                 f"connects two non-driven bodies — it has no angle")
            R0 = self._reference.get(c.name)
            return np.array([_axis_angle(_relative_rotation(T[i], ia, ib),
                                         joint.axis, R0) for i in range(len(T))])
        if c.source == "sample_field":
            key = a["key"]
            if key not in fields:
                raise KeyError(f"coordinate {c.name!r} needs recorded field {key!r}; "
                               f"samples carry {sorted(fields)}")
            return np.asarray(fields[key], float).reshape(len(C)) * \
                float(a.get("scale", 1.0))
        if c.source == "isomap":
            return iso[:, int(a.get("component", 0))]
        raise ValueError(f"unhandled coordinate source {c.source!r}")

    # -- state so a rebuild reproduces the same coordinate ---------------------
    def state(self) -> dict:
        out = {"reference": {k: v.tolist() for k, v in self._reference.items()}}
        if self._isomap is not None:
            out["isomap_points"] = self._isomap["points"].tolist()
            out["isomap_embedding"] = self._isomap["embedding"].tolist()
            out["isomap_k"] = self._isomap["k"]
        return out

    def load_state(self, state: dict):
        from scipy.spatial import cKDTree
        self._reference = {k: np.asarray(v, float)
                           for k, v in state.get("reference", {}).items()}
        if "isomap_points" in state:
            pts = np.asarray(state["isomap_points"], float)
            self._isomap = {"points": pts,
                            "embedding": np.asarray(state["isomap_embedding"], float),
                            "tree": cKDTree(pts), "k": int(state["isomap_k"])}
        self._fitted = True
        return self


# ─── differentiable joint angle (torch, on decoded transforms) ────────────────

def joint_angle_torch(transforms, joint, body_order):
    """Signed joint angle from a torch (..., n_bodies, 3, 4) transform stack.

    Differentiable, so `dθ/du` for an actuator's generalized force comes straight
    from autodiff through the decoder — no per-case kinematics.  Uses
    `atan2(axis·w, trace-1)` rather than `arccos`, which is numerically flat (and
    gradient-singular) exactly at θ=0 where a rest pose sits.
    """
    sample = transforms[..., 0, :, :3]
    identity = torch.eye(3, dtype=sample.dtype,
                         device=sample.device).expand_as(sample)

    def rotation(name):
        # a joint to a FIXED body (a grounded pivot) contributes identity, so the
        # grounded and body-to-body cases share one formula
        if name is None or name not in body_order:
            return identity
        return transforms[..., body_order.index(name), :, :3]

    Ra, Rb = rotation(joint.body_a), rotation(joint.body_b)
    R = Ra.transpose(-1, -2) @ Rb
    axis = torch.as_tensor(np.asarray(joint.axis, float), dtype=Ra.dtype,
                           device=Ra.device)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin2 = (w * axis).sum(-1)                     # 2 sinθ (about the joint axis)
    cos2 = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0     # 2 cosθ
    return torch.atan2(sin2, cos2)
