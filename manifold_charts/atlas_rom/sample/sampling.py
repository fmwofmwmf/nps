"""Production sampling contract for the physical patch-atlas pipeline.

Two physical sampler modes exist: ``analytic`` (an algebraic ideal-joint
manifold, projected by `ideal_oracle.IdealJointOracle`) and ``contact`` (a
contact-constrained manifold, projected by the full libuipc oracle in
`oracle.py`).  `AtlasCaseSpec.__post_init__` infers the mode from
`contact_model` when a case does not declare `sampling.sampler` explicitly.

Both modes run the SAME zero-prior advancing-front explorer: pop a frontier
point, take its tangent frame through the declared oracle, step along signed
tangent basis directions, PROJECT back onto the manifold, and keep the result
only if it registers real, witnessed progress.  The oracle is the only thing
that differs between the two modes -- the exploration policy, spacing control,
and closure/witness-edge logic are identical, and neither knows or cares
which physical mode produced the manifold it is walking.

The flood is the right general sampler because it is uniform in the
manifold's own metric and needs no global parameterization -- reset-to-rest
hops cannot traverse a mechanism's configuration space, and a raster over
declared coordinates jams the moment a coordinate is not independently
drivable.

A recorded cloud (``sampling.reuse``) is a separate input adapter: an
externally produced recording (frames + witnessed edges already computed) is
adopted verbatim, which is what lets a case migrate to this pipeline without
re-burning GPU hours.  It owns the shared ``samples.npz`` contract regardless
of which path filled it.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from atlas_rom.sample import dataset
from atlas_rom.spec import physical_pose as pp
from atlas_rom.spec.coordinates import CoordinateMap, outside_joint_limits
from atlas_rom.spec.physical_domain import violation as physical_domain_violation


def _paths(pattern: str, root: str) -> list[str]:
    paths = sorted(glob.glob(pattern if os.path.isabs(pattern)
                             else os.path.join(root, pattern)))
    if not paths:
        raise FileNotFoundError(f"sampling.reuse matched nothing: {pattern!r}")
    return paths


def _load_recording(pattern: str, root: str, body_order: list[str]):
    configs, frames, edge_chunks, field_chunks = [], [], [], {}
    for path in _paths(pattern, root):
        data = np.load(path, allow_pickle=True)
        key = next((k for k in ("configs", "centers", "transforms") if k in data.files), None)
        if key is None:
            raise KeyError(f"{path}: recording needs configs, centers, or transforms")
        q = np.asarray(data[key], float)
        if q.ndim > 2:
            q = q.reshape(len(q), -1)
        if "body_order" in data.files:
            source = [str(x) for x in data["body_order"].tolist()]
            if len(source) * 12 != q.shape[1]:
                raise ValueError(f"{path}: body_order disagrees with pose width")
            missing = [name for name in body_order if name not in source]
            if missing:
                raise ValueError(f"{path}: recording lacks bodies {missing}")
            take = [source.index(name) for name in body_order]
            q = q.reshape(len(q), len(source), 12)[:, take].reshape(len(q), -1)
        if "frames" not in data.files or "edges" not in data.files:
            raise ValueError(f"{path}: reusable recordings must provide frames and witnessed edges")
        f = np.asarray(data["frames"], float)
        e = np.asarray(data["edges"], int).reshape(-1, 2)
        if f.shape[:2] != q.shape:
            raise ValueError(f"{path}: frames shape {f.shape} does not match configurations {q.shape}")
        if e.size and (e.min() < 0 or e.max() >= len(q)):
            raise ValueError(f"{path}: edge index is outside the recording")
        offset = sum(len(chunk) for chunk in configs)
        configs.append(q); frames.append(f); edge_chunks.append(e + offset)
        for name in data.files:
            value = np.asarray(data[name])
            if value.ndim == 1 and len(value) == len(q):
                field_chunks.setdefault(name, []).append(value.astype(float))
    return (np.vstack(configs), np.vstack(frames), np.vstack(edge_chunks),
            {name: np.concatenate(values) for name, values in field_chunks.items()})


def _coordinates(case, fields: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for coordinate in case.coordinates:
        if coordinate.source != "sample_field":
            raise NotImplementedError(
                f"{case.name}: reusable {case.build.sampling.sampler} recordings require "
                f"a sampler-native coordinate for {coordinate.source!r}")
        key = str(coordinate.args.get("key", ""))
        if key not in fields:
            raise KeyError(f"{case.name}: recording is missing sample field {key!r}")
        values.append(fields[key])
    return np.column_stack(values)


# ─── the zero-prior advancing-front explorer ──────────────────────────────────
#
# Ported from the archived per-case explorer (`atlas_rom_experiments/sample.py`
# `_explore`) into production, unchanged in policy.  The only thing that moved
# is oracle selection: the archived version auto-detected analytic vs. contact
# from `ideal_oracle.supports(spec)`; here the mode is the case's already-
# validated `sampling.sampler` ("analytic" or "contact", see
# `AtlasCaseSpec.__post_init__`), so both physical samplers dispatch through
# the exact same walk.

# A tangent step is a LINEAR predictor: `project()` only pulls it back onto the
# manifold if it started near.  The bound that matters is therefore how far the
# predictor moves one body, not how far it moves the packed vector — a unit
# packed tangent spreads its length over every body, so a fixed ambient cap
# silently shrinks the usable step as a mechanism gains bodies (a 19-body stack
# got 1/2.4 of the step a single-body manifold got).  Capping per body is the
# same number for one body and scale-free above it.
MAX_BODY_DISPLACEMENT = 0.25

# `project()` rarely realizes the requested spacing exactly: the drive can be
# absorbed by a stiff coupled chain (undershoot) or a noisy du/dq estimate can
# ask for a huge step (overshoot).  Both used to kill a front — the candidate was
# rejected and the popped point was already gone, so that end of the curve simply
# stopped.  Rescale by the measured ratio and retry instead.
STEP_ATTEMPTS = 4
OVERSHOOT_FACTOR = 2.0      # realized spacing above this many `step`s: retry smaller
PROGRESS_FLOOR = 0.25       # ... below this: not a new sample at all
MAX_STALLS = 3              # failed pops of one 1-D front end before it retires
RESTORE_ATTEMPTS = 3        # settles allowed to bring the solver back to a pop
PROGRESS_SECONDS = 60.0     # heartbeat cadence while exploring

# A coordinate lift (`lifted_field`, below) is unbounded by design -- it exists
# only to pick a locally consistent step direction on a coordinate that wraps
# (an angle past its first turn) or is otherwise multivalued.  It must never be
# read as "distance from where we've already been": a periodic mechanism's
# PHYSICAL configuration repeats every time it winds around even though the
# lift keeps climbing.  CLOSURE_TOL_FACTOR sets how close (in the physical
# pose-twist metric, see physical_pose.py) a freshly projected candidate must
# land to an existing, already-accepted sample before it counts as the same
# physical state rather than a new one -- scaled to the walk's own realized
# step length, never an absolute constant, and only enabled once enough edges
# exist to measure that scale.
CLOSURE_TOL_FACTOR = 0.25
CLOSURE_MIN_EDGES = 8


def _explored_sample_field(spec):
    """The sample-field coordinate an explorer can construct online, if any.

    Imported recordings already carry arbitrary per-frame fields.  A live
    continuation does not.  The useful general case is an unwrapped body angle:
    a configuration contains its rotation only modulo 2*pi, but consecutive
    projected configurations are close enough that their signed increment is
    unambiguous.  Carrying those increments along the exploration tree produces
    the same lifted field a recorder would have supplied.

    Keep this deliberately narrow.  A field with no configuration-space meaning
    (for example a time stamp) cannot be reconstructed by an explorer and should
    still be imported rather than guessed.
    """
    if spec.dimension != 1 or len(spec.coordinates) != 1:
        return None
    coordinate = spec.coordinates[0]
    if coordinate.source != "sample_field":
        return None
    key = str(coordinate.args.get("key", ""))
    parts = key.split(":")
    if len(parts) != 3 or parts[0] != "recorded_angle" or parts[2] not in "xyz":
        raise ValueError(
            f"{spec.name}: exploration cannot reconstruct sample field {key!r}; "
            "use a recorded_angle:<body>:<axis> field or import a cloud")
    body = parts[1]
    if body not in spec.mechanism.body_order:
        raise ValueError(f"{spec.name}: sample field {key!r} names unknown driven body "
                         f"{body!r}")
    return key, spec.mechanism.body_order.index(body), "xyz".index(parts[2])


def _lifted_angle_delta(q0, q1, n_bodies, body_index, axis_index):
    """Signed, principal driver-angle increment from one nearby pose to another."""
    T0 = np.asarray(q0, float).reshape(n_bodies, 3, 4)[body_index, :, :3]
    T1 = np.asarray(q1, float).reshape(n_bodies, 3, 4)[body_index, :, :3]
    R = T1 @ T0.T
    trace = float(np.trace(R) - 1.0)
    sine = float(R[(axis_index + 2) % 3, (axis_index + 1) % 3]
                 - R[(axis_index + 1) % 3, (axis_index + 2) % 3])
    return float(np.arctan2(sine, trace))


def _revolute_field_predictor(spec, lifted_field):
    """Find the explicit hinge behind a recorded one-dimensional angle field."""
    if lifted_field is None:
        return None
    _key, body_index, axis_index = lifted_field
    body = spec.mechanism.body_order[body_index]
    for joint in spec.mechanism.joints:
        if joint.kind != "revolute" or body not in (joint.body_a, joint.body_b):
            continue
        joint_axis = np.asarray(joint.axis, float)
        if abs(joint_axis[axis_index]) < 1.0 - 1e-8:
            continue
        return (spec.mechanism.body_index(body), np.asarray(joint.pivot, float),
                joint_axis / np.linalg.norm(joint_axis))
    return None


def _rotate_target_about_pivot(q, n_bodies, body_index, pivot, axis, angle):
    """Compose a world-space hinge rotation into one packed body transform."""
    out = np.asarray(q, float).copy().reshape(n_bodies, 3, 4)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = pivot - R @ pivot
    T = np.eye(4)
    T[:3, :4] = out[body_index]
    out[body_index] = (delta @ T)[:3, :4]
    return out.reshape(-1)


def _planar_to_packed(configs, frames=None):
    """Convert planar rigid poses and their tangent frames to packed transforms."""
    q = np.asarray(configs, float)
    if q.ndim != 2 or q.shape[1] % 3:
        raise ValueError("planar_xytheta samples need q[N, 3*n_bodies]")
    n, ambient = q.shape
    bodies = ambient // 3
    blocks = q.reshape(n, bodies, 3)
    angle = blocks[..., 2]
    cosine, sine = np.cos(angle), np.sin(angle)
    packed = np.zeros((n, bodies, 3, 4), float)
    packed[..., 0, 0] = cosine
    packed[..., 0, 1] = -sine
    packed[..., 1, 0] = sine
    packed[..., 1, 1] = cosine
    packed[..., 2, 2] = 1.0
    packed[..., 0, 3] = blocks[..., 0]
    packed[..., 1, 3] = blocks[..., 1]
    packed = packed.reshape(n, 12 * bodies)
    if frames is None:
        return packed, None

    frame = np.asarray(frames, float)
    if frame.ndim != 3 or len(frame) != n:
        raise ValueError("planar_xytheta frames need shape [N, ambient, d]")
    if frame.shape[1] != ambient and frame.shape[2] == ambient:
        frame = frame.transpose(0, 2, 1)
    if frame.shape[1] != ambient:
        raise ValueError(
            f"planar frame ambient dimension is {frame.shape[1]}, expected {ambient}")
    d = frame.shape[2]
    source = frame.reshape(n, bodies, 3, d)
    target = np.zeros((n, bodies, 3, 4, d), float)
    target[..., 0, 3, :] = source[..., 0, :]
    target[..., 1, 3, :] = source[..., 1, :]
    theta = source[..., 2, :]
    target[..., 0, 0, :] = -sine[..., None] * theta
    target[..., 0, 1, :] = -cosine[..., None] * theta
    target[..., 1, 0, :] = cosine[..., None] * theta
    target[..., 1, 1, :] = -sine[..., None] * theta
    return packed, target.reshape(n, 12 * bodies, d)


def _recorded_frame_cloud(data, path, body_order):
    """Read the generic ``Recorder.save`` layout into packed relative poses.

    Older recordings store one 4x4 world transform per ``body__f<frame>``
    instead of a dense transform array.  Meshes imported from a scene have their
    rest-world transforms baked into vertices, whereas these transforms operate
    on node-local geometry.  Express each recorded pose relative to its first
    frame so either representation decodes the same motion.
    """
    if body_order is None or "body_names" not in data.files or "n_frames" not in data.files:
        return None
    source_order = [str(name) for name in np.asarray(data["body_names"]).reshape(-1)]
    missing = [name for name in body_order if name not in source_order]
    if missing:
        raise ValueError(f"{path}: recorder data lacks declared driven bodies "
                         f"{missing}; has {source_order}")
    n_frames = int(np.asarray(data["n_frames"]).reshape(-1)[0])
    if n_frames < 1:
        raise ValueError(f"{path}: recorder data has no frames")
    expected = [f"{name}__f{frame}" for name in body_order
                for frame in range(n_frames)]
    absent = [key for key in expected if key not in data.files]
    if absent:
        raise KeyError(f"{path}: incomplete recorder data; missing {absent[0]!r}")
    blocks = []
    for name in body_order:
        rest = np.asarray(data[f"{name}__f0"], float)
        if rest.shape != (4, 4):
            raise ValueError(f"{path}: {name!r} recorder transforms need shape (4, 4)")
        inv_rest = np.linalg.inv(rest)
        blocks.append(np.stack([np.asarray(data[f"{name}__f{frame}"], float)
                                @ inv_rest for frame in range(n_frames)]))
    q = np.stack(blocks, axis=1)[:, :, :3, :4].reshape(n_frames, -1)
    # A recorder preserves a traversal order even when no physical body angle is
    # one-to-one.  For a cyclic recording the final saved frame precedes the
    # first one by one sample step, so use n (not n - 1) to put that seam between
    # the two records.  Cases may declare this standard field as their periodic
    # coordinate without assuming an actuator angle is globally injective.
    fields = {"recorded_cycle_phase":
              2.0 * np.pi * np.arange(n_frames, dtype=float) / n_frames - np.pi}
    # Preserve unwrapped physical rotations too.  A recorder's frame index is a
    # valid loop parameter but often a poor dynamics coordinate: its spacing is
    # time-based, not configuration-based.  These fields let a case select a
    # real actuator angle even when it completes multiple turns per config loop.
    rotations = q.reshape(n_frames, len(body_order), 3, 4)[..., :3]
    for body_index, name in enumerate(body_order):
        R = rotations[:, body_index]
        trace = np.trace(R, axis1=1, axis2=2) - 1.0
        for axis_index, axis_name in enumerate("xyz"):
            sine = R[:, (axis_index + 2) % 3, (axis_index + 1) % 3] \
                   - R[:, (axis_index + 1) % 3, (axis_index + 2) % 3]
            angle = np.unwrap(np.arctan2(sine, trace))
            # Centre a periodic lift at zero, matching the runtime's symmetric
            # periodic wrap interval while retaining its full winding.
            angle -= 0.5 * (angle[0] + angle[-1])
            fields[f"recorded_angle:{name}:{axis_name}"] = angle
    return q, fields


def _load_cloud(pattern: str, root: str, representation="packed", body_order=None):
    """Adopt configurations (and frames/fields, when present) from npz file(s)."""
    paths = sorted(glob.glob(pattern if os.path.isabs(pattern)
                             else os.path.join(root, pattern)))
    if not paths:
        raise FileNotFoundError(f"sampling.seed_reuse matched nothing: {pattern!r}")
    configs, frames, fields, edge_chunks = [], [], {}, []
    config_keys = ("centers", "configs", "transforms")
    for path in paths:
        data = np.load(path, allow_pickle=True)
        key = next((k for k in config_keys if k in data.files), None)
        recorded = _recorded_frame_cloud(data, path, body_order)
        if key is None and recorded is None:
            raise KeyError(f"{path}: no {'/'.join(config_keys)} array "
                           "or Recorder frame layout")
        recorded_fields = {} if recorded is None else recorded[1]
        chunk = recorded[0] if recorded is not None else np.asarray(data[key], float)
        if chunk.ndim > 2:                  # (N, n_bodies, 3, 4) -> (N, 12*n_bodies)
            chunk = chunk.reshape(len(chunk), -1)
        # Older explorers commonly recorded every scene body, including fixed
        # scenery.  A generic ROM packs driven bodies only.  When the source
        # supplies its explicit order, select (and, if needed, reorder) just the
        # declared driven bodies rather than requiring a case-specific converter.
        if body_order is not None and "body_order" in data.files:
            source_order = [str(name) for name in np.asarray(data["body_order"]).tolist()]
            if len(source_order) * 12 != chunk.shape[1]:
                raise ValueError(f"{path}: body_order has {len(source_order)} names but "
                                 f"configs have {chunk.shape[1]} columns")
            missing = [name for name in body_order if name not in source_order]
            if missing:
                raise ValueError(f"{path}: sample body_order lacks declared driven "
                                 f"bodies {missing}; has {source_order}")
            take = [source_order.index(name) for name in body_order]
            chunk = chunk.reshape(len(chunk), len(source_order), 12)[:, take].reshape(len(chunk), -1)
        offset = sum(len(value) for value in configs)
        configs.append(chunk)
        for name, value in recorded_fields.items():
            fields.setdefault(name, []).append(np.asarray(value, float))
        frames.append(np.asarray(data["frames"], float)
                      if "frames" in data.files else None)
        if "edges" in data.files:
            edge_chunks.append(np.asarray(data["edges"], int).reshape(-1, 2)
                               + offset)
        for name in data.files:
            if name in config_keys or name == "frames":
                continue
            value = np.asarray(data[name])
            if value.ndim == 1 and len(value) == len(chunk):
                fields.setdefault(name, []).append(value.astype(float))
    q = np.vstack(configs)
    frames = (np.vstack(frames) if all(f is not None for f in frames) else None)
    if representation == "planar_xytheta":
        q, frames = _planar_to_packed(q, frames)
    elif representation != "packed":
        raise ValueError(f"unknown sample representation {representation!r}")
    fields = {k: np.concatenate(v) for k, v in fields.items()
              if sum(len(x) for x in v) == len(q)}
    edges = (np.vstack(edge_chunks) if len(edge_chunks) == len(paths) else None)
    return q, frames, fields, paths, edges


def _tree_path(adjacency, start, goal):
    """Vertex path from `start` to `goal` along the exploration graph.

    `project()` drives the mechanism from wherever the solver currently stands,
    and a target on the far side of the manifold is a drive it cannot complete:
    at best it lands short (the tangent probe then measures leftover travel), at
    worst the line search diverges on an inconsistent target and aborts the
    process.  The samples already explored ARE a connected path there, so relocate
    along them one settle at a time instead of asking for the jump.
    """
    if start == goal:
        return [start]
    previous, queue = {start: None}, [start]
    while queue:
        node = queue.pop(0)
        for neighbour in adjacency.get(node, ()):
            if neighbour in previous:
                continue
            previous[neighbour] = node
            if neighbour == goal:
                path = [goal]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return path[::-1]
            queue.append(neighbour)
    return []


def _spacing(spec, coord_map, a, b):
    """Separation between two configurations, in the units `step` is given in."""
    if spec.dimension != 1:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    pair = coord_map(np.asarray([np.ravel(a), np.ravel(b)], float))[:, 0]
    return float(abs(pair[0] - pair[1]))


def _predictor_cap(tangent, n_bodies, limit=MAX_BODY_DISPLACEMENT):
    """Largest tangent step whose predicted per-body pose change stays local."""
    blocks = np.asarray(tangent, float).reshape(n_bodies, 12)
    return float(limit) / max(float(np.linalg.norm(blocks, axis=1).max()), 1e-12)


def _select_oracle(spec, options, seed_hint, log, *, sdf_system=None):
    """Build the projection oracle for the case's declared sampler mode.

    `AtlasCaseSpec.__post_init__` has already validated
    `sampling.sampler in ("analytic", "contact")` and cross-checked it against
    `contact_model`, so this is a plain dispatch, not a fallback chain: the two
    physical samplers are the only kinds, and each is projected by exactly one
    oracle.
    """
    backend = "sdf" if sdf_system is not None else (options.oracle_backend or options.sampler)
    if backend == "sdf":
        if sdf_system is None:
            raise ValueError("oracle backend 'sdf' requires an SDFSystem")
        from atlas_rom.simulate.sdf_oracle import SDFPhysicalOracle
        return SDFPhysicalOracle(sdf_system), False
    if backend == "analytic":
        from atlas_rom.simulate import ideal_oracle
        if not ideal_oracle.supports(spec):
            raise ValueError(
                f"{spec.name}: sampling.sampler='analytic' but the mechanism is "
                f"not an ideal-joint-only case (contact_model={spec.contact_model!r}, "
                f"joint kinds {sorted({j.kind for j in spec.mechanism.joints})})")
        # Only the analytic oracle knows how to hold specific bodies fixed
        # during a correction; see `project`'s `hold` and `stationary_bodies`.
        return ideal_oracle.build_ideal_oracle(spec, log=log), True
    if backend != "contact":
        raise ValueError(f"{spec.name}: unsupported oracle backend {backend!r}")
    from atlas_rom.simulate.oracle import build_oracle
    settle = {k: v for k, v in
               (("probe_count", options.probe_count),
               ("drive_iters", options.drive_iters),
               ("relax_iters", options.relax_iters)) if v is not None}
    settle.update({k: v for k, v in (("max_kappa", options.max_kappa),
                                     ("newton_max_iter", options.newton_max_iter))
                   if v is not None})
    # Probe sizing belongs to the backend's physical-response implementation.
    # The explorer must not communicate a declared intrinsic dimension here.
    settle["d_hat"] = options.d_hat
    settle["actuated_bodies"] = options.actuated_bodies
    settle["initial_configuration"] = seed_hint
    settle["hold_drive_during_projection"] = options.hold_drive_during_projection
    settle["projection_mode"] = options.projection_mode
    settle["newton_velocity_tol"] = options.newton_velocity_tol
    settle["proximal_tol"] = options.proximal_tol
    settle["proximal_max_steps"] = options.proximal_max_steps
    return build_oracle(spec, log=log, **settle), False


def _reconcile_frames_with_secants(q, frames, edges, probe_diagnostics):
    """Reconcile measured frames against witnessed local motion only.

    A secant is evidence only when a vertex has enough independent incident
    walked transitions.  Otherwise the measurement is retained but marked
    uncertain; this never creates a tangent from an assumed topology.
    """
    q, frames = np.asarray(q, float), np.asarray(frames, float).copy()
    n, d = len(q), frames.shape[-1]
    adjacency = [[] for _ in range(n)]
    for a, b in np.asarray(edges, int).reshape(-1, 2):
        adjacency[a].append(b); adjacency[b].append(a)
    confidence = np.ones(n, float)
    angles = np.full(n, np.nan)
    secant_rank = np.zeros(n, int)
    for i in range(n):
        diag = probe_diagnostics.get(i, {})
        confidence[i] = float(diag.get("probe_confidence", 1.0))
        secants = []
        for j in adjacency[i]:
            delta = q[j] - q[i]
            norm = np.linalg.norm(delta)
            if norm > 1e-12:
                secants.append(delta / norm)
        if len(secants) < max(d, 2):
            confidence[i] *= 0.5
            continue
        S = np.asarray(secants).T
        U, values, _ = np.linalg.svd(S, full_matrices=False)
        tol = max(S.shape) * np.finfo(float).eps * values[0]
        rank = int(np.count_nonzero(values > tol))
        secant_rank[i] = rank
        if rank < d:
            confidence[i] *= 0.5
            continue
        span = U[:, :d]
        F, _ = np.linalg.qr(frames[i], mode="reduced")
        cosines = np.linalg.svd(F.T @ span, compute_uv=False).clip(0.0, 1.0)
        angle = float(np.degrees(np.arccos(cosines)).max())
        angles[i] = angle
        # A strong witnessed disagreement means the physical probe must not
        # supervise the AE unchecked.  Project only when the corrected span is
        # itself full rank; otherwise keep the raw measurement, low confidence.
        if angle > 12.0:
            projected = span @ (span.T @ F)
            corrected, r = np.linalg.qr(projected, mode="reduced")
            if np.linalg.matrix_rank(r) >= d:
                frames[i] = corrected[:, :d]
            else:
                # The probe can be wholly off the witnessed span (the failure
                # this repair targets).  With at least two local physical
                # transitions, the leading local secant directions are the
                # only supported replacement; retain the low confidence so
                # downstream tangent supervision does not overstate certainty.
                frames[i] = span
            confidence[i] *= np.exp(-angle / 20.0)
        else:
            confidence[i] *= np.exp(-angle / 90.0)
    return frames, {"_frame_confidence": confidence,
                    "_frame_probe_secant_angle_deg": angles,
                    "_frame_secant_rank": secant_rank.astype(float),
                    "_frame_probe_scale_disagreement": np.asarray([
                        probe_diagnostics.get(i, {}).get("probe_scale_disagreement", np.nan)
                        for i in range(n)], float),
                    "_frame_probe_settle_repeatability": np.asarray([
                        probe_diagnostics.get(i, {}).get("probe_settle_repeatability", np.nan)
                        for i in range(n)], float)}


def _explore_legacy(spec, log=print, ledger_path=None):
    """Advancing-front flood on the manifold, through the declared oracle."""
    from atlas_rom.simulate.oracle import tangent_frame_for_spec

    options = spec.build.sampling
    tangent_rel_tol = options.tangent_rel_tol
    if spec.seeds:
        seeds = [np.asarray(s, float) for s in spec.seeds]
    elif options.seed_reuse:
        seed_q, _frames, _fields, paths, _edges = _load_cloud(
            options.seed_reuse, root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            representation=options.representation,
            body_order=spec.mechanism.body_order)
        if not len(seed_q):
            raise ValueError(f"{spec.name}: seed_reuse {options.seed_reuse!r} is empty")
        index = int(options.seed_index) % len(seed_q)
        seeds = [seed_q[index]]
        log(f"sampling: seeded from {paths[0]} frame {index}/{len(seed_q)}")
    else:
        seeds = None

    oracle, use_hold = _select_oracle(
        spec, options, None if seeds is None else seeds[0], log)
    if ledger_path is not None:
        # The ledger is a sampler artifact, not a routing or topology input.
        # Reopening it makes identical physical measurements reusable across a
        # resumed sampling run while preserving every original observation.
        from atlas_rom.simulate.oracle_base import ProjectionLedger
        oracle.ledger = ProjectionLedger(ledger_path)
    rng = np.random.default_rng(options.seed)
    if seeds is None:
        seeds = [oracle.seed()]
    accepted, frontier, edges = [], [], []
    for seed in seeds:
        q = oracle.project(seed)
        accepted.append(q)
        frontier.append(len(accepted) - 1)

    # Physical-configuration closure state (see CLOSURE_TOL_FACTOR above): a
    # twist vector per accepted sample, all relative to ONE fixed reference
    # pose (the first seed) so distances between any two entries stay
    # comparable for the whole run, plus the physical length of each walked
    # (non-closure) edge, from which the closure tolerance is derived.
    closure_reference = np.asarray(accepted[0], float)
    closure_twists = [pp.pose_twists(np.asarray(p, float)[None, :],
                                     reference=closure_reference)[0]
                      for p in accepted]
    closure_edge_lengths = []

    if spec.dimension == 1:
        # Carry the side of a 1-D continuation explicitly.  Without this tag,
        # alternating min/max selection can reuse the same newly-created point
        # for both fronts and walk back into the seed.
        frontier = [(index, side, 0) for index in frontier for side in (-1, 1)]

    # A recorder can supply an unwrapped actuator angle as a sample field.  The
    # explorer must create that same lift while it walks: every accepted edge is
    # local, so its principal angle increment is unambiguous even when the full
    # configuration loop winds through more than one turn.
    lifted_field = _explored_sample_field(spec)
    lifted_u = ({index: 0.0 for index in range(len(accepted))}
                if lifted_field is not None else None)
    field_predictor = _revolute_field_predictor(spec, lifted_field)

    def coordinate_at(index):
        if lifted_u is not None:
            return lifted_u[index]
        return float(coord_map(np.asarray(accepted[index]).reshape(1, -1))[0, 0])

    def coordinate_near(origin, value):
        """Coordinate of ``value``, using ``origin`` to choose an angle lift."""
        if lifted_field is None:
            return float(coord_map(np.asarray(value).reshape(1, -1))[0, 0])
        _key, body, axis = lifted_field
        return coordinate_at(origin) + _lifted_angle_delta(
            accepted[origin], value, spec.n_bodies, body, axis)

    def inside_domains(value, coordinate=None):
        transforms = np.asarray(value, float).reshape(spec.n_bodies, 3, 4)
        if outside_joint_limits(spec, transforms) is not None:
            return False
        for bound in spec.mechanism.workspace_bounds:
            body = spec.mechanism.body_index(bound.body)
            axis = "xyz".index(bound.axis)
            component = float(transforms[body, axis, 3])
            if component < bound.lo or component > bound.hi:
                return False
        values = (coord_map(np.asarray(value).reshape(1, -1))[0]
                  if coordinate is None else (coordinate,))
        return all(c.topology != "bounded" or c.lo <= component <= c.hi
                   for component, c in zip(values, spec.coordinates))

    # Global coordinates are also hard workspace limits.  Checking them during
    # continuation prevents a bounded chart from spilling into another assembly
    # branch before the later atlas coverage audit sees it.
    coord_map = CoordinateMap(spec).fit(np.asarray(accepted))

    target_label = "unbounded" if options.n_samples is None else str(options.n_samples)
    at_sample_cap = lambda: (options.n_samples is not None
                             and len(accepted) >= options.n_samples)
    log(f"sampling[{options.sampler}]: {len(seeds)} seed(s) projected; "
        f"target {target_label}, step {options.step}, budget {options.max_seconds:.0f}s")

    started = time.perf_counter()
    reject = {"domain": 0, "non_finite": 0, "backward": 0, "spacing": 0,
              "rank": 0}
    frame_cache = {}
    frame_diagnostics = {}
    endpoint_turn = -1
    solver_at = len(accepted) - 1        # sample the solver was last driven to
    adjacency = {}
    # Continuation failures are invisible in the summary counts — set
    # ATLAS_SAMPLE_DEBUG=1 to see which pop, which direction and which realized
    # spacing retired a front.
    debug = bool(os.environ.get("ATLAS_SAMPLE_DEBUG"))
    # Until this loop ends it prints nothing, so a long run — a 3-D manifold can
    # take hours — is indistinguishable from a hung one.  A periodic heartbeat
    # reports the sample count, the rate and the frontier depth, which is enough
    # to tell "still growing" from "grinding on a retired front" and to project
    # whether the budget or the target will stop the run.
    next_heartbeat = started + PROGRESS_SECONDS
    while frontier and not at_sample_cap():
        now = time.perf_counter()
        if now - started > options.max_seconds:
            log(f"sampling: budget exhausted at {len(accepted)} samples")
            break
        if now >= next_heartbeat:
            rate = len(accepted) / max(now - started, 1e-9)
            # Rejections are the difference between "slow per pop" and "fast per
            # pop but throwing most steps away", which need opposite fixes.
            why = ", ".join(f"{k} {v}" for k, v in reject.items() if v) or "none"
            log(f"sampling: {len(accepted)}/{target_label} samples, "
                f"{rate:.3f}/s, frontier {len(frontier)}, rejected [{why}], "
                f"{now - started:.0f}s of {options.max_seconds:.0f}s")
            next_heartbeat = now + PROGRESS_SECONDS
        if spec.dimension == 1:
            # Grow the two ends of a bounded curve first.  FIFO exploration
            # spends the sample budget re-expanding interior points and can hit
            # n_samples before either coordinate endpoint is certified.
            #
            # Stay on one end until it stops growing.  Alternating every pop made
            # each `project()` drive the mechanism the full width of the manifold
            # (the two fronts are at opposite ends), which no fixed drive-iteration
            # budget can complete: the probes then measure leftover travel instead
            # of the tangent, and both signed steps land back INSIDE the explored
            # region, retiring the front while the manifold still continues.
            wanted_side = endpoint_turn
            positions = [i for i, item in enumerate(frontier)
                         if item[1] == wanted_side]
            if not positions:
                wanted_side = -wanted_side
            positions = [i for i, item in enumerate(frontier)
                          if item[1] == wanted_side]
            frontier_u = np.asarray([coordinate_at(frontier[i][0])
                                     for i in positions])
            local = int(np.argmin(frontier_u) if wanted_side < 0
                        else np.argmax(frontier_u))
            point_index, side, stalls = frontier.pop(positions[local])
            grow_low = side < 0
            endpoint_turn = wanted_side
        else:
            point_index = frontier.pop(0)
        grew = False
        point = accepted[point_index]
        base_coordinate = coordinate_at(point_index) if spec.dimension == 1 else None
        # `project()` drives from wherever the solver currently stands, so a pop
        # that moves elsewhere on the manifold has to travel there first.  Spend
        # that travel here, once, along explored ground, instead of inside every
        # tangent probe.
        if point_index != solver_at:
            via = _tree_path(adjacency, solver_at, point_index)
            if debug and len(via) > 2:
                log(f"relocate {solver_at} -> {point_index} "
                    f"via {len(via) - 2} explored sample(s)")
            for node in via[1:-1]:
                oracle.project(accepted[node])
        solver_at = point_index
        for _ in range(RESTORE_ATTEMPTS):
            landed = oracle.project(point)
            restore_spacing = (abs(coordinate_near(point_index, landed) - base_coordinate)
                               if spec.dimension == 1 else
                               _spacing(spec, coord_map, landed, point))
            if restore_spacing \
                    < PROGRESS_FLOOR * float(options.step):
                break
        basis, _sig, dim, measurement = tangent_frame_for_spec(
            spec, oracle, point, rel_tol=tangent_rel_tol, return_diagnostics=True)
        frame_diagnostics[point_index] = measurement
        if dim > spec.dimension:
            # The probe's singular-value gap is measured through a stochastic
            # settle, so it occasionally reports one direction too many.  Beyond
            # the declared dimension those columns are projection noise, and a
            # full-size step along a noise direction in a many-body packing is an
            # inconsistent drive target — the solver's line search diverges and
            # takes the whole process down.  Keep the leading declared directions.
            basis, dim = np.asarray(basis)[:, :spec.dimension], spec.dimension
        tree = cKDTree(np.asarray(accepted))
        # Every tangent axis must be probed in both directions.  Independent
        # random draws can repeat one sign (fatal for a 1-D frontier: both ends
        # may probe inward and falsely report completion after only a few points).
        directions = np.vstack((np.eye(dim), -np.eye(dim))) if dim else []
        if dim > 1:
            directions = directions[rng.permutation(len(directions))]
        elif spec.dimension == 1 and dim == 1:
            # A side-tagged 1-D front only ever accepts the outward step, and the
            # sign of a tangent basis vector is arbitrary.  Read the outward sign
            # off the coordinate derivative (pure arithmetic) instead of spending
            # a settle to discover the inward one and reject it — that also left
            # the solver sitting inside the explored region before the next pop.
            probe = np.asarray(basis)[:, 0]
            eps = 1e-4
            slope = coordinate_near(point_index, point + eps * probe) - base_coordinate
            outward = -1.0 if grow_low else 1.0
            directions = np.asarray([[outward * np.sign(slope) or 1.0]])
        for direction in directions:
            tangent = np.asarray(basis) @ direction
            # A stepping direction that leaves some bodies' block at zero to
            # first order is a signal, not noise: on a mechanism with more than
            # one assembly branch through the same region, an unconstrained
            # least-norm correction can undo exactly that motion by drifting
            # onto a cheaper, different branch (measured on `rotate` — see
            # `IdealJointOracle.project`'s `hold` docstring).  Holding those
            # bodies fixed during the correction keeps the step on the branch
            # the tangent direction actually describes.  A no-op whenever every
            # body has a nonzero component, which is the ordinary case.
            hold = oracle.stationary_bodies(tangent) if use_hold else None
            body_cap = (MAX_BODY_DISPLACEMENT
                        if options.predictor_body_displacement is None
                        else float(options.predictor_body_displacement))
            if body_cap <= 0.0:
                raise ValueError("sampling.predictor_body_displacement must be positive")
            cap = _predictor_cap(tangent, spec.n_bodies, body_cap)
            tangent_step = min(float(options.step), cap)
            if spec.dimension == 1:
                # For a declared 1-D coordinate, `step` is intrinsic spacing.
                # A unit packed-pose tangent has a scale that grows with the
                # number of bodies, so using it directly made many samples
                # separated in q while nearly identical in u.
                eps = 1e-4
                base_u = base_coordinate
                probe_u = coordinate_near(point_index, point + eps * tangent)
                du_dq = abs(probe_u - base_u) / eps
                if du_dq > 1e-10:
                    tangent_step = min(cap, float(options.step) / du_dq)
                # An explicit revolute driver is a better predictor than an
                # ambient tangent: command precisely one intrinsic angle and
                # let its contact-coupled followers settle.  The tangent is
                # still measured above for rank validation and fallback.
                if field_predictor is not None:
                    tangent_step = float(options.step)
            accept = None
            for _attempt in range(STEP_ATTEMPTS):
                if field_predictor is None:
                    candidate = point + tangent_step * tangent
                else:
                    body, pivot, axis = field_predictor
                    outward = -1.0 if grow_low else 1.0
                    candidate = _rotate_target_about_pivot(
                        point, spec.n_bodies, body, pivot, axis,
                        outward * tangent_step)
                candidate_coordinate = (coordinate_near(point_index, candidate)
                                        if spec.dimension == 1 else None)
                if not inside_domains(candidate, candidate_coordinate):
                    reject["domain"] += 1
                    break
                projected = (oracle.project(candidate, hold=hold) if use_hold
                            else oracle.project(candidate))
                if not np.isfinite(projected).all():
                    reject["non_finite"] += 1
                    break
                projected_coordinate = (coordinate_near(point_index, projected)
                                        if spec.dimension == 1 else None)
                if not inside_domains(projected, projected_coordinate):
                    reject["domain"] += 1
                    break
                if spec.dimension == 1:
                    outward = -1.0 if grow_low else 1.0
                    if outward * (projected_coordinate - base_coordinate) <= 0.0:
                        reject["backward"] += 1
                        break
                    existing_u = np.asarray([coordinate_at(index)
                                             for index in range(len(accepted))])
                    realized = float(np.min(np.abs(existing_u
                                                   - projected_coordinate)))
                else:
                    realized = float(tree.query(projected, k=1)[0])
                # The spacing the projection actually delivered is the only
                # measure of whether this step was the right size.
                if realized > OVERSHOOT_FACTOR * float(options.step) \
                        or realized < 0.8 * float(options.step):
                    rescale = float(np.clip(
                        float(options.step) / max(realized, 1e-12), 0.5, 2.0))
                    retry = min(cap, tangent_step * rescale)
                    if realized >= PROGRESS_FLOOR * float(options.step) and (
                            accept is None
                            or abs(realized - float(options.step))
                            < abs(accept[1] - float(options.step))):
                        # A mis-sized but genuinely new configuration is still a
                        # sample; hold the closest-to-`step` one seen so far.
                        accept = (projected, realized, projected_coordinate)
                    if debug:
                        log(f"  attempt ts={tangent_step:.4f} cap={cap:.4f} "
                            f"realized={realized:.5f} -> retry {retry:.4f}")
                    if abs(retry - tangent_step) > 1e-12 * max(retry, 1.0):
                        tangent_step = retry
                        continue
                    # Already at the cap (or the rescale is a no-op): a farther
                    # sample is not reachable from here, so take what progress
                    # this direction did make rather than abandoning the front.
                    if accept is None:
                        reject["spacing"] += 1
                    break
                accept = (projected, realized, projected_coordinate)
                break
            if accept is None:
                continue
            projected = accept[0]
            # A bounded coordinate can terminate at a rank-changing pose. Do
            # not put that singular stratum into a regular chart whose declared
            # dimension is one; it is a boundary witness, not a 1-D sample.
            frame, _sig, frame_dim, measurement = tangent_frame_for_spec(
                spec, oracle, projected, rel_tol=tangent_rel_tol, return_diagnostics=True)
            if frame_dim < spec.dimension:
                reject["rank"] += 1
                continue
            # Contact projection can expose compliant/noise directions beyond a
            # case's declared kinematic manifold dimension.  Continuation above
            # already keeps only the declared leading directions; applying a
            # stricter exact-rank test here used to discard every such candidate
            # and leave a one-sample atlas.
            frame = np.asarray(frame)[:, :spec.dimension]

            # Physical closure check: does this PROJECTED candidate already
            # coincide with an existing, already-accepted sample in physical
            # pose space -- regardless of how far its lifted/local coordinate
            # says it travelled? This is a local proximity test only (never a
            # winding-number, period, or Betti-number calculation): a tight
            # tolerance, scaled to this walk's own realized step length,
            # naturally excludes the point we just stepped FROM (which sits a
            # full step away) while catching a genuine physical revisit (e.g.
            # a periodic mechanism past its first turn).
            closure_target = None
            if len(closure_edge_lengths) >= CLOSURE_MIN_EDGES:
                closure_tol = CLOSURE_TOL_FACTOR * float(np.median(closure_edge_lengths))
                proj_twist = pp.pose_twists(np.asarray(projected, float)[None, :],
                                            reference=closure_reference)[0]
                twist_arr = np.asarray(closure_twists)
                dist = np.sqrt(np.mean((twist_arr - proj_twist[None, :]) ** 2, axis=1))
                dist[point_index] = np.inf   # never "close" onto the point we came from
                idx = int(np.argmin(dist))
                if dist[idx] <= closure_tol:
                    closure_target = idx

            if closure_target is not None:
                # Do not append an indistinguishable duplicate sample: link
                # the closure and retire this continuation instead. No new
                # frontier entry is pushed, so the front this direction was
                # growing simply has nothing left to pop next time.
                edges.append((point_index, closure_target))
                adjacency.setdefault(point_index, []).append(closure_target)
                adjacency.setdefault(closure_target, []).append(point_index)
                if debug:
                    log(f"closure: pop {point_index} -> existing {closure_target} "
                        f"(physical revisit, dist={dist[closure_target]:.3g} "
                        f"<= tol {closure_tol:.3g}); retiring this front")
                grew = True
                continue

            accepted.append(projected)
            target_index = len(accepted) - 1
            if lifted_u is not None:
                lifted_u[target_index] = accept[2]
            # The frame was just measured at this exact configuration; the final
            # pass would otherwise re-probe every sample a second time.
            frame_cache[target_index] = np.asarray(frame, float)
            frame_diagnostics[target_index] = measurement
            frontier.append((target_index, side, 0) if spec.dimension == 1
                            else target_index)
            edges.append((point_index, target_index))
            adjacency.setdefault(point_index, []).append(target_index)
            adjacency.setdefault(target_index, []).append(point_index)
            new_twist = pp.pose_twists(np.asarray(projected, float)[None, :],
                                       reference=closure_reference)[0]
            closure_twists.append(new_twist)
            closure_edge_lengths.append(
                float(np.sqrt(np.mean((new_twist - closure_twists[point_index]) ** 2))))
            solver_at = target_index
            grew = True
            tree = cKDTree(np.asarray(accepted))
            if at_sample_cap():
                break
        # A 1-D front has exactly one live point per side, so a pop that grows
        # nothing used to END that side for good.  `project()` is a stochastic
        # settle, not an oracle of where the manifold stops: one failed attempt
        # is not a boundary certificate.  Put the point back and let a later,
        # differently-seeded attempt try again — a real boundary still stops the
        # side, it just takes MAX_STALLS attempts to say so.
        if spec.dimension == 1 and not grew:
            if debug:
                log(f"pop side={side} u={base_coordinate:.5f} dim={dim} grew "
                    f"nothing (stall {stalls + 1}/{MAX_STALLS})")
            if stalls + 1 < MAX_STALLS:
                frontier.append((point_index, side, stalls + 1))
            # Hand the solver to the other end rather than re-driving this one
            # from the same state that just failed.
            endpoint_turn = -side
    elapsed = time.perf_counter() - started
    log(f"sampling: {len(accepted)} samples, {sum(reject.values())} rejected "
        f"({', '.join(f'{k} {v}' for k, v in reject.items() if v) or 'none'}), "
        f"{elapsed:.1f}s ({len(accepted) / max(elapsed, 1e-9):.2f}/s)")
    if not frontier and not at_sample_cap():
        log(f"sampling: frontier emptied at {len(accepted)} samples: the "
            f"explored extent is where continuation stopped, not the budget")
    # A tangent rank that disagrees with the declared dimension is a singularity
    # or a wrong spec, and either way it must not stay buried in the per-call
    # log.  Measure-zero sets get hit occasionally; a large share does not.
    mismatches = getattr(oracle, "dim_mismatches", 0)
    if mismatches:
        log(f"sampling: {mismatches} configuration(s) measured a dimension other "
            f"than the declared {spec.build.expect.dimension}: singular set if "
            f"a handful, wrong expect.dimension if it is most of them")
    # Frames belong to samples, not frontier-pop events.  The old implementation
    # accumulated one frame per expanded point and then discarded the mismatched
    # array entirely.  Certified mesh reconstruction requires a tangent frame at
    # every accepted configuration, so evaluate them after exploration closes.
    accepted = np.asarray(accepted)
    frames = []
    expected_dim = spec.dimension
    for index, point in enumerate(accepted):
        if index in frame_cache:
            frames.append(frame_cache[index])
            continue
        basis, _sig, dim, measurement = tangent_frame_for_spec(
            spec, oracle, point, rel_tol=tangent_rel_tol, return_diagnostics=True)
        if dim < expected_dim:
            raise ValueError(
                f"sampling: sample {index} has tangent dimension {dim}, "
                f"need at least {expected_dim}")
        frames.append(np.asarray(basis, float)[:, :expected_dim])
        frame_diagnostics[index] = measurement
    fields = ({lifted_field[0]: np.asarray([lifted_u[index]
                                             for index in range(len(accepted))])}
              if lifted_field is not None else {})
    frames, frame_fields = _reconcile_frames_with_secants(
        accepted, np.asarray(frames), np.asarray(edges, int).reshape(-1, 2), frame_diagnostics)
    fields.update(frame_fields)
    return accepted, frames, np.asarray(edges, int).reshape(-1, 2), fields


def _seed_resume_checkpoint(source: str, checkpoint_dir: str, *, seed: int) -> None:
    """Re-open a physical front using only recorded poses and witnessed edges."""
    source_path = os.path.join(source, dataset.SAMPLES) if os.path.isdir(source) else source
    with np.load(source_path, allow_pickle=False) as saved:
        poses = np.asarray(saved["q"], float)
        edges = np.asarray(saved["edges"] if "edges" in saved.files else [], int).reshape(-1, 2)
    if len(poses) < 2:
        raise ValueError("resume source needs at least two physical observations")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # A bare re-open: only recorded poses/edges survive.  ``explore``'s resume
    # reader also expects the per-node probe bookkeeping keys, so write them as
    # empty -- it re-measures every node's frame and re-seeds the frontier from
    # scratch (see the ``index not in records`` bootstrap there).
    np.savez_compressed(os.path.join(checkpoint_dir, "frontier_state.npz"),
                        poses=poses, edges=edges,
                        frontier=np.asarray([], dtype=object),
                        probed=np.asarray({}, dtype=object),
                        inference_probes=np.asarray({}, dtype=object),
                        records=np.asarray({}, dtype=object),
                        dirty=np.asarray([], dtype=int),
                        cone_angles=np.asarray({}, dtype=object))
    rng_state = np.random.default_rng(seed).bit_generator.state
    with open(os.path.join(checkpoint_dir, "frontier_state.json"), "w", encoding="utf-8") as fh:
        import json
        json.dump({"rng_state": rng_state, "resumed_from": os.path.abspath(source_path)}, fh, indent=2)


def _explore(spec, log=print, ledger_path=None, *, checkpoint_dir=None, resume_from=None,
             resume_and_repair=False):
    """Shared physical front for analytic and contact oracle backends."""
    from atlas_rom.simulate.oracle_base import LegacyPhysicalOracleAdapter, PhysicalObservation, ProjectionLedger
    from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions, explore

    options = spec.build.sampling
    solver_oracle, _use_hold = _select_oracle(spec, options, None, log)
    if ledger_path is not None:
        solver_oracle.ledger = ProjectionLedger(ledger_path)
    def physical_domain(result):
        transforms = np.asarray(result, float).reshape(spec.n_bodies, 3, 4)
        diagnostics = physical_domain_violation(spec, transforms)
        return diagnostics is None, diagnostics or {}

    backend_validity = solver_oracle.response_validity

    def response_validity(result):
        valid, diagnostics = backend_validity(result)
        return (valid, diagnostics) if not valid else physical_domain(result)

    solver_oracle.response_validity = response_validity
    physical_oracle = LegacyPhysicalOracleAdapter(solver_oracle)
    if resume_from is not None:
        if checkpoint_dir is None:
            raise ValueError("resuming a physical front requires a checkpoint directory")
        # An interrupted physical run already has the complete frontier
        # checkpoint (poses, open cones, and per-node probes).  Preserve it;
        # only a completed external sample set needs conversion to a bare
        # frontier checkpoint.
        source_frontier = os.path.join(resume_from, "frontier_state.npz")
        target_frontier = os.path.join(checkpoint_dir, "frontier_state.npz")
        if not (os.path.samefile(source_frontier, target_frontier)
                if os.path.exists(source_frontier) and os.path.exists(target_frontier)
                else os.path.abspath(source_frontier) == os.path.abspath(target_frontier)):
            _seed_resume_checkpoint(resume_from, checkpoint_dir, seed=options.seed)
        log(f"sampling: resuming physical frontier from {resume_from}")
    seed_values = [np.asarray(seed, float) for seed in spec.seeds]
    if not seed_values and options.seed_reuse:
        seed_q, _frames, _fields, paths, _edges = _load_cloud(
            options.seed_reuse, root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            representation=options.representation,
            body_order=spec.mechanism.body_order)
        if not len(seed_q):
            raise ValueError(f"{spec.name}: seed_reuse {options.seed_reuse!r} is empty")
        seed_values = [seed_q[int(options.seed_index) % len(seed_q)]]
        log(f"sampling: seeded from {paths[0]} frame {options.seed_index}/{len(seed_q)}")
    seeds = ([PhysicalObservation(value) for value in seed_values]
             if seed_values else None)
    q, edges, frames, _records = explore(
        physical_oracle,
        options=AdvancingFrontOptions(step=options.step,
                                      max_samples=options.n_samples,
                                      max_seconds=options.max_seconds,
                                      probe_scale=options.tangent_eps,
                                      rank_gap_ratio=options.rank_gap_ratio,
                                      checkpoint_seconds=options.checkpoint_seconds,
                                      spacing_fraction=0.6,
                                      seed=options.seed,
                                      report_every=50,
                                      max_probes_per_node=options.max_probes_per_node),
        seeds=seeds, log=log, checkpoint_dir=checkpoint_dir, resume_from=resume_from,
        repair_inference=resume_and_repair)
    cache = solver_oracle.ledger.statistics
    log("projection cache: " + ", ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in cache.items()))
    if len(frames) != len(q):
        raise RuntimeError("physical advancing front did not produce one frame per pose")
    return q, np.asarray(frames), edges, {}


def prepare(case, work: str, root: str, *, log=print, sdf_system=None,
            write_ledger: bool | None = None, resume_from: str | None = None,
            resume_and_repair: bool = False) -> dict:
    """Run the selected physical sampler and write the common SampleSet artifact.

    Both modes accept an externally recorded cloud through ``reuse``.  Absent
    that, both run the zero-prior advancing-front explorer above, dispatched
    to the case's declared `sampling.sampler` oracle.

    ``write_ledger`` controls the on-disk ``projection_ledger.jsonl`` audit
    trail.  It is a JSON line per oracle probe (tens per accepted sample), so on
    a dim-3 case it reaches hundreds of MB and dominates the wall clock.  The
    in-memory reuse cache is kept either way.  ``None`` defers to the
    ``ATLAS_ROM_NO_LEDGER`` environment variable (set to disable), else on.
    """
    if write_ledger is None:
        write_ledger = not os.environ.get("ATLAS_ROM_NO_LEDGER")
    ledger_path = (os.path.join(work, "projection_ledger.jsonl")
                   if write_ledger else None)
    if sdf_system is not None:
        # Pure region path: its decisions consume only SDF responses.  It uses
        # the same physical front as analytic/contact; SDF tracing remains in
        # the selected backend.
        from atlas_rom.simulate.oracle_base import PhysicalObservation, ProjectionLedger
        from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions, explore
        if sdf_system.observe is None:
            raise ValueError("SDF sampling requires a physical pose observation callback")
        oracle, _ = _select_oracle(case, case.build.sampling, None, log,
                                   sdf_system=sdf_system)
        oracle.backend.ledger = ProjectionLedger(ledger_path)
        observations = [PhysicalObservation(oracle._remember(seed),
                                            {"state": np.asarray(seed, float)})
                        for seed in sdf_system.seeds]
        poses, edges, frames, _records = explore(
            oracle,
            options=AdvancingFrontOptions(step=sdf_system.step,
                                          max_samples=sdf_system.max_samples,
                                          max_seconds=sdf_system.max_seconds,
                                          probe_scale=sdf_system.tangent_eps,
                                          rank_gap_ratio=case.build.sampling.rank_gap_ratio),
            seeds=observations)
        cache = oracle.backend.ledger.statistics
        log("projection cache: " + ", ".join(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in cache.items()))
        states = np.asarray([oracle._state_by_pose.get(
            np.asarray(pose, float).reshape(-1).tobytes(), pose) for pose in poses], float)
        observed = [np.asarray(sdf_system.observe(state), float) for state in states]
        if any(pose.shape != observed[0].shape for pose in observed):
            raise ValueError("SDF observer returned inconsistent physical-pose shapes")
        q = np.asarray(observed, float).reshape(len(states), -1)
        if q.shape[1] != 12 * case.n_bodies:
            raise ValueError("SDF observer must emit one 3x4 transform per driven body")
        if len(frames) != len(q):
            raise RuntimeError("shared physical front returned incomplete SDF frames")
        os.makedirs(work, exist_ok=True)
        # There are deliberately no supplied chart coordinates: the default
        # physical-chart encoder owns that representation.
        dataset.save_samples(work, q, np.empty((len(q), 0)), frames=frames, edges=edges,
                             fields={})
        import json
        with open(dataset.path_in(work, dataset.COORDINATE_STATE), "w", encoding="utf-8") as fh:
            json.dump({"representation": "physical_pose_observations",
                       "fingerprint": sdf_system.fingerprint}, fh)
        if sdf_system.state_from_observation is None:
            raise ValueError("SDF sampling requires pose-to-state evidence for certified chord completion")
        from atlas_rom.sample import sample_graph

        class ObservedSDFResidual:
            def residual(self, pose):
                state = np.asarray(sdf_system.state_from_observation(pose), float)
                if sdf_system.surface:
                    try:
                        projected = oracle.project(state)
                    except (ValueError, RuntimeError):
                        # A midpoint outside an open physical surface subset
                        # (for example, through a puncture) is an off-sheet
                        # chord.  It is evidence against the edge, not a
                        # sampling error that aborts certification.
                        return np.full(np.asarray(pose, float).size, np.inf)
                    return np.asarray(sdf_system.observe(projected), float).reshape(-1) - np.asarray(pose, float)
                return np.asarray([max(float(sdf_system.sdf(state)) - sdf_system.tolerance, 0.0)])

        certified, graph_report = sample_graph.certified_edges(
            q, edges, ObservedSDFResidual(), k=16, log=log)
        sample_graph.save(os.path.join(work, sample_graph.ARTIFACT), certified, graph_report)
        from atlas_rom.sample import new_dataset
        built = new_dataset.build(work, log=log)
        log(f"sampling[sdf]: {len(q)} states, {len(edges)} witnessed edges")
        return {"sampler": "sdf", "n_samples": len(q), "n_edges": len(edges),
                "dimension": int(frames.shape[2]),
                "sdf_queries": oracle.query_count,
                "frame_source": "directional_response_pose_secants", "new_dataset": built}
    options = case.build.sampling
    if options.reuse:
        q, frames, edges, fields = _load_recording(options.reuse, root, case.mechanism.body_order)
        if q.shape[1] != 12 * case.n_bodies:
            raise ValueError(f"{case.name}: sampler produced {q.shape[1]} pose columns, "
                             f"need {12 * case.n_bodies}")
        u = _coordinates(case, fields)
        dataset.save_samples(work, q, u, frames=frames, edges=edges, fields=fields)
        log(f"sampling[{options.sampler}]: {len(q)} states, {len(edges)} witnessed edges")
        return {"sampler": options.sampler, "n_samples": len(q), "n_edges": len(edges),
                "dimension": int(frames.shape[2])}

    if options.fault_tolerant:
        raise NotImplementedError(
            f"{case.name}: sampling.fault_tolerant needs the archived continuation "
            "supervisor, which is not part of the production explorer yet")

    q, frames, edges, fields = _explore(case, log=log, ledger_path=ledger_path,
                                        checkpoint_dir=work, resume_from=resume_from,
                                        resume_and_repair=resume_and_repair)
    expected = 12 * case.n_bodies
    if q.shape[1] != expected:
        raise ValueError(
            f"{case.name}: samples carry {q.shape[1]} config columns but the "
            f"mechanism declares {case.n_bodies} driven bodies (12*n = {expected}). "
            f"The spec's body order and the sample source disagree.")

    # Exploration emits physical observations only.  Declared coordinates are
    # downstream chart metadata, never a sampling input or sampling output.
    u = np.empty((len(q), 0), dtype=float)
    dataset.save_samples(work, q, u, frames=frames, edges=edges, fields={})
    import json
    with open(dataset.path_in(work, dataset.COORDINATE_STATE), "w",
              encoding="utf-8") as fh:
        json.dump({"representation": "physical_pose_observations",
                   "strategy": options.exploration_strategy,
                   "oracle_backend": options.oracle_backend or options.sampler}, fh)
    # Certification is evaluator-only: it consumes physical residual evidence
    # along witnessed chords, never an analytic tangent/Jacobian or declared
    # configuration dimension.  Keep the same artifact contract as the SDF
    # path so downstream training cannot distinguish a privileged branch.
    from atlas_rom.sample import new_dataset, sample_graph
    certificate_oracle, _unused = _select_oracle(case, options, None, log)
    certified, graph_report = sample_graph.certified_edges(q, edges, certificate_oracle, k=16, log=log)
    sample_graph.save(os.path.join(work, sample_graph.ARTIFACT), certified, graph_report)
    built = new_dataset.build(work, log=log)
    log(f"sampling[{options.sampler}]: {len(q)} states, {len(edges)} witnessed edges")
    return {"sampler": options.sampler, "n_samples": len(q), "dimension": int(frames.shape[2]),
            "n_edges": len(edges), "new_dataset": built}
