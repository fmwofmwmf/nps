"""factored.py — sample a Riemannian product M = A x B factor by factor.

On a product, a pose is fixed by one point of each factor, so the sample set
can be one closed front per factor plus a certified way of composing them,
instead of one front over the whole product.  The sample count becomes a SUM
over factors instead of a product.  Everything here is inferred from
`PhysicalSystem` responses; nothing reads a declared coordinate, dimension,
joint, body role, or evaluator reference.

    1. DETECT    a bounded partial front around the seed, then
                 `product_split.detect_product_split` on its inferred frames.
                 The split is carried to every node by transport.
    2. CERTIFY   `product_split.certify_split` at the seed: A-then-B closes on
                 B-then-A under refinement; an interval factor's width is the
                 same wherever it is measured.
    3. CLASSIFY  each body by evidence: does its pose block move along A's
                 directions, B's, both, or neither, at any partial-front node.
    4. LEAVES    the UNCHANGED advancing front, run on a `LeafSystem` that
                 confines commands to one factor's leaf through the seed.  A
                 projected step drifts off a leaf at second order; the wrapper
                 Newton-corrects every response by holding the other factor's
                 own bodies (those that move with it alone) at their seed
                 poses.  The front's own `frontier_complete` then means "this
                 leaf closed".
    5. COMPOSE   cross samples off both leaves, placed farthest-point over
                 (a, b), each found by Newton on the A-only and B-only blocks.
                 The model under test, per body,
                     T(a, b) = T(a0, b) T(a0, b0)^-1 T(a, b0)
                 (or the other order), must reproduce them to the Newton
                 tolerance, and keep doing so as that tolerance refines.

`frontier_complete` of the factored audit is the AND of: a product split
detected, the stage-3 certificate passing, every leaf front complete, and the
composition certificate passing.  Any failure returns `factored: False` with
the reason, so the caller samples the full product instead.

METRIC.  "Product" is a statement about a metric.  By default it is the packed
pose metric.  With `FactoredOptions.body_moments` (each body's 4x4 second
moment  Mbar = int rho [p;1][p;1]^T) it is the KINETIC metric: a body's pose
rows move a point p by dA [p;1], so its kinetic energy is sum_r dA_r Mbar dA_r^T,
a flat ambient metric that is the pose metric after each block row is
multiplied by a Cholesky factor of Mbar.  Detection (1), the orthogonality
test of the refined split, and the stage-3 walks (2) then run in those
weighted coordinates, so a certified split is one the mass matrix also splits
along.  Leaves (4) and composition (5) do not depend on a metric -- composition
is about body motions -- and stay in pose coordinates.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions, explore
from atlas_rom.sample.product_split import (
    ProductSplitOptions, certify_split, detect_product_split, infer_frame, metric_frame, transport)
from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse

BLOCK = 12                                   # packed 3x4 pose per body


@dataclass
class FactoredOptions:
    detect_samples: int = 200                # partial front size for detection
    detect_seconds: float | None = 900.
    body_motion_rel: float = 1e-4            # below this fraction of a factor's largest block motion = still
    coarse_ratio: float = .25                # first pass, on the approximate split
    anchor_tol: float = 1e-8                 # Newton tolerance on anchor blocks (inf-norm)
    refine_factor: float = 1e-2              # composition refinement: tolerance x this
    max_newton: int = 40
    cross_min: int = 50
    cross_fraction: float = .05
    refine_pairs: int = 10
    order_gap: float = 1e3                   # wrong composition order must be this much worse
    certify: bool = True
    certify_substeps: tuple[int, ...] = (2, 4, 8)
    gap_ratio_pass: float = .6
    split: ProductSplitOptions = dataclasses.field(default_factory=ProductSplitOptions)
    # (n_bodies, 4, 4) second moments in the system's body order: detect and
    # certify the split in the kinetic metric.  None = packed pose metric.
    body_moments: np.ndarray | None = None
    # Flat-case body null-space split: a body's still directions must sit
    # below this spectral gap; candidate factors this orthogonal (max |cos|).
    null_gap: float = 1e3
    null_orth_tol: float = 1e-3


def _key(pose) -> bytes:
    return np.asarray(pose, float).reshape(-1).tobytes()


def _orth(matrix: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.svd(matrix, full_matrices=False)[0][:, :k]


# ── metric ───────────────────────────────────────────────────────────────────
class KineticMetric:
    """Packed poses <-> kinetic coordinates: every row of body i's 3x4 block
    times L_i, where Mbar_i = L_i L_i^T.  Euclidean length in kinetic
    coordinates is sqrt(2 x kinetic energy) of the pose change."""

    def __init__(self, moments):
        moments = np.asarray(moments, float)
        self.L = np.linalg.cholesky(moments)
        self.L_inv = np.linalg.inv(self.L)

    def _apply(self, M, x):
        """x: (ambient,) or (ambient, k)."""
        x = np.asarray(x, float)
        flat = x.reshape(x.shape[0], -1)
        n = len(M)
        return np.einsum("nrck,ncd->nrdk", flat.reshape(n, 3, 4, -1), M).reshape(x.shape)

    def weigh(self, x):
        return self._apply(self.L, x)

    def unweigh(self, x):
        return self._apply(self.L_inv, x)


def block_columns(bodies) -> np.ndarray:
    return np.concatenate([BLOCK * int(b) + np.arange(BLOCK) for b in bodies]) if len(bodies) \
        else np.empty(0, int)


# ── 3: body classes ──────────────────────────────────────────────────────────
def classify_bodies(node_bases, n_bodies: int, rel: float):
    """Per factor, every body's largest block motion along that factor's
    directions over all nodes; a body 'moves with' a factor when that exceeds
    `rel` of the factor's largest.  Returns per-body labels and the bodies
    that move with each factor ALONE."""
    nodes = [b for b in node_bases if b is not None]
    n_factors = len(nodes[0])
    motion = np.zeros((n_factors, n_bodies))
    for bases in nodes:
        for f, basis in enumerate(bases):
            blocks = np.asarray(basis).reshape(n_bodies, BLOCK, -1)
            motion[f] = np.maximum(motion[f], np.linalg.norm(blocks, axis=(1, 2)))
    moves = motion > rel * motion.max(axis=1, keepdims=True)
    labels = []
    for i in range(n_bodies):
        which = [f for f in range(n_factors) if moves[f, i]]
        labels.append("still" if not which else ("mixed" if len(which) > 1 else f"factor{which[0]}"))
    only = [[i for i in range(n_bodies) if moves[f, i] and moves[:, i].sum() == 1]
            for f in range(n_factors)]
    return {"labels": labels, "only": only, "motion": motion.tolist()}


def coarse_only(node_bases, n_bodies: int, ratio: float):
    """Bodies whose motion along the other factor is at most `ratio` of their
    motion along this one, at every node.  Tolerates a split that is only
    approximately right (the boundary fit is), which `classify_bodies` at a
    tight cut does not."""
    nodes = [b for b in node_bases if b is not None]
    worst = np.zeros((2, n_bodies))
    for bases in nodes:
        motion = np.stack([np.linalg.norm(np.asarray(b).reshape(n_bodies, BLOCK, -1), axis=(1, 2))
                           for b in bases])
        for f in range(2):
            worst[f] = np.maximum(worst[f], motion[1 - f] / np.maximum(motion[f], 1e-300))
    return [[i for i in range(n_bodies) if worst[f, i] <= ratio] for f in range(2)]


def body_groups(frames, n_bodies: int, gap_ratio: float, overlap_tol: float = 1e-4,
                strong: float = .1):
    """A split read off the pose blocks themselves, for DECOUPLED subsystems.

    Per node, each body's block of the tangent frame has a row space: the
    tangent directions that move it.  Bodies whose row spaces overlap at any
    node move together and are grouped (union-find).  When two or more groups
    remain, their spans are orthogonal and fill the tangent space at every
    node, the manifold is a product extrinsically -- no curvature or boundary
    evidence needed.  A body that moves with both factors (a steered, driven
    wheel) overlaps every group and merges them into one, so this finds
    nothing there and the caller falls back to `detect_product_split`.

    A body barely moving at a node (a piston at dead centre) has a block whose
    direction is mostly measurement noise, so overlap is only tested where
    both bodies move at least `strong` of their own peak, and a group's span
    is taken from its summed block Gram, where such bodies weigh little.

    Returns None, or (groups, node_bases) with node_bases[k] = [ambient basis
    of group 0, ambient basis of all other groups] -- two factors."""
    n, _, d = frames.shape
    parent = list(range(n_bodies))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    blocks = frames.reshape(n, n_bodies, BLOCK, d)
    strength = np.linalg.norm(blocks, axis=(2, 3))                    # (n, bodies)
    peak = strength.max(axis=0)
    spaces = []
    for k in range(n):
        rows = []
        for i in range(n_bodies):
            _, s, vt = np.linalg.svd(blocks[k, i], full_matrices=False)
            rank = int(np.sum(s > s[0] / gap_ratio)) if s[0] > 1e-12 else 0
            rows.append(vt[:rank].T)
        spaces.append(rows)
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                if strength[k, i] < strong * peak[i] or strength[k, j] < strong * peak[j]:
                    continue
                if rows[i].shape[1] and rows[j].shape[1] and                         np.abs(rows[i].T @ rows[j]).max() > overlap_tol:
                    parent[find(i)] = find(j)
    moving = [i for i in range(n_bodies) if peak[i] > 1e-12]
    roots = sorted({find(i) for i in moving})
    groups = [[i for i in moving if find(i) == r] for r in roots]
    if len(groups) < 2:
        return None
    node_bases = []
    for k in range(n):
        spans = []
        for members in (groups[0], [i for g in groups[1:] for i in g]):
            gram = sum(blocks[k, i].T @ blocks[k, i] for i in members)
            w, V = np.linalg.eigh(gram)
            spans.append(V[:, w > w.max() / gap_ratio ** 2])
        if sum(s.shape[1] for s in spans) != d or np.abs(spans[0].T @ spans[1]).max() > overlap_tol:
            return None
        node_bases.append([frames[k] @ s for s in spans])
    return groups, node_bases


def body_null_split(frame, n_bodies: int, gap: float, weigh=None, orth_tol: float = 1e-3):
    """Propose a two-factor split from the bodies alone, at one node.

    On a product A x B, a body that moves with B alone is still along every
    direction of A: its tangent block's null space IS A.  So collect each
    body's still directions (null space under a decisive spectral gap), group
    bodies with the same one, and look for two groups whose null spaces are
    complementary and orthogonal in the metric `weigh` maps into (None =
    pose).  Returns {"only": [bodies moving with factor 0 alone, factor 1
    alone] or None, "dims": [dim A, dim B], "report": ...}, larger factor
    first.  A proposal only: `refine_split`, stage 3 and the composition
    certificate decide."""
    d = frame.shape[1]
    blocks = frame.reshape(n_bodies, BLOCK, d)
    peak = float(np.linalg.norm(blocks, axis=(1, 2)).max())
    groups = []                                            # [null basis (d, k), bodies]
    for i in range(n_bodies):
        if np.linalg.norm(blocks[i]) <= 1e-9 * peak:
            continue                                       # still everywhere: says nothing
        _, s, vt = np.linalg.svd(blocks[i], full_matrices=True)
        s = np.concatenate((s, np.zeros(d - len(s))))
        cut = [r for r in range(1, d) if s[r - 1] >= gap * max(s[r], 1e-300)]
        if not cut:
            continue                                       # no still directions, or no clean gap
        null = vt[cut[0]:].T
        for group in groups:
            basis = group[0]
            if basis.shape == null.shape and np.linalg.norm(null - basis @ (basis.T @ null)) < 10. / gap:
                group[1].append(i)
                break
        else:
            groups.append([null, [i]])
    candidates = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            na, nb = groups[a][0], groups[b][0]
            if na.shape[1] + nb.shape[1] != d:
                continue
            A, B = frame @ na, frame @ nb
            if weigh is not None:
                A, B = weigh(A), weigh(B)
            cos = float(np.abs(_orth(A, na.shape[1]).T @ _orth(B, nb.shape[1])).max())
            if cos <= orth_tol:
                candidates.append((cos, a, b))
    report = {"groups": [{"still_dims": int(g[0].shape[1]), "bodies": g[1]} for g in groups],
              "candidates": [{"cos": c, "groups": [a, b]} for c, a, b in candidates]}
    if not candidates:
        return {"only": None, "dims": None, "report": report}
    _, a, b = min(candidates)
    if groups[a][0].shape[1] < groups[b][0].shape[1]:
        a, b = b, a
    # Factor 0 = group a's still directions; the bodies moving with it alone
    # are the ones still along factor 1, i.e. group b.
    return {"only": [groups[b][1], groups[a][1]], "dims": [groups[a][0].shape[1], groups[b][0].shape[1]],
            "report": {**report, "chosen": [a, b]}}


def refine_split(frames_by_node, node_bases, only, dims, gap_ratio, weigh=None):
    """The exact split at every node: factor f's directions are the tangent
    null space of the blocks of the bodies that move with the OTHER factor
    alone.  Those subspaces need no metric; their orthogonality is tested in
    the metric `weigh` maps into (None = pose).  Returns (node_bases, report),
    node_bases pose-orthonormal per factor, or None on failure."""
    refined = [None] * len(node_bases)
    worst_gap, worst_orth = np.inf, 0.0
    for k, bases in enumerate(node_bases):
        if bases is None:
            continue
        frame = frames_by_node[k]
        out = []
        for f in range(2):
            cols = block_columns(only[1 - f])
            _, s, vt = np.linalg.svd(frame[cols], full_matrices=True)
            s = np.concatenate((s, np.zeros(frame.shape[1] - len(s))))
            k_null = dims[f]
            kept, null = s[:frame.shape[1] - k_null], s[frame.shape[1] - k_null:]
            gap = float(kept.min() / max(null.max(), 1e-300)) if len(kept) else np.inf
            worst_gap = min(worst_gap, gap)
            out.append(frame @ vt[frame.shape[1] - k_null:].T)
        measured = out if weigh is None else [_orth(weigh(b), b.shape[1]) for b in out]
        worst_orth = max(worst_orth, float(np.abs(measured[0].T @ measured[1]).max()))
        refined[k] = out
    # Null spaces known to a relative gap g are known to ~1/g in angle; a
    # product's two factors must be orthogonal to that precision, not merely
    # roughly.
    tolerance = 1e3 / worst_gap if worst_gap > 0 else np.inf
    report = {"null_space_gap_min": worst_gap, "orthogonality_max": worst_orth,
              "orthogonality_tolerance": tolerance}
    ok = worst_gap >= gap_ratio and worst_orth <= tolerance
    return (refined if ok else None), {**report, "pass": bool(ok)}


# ── 4: leaf-confined system ──────────────────────────────────────────────────
class LeafSystem:
    """A `PhysicalSystem` whose responses stay on one factor's leaf.

    Every command is projected onto the leaf tangent carried from the node it
    starts at; every response is then pulled back onto the leaf by Newton on
    the anchor blocks (bodies that move with the OTHER factor alone), held at
    their seed values.

    A node starts with the leaf tangent and frame of the node that produced
    it.  Neither is re-measured: a slightly stale tangent only aims commands a
    little off the leaf, which the correction removes (it costs Newton
    iterations, never correctness).  Instead, whenever the front's own paired
    probes +u, -u at one scale come back corrected, their central secant is a
    leaf tangent vector to O(t^2), and the node's tangent is refreshed from
    those -- the experiments the front runs anyway, at no extra cost."""

    def __init__(self, base, seed, frame, leaf, anchor_cols, *, tol, max_newton, probe_h):
        self.base = base
        self.seed_observation = seed
        self.dimension = frame.shape[1]
        self.anchor_cols = np.asarray(anchor_cols, int)
        self.target = np.asarray(seed.pose, float).reshape(-1)[self.anchor_cols]
        self.tol, self.max_newton, self.probe_h = tol, max_newton, probe_h
        self._node = {_key(seed.pose): (frame, leaf)}
        self._origin: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self._probes: dict[bytes, dict[tuple[bytes, float], np.ndarray]] = {}
        self._secants: dict[bytes, list[np.ndarray]] = {}
        self.newton_commands = 0
        self.corrections_unconverged = 0

    def _complement(self, frame, leaf):
        return _orth(frame - leaf @ (leaf.T @ frame), self.dimension - leaf.shape[1])

    def seed(self):
        return self.seed_observation

    def restore(self, observation):
        key = _key(observation.pose)
        if key not in self._node:
            if key not in self._origin:
                raise ValueError("leaf session can only start from a response this leaf produced")
            self._node[key] = self._origin[key]
        frame, leaf = self._node[key]
        return self.base.restore(observation), observation, frame, leaf, key

    def command(self, session, direction, scale, *, parent=None):
        base_session, observation, frame, leaf, node = session
        along = leaf @ (leaf.T @ np.asarray(direction, float).reshape(-1))
        norm = float(np.linalg.norm(along))
        if norm <= 1e-12:
            return PhysicalResponse(observation, True, True, {"off_leaf_command": True}, parent)
        unit, length_cmd = along / norm, scale * norm
        response = self.base.command(base_session, unit, length_cmd, parent=parent)
        if not (response.valid and response.converged):
            return response
        complement = self._complement(frame, leaf)
        jacobian = complement[self.anchor_cols]
        current = response
        for _ in range(self.max_newton):
            residual = self.target - np.asarray(current.observation.pose, float).reshape(-1)[self.anchor_cols]
            if np.max(np.abs(residual)) <= self.tol:
                break
            step = complement @ np.linalg.lstsq(jacobian, residual, rcond=None)[0]
            length = float(np.linalg.norm(step))
            if length <= 1e-15:
                break
            self.newton_commands += 1
            current = self.base.command(self.base.restore(current.observation), step / length, length,
                                        parent=parent)
            if not (current.valid and current.converged):
                return current
        else:
            self.corrections_unconverged += 1
            return PhysicalResponse(current.observation, False, current.valid,
                                    {**current.diagnostics, "leaf_correction_unconverged": True}, parent)
        self._origin.setdefault(_key(current.observation.pose), (frame, leaf))
        self._pair(node, unit, length_cmd, np.asarray(current.observation.pose, float).reshape(-1))
        return PhysicalResponse(current.observation, True, current.valid, current.diagnostics, parent)

    def _pair(self, node, unit, t, pose):
        """Refresh the node's leaf tangent from a completed +u / -u probe pair."""
        probes = self._probes.setdefault(node, {})
        # `+ 0.0` turns -0.0 into 0.0: a negated direction's zero entries (the
        # blocks of bodies it does not move) are otherwise different bytes.
        opposite = probes.pop(((-unit + 0.0).tobytes(), t), None)
        if opposite is None:
            probes[((unit + 0.0).tobytes(), t)] = pose
            return
        secants = self._secants.setdefault(node, [])
        secants.append((pose - opposite) / (2.0 * t))
        frame, leaf = self._node[node]
        k = leaf.shape[1]
        if len(secants) >= k:
            u, s, _ = np.linalg.svd(np.column_stack(secants), full_matrices=False)
            if s[k - 1] > 1e3 * s[k] if len(s) > k else s[k - 1] > 0:
                self._node[node] = (frame, u[:, :k])

    def command_many(self, session, directions, scale, *, parent=None):
        """The whole set at once when the base answers batches (`command_batch`):
        the commands go out together, then the correction runs in lockstep
        rounds over the responses still off the leaf.  Same responses as
        commanding one by one; a base without batches does exactly that."""
        batch = getattr(self.base, "command_batch", None)
        if batch is None:
            return [self.command(session, d, scale, parent=parent) for d in directions]
        base_session, observation, frame, leaf, node = session
        out = [None] * len(directions)
        jobs = []
        for i, d in enumerate(directions):
            along = leaf @ (leaf.T @ np.asarray(d, float).reshape(-1))
            norm = float(np.linalg.norm(along))
            if norm <= 1e-12:
                out[i] = PhysicalResponse(observation, True, True, {"off_leaf_command": True}, parent)
            else:
                jobs.append((i, along / norm, scale * norm))
        current = dict(zip((i for i, *_ in jobs),
                           batch([(base_session, u, t) for _, u, t in jobs], parent=parent)))
        complement = self._complement(frame, leaf)
        solve = np.linalg.pinv(complement[self.anchor_cols])
        active, finished = [], []
        for i, response in current.items():
            (active if response.valid and response.converged else finished).append(i)
        for i in finished:
            out[i] = current[i]
        settled = []
        for _ in range(self.max_newton):
            steps = []
            for i in active:
                pose = np.asarray(current[i].observation.pose, float).reshape(-1)
                residual = self.target - pose[self.anchor_cols]
                step = complement @ (solve @ residual)
                length = float(np.linalg.norm(step))
                if np.max(np.abs(residual)) <= self.tol or length <= 1e-15:
                    settled.append(i)
                else:
                    steps.append((i, step / length, length, pose))
            if not steps:
                active = []
                break
            self.newton_commands += len(steps)
            responses = batch([(pose, u, length) for _, u, length, pose in steps], parent=parent)
            active = []
            for (i, *_), response in zip(steps, responses):
                current[i] = response
                if response.valid and response.converged:
                    active.append(i)
                else:
                    out[i] = response
        for i in active:                              # ran out of Newton iterations
            self.corrections_unconverged += 1
            r = current[i]
            out[i] = PhysicalResponse(r.observation, False, r.valid,
                                      {**r.diagnostics, "leaf_correction_unconverged": True}, parent)
        units = {i: (u, t) for i, u, t in jobs}
        for i in settled:
            r = current[i]
            pose = np.asarray(r.observation.pose, float).reshape(-1)
            self._origin.setdefault(_key(pose), (frame, leaf))
            self._pair(node, units[i][0], units[i][1], pose)
            out[i] = PhysicalResponse(r.observation, True, r.valid, r.diagnostics, parent)
        return out


# ── 5: composition certificate ───────────────────────────────────────────────
def _poses44(q, n_bodies):
    T = np.zeros((n_bodies, 4, 4))
    T[:, :3, :] = np.asarray(q, float).reshape(n_bodies, 3, 4)
    T[:, 3, 3] = 1.0
    return T


def _farthest_pairs(qa, qb, count, rng):
    """Farthest-point pairs (i, j), i, j >= 1, in the product of leaf distances."""
    da = np.linalg.norm(qa[1:, None] - qa[None, 1:], axis=2)
    db = np.linalg.norm(qb[1:, None] - qb[None, 1:], axis=2)
    pairs = [(int(rng.integers(len(qa) - 1)), int(rng.integers(len(qb) - 1)))]
    best = np.full((len(qa) - 1, len(qb) - 1), np.inf)
    while len(pairs) < count:
        i, j = pairs[-1]
        best = np.minimum(best, np.sqrt(da[i][:, None] ** 2 + db[j][None, :] ** 2))
        i, j = np.unravel_index(int(np.argmax(best)), best.shape)
        if best[i, j] <= 0.0:
            break
        pairs.append((int(i), int(j)))
    return [(i + 1, j + 1) for i, j in pairs]


def _reach(system, start, target, cols, dimension, tol, max_newton, step_max, probe_h):
    """Damped Gauss-Newton from `start` until the pose at `cols` equals
    `target`.  None if it fails.

    The frame is measured at the start and re-measured only when a step stops
    halving the residual: a frame from a nearby pose aims the step slightly
    off, which costs an iteration, never correctness -- the stopping test is
    on the anchors themselves."""
    obs = start
    frame = infer_frame(system, obs, dimension, probe_h)
    fresh, last = True, np.inf
    for _ in range(max_newton):
        pose = np.asarray(obs.pose, float).reshape(-1)
        residual = target - pose[cols]
        error = float(np.max(np.abs(residual)))
        if error <= tol:
            return obs
        if not fresh and error > .5 * last:
            new_frame = infer_frame(system, obs, dimension, probe_h, guide=frame)
            frame, fresh = (new_frame if new_frame is not None else frame), True
        if frame is None:
            return None
        step = frame @ np.linalg.lstsq(frame[cols], residual, rcond=None)[0]
        length = float(np.linalg.norm(step))
        if length <= 1e-15:
            return None
        response = system.command(system.restore(obs), step / length, min(length, step_max))
        if not (response.valid and response.converged):
            return None
        obs, last, fresh = response.observation, error, False
    return None


def cross_sample(system, start, predicted, target, cols, dimension, tol, max_newton, step_max, probe_h):
    """One cross sample: the pose whose anchor blocks equal `target`, reached
    by `_reach` from leaf pose `start` -- or, when given, from the projection
    of the composed `predicted` pose, which is one projection away if the
    composition is right.  Starting there cannot pass a wrong composition:
    the pose is still fixed by its anchors, and is compared with the
    prediction afterwards.  Returns the pose or None.  Module-level, so a pool
    worker can run it on its own system."""
    obs = PhysicalObservation(np.asarray(start, float))
    if predicted is not None:
        delta = np.asarray(predicted, float) - obs.pose
        length = float(np.linalg.norm(delta))
        if length > 1e-15:
            response = system.command(system.restore(obs), delta / length, length)
            if response.valid and response.converged:
                obs = response.observation
    reached = _reach(system, obs, target, cols, dimension, tol, max_newton, step_max, probe_h)
    return None if reached is None else np.asarray(reached.pose, float).reshape(-1)


def composition_certificate(system, qa, qb, anchor_cols, n_bodies, dimension, options: FactoredOptions,
                            *, step_max, probe_h, rng, has_mixed=True):
    """Cross samples off the two leaves against the composed prediction."""
    count = min(max(options.cross_min, int(np.ceil(options.cross_fraction * len(qa) * len(qb)))),
                (len(qa) - 1) * (len(qb) - 1))
    pairs = _farthest_pairs(qa, qb, count, rng)
    cols = np.concatenate(anchor_cols)
    rest_inv = np.linalg.inv(_poses44(qa[0], n_bodies))

    def predict(i, j):
        A, B = _poses44(qa[i], n_bodies), _poses44(qb[j], n_bodies)
        return B @ rest_inv @ A, A @ rest_inv @ B

    def residuals(pose, i, j):
        T = _poses44(pose, n_bodies)
        return [np.linalg.norm((p - T)[:, :3, :], axis=(1, 2)) for p in predict(i, j)]

    def target(i, j):
        value = np.empty(len(cols))
        value[:len(anchor_cols[0])] = qa[i][anchor_cols[0]]
        value[len(anchor_cols[0]):] = qb[j][anchor_cols[1]]
        return value

    # Independent jobs; a pooled system runs each whole walk in a worker.
    run = getattr(system, "map_jobs", None)

    def solve(jobs):
        args = [(start, pred, target(i, j), cols, dimension, tol, options.max_newton, step_max, probe_h)
                for i, j, start, pred, tol in jobs]
        if run is not None:
            return run(cross_sample, args)
        return [cross_sample(system, *a) for a in args]

    # The order is decided on the first pairs reached WITHOUT a prediction to
    # start from; the rest start at that order's prediction.
    found, res, order, k = [], [], None, 0
    while order is None and k < len(pairs):
        i, j = pairs[k]
        k += 1
        pose = solve([(i, j, qa[i], None, options.anchor_tol)])[0]
        if pose is not None:
            found.append((i, j, pose))
            res.append(residuals(pose, i, j))
            order = int(np.argmin(np.max(res[-1], axis=1)))
    rest = pairs[k:]
    if order is not None and rest:
        guesses = [predict(i, j)[order][:, :3, :].reshape(-1) for i, j in rest]
        for (i, j), pose in zip(rest, solve([(i, j, qa[i], g, options.anchor_tol)
                                             for (i, j), g in zip(rest, guesses)])):
            if pose is not None:
                found.append((i, j, pose))
                res.append(residuals(pose, i, j))
    report = {"pairs_requested": len(pairs), "pairs_reached": len(found)}
    if not found:
        return {**report, "pass": False, "reason": "no cross sample reached"}
    res = np.asarray(res)                                   # (pairs, 2 orders, bodies)
    order = int(np.argmin(res.max(axis=(0, 2))))
    best, other = res[:, order].max(), res[:, 1 - order].max()
    fine = []
    refine = found[:options.refine_pairs]
    for (i, j, _), tight in zip(refine, solve([(i, j, pose, None, options.anchor_tol * options.refine_factor)
                                               for i, j, pose in refine])):
        if tight is not None:
            fine.append(residuals(tight, i, j)[order])
    fine_max = float(np.max(fine)) if fine else None
    coarse_max = float(res[:len(fine), order].max()) if fine else None
    refines = fine_max is not None and (fine_max <= .1 * coarse_max or
                                        fine_max <= 1e3 * options.anchor_tol * options.refine_factor)
    # With no body moving with both factors the two orders are the same map
    # (each body comes from its own leaf), so there is no order to decide.
    decisive = (not has_mixed) or other >= options.order_gap * max(best, 1e-300)
    report.update({"order": ["B rest^-1 A", "A rest^-1 B"][order] if has_mixed else "either (no mixed bodies)",
                   "residual_max": float(best), "residual_median": float(np.median(res[:, order])),
                   "other_order_max": float(other),
                   "refine_pairs": len(fine), "refined_residual_max": fine_max,
                   "unrefined_residual_max_same_pairs": coarse_max,
                   "pass": bool(refines and decisive and len(found) == len(pairs))})
    return {**report, "_cross": [(i, j, pose) for i, j, pose in found]}


# ── orchestration ────────────────────────────────────────────────────────────
def factored_explore(system, front: AdvancingFrontOptions, options: FactoredOptions | None = None,
                     log=print, on_leaf=None):
    """Detect, certify, and sample a product factor by factor.

    Returns a dict with "audit" always, "detect" (the partial front's q,
    edges, frames) once detection ran, and on success "leaves" (per factor:
    q, edges, frames, front audit) and "cross" (pairs and poses).
    `on_leaf(f, leaf)` is called as each leaf front closes, so a caller can
    save it before the composition certificate runs."""
    detect = {}
    result = _factored_explore(system, front, options, log, detect, on_leaf)
    if detect:
        result["detect"] = detect
    return result


def _factored_explore(system, front, options, log, detect, on_leaf=None):
    options = options or FactoredOptions()
    rng = np.random.default_rng(front.seed)
    seed = system.seed()
    detect_front = dataclasses.replace(front, max_samples=options.detect_samples,
                                       max_seconds=options.detect_seconds)
    # The partial front is sampled and its frames measured in pose units:
    # every command a black-box system takes is projected in ITS metric, so a
    # command along a random kinetic direction is a long, mostly off-manifold
    # pose move (light bodies are magnified by cond(L)), and the probes land
    # far away.  Only the ANALYSIS -- detection, orthogonality, stage-3
    # transport and gaps -- runs in kinetic coordinates; stage 3 commands
    # tangent directions only, where the two metrics just rescale lengths.
    metric = KineticMetric(options.body_moments) if options.body_moments is not None else None
    q, edges, frames, detect_audit = explore(system, options=detect_front, seeds=[seed], log=log)
    detect.update(q=q, edges=edges, frames=frames)
    root = int(np.argmin(np.linalg.norm(q - np.asarray(seed.pose, float).reshape(-1), axis=1)))
    audit = {"factored": False, "frontier_complete": False, "detect_samples": int(len(q)),
             "metric": "kinetic" if metric else "pose"}
    if len(q) < 2:
        return {"audit": {**audit, "reason": "the partial front produced fewer than two samples"}}
    if metric:
        q_m = metric.weigh(q.T).T
        frames_m = np.stack([_orth(metric.weigh(f), f.shape[1]) for f in frames])
        # Kinetic length per unit pose length along the front's own edges:
        # stage-3 walk lengths, set in `front.step` units, keep their size.
        walked = np.asarray(edges, int).reshape(-1, 2)
        unit = float(np.median(np.linalg.norm(q_m[walked[:, 0]] - q_m[walked[:, 1]], axis=1)
                               / np.linalg.norm(q[walked[:, 0]] - q[walked[:, 1]], axis=1)))
        audit["kinetic_per_pose_length"] = unit
    else:
        q_m, frames_m, unit = q, frames, 1.0
    if np.linalg.norm(q[root] - np.asarray(seed.pose, float).reshape(-1)) > 1e-9:
        return {"audit": {**audit, "reason": "seed is not a regular partial-front sample"}}
    # The front records where its own probes and steps answered "physical
    # boundary"; a one-sided node without such evidence is only where the
    # partial front stopped exploring.
    evidence = [bool(r["diagnostics"].get("boundary_probes") or r["diagnostics"].get("boundary_steps"))
                for r in detect_audit[:len(q)]]

    def physical_boundary(nodes, outward):
        return [evidence[i] for i in nodes]

    d = frames.shape[2]
    n_bodies = q.shape[1] // BLOCK
    def to_pose(node_bases_m):
        """Metric-coordinate factor bases -> pose-orthonormal tangent bases."""
        if not metric:
            return node_bases_m
        return [None if b is None else [_orth(metric.unweigh(x), x.shape[1]) for x in b]
                for b in node_bases_m]

    decoupled = body_groups(frames_m, n_bodies, options.split.gap_ratio)
    if decoupled is not None:
        groups, node_bases = decoupled
        split = {"split": {"kind": "body_groups", "factors": [b.shape[1] for b in node_bases[root]],
                           "groups": [groups[0], [i for g in groups[1:] for i in g]]}}
        audit["split"] = split
        log(f"factored: partial front {len(q)} samples, decoupled body groups {groups}")
        node_bases = to_pose(node_bases)
    else:
        # The front spaced its samples in pose units: pick neighbours there.
        split = detect_product_split(q_m, frames_m, edges, options=options.split, root=root,
                                     confirm_boundary=physical_boundary, graph_q=q if metric else None)
        node_bases, bases_root = split.pop("_node_bases"), split.pop("_bases")
        audit["split"] = split
        log(f"factored: partial front {len(q)} samples, split {split['split']}")
        if split["split"]["kind"] == "flat_ambiguous":
            # Flat, so every direction is parallel and nothing local picks the
            # factors; the partial front saw too little boundary to.  The
            # bodies can: a product's single-factor bodies are still along
            # the whole other factor.
            proposal = body_null_split(frames[root], n_bodies, options.null_gap,
                                       metric.weigh if metric else None, options.null_orth_tol)
            audit["body_null_split"] = proposal["report"]
            log(f"factored: flat, split proposed from body null spaces {proposal['report']}")
            if proposal["only"] is None:
                return {"audit": {**audit, "reason": "flat, and no body null spaces pin a two-factor split"}}
            split["split"] = {"kind": "flat_bodies", "factors": proposal["dims"]}
            dims, coarse = proposal["dims"], proposal["only"]
            node_bases = [[] for _ in range(len(q))]          # every node: refine measures its own
        elif split["split"]["kind"] not in ("flat_boundary", "holonomy") or len(bases_root) != 2:
            return {"audit": {**audit, "reason": f"no two-factor product split ({split['split']['kind']})"}}
        else:
            dims = [b.shape[1] for b in bases_root]
            # Per-body motion ratios, in the metric the split was found in.
            coarse = coarse_only(node_bases, n_bodies, options.coarse_ratio)
        audit["coarse_only"] = coarse
        if not all(coarse):
            return {"audit": {**audit, "reason": "a factor has no body that moves with it alone"}}
        refined, refine_report = refine_split(frames, node_bases, coarse, dims, options.split.gap_ratio,
                                              weigh=metric.weigh if metric else None)
        audit["refine"] = refine_report
        if refined is None:
            why = ("their directions are not orthogonal: not a Riemannian product"
                   if refine_report["null_space_gap_min"] >= options.split.gap_ratio
                   else "their blocks leave no clean null space")
            return {"audit": {**audit, "reason": f"single-factor bodies do not pin the split ({why})"}}
        node_bases = refined
    bases_root = [frames[root].T @ b for b in node_bases[root]]
    # A refined factor's directions are known to ~1/gap (its null-space gap),
    # so a body moving less than that along it is still, not mixed.  In the
    # kinetic metric a light body is seen faintly and the gap is smaller.
    motion_rel = options.body_motion_rel
    if "refine" in audit:
        motion_rel = max(motion_rel, 10. / audit["refine"]["null_space_gap_min"])
    audit["body_motion_rel"] = motion_rel
    classes = classify_bodies(node_bases, n_bodies, motion_rel)
    audit["bodies"] = classes
    anchors = [block_columns(classes["only"][f]) for f in range(2)]
    seed_frame = frames[root]
    leaf_bases = node_bases[root]
    for f in range(2):
        other = 1 - f
        if not len(anchors[other]):
            return {"audit": {**audit, "reason": f"no body moves with factor {other} alone; "
                                                 f"factor {f}'s leaf cannot be held"}}
        complement = leaf_bases[other]
        if np.linalg.matrix_rank(complement[anchors[other]], tol=1e-8) < complement.shape[1]:
            return {"audit": {**audit, "reason": f"factor {other}'s own bodies do not see all of it"}}
    probe_h = .25 * front.step
    if options.certify:
        # Stage 3 walks parallel fields, so it runs in the detection metric.
        # Bases in coordinates of the frame stage 3 measures in: the pose
        # frame, or its kinetic orthonormalisation `metric_frame`.
        if metric:
            def c_basis(k, b):
                return metric_frame(metric, frames[k]).T @ _orth(metric.weigh(b), b.shape[1])
        else:
            def c_basis(k, b):
                return frames[k].T @ b
        c_bases = [c_basis(root, b) for b in node_bases[root]]
        width_bases = None
        # A boundary-selected split puts the boundary normal, the interval
        # factor, second.
        if split["split"]["kind"] == "flat_boundary" and bases_root[1].shape[1] == 1:
            f = 1
            far = np.argsort(-np.linalg.norm(q_m - q_m[root], axis=1))
            picks = [root] + [int(k) for k in far if node_bases[k] is not None][:2]
            width_bases = [(PhysicalObservation(q[k]), frames[k], c_basis(k, node_bases[k][f])[:, 0])
                           for k in picks]
        step_m = unit * front.step                  # the front's step, in the detection metric
        stage3 = certify_split(system, seed, frames[root], c_bases, metric=metric, length=.6 * step_m,
                               substeps=options.certify_substeps, width_bases=width_bases,
                               width_substep=.1 * step_m,
                               width_max_steps=int(np.ceil(4 * split["boundary"]["graph_width_mean"]
                                                           / (.1 * step_m))) + 10
                               if width_bases else 10000, rng=rng)
        gaps = [g for g in stage3["commuting_gap"] if g is not None]
        floor = stage3.get("gap_noise_floor")
        commutes = bool(gaps) and (stage3.get("gap_refinement_ratio", 1.) <= options.gap_ratio_pass
                                   or (floor is not None and gaps[-1] <= 10. * floor))
        width_ok = True
        if width_bases:
            widths = [w for w in stage3.get("widths", []) if w is not None]
            resolution = .1 * step_m / 2 ** 12
            width_ok = len(widths) == len(width_bases) and \
                stage3["width_cv"] <= 10 * resolution / float(np.mean(widths))
        stage3["pass"] = bool(commutes and width_ok)
        audit["stage3"] = stage3
        log(f"factored: stage-3 certificate {'PASS' if stage3['pass'] else 'FAIL'} {stage3}")
        if not stage3["pass"]:
            return {"audit": {**audit, "reason": "stage-3 certificate failed"}}
    leaves = []
    for f in range(2):
        leaf_system = LeafSystem(system, seed, seed_frame, leaf_bases[f], anchors[1 - f],
                                 tol=options.anchor_tol, max_newton=options.max_newton, probe_h=probe_h)
        lq, le, lf, laudit = explore(leaf_system, options=front, seeds=[seed], log=log)
        summary = laudit[-1]
        leaves.append({"q": lq, "edges": le, "frames": lf, "front_audit": laudit,
                       "frontier_complete": bool(summary["frontier_complete"]),
                       "budget_exhausted": bool(summary["budget_exhausted"]),
                       "newton_commands": leaf_system.newton_commands,
                       "corrections_unconverged": leaf_system.corrections_unconverged})
        log(f"factored: leaf {f}: {len(lq)} samples, rank {lf.shape[2] if lf.ndim == 3 else 0}, "
            f"complete={summary['frontier_complete']}")
        if on_leaf is not None:
            on_leaf(f, leaves[-1])
    audit["leaves"] = [{k: v for k, v in leaf.items() if k not in ("q", "edges", "frames", "front_audit")}
                       | {"n_samples": int(len(leaf["q"])),
                          "rank": int(leaf["frames"].shape[2]) if leaf["frames"].ndim == 3 else 0}
                       for leaf in leaves]
    qa, qb = leaves[0]["q"], leaves[1]["q"]
    for leaf, name in ((qa, 0), (qb, 1)):
        if np.linalg.norm(leaf[0] - q[root]) > 1e-9:
            return {"audit": {**audit, "reason": f"leaf {name} does not start at the seed"}}
    if len(qa) < 2 or len(qb) < 2:
        return {"audit": {**audit, "reason": "a leaf front produced fewer than two samples"}, "leaves": leaves}
    composition = composition_certificate(system, qa, qb, anchors, n_bodies, d, options,
                                          step_max=front.step, probe_h=probe_h, rng=rng,
                                          has_mixed="mixed" in classes["labels"])
    cross = composition.pop("_cross", [])
    audit["composition"] = composition
    log(f"factored: composition {'PASS' if composition['pass'] else 'FAIL'} {composition}")
    complete = (all(leaf["frontier_complete"] for leaf in leaves) and composition["pass"]
                and audit.get("stage3", {"pass": True})["pass"])
    audit.update({"factored": bool(composition["pass"]), "frontier_complete": bool(complete),
                  "budget_exhausted": any(leaf["budget_exhausted"] for leaf in leaves),
                  "sample_count": int(len(qa) + len(qb) + len(cross)),
                  "product_count": int(len(qa) * len(qb))})
    if not composition["pass"]:
        audit["reason"] = "composition certificate failed"
    return {"audit": audit, "leaves": leaves,
            "cross": {"pairs": np.asarray([(i, j) for i, j, _ in cross], int).reshape(-1, 2),
                      "q": np.asarray([p for *_, p in cross], float)}}
