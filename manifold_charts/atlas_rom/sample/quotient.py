"""quotient.py — collapse repeated physical states in a walked sample cloud
*before* chart growth, so the atlas is built on the physical configuration
graph, not on how many times or in what order the explorer visited it.

The problem this fixes: a periodic mechanism revisits the same physical
state every time it winds around, but the explorer's own bookkeeping (row
index, traversal order, phase coordinate) treats each visit as a distinct
sample.  Growing charts and routing rollouts on that raw walk therefore
scatters one physical neighbourhood across many disconnected regions of the
sample-index graph -- exactly the long-range routing failure jansen's atlas
showed (farthest-point trajectories failed while short local replays near
any single point passed cleanly).  Quotienting first means "the same state,
walked twice" becomes one graph vertex with two incident walked edges,
instead of two vertices an unrelated distance apart in traversal order.

This is a LOCAL equivalence test, never a global topology computation: two
samples merge only when direct local evidence -- physical pose distance,
tangent-subspace agreement, intrinsic dimension, and (when available) an
oracle's own retraction/residual certificate -- says they are the same
state.  Nothing here ever asks "does this graph contain a cycle" or infers
Betti numbers; the cycle in a periodic mechanism's quotient graph is a
CONSEQUENCE of local merges, never an input to them.

Pipeline position (see atlas_rom/build.py's ``STAGES``):

    samples.npz (walked q, frames, edges)
        -> quotient.npz   (this module)
        -> certified_graph.npz  (sample_graph.py, now over quotient vertices)
        -> new_dataset.npz      (new_dataset.py)
        -> patch_atlas cover/training
"""
from __future__ import annotations

import os
import numpy as np

from atlas_rom.spec import physical_pose as pp

ARTIFACT = "quotient.npz"


# ─── local geometry helpers ─────────────────────────────────────────────────

def _orthonormal(frame):
    F = np.asarray(frame, float)
    if F.ndim == 1:
        F = F[:, None]
    Q, _ = np.linalg.qr(F, mode="reduced")
    return Q


def _subspace_distance(Qi, Qj):
    """sin(largest principal angle) between two orthonormal bases -- 0 for
    identical subspaces, 1 for orthogonal ones.  A dimension mismatch (two
    frames that do not even agree on intrinsic dimension) is an automatic,
    maximal disagreement rather than a silent broadcast/crash."""
    if Qi.shape[1] != Qj.shape[1]:
        return 1.0
    s = np.linalg.svd(Qi.T @ Qj, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - np.clip(s.min(), -1.0, 1.0) ** 2)))


def _knn_pairs(D, k):
    n = len(D)
    k = max(1, min(k, n - 1))
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    a = np.repeat(np.arange(n), k)
    b = idx.ravel()
    return np.unique(np.sort(np.c_[a, b], axis=1), axis=0)


# ─── step 1: local equivalence candidates ──────────────────────────────────

def local_equivalence_candidates(q, frames, *, k=16, pose_tol, tangent_tol_deg,
                                 oracle=None, distance_matrix=None, log=print):
    """Candidate i~j edges where ALL available local evidence agrees:
    physical pose distance, tangent-subspace agreement, matching intrinsic
    dimension, and -- when an oracle is supplied -- mutual retraction (the
    candidates' midpoint projects back close to BOTH endpoints, not off to a
    third state) as an optional physical-validity signature. Returns
    ``(pairs, diagnostics)`` with ``pairs`` sorted by ascending pose
    distance, each entry ``(i, j, pose_distance)``."""
    q = np.asarray(q, float)
    n = len(q)
    D = (np.asarray(distance_matrix, float) if distance_matrix is not None
         else pp.physical_distance_matrix(q))
    cand = _knn_pairs(D, k)
    Qs = [_orthonormal(frames[i]) for i in range(n)]
    # `D` is the twist-RMS physical-pose metric (see physical_pose.py), not
    # raw ambient Euclidean distance -- the retraction check below must stay
    # in that SAME metric (relative to the same reference pose q[0]) or its
    # threshold is being compared against the wrong units entirely.
    reference = q[0]
    twists = pp.pose_twists(q, reference=reference)

    def _twist_distance(point, twist_b):
        twist_a = pp.pose_twists(point[None, :], reference=reference)[0]
        return float(np.sqrt(np.mean((twist_a - twist_b) ** 2)))

    diag = {"n_candidates": int(len(cand)), "n_pose_ok": 0, "n_dim_ok": 0,
           "n_tangent_ok": 0, "n_retraction_checked": 0, "n_retraction_ok": 0}
    tol_sine = float(np.sin(np.radians(tangent_tol_deg)))
    accepted = []
    for i, j in cand:
        dij = float(D[i, j])
        if dij > pose_tol:
            continue
        diag["n_pose_ok"] += 1
        if Qs[i].shape[1] != Qs[j].shape[1]:
            continue
        diag["n_dim_ok"] += 1
        if _subspace_distance(Qs[i], Qs[j]) > tol_sine:
            continue
        diag["n_tangent_ok"] += 1
        if oracle is not None and hasattr(oracle, "project"):
            diag["n_retraction_checked"] += 1
            mid = 0.5 * (q[i] + q[j])
            proj = np.asarray(oracle.project(mid), float)
            if (_twist_distance(proj, twists[i]) > pose_tol
                    or _twist_distance(proj, twists[j]) > pose_tol):
                continue
            diag["n_retraction_ok"] += 1
        accepted.append((int(i), int(j), dij))
    accepted.sort(key=lambda t: t[2])
    log(f"quotient: {diag['n_candidates']} kNN candidates -> {diag['n_pose_ok']} pose-ok "
       f"-> {diag['n_tangent_ok']} tangent-ok -> {len(accepted)} accepted "
       f"(pose_tol={pose_tol:.4g}, tangent_tol={tangent_tol_deg:.1f} deg)")
    return accepted, diag


# ─── step 2: conservative classing (complete-linkage protected) ───────────

class _Class:
    __slots__ = ("members", "rep_twist", "rep_Q", "radius", "tangent_spread_deg")

    def __init__(self, index, twist, Q):
        self.members = [index]
        self.rep_twist = twist
        self.rep_Q = Q
        self.radius = 0.0
        self.tangent_spread_deg = 0.0


def quotient_classes(q, frames, accepted_pairs, *, pose_tol, tangent_tol_deg, log=print):
    """Connected components of the accepted equivalence edges, but a sample
    joins a class only if it stays compatible with that class's
    REPRESENTATIVE (not merely the one accepted edge that reached it) and
    the class's tangent spread stays bounded after joining. This is what
    stops a chain of individually-plausible pairwise merges from collapsing
    an entire curve end to end (complete-linkage protection, not naive
    single-linkage union-find). An edge that fails this representative-level
    check is REJECTED outright -- it never forces an ambiguous merge."""
    n = len(q)
    twists = pp.pose_twists(q, reference=q[0])
    Qs = [_orthonormal(frames[i]) for i in range(n)]

    owner = list(range(n))
    classes = {i: _Class(i, twists[i], Qs[i]) for i in range(n)}
    rejected = []

    def find(i):
        return owner[i]

    tol_sine = float(np.sin(np.radians(tangent_tol_deg)))
    d = Qs[0].shape[1] if n else 0

    for i, j, _dij in accepted_pairs:
        ci, cj = find(i), find(j)
        if ci == cj:
            continue
        A, B = classes[ci], classes[cj]
        # small-to-large: always absorb the smaller class into the larger,
        # so `owner` updates stay O(n log n) amortized over the whole run.
        if len(B.members) > len(A.members):
            A, B, ci, cj = B, A, cj, ci

        d_rep = float(np.sqrt(np.mean((A.rep_twist - B.rep_twist) ** 2)))
        if d_rep > pose_tol:
            rejected.append((i, j, "representative_pose"))
            continue
        if _subspace_distance(A.rep_Q, B.rep_Q) > tol_sine:
            rejected.append((i, j, "representative_tangent"))
            continue

        new_members = A.members + B.members
        new_twist = ((len(A.members) * A.rep_twist + len(B.members) * B.rep_twist)
                     / len(new_members))
        radius = max(float(np.sqrt(np.mean((twists[m] - new_twist) ** 2)))
                    for m in new_members)
        if radius > pose_tol:
            rejected.append((i, j, "spread_pose"))
            continue

        P = np.zeros((Qs[0].shape[0], Qs[0].shape[0]))
        for m in new_members:
            P += Qs[m] @ Qs[m].T
        P /= len(new_members)
        eigval, eigvec = np.linalg.eigh(P)
        new_Q = eigvec[:, -d:] if d else eigvec[:, :0]
        spread_deg = max(
            (float(np.degrees(np.arcsin(np.clip(_subspace_distance(new_Q, Qs[m]),
                                                 0.0, 1.0)))) for m in new_members),
            default=0.0)
        if spread_deg > tangent_tol_deg:
            rejected.append((i, j, "spread_tangent"))
            continue

        A.members = new_members
        A.rep_twist = new_twist
        A.rep_Q = new_Q
        A.radius = radius
        A.tangent_spread_deg = spread_deg
        del classes[cj]
        for m in B.members:
            owner[m] = ci

    log(f"quotient: {len(accepted_pairs)} accepted candidates -> "
       f"{len(rejected)} rejected at classing (ambiguous merges refused), "
       f"{len(classes)} classes from {n} samples")
    return owner, classes, rejected


# ─── step 3: quotient graph + full artifact ────────────────────────────────

def build_quotient(q, frames, walked_edges, *, oracle=None, k=16, pose_tol=None,
                   tangent_tol_deg=15.0, distance_matrix=None, log=print):
    """The full quotient pipeline. ``pose_tol``, left ``None``, is derived
    from the data's own resolution (a fraction of the median walked-edge
    physical-pose length) rather than an arbitrary absolute constant, so it
    scales with however densely this particular case was sampled."""
    q = np.asarray(q, float)
    frames = np.asarray(frames, float)
    n = len(q)
    walked = np.asarray(walked_edges, int).reshape(-1, 2)
    D = (np.asarray(distance_matrix, float) if distance_matrix is not None
         else pp.physical_distance_matrix(q))

    if pose_tol is None:
        if len(walked):
            lengths = D[walked[:, 0], walked[:, 1]]
            pose_tol = 0.25 * float(np.median(lengths))
        elif n > 1:
            pose_tol = 0.1 * float(np.median(D[D > 0]))
        else:
            pose_tol = 1e-6
        pose_tol = max(pose_tol, 1e-9)

    accepted, cand_diag = local_equivalence_candidates(
        q, frames, k=k, pose_tol=pose_tol, tangent_tol_deg=tangent_tol_deg,
        oracle=oracle, distance_matrix=D, log=log)
    owner, classes, rejected = quotient_classes(
        q, frames, accepted, pose_tol=pose_tol, tangent_tol_deg=tangent_tol_deg, log=log)

    class_of = np.asarray(owner, int)
    old_to_new = {old: new for new, old in enumerate(sorted(classes))}
    class_of = np.asarray([old_to_new[c] for c in class_of], int)
    ordered = sorted(classes.items(), key=lambda kv: old_to_new[kv[0]])

    n_classes = len(ordered)
    ambient = q.shape[1]
    d = frames.shape[-1] if frames.ndim == 3 else 1
    rep_q = np.zeros((n_classes, ambient))
    rep_frames = np.zeros((n_classes, ambient, d))
    radius = np.zeros(n_classes)
    tangent_spread_deg = np.zeros(n_classes)
    member_offsets = [0]
    member_indices = []
    for c, cls in ordered:
        idx = old_to_new[c]
        rep_q[idx] = pp.se3_lift(q[0], cls.rep_twist[None, :])[0]
        rep_frames[idx] = cls.rep_Q if d > 1 else cls.rep_Q.reshape(-1, 1)
        radius[idx] = cls.radius
        tangent_spread_deg[idx] = cls.tangent_spread_deg
        member_indices.extend(int(m) for m in cls.members)
        member_offsets.append(len(member_indices))
    member_indices = np.asarray(member_indices, int)
    member_offsets = np.asarray(member_offsets, int)

    mapped = class_of[walked]
    self_loops = mapped[:, 0] == mapped[:, 1]
    quotient_edges = np.unique(np.sort(mapped[~self_loops], axis=1), axis=0) \
        if (~self_loops).any() else np.empty((0, 2), int)
    collapsed_edges = walked[self_loops]

    diagnostics = {
        "n_original_samples": int(n), "n_quotient_vertices": int(n_classes),
        "n_walked_edges": int(len(walked)),
        "n_walked_edges_collapsed_to_self_loop": int(self_loops.sum()),
        "n_quotient_edges": int(len(quotient_edges)),
        "pose_tol": float(pose_tol), "tangent_tol_deg": float(tangent_tol_deg),
        "max_within_class_pose_distance": float(radius.max()) if n_classes else 0.0,
        "max_tangent_disagreement_deg": float(tangent_spread_deg.max()) if n_classes else 0.0,
        "n_classing_rejections": len(rejected),
        "candidate_diagnostics": cand_diag,
    }
    log(f"quotient: {n} samples -> {n_classes} quotient vertices "
       f"({diagnostics['n_walked_edges_collapsed_to_self_loop']}/{len(walked)} walked "
       "edges collapsed to a self-loop, i.e. re-visited an already-known state); "
       f"{len(quotient_edges)} quotient edges, max in-class pose radius "
       f"{diagnostics['max_within_class_pose_distance']:.4g}, max tangent spread "
       f"{diagnostics['max_tangent_disagreement_deg']:.2f} deg")

    return {
        "q": rep_q, "frames": rep_frames, "radius": radius,
        "tangent_spread_deg": tangent_spread_deg,
        "class_of": class_of, "member_offsets": member_offsets,
        "member_indices": member_indices,
        "quotient_edges": quotient_edges, "collapsed_edges": collapsed_edges,
        "diagnostics": diagnostics,
    }


def effective_samples(work, q, frames, edges, log=print):
    """If ``work/quotient.npz`` exists, return the QUOTIENT vertices'
    (q, frames, quotient_edges, class_of) in place of the raw walked
    (q, frames, edges) -- everything downstream of the samples stage
    (certified_graph, new_dataset, patch_atlas) already treats its input
    generically as "q, frames, edges", so substituting the quotient here is
    the whole integration.  When no quotient artifact exists (an older work
    dir, or a case that skipped quotienting), the raw walked triple passes
    through UNCHANGED and ``class_of`` is the identity map, so every
    already-built case keeps working with zero behavior change."""
    path = os.path.join(work, ARTIFACT)
    if not os.path.exists(path):
        n = len(q)
        return (np.asarray(q, float), np.asarray(frames, float),
               np.asarray(edges if edges is not None else [], int).reshape(-1, 2),
               np.arange(n))
    art = load(path)
    d = art["diagnostics"]
    log(f"quotient: using {d['n_quotient_vertices']} quotient vertices from "
       f"{d['n_original_samples']} walked samples (quotient.npz)")
    return art["q"], art["frames"], art["quotient_edges"], art["class_of"]


def members_of(artifact, class_index):
    """Original sample indices belonging to one quotient vertex."""
    lo, hi = artifact["member_offsets"][class_index], artifact["member_offsets"][class_index + 1]
    return artifact["member_indices"][lo:hi]


# ─── artifact io ────────────────────────────────────────────────────────────

def save(path, artifact):
    import json
    np.savez_compressed(
        path, q=artifact["q"], frames=artifact["frames"], radius=artifact["radius"],
        tangent_spread_deg=artifact["tangent_spread_deg"], class_of=artifact["class_of"],
        member_offsets=artifact["member_offsets"], member_indices=artifact["member_indices"],
        quotient_edges=artifact["quotient_edges"], collapsed_edges=artifact["collapsed_edges"],
        diagnostics=np.asarray(json.dumps(artifact["diagnostics"])))
    return path


def load(path):
    import json
    data = np.load(path, allow_pickle=True)
    return {
        "q": np.asarray(data["q"], float), "frames": np.asarray(data["frames"], float),
        "radius": np.asarray(data["radius"], float),
        "tangent_spread_deg": np.asarray(data["tangent_spread_deg"], float),
        "class_of": np.asarray(data["class_of"], int),
        "member_offsets": np.asarray(data["member_offsets"], int),
        "member_indices": np.asarray(data["member_indices"], int),
        "quotient_edges": np.asarray(data["quotient_edges"], int),
        "collapsed_edges": np.asarray(data["collapsed_edges"], int),
        "diagnostics": json.loads(str(data["diagnostics"])),
    }
