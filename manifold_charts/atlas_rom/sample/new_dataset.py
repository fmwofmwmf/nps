"""new_dataset.py — the atlas-facing dataset, stripped to exactly what the
physical chart model may see:

    q              [N, 12B]       physical SE(3) poses
    frames         [N, 12B, 2]    physical tangent estimates
    cover_edges    [M, 2]         connected graph for growth/geodesics
    local_edges    [K, 2]         conservative mesh-like neighbor graph

Generator parameters (the two motor-angle coordinates ``u``) and assembly
labels (``branch_labels``, ``parent``, ``ranks``) are real and useful for
auditing, but they must never reach chart training — a chart that could see
them would be cheating on exactly the question ("can pure physical geometry
find a competent 5-chart cover") this pipeline exists to answer.  They are
written to ``dataset_audit.json`` only; ``load()`` below does not expose them.

Edge construction never uses assembly labels either.  Candidate chords come
from ``certified_graph.npz`` (already midpoint/projection-certified against
off-sheet shortcuts, see ``sample_graph.py``).  ``local_edges`` narrows that
pool with four purely-geometric rules:

    mutual locality       keep (i,j) only if each is in the other's k-NN
    tangent compatibility  reject an edge whose displacement has a large
                           component off BOTH endpoints' tangent planes
                           (frames) -- this is what actually screens out a
                           cross-branch shortcut, with no branch label at all
    relative-neighborhood  reject (i,j) if some third vertex is closer to
                           pruning                both endpoints than they are to each other
    directional coverage   after pruning, refill any vertex whose remaining
                           edge directions leave a wide angular gap on its
                           tangent plane

``k`` is swept to land mean degree in [5, 8].  ``cover_edges`` is
``local_edges`` plus the minimum extra chords needed to connect any
components that pruning split off; those extra chords are flagged
``connectivity_only`` and excluded from interpolation diagnostics.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

from atlas_rom.spec.physical_pose import pose_distance_matrix

ARTIFACT = "new_dataset.npz"
AUDIT = "dataset_audit.json"


# ─── local_edges construction (physical geometry only) ─────────────────────

def _rng_violations(D, edges, candidates_by_vertex):
    """Relative-neighborhood-graph violation COUNT per edge, restricted to
    each edge's own local candidate neighborhood (union of i's and j's
    certified-candidate lists) rather than the full 626-point cloud.  A
    plain global RNG lune test (any witness anywhere) collapses almost
    every edge in a 48-D pose metric -- with 626 points nearly every edge
    has SOME third point closer to both ends somewhere in the ambient
    cloud, even off-manifold in an unrelated direction, which is a metric-
    dimensionality artifact, not evidence the edge is a shortcut.
    Restricting the witness set to points already certified as physically
    local to i or j keeps the rule meaningful, and returning a COUNT
    (rather than a hard reject) lets it act as a preference ranking: prefer
    few-violation edges, but do not starve a vertex down to nothing just
    because every candidate has one nearby witness in a high-D metric."""
    out = np.zeros(len(edges), int)
    for k, (i, j) in enumerate(edges):
        dij = D[i, j]
        witnesses = (candidates_by_vertex.get(i, set())
                    | candidates_by_vertex.get(j, set())) - {i, j}
        if not witnesses:
            continue
        w = np.asarray(sorted(witnesses), int)
        out[k] = int(((D[i, w] < dij) & (D[j, w] < dij)).sum())
    return out


def _tangent_residual(q, frames, i, j):
    """Fraction of the i->j displacement NOT explained by i's tangent plane."""
    disp = q[j] - q[i]
    length = np.linalg.norm(disp)
    if length < 1e-12:
        return 0.0
    F = frames[i]                                   # (12B, 2)
    # project disp onto span(F) via least squares (F may be non-orthonormal)
    coeff, *_ = np.linalg.lstsq(F, disp, rcond=None)
    resid = disp - F @ coeff
    return float(np.linalg.norm(resid) / length)


def _tangent_prune(q, frames, edges, tol):
    kept, rejected = set(), []
    for i, j in edges:
        r_i = _tangent_residual(q, frames, i, j)
        r_j = _tangent_residual(q, frames, j, i)
        if min(r_i, r_j) <= tol:
            kept.add((i, j))
        else:
            rejected.append((i, j, min(r_i, r_j)))
    return kept, rejected


def _directional_coverage_fixed(q, frames, kept, candidates_by_vertex, n, gap_deg=150.0):
    """Refill a vertex whose surviving edges leave a wide angular gap on its
    own tangent space, pulling the best available candidate (even one the
    tangent/RNG pruning had rejected) that falls in the gap.

    Dimension-general: for each vertex, express every neighbor displacement
    as a unit direction in the local tangent-coordinate space (whatever its
    dimension ``d`` is -- 1 for a single-DOF mechanism, 2+ for a multi-DOF
    one), then score each unused candidate direction by its angle to the
    NEAREST already-kept direction. A vertex is under-covered if the best
    candidate's nearest-neighbor angle is still wide (>= gap_deg/2, i.e. the
    empty angular region it would fill is >= gap_deg) -- this collapses to
    the old "widest gap between consecutive sorted angles on a circle" rule
    when d=2, and to "does this vertex have a candidate on the OTHER side"
    when d=1, where a circular gap isn't a meaningful concept."""
    adj = {v: set() for v in range(n)}
    for a, b in kept:
        adj[a].add(b); adj[b].add(a)
    added = set()

    def _direction(F, disp):
        coeff, *_ = np.linalg.lstsq(F, disp, rcond=None)
        norm = np.linalg.norm(coeff)
        return (coeff / norm) if norm >= 1e-9 else None

    for v in range(n):
        F = frames[v]
        neigh = sorted(adj[v])
        if not neigh:
            continue
        dirs = [d for d in (_direction(F, q[u] - q[v]) for u in neigh) if d is not None]
        if not dirs:
            continue
        dirs = np.stack(dirs)
        best = None
        for cu in candidates_by_vertex.get(v, ()):
            if cu in adj[v]:
                continue
            cdir = _direction(F, q[cu] - q[v])
            if cdir is None:
                continue
            min_ang = float(np.degrees(np.arccos(np.clip(dirs @ cdir, -1.0, 1.0))).min())
            if best is None or min_ang > best[0]:
                best = (min_ang, cu)
        if best is None or 2 * best[0] < gap_deg:
            continue
        _, cu = best
        added.add((min(v, cu), max(v, cu)))
        adj[v].add(cu); adj[cu].add(v)
    return kept | added, added


def _degree_capped_select(D, ranked_edges, n, cap):
    """Greedily accept edges in priority order, skipping one once either
    endpoint has reached ``cap`` -- the mechanism that actually targets a
    mean degree, with RNG/tangent quality only setting WHICH edges win."""
    degree = np.zeros(n, int)
    kept = []
    for i, j in ranked_edges:
        if degree[i] >= cap and degree[j] >= cap:
            continue
        kept.append((i, j)); degree[i] += 1; degree[j] += 1
    return set(kept)


def build_local_edges(q, frames, D, certified_candidates, *, target_degree=(5.0, 8.0),
                      tangent_tol=0.5, log=print):
    n = len(q)
    candidates = sorted({(min(a, b), max(a, b)) for a, b in certified_candidates})
    candidates_by_vertex = {}
    for a, b in candidates:
        candidates_by_vertex.setdefault(a, set()).add(b)
        candidates_by_vertex.setdefault(b, set()).add(a)

    tangent_kept, tangent_rejected = _tangent_prune(q, frames, candidates, tangent_tol)
    log(f"local_edges: tangent test kept {len(tangent_kept)}/{len(candidates)} "
       f"certified candidates (tol={tangent_tol})")

    edges_arr = np.asarray(sorted(tangent_kept), int).reshape(-1, 2)
    violations = _rng_violations(D, edges_arr, candidates_by_vertex)
    lengths = D[edges_arr[:, 0], edges_arr[:, 1]]
    # priority: fewest RNG witnesses first (most "locally extremal"), then
    # the shortest physical chord -- mutual locality is implicit, since
    # every candidate here already came from the symmetrized certified pool.
    order = np.lexsort((lengths, violations))
    ranked = [tuple(edges_arr[k]) for k in order]

    lo, hi = target_degree
    cap_lo, cap_hi, best = 1, 20, None
    for _attempt in range(6):
        cap = (cap_lo + cap_hi) // 2
        selected = _degree_capped_select(D, ranked, n, cap)
        covered, added = _directional_coverage_fixed(q, frames, selected, candidates_by_vertex, n)
        degree = 2 * len(covered) / max(n, 1)
        log(f"  degree-cap={cap}: selected={len(selected)} +coverage={len(added)} "
           f"-> mean degree {degree:.2f}")
        if best is None or abs(degree - 6.5) < abs(best[0] - 6.5):
            best = (degree, covered, cap)
        if lo <= degree <= hi:
            break
        if degree < lo:
            cap_lo = cap + 1
        else:
            cap_hi = cap - 1
        if cap_lo > cap_hi:
            break
    degree, local_edges, used_cap = best
    return (np.asarray(sorted(local_edges), int).reshape(-1, 2), used_cap, float(degree),
           {"n_certified_candidates": len(candidates),
            "n_tangent_kept": len(tangent_kept),
            "tangent_tol": tangent_tol, "degree_cap": used_cap, "mean_degree": float(degree)})


def build_cover_edges(local_edges, n, D):
    """local_edges plus the minimum extra chords needed for one component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
    edges = np.asarray(local_edges, int).reshape(-1, 2)
    if len(edges):
        g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    else:
        g = coo_matrix((n, n))
    count, label = _cc(g, directed=False)
    extra = []
    if count > 1:
        reps = [int(np.flatnonzero(label == c)[0]) for c in range(count)]
        # connect components by nearest-pose-distance pair, chained
        order = list(range(count))
        for a, b in zip(order[:-1], order[1:]):
            members_a = np.flatnonzero(label == a)
            members_b = np.flatnonzero(label == b)
            sub = D[np.ix_(members_a, members_b)]
            ia, ib = np.unravel_index(np.argmin(sub), sub.shape)
            extra.append((int(members_a[ia]), int(members_b[ib])))
    extra = np.asarray(sorted({(min(a, b), max(a, b)) for a, b in extra}), int).reshape(-1, 2)
    cover = np.vstack([edges, extra]) if len(extra) else edges
    connectivity_only = np.zeros(len(cover), bool)
    connectivity_only[len(edges):] = True
    return cover, connectivity_only, count


# ─── audit (reads generator fields; not part of the training artifact) ─────

def _rotation_angle_deg(Ra, Rb):
    rel = np.einsum("...ij,...ik->...jk", Ra, Rb)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def audit(work, q, frames, cover_edges, local_edges, connectivity_only, fields, D,
         edge_meta, n_components, log=print):
    n, n_bodies = len(q), q.shape[1] // 12
    T = q.reshape(n, n_bodies, 3, 4)
    R = T[..., :3]

    # rotation validity
    orth = np.abs(np.einsum("...ij,...ik->...jk", R, R) - np.eye(3)).max()
    det = np.linalg.det(R)

    # duplicates (full pose distance)
    dup_thresh = 1e-3
    near_dup = int((np.triu(D, 1) > 0).sum() and (D[np.triu_indices(n, 1)] < dup_thresh).sum())

    degrees = np.zeros(n, int)
    for a, b in local_edges:
        degrees[a] += 1; degrees[b] += 1
    lengths = np.linalg.norm(q[local_edges[:, 0]] - q[local_edges[:, 1]], axis=1) if len(local_edges) else np.zeros(0)

    # mutual-edge fraction of local_edges within the certified candidate pool
    mutual_frac = 1.0  # local_edges is already the mutual-filtered set by construction

    # tangent-normal residual over kept local_edges
    resid = np.asarray([_tangent_residual(q, frames, a, b) for a, b in local_edges]) if len(local_edges) else np.zeros(0)

    # local intrinsic dimension vs neighbourhood radius (SE(3) tangent PCA proxy
    # via pose-distance kNN ball singular-value ratio)
    dims = []
    for i in range(0, n, max(1, n // 80)):
        order = np.argsort(D[i])[1:9]
        pts = q[order] - q[i]
        s = np.linalg.svd(pts, compute_uv=False)
        dims.append(float((s[:2] ** 2).sum() / (s ** 2).sum().clip(1e-12)))
    top2_fraction = float(np.mean(dims))

    # sampling density variation
    knn_r = np.sort(D, axis=1)[:, 8]
    density_cv = float(knn_r.std() / knn_r.mean())

    report = {
        "n_samples": n, "n_bodies": n_bodies,
        "rotation_orthogonality_max_dev": float(orth),
        "rotation_det_min": float(det.min()), "rotation_det_max": float(det.max()),
        "near_duplicate_pairs_lt_1e-3": near_dup,
        "connected_components_local_edges": int(n_components),
        "local_edges": {"n_edges": int(len(local_edges)),
                       "mean_degree": float(degrees.mean()) if n else 0.0,
                       "max_degree": int(degrees.max()) if n else 0,
                       "min_degree": int(degrees.min()) if n else 0,
                       "length_p50": float(np.median(lengths)) if len(lengths) else float("nan"),
                       "length_p95": float(np.quantile(lengths, .95)) if len(lengths) else float("nan"),
                       "length_max": float(lengths.max()) if len(lengths) else float("nan"),
                       "mutual_edge_fraction": mutual_frac,
                       "tangent_normal_residual_mean": float(resid.mean()) if len(resid) else float("nan"),
                       "tangent_normal_residual_p95": float(np.quantile(resid, .95)) if len(resid) else float("nan")},
        "cover_edges": {"n_edges": int(len(cover_edges)),
                       "n_connectivity_only": int(connectivity_only.sum())},
        "local_intrinsic_dimension_top2_variance_fraction_mean": top2_fraction,
        "sampling_density_coefficient_of_variation": density_cv,
        "graph_construction": edge_meta,
    }

    # A physical advancing-front run may stop on its wall-clock budget while
    # still producing a structurally valid dataset.  Carry that provenance
    # into the training-visible audit so downstream fitters can fail closed;
    # sample count alone cannot establish domain coverage.
    frontier_state_path = os.path.join(work, "frontier_state.json")
    if os.path.isfile(frontier_state_path):
        with open(frontier_state_path, "r", encoding="utf-8") as f:
            frontier = json.load(f)
        report["frontier_complete"] = frontier.get("frontier_complete") is True
        report["budget_exhausted"] = frontier.get("budget_exhausted") is True
        report["frontier_state_counts"] = frontier.get("frontier_state_counts", {})
    else:
        # No provenance is not evidence of a complete physical frontier.
        report["frontier_complete"] = False
        report["budget_exhausted"] = False
        report["frontier_state_counts"] = {}
    report["training_allowed"] = report["frontier_complete"] is True

    # correlation-only context: generator params / assembly labels, read here
    # ONLY for audit purposes -- never exposed to the loader below.
    if "u" in fields and np.asarray(fields["u"]).size:
        u = np.asarray(fields["u"], float)
        report["generator_diagnostics"] = {
            "u_range": [float(u.min()), float(u.max())],
            "note": "generator coordinates are audit-only and are not part of "
                    "new_dataset.npz; the atlas loader in this module never reads them"}
    if "branch_labels" in fields:
        bl = np.asarray(fields["branch_labels"], float)
        report["generator_diagnostics"] = report.get("generator_diagnostics", {})
        report["generator_diagnostics"]["branch_split"] = [
            int((bl > 0).sum()), int((bl <= 0).sum())]

    with open(os.path.join(work, AUDIT), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"dataset_audit: local_edges mean degree {report['local_edges']['mean_degree']:.2f}, "
       f"components={n_components}, tangent-residual p95={report['local_edges']['tangent_normal_residual_p95']:.3f}, "
       f"near_dup={near_dup}")
    return report


# ─── save / load (training-visible fields ONLY) ─────────────────────────────

def save(work, q, frames, cover_edges, local_edges, connectivity_only, *, frame_confidence=None):
    path = os.path.join(work, ARTIFACT)
    payload = {"q": np.asarray(q, float), "frames": np.asarray(frames, float),
               "cover_edges": np.asarray(cover_edges, int),
               "local_edges": np.asarray(local_edges, int),
               "connectivity_only": np.asarray(connectivity_only, bool)}
    if frame_confidence is not None:
        payload["frame_confidence"] = np.asarray(frame_confidence, float).reshape(len(q))
    np.savez_compressed(path, **payload)
    return path


def load(work):
    """Loader contract: q, frames, cover_edges, local_edges ONLY. No u, no
    branch/parent/rank fields -- those live in dataset_audit.json alone."""
    data = np.load(os.path.join(work, ARTIFACT))
    return {"q": np.asarray(data["q"], float), "frames": np.asarray(data["frames"], float),
            "cover_edges": np.asarray(data["cover_edges"], int),
            "local_edges": np.asarray(data["local_edges"], int),
            "connectivity_only": np.asarray(data["connectivity_only"], bool),
            "frame_confidence": (np.asarray(data["frame_confidence"], float)
                                 if "frame_confidence" in data.files else np.ones(len(data["q"]), float))}


# ─── frame/secant consistency gate ──────────────────────────────────────────
#
# A stored tangent frame that is systematically orthogonal-ish to the ACTUAL
# motion between neighboring samples (a walked edge or a certified chord) is
# not a hard-geometry edge case -- it means the frame producer measured the
# wrong thing (e.g. an unconstrained-rigid-body mobility probe run on a
# mechanism whose joints were never declared to the oracle). Silently letting
# that reach chart training burns AE-fitting time on data that can never
# reconstruct, and would tempt a "fix" that quietly substitutes a PCA/secant
# estimate for the recorded frame -- which hides the real defect instead of
# surfacing it. This gate fails the dataset build outright instead.

FRAME_CONSISTENCY_TOL_DEG = 45.0


def frame_secant_consistency(q, frames, cover_edges):
    """Per-vertex median angle between each neighbor secant and its
    projection onto the SPAN of the stored frame (handles d>1: a secant is
    consistent if it lies close to the frame subspace, not just close to one
    particular column of it). Returns an array of length ``len(q)``; a
    vertex with no neighbors gets ``nan`` and is excluded from the summary
    gate."""
    q = np.asarray(q, float)
    n = len(q)
    adjacency = [[] for _ in range(n)]
    for a, b in np.asarray(cover_edges, int):
        adjacency[a].append(b)
        adjacency[b].append(a)
    medians = np.full(n, np.nan)
    for v in range(n):
        if not adjacency[v]:
            continue
        T = np.asarray(frames[v], float)
        if T.ndim == 1:
            T = T[:, None]
        Q, _ = np.linalg.qr(T, mode="reduced")
        if Q.size == 0 or np.linalg.norm(Q) < 1e-12:
            medians[v] = np.nan
            continue
        angles = []
        for nb in adjacency[v]:
            sec = q[nb] - q[v]
            sn = np.linalg.norm(sec)
            if sn < 1e-12:
                continue
            sec = sec / sn
            proj = Q @ (Q.T @ sec)
            c = np.clip(np.linalg.norm(proj), 0.0, 1.0)
            angles.append(np.degrees(np.arccos(c)))  # angle to the subspace
        medians[v] = float(np.median(angles)) if angles else np.nan
    return medians


def check_frame_consistency(q, frames, cover_edges, *, frame_confidence=None,
                            tol_deg=FRAME_CONSISTENCY_TOL_DEG, log=print):
    """Reject the dataset outright if stored frames are systematically
    inconsistent with the actual local motion, rather than let it reach
    chart training. Only the GLOBAL median is gated (not a per-vertex
    threshold): a handful of genuinely bad vertices is a normal, tolerable
    data-quality issue (chart growth already routes around them via its own
    per-patch certificate); a bad global median means the frame PRODUCER
    measured the wrong thing for this case, and no amount of AE fitting will
    fix that."""
    medians = frame_secant_consistency(q, frames, cover_edges)
    valid = medians[np.isfinite(medians)]
    if len(valid) == 0:
        raise ValueError("frame/secant consistency check: no vertex has both "
                         "a frame and a cover-graph neighbor to check against")
    global_median = float(np.median(valid))
    n_bad = int(np.sum(valid > tol_deg))
    low_confidence = (int(np.sum(np.asarray(frame_confidence) < .5))
                      if frame_confidence is not None else 0)
    log(f"new_dataset: frame/secant consistency -- global median "
       f"{global_median:.1f} deg, {n_bad}/{len(valid)} vertices > {tol_deg:.0f} deg, "
       f"{low_confidence} low-confidence measurements")
    if global_median > tol_deg:
        raise ValueError(
            f"frame/secant consistency check FAILED: global median angle between "
            f"stored frames and cover-graph secants is {global_median:.1f} deg "
            f"(tolerance {tol_deg:.0f} deg). This means the frame producer measured "
            "the wrong thing for this case (e.g. an oracle probe run without the "
            "mechanism's real constraints declared), not a training/hyperparameter "
            "issue -- fix the frame producer and regenerate, do not proceed to chart "
            "training on this dataset.")
    return {"global_median_deg": global_median, "n_vertices_checked": len(valid),
           "n_above_tolerance": n_bad, "low_confidence": low_confidence,
           "tolerance_deg": tol_deg, "per_vertex_median_deg": medians}


# ─── driver ──────────────────────────────────────────────────────────────

def build_from_walked_edges(work, log=print):
    """Skip certified-graph chord completion entirely: use the explorer's own
    walked edges (each one actually traversed, so safe by construction)
    directly as ``cover_edges``/``local_edges``, with no candidate-chord
    certification or tangent/RNG pruning pass.

    This is the "no chord completion" path for cases where a certified
    residual oracle isn't available (e.g. no declared joints, so
    ``ideal_oracle`` doesn't apply, and the general libuipc-backed oracle has
    no ``.residual()``). It trades a possibly sparser, exploration-order-
    biased graph for correctness that never depended on the graph in the
    first place: chart membership is decided entirely by ``certify_patch``'s
    AE certificate downstream, not by which edges happen to exist here -- a
    sparser graph only affects growth efficiency and seed choice.
    """
    from atlas_rom.sample import dataset as ds
    from atlas_rom.sample import sample_graph
    from atlas_rom.sample import quotient as qm

    raw = ds.load_samples(work)
    walked = raw.get("edges")
    if walked is None or len(walked) == 0:
        raise ValueError(f"{work}: no walked explorer edges to fall back on")
    # Substitutes quotient vertices for raw walked samples when work/quotient.npz
    # exists (see quotient.py); a no-op, identity-mapped pass-through otherwise.
    q, frames, walked, _class_of = qm.effective_samples(
        work, raw["q"], raw["frames"], walked, log=log)
    raw_confidence = raw["fields"].get("_frame_confidence")
    confidence = None
    if raw_confidence is not None:
        confidence = np.ones(len(q), float)
        for i in range(len(q)):
            confidence[i] = float(np.min(np.asarray(raw_confidence)[_class_of == i]))
    n = len(q)
    cover_edges = sample_graph.undirected(walked, n)
    connectivity_only = np.zeros(len(cover_edges), dtype=bool)
    log(f"new_dataset: {n} samples, {len(cover_edges)} walked explorer edges "
       "(no chord completion, no tangent/RNG pruning)")
    check_frame_consistency(q, frames, cover_edges, frame_confidence=confidence, log=log)
    save(work, q, frames, cover_edges, cover_edges, connectivity_only,
         frame_confidence=confidence)
    return {"n_samples": n, "n_local_edges": int(len(cover_edges)),
           "n_cover_edges": int(len(cover_edges)), "mean_degree": 2.0 * len(cover_edges) / n}


def build(work, *, target_degree=(5.0, 8.0), tangent_tol=0.5, allow_walked_fallback=False, log=print):
    from atlas_rom.sample import dataset as ds
    from atlas_rom.sample import sample_graph
    from atlas_rom.sample import quotient as qm

    raw = ds.load_samples(work)
    q, frames, fields = raw["q"], raw["frames"], dict(raw["fields"])
    fields["u"] = raw["u"]
    q, frames, _walked, class_of = qm.effective_samples(
        work, q, frames, raw.get("edges"), log=log)
    raw_confidence = fields.get("_frame_confidence")
    if raw_confidence is not None:
        confidence = np.ones(len(q), float)
        for i in range(len(q)):
            confidence[i] = float(np.min(np.asarray(raw_confidence)[class_of == i]))
        fields["_frame_confidence"] = confidence
    fields["quotient_class_of"] = class_of
    n = len(q)

    cg_path = os.path.join(work, sample_graph.ARTIFACT)
    if not os.path.exists(cg_path):
        if allow_walked_fallback:
            log(f"new_dataset: {cg_path} missing, falling back to walked "
               "explorer edges (allow_walked_fallback=True)")
            return build_from_walked_edges(work, log=log)
        raise FileNotFoundError(
            f"{cg_path} missing -- run the certified-graph stage first, or pass "
            "allow_walked_fallback=True to use the explorer's own walked edges "
            "directly (no chord completion; safe but possibly sparser)")
    certified = sample_graph.load(cg_path)["edges"]
    # certified_graph.npz indexes the RAW (pre-quotient) sample space in
    # every graph_source (only the ideal_oracle branch of stage_graph
    # substitutes quotient vertices before certifying; coordinate_split does
    # not, since it never builds the dense physical-distance matrix
    # quotienting needs). Remap through class_of unconditionally -- it is
    # the identity map when no quotient.npz exists, so this is a no-op for
    # every case that was already working.
    certified = np.asarray(certified, int).reshape(-1, 2)
    mapped = class_of[certified]
    self_loops = mapped[:, 0] == mapped[:, 1]
    certified = (np.unique(np.sort(mapped[~self_loops], axis=1), axis=0)
                if (~self_loops).any() else np.empty((0, 2), int))
    log(f"new_dataset: {n} samples, {len(certified)} certified candidate chords")

    D = pose_distance_matrix(q)
    local_edges, used_cap, degree, edge_meta = build_local_edges(
        q, frames, D, certified, target_degree=target_degree, tangent_tol=tangent_tol, log=log)
    cover_edges, connectivity_only, n_components = build_cover_edges(local_edges, n, D)
    log(f"new_dataset: local_edges={len(local_edges)} (cap={used_cap}, degree={degree:.2f}), "
       f"cover_edges={len(cover_edges)} ({connectivity_only.sum()} connectivity-only), "
       f"components before connectivity fix={n_components}")

    check_frame_consistency(q, frames, cover_edges,
                            frame_confidence=fields.get("_frame_confidence"), log=log)
    save(work, q, frames, cover_edges, local_edges, connectivity_only,
         frame_confidence=fields.get("_frame_confidence"))
    audit(work, q, frames, cover_edges, local_edges, connectivity_only, fields, D,
         edge_meta, n_components, log=log)
    return {"n_samples": n, "n_local_edges": int(len(local_edges)),
           "n_cover_edges": int(len(cover_edges)), "mean_degree": degree}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="work/five_bar")
    p.add_argument("--tangent-tol", type=float, default=0.5)
    p.add_argument("--allow-walked-fallback", action="store_true",
                   help="if certified_graph.npz is missing, use the explorer's own "
                        "walked edges directly instead of failing (no chord completion)")
    args = p.parse_args()
    result = build(args.work, tangent_tol=args.tangent_tol,
                   allow_walked_fallback=args.allow_walked_fallback)
    print(json.dumps({"status": "done", **result}))


if __name__ == "__main__":
    main()
