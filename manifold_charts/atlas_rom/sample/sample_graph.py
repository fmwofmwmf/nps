"""sample_graph.py — the certified neighbourhood graph of a sampled manifold.

The explorer records ONE PARENT LINK per accepted sample, so `samples["edges"]`
is a spanning TREE.  A tree is a legitimate record of what was walked and a
terrible model of the manifold: tree distance is search order, not geometry —
siblings detour through their common ancestor — so anything built on it
(geodesics, a Laplacian, a projection cover) inherits the search, not the shape.
Isomap on the bare three_rrs tree returns a flat spectrum (0.325, 0.115, 0.097,
...) with no 3-D structure at all, where the completed graph gives 0.371, 0.311,
0.281, 0.010.

The completion cannot be plain kNN, because kNN is exactly the rule that fuses
two assembly modes wherever they pass close — the failure this whole line of work
exists to avoid.  Every candidate chord is certified by its MIDPOINT instead:

  * inside ONE sheet the midpoint is a secant's, off the manifold by curvature
    times length^2, so `residual / length^2` is bounded by the local curvature;
  * a chord ACROSS sheets has its midpoint in empty configuration space, and its
    residual does not shrink with length at all.

The cut is not invented.  The tree edges are chords the explorer actually walked,
so their residual ratios ARE the same-sheet distribution for this mechanism, and
the threshold is a margin above their 99th percentile.  A case with no analytic
residual cannot use this and says so rather than falling back to kNN.

This module is the one place that rule lives.  `intrinsic_probe` uses it to
decide whether a declared coordinate is single-valued; `build` writes its output
as `certified_graph.npz`, and the canonical stage builds z, the projection cover
and the retraction graph on it.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ARTIFACT = "certified_graph.npz"


def certified_edges(q, edges, oracle, k=16, audit=400, log=print,
                    distance_matrix=None):
    """The explorer's tree, plus every short chord that stays on ONE sheet.

    Returns `(edges, report)`.  `edges` contains the original tree links
    unchanged — they were walked, so they are certified by construction — with
    the accepted chords appended.

    The residual test stands in for a full projection because 688k projections is
    twenty minutes and 688k residuals is seconds; `audit` re-checks a subset the
    expensive way and reports the agreement.

    That audit is NOT random.  Residual proportional to length^2 is a sound
    same-sheet test in general position, but near a fold two sheets approach each
    other and their midpoint can be genuinely low-residual — or project onto a
    third solution entirely — so the test's errors all live near its threshold.
    A uniform sample spends its budget where the verdict was never in doubt.
    Half the audit therefore goes to the chords whose ratio sits closest to the
    cut, and half stays uniform so a systematic error away from the threshold
    still has a way of showing up.
    """
    q = np.asarray(q, float)
    # Do not retain the explorer's continuation tree: all topology below is
    # rebuilt from physical pose distance and projection-certified connections.
    edges = np.asarray(edges if edges is not None else [], int).reshape(-1, 2)

    if len(q) < 2:
        return edges, {
            "graph_source": "physical_pose_distance_plus_oracle_projection_v2",
            "n_tree_edges": int(len(edges)), "n_candidates": 0,
            "n_certified_chords": 0, "rejected_fraction": 0.0,
            "cut": None, "walked_median_ratio": None, "walked_p99_ratio": None,
            "audit_size": 0, "audit_agreement": 1.0,
            "audit_agreement_marginal": 1.0, "knn_k": int(k),
        }

    def ratio(pairs):
        mid = 0.5 * (q[pairs[:, 0]] + q[pairs[:, 1]])
        length = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
        res = np.array([np.linalg.norm(oracle.residual(m), np.inf) for m in mid])
        return res / np.maximum(length ** 2, 1e-12), length

    if distance_matrix is None:
        tree = cKDTree(q)
        _, idx = tree.query(q, k=min(k + 1, len(q)))
    else:
        distances = np.asarray(distance_matrix, float)
        if distances.shape != (len(q), len(q)):
            raise ValueError("physical distance matrix has the wrong shape")
        idx = np.argsort(distances, axis=1)[:, :min(k + 1, len(q))]
    a = np.repeat(np.arange(len(q)), idx.shape[1] - 1)
    cand = np.unique(np.sort(np.c_[a, idx[:, 1:].ravel()], axis=1), axis=0)
    drift, length = ratio(cand)
    # ``residual`` is the physical midpoint deviation.  For SDF observations
    # it is measured by re-evaluating the SDF from the observed pose; for pose
    # mechanisms it remains the closure residual.  Neither path accepts an
    # ambient kNN edge without this physical check.
    ok = drift < 0.25 * length
    log(f"chords: {len(cand)} physical-kNN candidates, {ok.sum()} certified "
        f"({1 - ok.mean():.1%} rejected as off-sheet)")

    rng = np.random.default_rng(0)
    audit = min(audit, len(cand))
    margin = np.abs(drift / np.maximum(length, 1e-12) - .25)
    near = np.argsort(margin)[:audit // 2]
    rest = np.setdiff1d(np.arange(len(cand)), near)
    pick = np.r_[near, rng.choice(rest, size=audit - len(near), replace=False)]
    strict = drift[pick] < 0.25 * length[pick]
    agree = strict == ok[pick]
    report = {
        "graph_source": "physical_pose_distance_plus_oracle_projection_v2",
        "n_tree_edges": 0,
        "n_candidates": int(len(cand)),
        "n_certified_chords": int(ok.sum()),
        "rejected_fraction": float(1.0 - ok.mean()),
        "cut": None,
        "walked_median_ratio": None,
        "walked_p99_ratio": None,
        "audit_size": int(len(pick)),
        "audit_agreement": float(agree.mean()),
        "audit_agreement_marginal": (float(agree[:len(near)].mean())
                                      if len(near) else 1.0),
        "knn_k": int(k),
    }
    return np.vstack([edges, cand[ok]]), report


def undirected(edges, n):
    """Deduplicated, self-loop-free, sorted (i<j) edge list."""
    edges = np.asarray(edges, int).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if not len(edges):
        return np.empty((0, 2), int)
    out = np.unique(np.sort(edges, axis=1), axis=0)
    if out.min() < 0 or out.max() >= n:
        raise ValueError("edge references a sample outside the cloud")
    return out


def adjacency(edges, n, weights=None):
    """Symmetric CSR adjacency; unit weights unless `weights` is given."""
    edges = np.asarray(edges, int).reshape(-1, 2)
    w = (np.ones(len(edges)) if weights is None
         else np.asarray(weights, float).reshape(len(edges)))
    return coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]],
                                     np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()


def neighbour_lists(edges, n):
    """Per-sample sorted neighbour index arrays, the mesh-free `_mesh_neighbours`."""
    sets = [set() for _ in range(n)]
    for a, b in np.asarray(edges, int).reshape(-1, 2):
        sets[int(a)].add(int(b))
        sets[int(b)].add(int(a))
    return [np.asarray(sorted(s), int) for s in sets]


def largest_component(edges, n):
    """Boolean mask of the largest connected component, and its edge subset.

    Everything downstream — the Laplacian, the cover, the retraction graph —
    requires one component, and a certified graph can leave a handful of samples
    stranded where the explorer reached them by a single step that the chord test
    then had no company for.
    """
    count, label = connected_components(adjacency(edges, n), directed=False)
    keep = label == np.argmax(np.bincount(label))
    sub = np.asarray(edges, int).reshape(-1, 2)
    sub = sub[keep[sub[:, 0]] & keep[sub[:, 1]]]
    return keep, sub, int(count)


def hop_ball(edges, n, hops):
    """Boolean CSR of `hop distance <= hops`, the sparse replacement for a dense
    truncated all-pairs BFS.

    `canonical.certify` needs one predicate — is this pair FAR on the manifold —
    and the dense `(N, N)` hop matrix that used to answer it is 36M entries at
    6000 samples and a 1.7 GB temporary in the distance step that follows.  On a
    d-dimensional cloud the hop ball itself stays small (a few hundred entries
    per row at four hops), so the sparse form is both smaller and exact.
    """
    from scipy.sparse import identity
    a = (adjacency(edges, n) + identity(n, format="csr")).astype(bool)
    ball = a.copy()
    for _ in range(int(hops) - 1):
        ball = (ball @ a).astype(bool)
    ball.setdiag(True)
    return ball.tocsr()


# ─── artifact io ──────────────────────────────────────────────────────────────

def save(path, edges, report, keep=None):
    np.savez_compressed(path, edges=np.asarray(edges, int),
                        keep=(np.ones(0, bool) if keep is None
                              else np.asarray(keep, bool)),
                        report=np.asarray(report, dtype=object))


def load(path):
    data = np.load(path, allow_pickle=True)
    return {"edges": np.asarray(data["edges"], int),
            "keep": (np.asarray(data["keep"], bool)
                     if data["keep"].size else None),
            "report": data["report"].item()}
