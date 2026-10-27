"""product_split.py — is the configuration manifold a Riemannian product?

A manifold splits isometrically, M = A x B, exactly when its tangent bundle
splits into two orthogonal pieces that parallel transport preserves (de Rham).
This module looks for that split in simulator evidence only: sampled poses,
the tangent frames the advancing front inferred from responses, and fresh
`PhysicalSystem` responses.  It never reads a declared coordinate, dimension,
joint, body grouping, or evaluator reference.  It reports evidence; it does not
change how anything is sampled or charted.

STAGE 1 — local evidence, from a sample set
    Neighbouring frames are related by their orthogonal Procrustes rotation, a
    discrete Levi-Civita transport.  Its graph connection Laplacian (vector
    diffusion maps, Singer & Wu 2012) has a near-zero eigenvector for every
    PARALLEL vector field; a parallel field splits off a flat 1-D factor.  The
    count is read off a spectral gap, measured against the scalar Laplacian's
    first eigenvalue on the same graph so it is scale free.  Small loops give
    holonomy rotations; the subspaces every holonomy preserves are the split of
    a curved product, found as the commutant of the holonomy generators.

STAGE 2 — global evidence, for the flat case
    On a FLAT manifold every transport is trivial, every split commutes, and
    local evidence cannot choose one.  A boundary can: a product with an
    interval factor has inward boundary normals that are one parallel field
    (one direction in transported coordinates, up to sign), and a width
    between its two faces that is the same everywhere.

STAGE 3 — physical certificate, with fresh responses
    Walk A-then-B and B-then-A from a pose, re-measuring the frame by probes at
    every substep and carrying the split along by Procrustes (a constant
    ambient command direction would follow P·v, not a parallel field, and on a
    product that is not an EXTRINSIC product — a rotation composed of two
    rotations — it would falsely fail).  The closure gap must shrink as the
    substeps refine.  For an interval factor, the width to the boundary along
    it must not depend on where it is measured.  Both are refinement or
    invariance statements, not thresholds on a single number.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix, diags, eye as sparse_eye
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh

from atlas_rom.sample.physical_advancing_front import _rank_by_gap


@dataclass
class ProductSplitOptions:
    k_neighbours: int | None = None       # default 2d + 2
    gap_ratio: float = 10.0               # parallel-field and commutant spectral gap
    n_eigen_extra: int = 2                # eigenvalues computed beyond d
    max_loops: int = 20000                # triangles used for holonomy
    boundary_score: float = 0.25          # max -cos of any neighbour against the inward mean
    boundary_alignment: float = 0.9       # fraction of normal variance on one axis
    seed: int = 0


# ── transports ───────────────────────────────────────────────────────────────
def polar(matrix: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix (the rotation part of `matrix`)."""
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def transport(frame_to: np.ndarray, frame_from: np.ndarray) -> np.ndarray:
    """d x d rotation taking coordinates in `frame_from` to `frame_to`."""
    return polar(frame_to.T @ frame_from)


def neighbour_graph(q: np.ndarray, edges=None, k: int = 8, chunk: int = 512) -> np.ndarray:
    """Undirected edges: each node's k nearest neighbours, plus any walked edges.

    Brute-force distances in bounded chunks (`chunk` x N at a time), so memory
    stays O(chunk * N) whatever the sample count.
    """
    q = np.asarray(q, float)
    n = len(q)
    k = min(k, n - 1)
    sq = np.einsum("ij,ij->i", q, q)
    pairs = []
    for start in range(0, n, chunk):
        block = q[start:start + chunk]
        dist = sq[start:start + chunk, None] + sq[None, :] - 2.0 * block @ q.T
        rows = np.arange(len(block))
        dist[rows, start + rows] = np.inf
        nearest = np.argpartition(dist, k, axis=1)[:, :k]
        pairs.append(np.column_stack((np.repeat(start + rows, k), nearest.reshape(-1))))
    if edges is not None and len(edges):
        pairs.append(np.asarray(edges, int).reshape(-1, 2))
    pairs = np.sort(np.vstack(pairs), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


# ── stage 1: parallel fields and holonomy ────────────────────────────────────
def parallel_fields(frames: np.ndarray, edges: np.ndarray, n_eigen: int):
    """Smallest eigenpairs of the normalised connection Laplacian, scaled by
    the scalar Laplacian's first non-zero eigenvalue on the same graph.

    Returns (mu, fields) with fields shaped (N, d, n_eigen) in each node's own
    frame coordinates."""
    n, _, d = frames.shape
    i, j = edges[:, 0], edges[:, 1]
    O = np.stack([transport(frames[a], frames[b]) for a, b in edges])     # j-coords -> i-coords
    rows, cols, vals = [], [], []
    for (a, b), R in zip(edges, O):
        for blk_r, blk_c, M in ((a, b, R), (b, a, R.T)):
            rr, cc = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
            rows.append((blk_r * d + rr).ravel())
            cols.append((blk_c * d + cc).ravel())
            vals.append(M.ravel())
    W = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n * d, n * d))
    degree = np.bincount(np.concatenate((i, j)), minlength=n).astype(float)
    inv_sqrt = diags(np.repeat(1.0 / np.sqrt(degree), d))
    L = sparse_eye(n * d) - inv_sqrt @ W @ inv_sqrt
    A = csr_matrix((np.ones(2 * len(edges)), (np.concatenate((i, j)), np.concatenate((j, i)))),
                   shape=(n, n))
    s_inv = diags(1.0 / np.sqrt(degree))
    Ls = sparse_eye(n) - s_inv @ A @ s_inv
    scalar = np.sort(eigsh(Ls, k=2, sigma=-1e-6, which="LM", return_eigenvectors=False))
    values, vectors = eigsh(L, k=n_eigen, sigma=-1e-6, which="LM")
    order = np.argsort(values)
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    fields = (np.repeat(1.0 / np.sqrt(degree), d)[:, None] * vectors).reshape(n, d, n_eigen)
    return values / max(float(scalar[1]), 1e-300), fields, O


def parallel_count(mu: np.ndarray, d: int, gap_ratio: float) -> int:
    """Number of eigenvalues below the widest decisive gap among the first d+1."""
    mu = np.asarray(mu, float)[:d + 1]
    floor = max(float(mu[-1]) * np.finfo(float).eps, 1e-300)
    ratios = np.maximum(mu[1:], floor) / np.maximum(mu[:-1], floor)
    k = int(np.argmax(ratios))
    return k + 1 if ratios[k] >= gap_ratio else 0


def tree_transports(n: int, edges: np.ndarray, O: np.ndarray, root: int = 0):
    """Per node, the rotation from its frame coordinates to the ROOT's, along a
    BFS tree.  Nodes the tree does not reach get None."""
    adjacency = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adjacency[a].append((b, e, False))
        adjacency[b].append((a, e, True))
    T = [None] * n
    T[root] = np.eye(O.shape[1])
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b, e, flipped in adjacency[a]:
            if T[b] is None:
                to_a = O[e] if not flipped else O[e].T       # b-coords -> a-coords
                T[b] = T[a] @ to_a
                queue.append(b)
    return T


def holonomy_generators(q, edges, O, T, max_loops, rng):
    """Skew parts of triangle holonomies, in root coordinates, per unit area."""
    n = len(q)
    lookup = {}
    for e, (a, b) in enumerate(edges):
        lookup[(a, b)] = O[e]                      # b -> a
        lookup[(b, a)] = O[e].T                    # a -> b
    nbrs = [set() for _ in range(n)]
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    triangles = [(a, b, c) for a, b in edges for c in nbrs[a] & nbrs[b] if c > b]
    if len(triangles) > max_loops:
        triangles = [triangles[t] for t in rng.choice(len(triangles), max_loops, replace=False)]
    generators, areas = [], []
    for a, b, c in triangles:
        if T[a] is None:
            continue
        # a -> b -> c -> a, as a map on a's coordinates
        H = lookup[(a, c)] @ lookup[(c, b)] @ lookup[(b, a)]
        u, v = q[b] - q[a], q[c] - q[a]
        area = 0.5 * np.sqrt(max(float((u @ u) * (v @ v) - (u @ v) ** 2), 0.0))
        if area <= 0.0:
            continue
        S = 0.5 * (H - H.T)
        generators.append(T[a] @ S @ T[a].T)
        areas.append(area)
    return np.asarray(generators), np.asarray(areas)


def invariant_split(generators: np.ndarray, d: int, gap_ratio: float):
    """Subspaces every holonomy generator preserves: eigenspaces of a generic
    symmetric X in the commutant {X : G X = X G for all G}.

    Returns (commutant_dim, blocks) with blocks a list of (d, k) bases in root
    coordinates."""
    basis = []
    for a in range(d):
        for b in range(a, d):
            E = np.zeros((d, d))
            E[a, b] = E[b, a] = 1.0
            basis.append(E)
    # Accumulate the (params x params) Gram of the stacked commutator rows, so
    # memory does not grow with the loop count.
    gram = np.zeros((len(basis), len(basis)))
    for G in generators:
        rows = np.stack([(G @ E - E @ G).ravel() for E in basis], 1)
        gram += rows.T @ rows
    w, vecs = np.linalg.eigh(gram)
    s = np.sqrt(np.maximum(w[::-1], 0.0))                  # singular values, descending
    vt = vecs[:, ::-1].T
    rank = _rank_by_gap(s, gap_ratio)
    rank = len(basis) if rank is None else rank
    null = vt[rank:]
    dim = len(null)
    if dim <= 1:
        return dim, [np.eye(d)]
    rng = np.random.default_rng(1)
    X = sum(c * sum(w * E for w, E in zip(v, basis)) for c, v in zip(rng.standard_normal(dim), null))
    w, V = np.linalg.eigh(X)
    spread = max(float(np.ptp(w)), 1e-300)
    groups, current = [], [0]
    for k in range(1, d):
        if (w[k] - w[k - 1]) / spread < 1e-6:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)
    return dim, [V[:, g] for g in groups]


# ── stage 2: boundary-selected split of a flat manifold ──────────────────────
def boundary_normals(q, frames, edges, score_cut):
    """Nodes whose neighbours all lie on one side, and their inward unit
    normals in each node's frame coordinates.

    The candidate inward direction is the mean unit secant.  A node is on the
    boundary when no neighbour lies against it by more than `score_cut` (the
    largest -cos between a secant and that direction): an interior node has a
    neighbour roughly opposite (-cos ~ 1), a boundary node only neighbours
    along the boundary or inward (-cos ~ 0).  The mean alone is diluted by
    neighbours along the boundary, so it only supplies the direction."""
    n, _, d = frames.shape
    secants = [[] for _ in range(n)]
    for a, b in edges:
        for s, t in ((a, b), (b, a)):
            v = frames[s].T @ (q[t] - q[s])
            norm = np.linalg.norm(v)
            if norm > 0:
                secants[s].append(v / norm)
    normals = np.zeros((n, d))
    against = np.ones(n)
    for s, vs in enumerate(secants):
        if not vs:
            continue
        vs = np.asarray(vs)
        mean = vs.mean(0)
        norm = np.linalg.norm(mean)
        if norm <= 1e-9:
            continue
        normals[s] = mean / norm
        against[s] = float(np.max(-(vs @ normals[s])))
    nodes = np.flatnonzero(against < score_cut)
    return nodes, normals[nodes], against


def boundary_split(q, frames, edges, T, options: ProductSplitOptions):
    """Flat-case split chosen by the boundary, with its width spread."""
    nodes, normals, _ = boundary_normals(q, frames, edges, options.boundary_score)
    reached = [k for k, i in enumerate(nodes) if T[i] is not None]
    if len(reached) < 2:
        return {"n_boundary_nodes": int(len(nodes)), "split": None}
    root_normals = np.stack([T[nodes[k]] @ normals[k] for k in reached])
    w, V = np.linalg.eigh(root_normals.T @ root_normals)
    alignment = float(w[-1] / max(w.sum(), 1e-300))
    axis = V[:, -1]
    report = {"n_boundary_nodes": int(len(nodes)), "normal_alignment": alignment, "split": None}
    if alignment < options.boundary_alignment:
        return report
    side = np.sign(root_normals @ axis)
    faces = [nodes[[k for k, s in zip(reached, side) if s == sign]] for sign in (1.0, -1.0)]
    if min(len(f) for f in faces) == 0:
        report["faces"] = [int(len(f)) for f in faces]
        return report
    lengths = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1)
    graph = csr_matrix((np.concatenate((lengths, lengths)),
                        (np.concatenate((edges[:, 0], edges[:, 1])),
                         np.concatenate((edges[:, 1], edges[:, 0])))), shape=(len(q), len(q)))
    dist = [dijkstra(graph, indices=f, min_only=True) for f in faces]
    interior = np.setdiff1d(np.arange(len(q)), nodes)
    width = dist[0][interior] + dist[1][interior]
    width = width[np.isfinite(width)]
    d = frames.shape[2]
    complement = np.linalg.svd(np.eye(d) - np.outer(axis, axis))[0][:, :d - 1]
    # Graph geodesics zigzag along k-NN edges, so this spread has a
    # discretisation floor of several percent even on an exact product (0.04
    # on a gridded cylinder, 0.13 on a gridded T^2 x I).  It is a coarse
    # report; the width certificate is `width_along` in stage 3.
    report.update({"faces": [int(len(f)) for f in faces],
                   "graph_width_mean": float(width.mean()),
                   "graph_width_cv": float(width.std() / width.mean()),
                   "split": [complement, axis[:, None]]})
    return report


def detect_product_split(q, frames, edges=None, *, options: ProductSplitOptions | None = None,
                         root: int = 0):
    """Stages 1 and 2 on a sample set.  Returns a JSON-ready report plus the
    split bases (root frame coordinates) under the key "_bases"."""
    options = options or ProductSplitOptions()
    q = np.asarray(q, float)
    frames = np.asarray(frames, float)
    n, _, d = frames.shape
    rng = np.random.default_rng(options.seed)
    graph = neighbour_graph(q, edges, options.k_neighbours or 2 * d + 2)
    # A disconnected graph has a zero scalar eigenvalue per component, which
    # would make every connection eigenvalue look large against it.  Work on
    # the root's component and say how much was left out.
    _, label = connected_components(
        csr_matrix((np.ones(len(graph)), (graph[:, 0], graph[:, 1])), shape=(n, n)), directed=False)
    keep = np.flatnonzero(label == label[root])
    if len(keep) < n:
        remap = -np.ones(n, int)
        remap[keep] = np.arange(len(keep))
        graph = remap[graph[(label[graph[:, 0]] == label[root])]]
        q, frames, root = q[keep], frames[keep], int(remap[root])
    n_all, n = n, len(keep)
    mu, fields, O = parallel_fields(frames, graph, d + options.n_eigen_extra)
    p = parallel_count(mu, d, options.gap_ratio)
    T = tree_transports(n, graph, O, root)
    report = {"dimension": int(d), "n_nodes": int(n_all), "n_component": int(n),
              "n_edges": int(len(graph)),
              "reached": int(sum(t is not None for t in T)),
              "parallel": {"mu": mu.tolist(), "count": int(p)}}
    if p == d:
        report["kind"] = "flat"
        boundary = boundary_split(q, frames, graph, T, options)
        bases = boundary.pop("split")
        report["boundary"] = boundary
        if bases is None:
            report["split"] = {"kind": "flat_ambiguous", "factors": [d]}
            return {**report, "_bases": [np.eye(d)]}
        report["split"] = {"kind": "flat_boundary", "factors": [b.shape[1] for b in bases]}
        return {**report, "_bases": bases}
    G, areas = holonomy_generators(q, graph, O, T, options.max_loops, rng)
    dim, blocks = invariant_split(G / areas[:, None, None] if len(G) else np.zeros((1, d, d)),
                                  d, options.gap_ratio)
    report["kind"] = "curved"
    report["holonomy"] = {"n_loops": int(len(G)), "commutant_dim": int(dim),
                          "curvature_median": float(np.median(np.linalg.norm(G, axis=(1, 2)) / areas))
                          if len(G) else 0.0}
    if len(blocks) == 1:
        report["split"] = {"kind": "irreducible", "factors": [d]}
    else:
        report["split"] = {"kind": "holonomy", "factors": [b.shape[1] for b in blocks]}
    return {**report, "_bases": blocks}


# ── stage 3: physical certificate ────────────────────────────────────────────
def infer_frame(system, observation, d, h, guide=None, n_random=None, rng=None):
    """Tangent frame at a pose from central-difference probe responses.

    `guide` (ambient, d) directions top up random ones, as the front's walked
    secants do; d itself comes from the sample set's inferred frames."""
    rng = rng or np.random.default_rng(0)
    pose = np.asarray(observation.pose, float).reshape(-1)
    count = n_random if n_random is not None else max(d, int(np.ceil(np.log2(pose.size + 1))))
    directions = np.linalg.qr(rng.standard_normal((pose.size, count)))[0]
    if guide is not None:
        directions = np.column_stack((guide, directions))
    session = system.restore(observation)
    secants = []
    for column in directions.T:
        plus, minus = system.command_many(session, (column, -column), h)
        if plus.valid and minus.valid and plus.converged and minus.converged:
            secants.append((np.asarray(plus.observation.pose, float).reshape(-1)
                            - np.asarray(minus.observation.pose, float).reshape(-1)) / (2 * h))
    if len(secants) < d:
        return None
    u = np.linalg.svd(np.column_stack(secants), full_matrices=False)[0]
    return u[:, :d]


def _march(system, observation, frame, basis, column, length, substeps, h, rng):
    """Follow basis[:, column] (frame coordinates) for `length`, re-measuring
    the frame and transporting `basis` at every substep.  None at a boundary."""
    step = length / substeps
    for _ in range(substeps):
        direction = frame @ basis[:, column]
        direction /= np.linalg.norm(direction)
        response = system.command(system.restore(observation), np.sign(step) * direction, abs(step))
        if not (response.valid and response.converged):
            return None
        new_frame = infer_frame(system, response.observation, frame.shape[1], h,
                                guide=frame, rng=rng)
        if new_frame is None:
            return None
        basis = transport(new_frame, frame) @ basis
        observation, frame = response.observation, new_frame
    return observation, frame, basis


def commuting_gap(system, observation, frame, basis_a, basis_b, length, substeps, *, h=None, rng=None):
    """|pose(A then B) - pose(B then A)| / length for the first column of each
    factor, at each substep count."""
    rng = rng or np.random.default_rng(0)
    basis = np.column_stack((basis_a[:, :1], basis_b[:, :1]))
    gaps = []
    for m in substeps:
        hh = h if h is not None else 0.25 * length / m
        ends = []
        for first, second in ((0, 1), (1, 0)):
            mid = _march(system, observation, frame, basis, first, length, m, hh, rng)
            end = None if mid is None else _march(system, *mid, second, length, m, hh, rng)
            ends.append(None if end is None else np.asarray(end[0].pose, float).reshape(-1))
        gaps.append(None if any(e is None for e in ends)
                    else float(np.linalg.norm(ends[0] - ends[1]) / length))
    return gaps


def width_along(system, observation, frame, axis, *, substep, h=None, bisect=12, max_steps=10000,
                rng=None):
    """Walked length from one boundary to the other along a transported axis
    (frame coordinates), each end located by bisection on the last substep."""
    rng = rng or np.random.default_rng(0)
    hh = h if h is not None else 0.25 * substep
    total = 0.0
    for sign in (1.0, -1.0):
        obs, fr, basis = observation, frame, sign * np.asarray(axis, float).reshape(-1, 1)
        for _ in range(max_steps):
            nxt = _march(system, obs, fr, basis, 0, substep, 1, hh, rng)
            if nxt is None:
                lo, hi = 0.0, substep
                for _ in range(bisect):
                    mid = 0.5 * (lo + hi)
                    direction = fr @ basis[:, 0]
                    response = system.command(system.restore(obs), direction / np.linalg.norm(direction), mid)
                    lo, hi = (mid, hi) if (response.valid and response.converged) else (lo, mid)
                total += lo
                break
            obs, fr, basis = nxt
            total += substep
        else:
            return None
    return total


def certify_split(system, observation, frame, bases, *, length, substeps=(4, 8, 16),
                  width_bases=None, width_substep=None, rng=None):
    """Stage 3 report.  `bases` are the two factors in `frame` coordinates;
    `width_bases` optional (observation, frame, axis) triples, one per base
    pose, to measure an interval factor's width at."""
    rng = rng or np.random.default_rng(0)
    gaps = commuting_gap(system, observation, frame, bases[0], bases[1], length, substeps, rng=rng)
    report = {"length": float(length), "substeps": list(substeps), "commuting_gap": gaps}
    valid = [g for g in gaps if g is not None]
    if len(valid) >= 2 and valid[-2] > 0:
        report["gap_refinement_ratio"] = float(valid[-1] / valid[-2])
    if width_bases:
        widths = [width_along(system, obs, fr, axis, substep=width_substep, rng=rng)
                  for obs, fr, axis in width_bases]
        report["widths"] = widths
        finite = np.asarray([w for w in widths if w is not None], float)
        if len(finite) >= 2:
            report["width_cv"] = float(finite.std() / finite.mean())
    return report


# ── entrypoint: stages 1-2 on a saved sample set ─────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", required=True, help="samples.npz with q, frames, edges")
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--gap-ratio", type=float, default=ProductSplitOptions.gap_ratio)
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    with np.load(args.samples, allow_pickle=False) as data:
        q, frames = np.asarray(data["q"], float), np.asarray(data["frames"], float)
        edges = np.asarray(data["edges"], int) if "edges" in data.files else None
    t_load = time.perf_counter()
    report = detect_product_split(q, frames, edges, options=ProductSplitOptions(gap_ratio=args.gap_ratio))
    report["_bases"] = [b.tolist() for b in report["_bases"]]
    report["bases_root"] = report.pop("_bases")
    t_done = time.perf_counter()
    report["timing"] = {"load_seconds": t_load - t0, "detect_seconds": t_done - t_load,
                        "started_at": started,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "wall_seconds": t_done - t0}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: report[k] for k in ("kind", "split", "parallel")}, indent=2))


if __name__ == "__main__":
    main()
