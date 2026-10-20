"""Graph-free competitive physical chart autoencoders.

This fitter deliberately never reads ``ChartFitInput.cover_edges``.  Chart
structure is inferred from reconstruction-derived competence on simulator
poses; overlap is a learned, physical agreement relation rather than a graph
collar or stored membership lookup.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from atlas_rom.charts.chart_contract import ChartFitConfig, ChartFitInput
from atlas_rom.charts.latent_support import in_local_support, local_support_radii
from atlas_rom.charts.validity import ARTIFACT, ARTIFACT_FORMAT, CONSTRAINT_TOL, _json_safe
from atlas_rom.charts.registry import register

# Normalized E-step shell-priority policy constants (see ``_shell_policy``).
# Fixed at their validated defaults -- only ``policy_eta_absorb`` (the absorption
# gain A) is still a tunable knob, and it is gated to post-coverage.
_POLICY_ETA_ATTACH = 1.0    # eta_H: weight on physical attachment H_i
_POLICY_ETA_OVERLAP = 1.0   # eta_X: weight on the triple-overlap guard X_i
_POLICY_ABSORB_TEMP = 1.0   # T_E: softmax temperature over redundant-chart mass
_POLICY_EPS = 1e-3          # eps: residual exploration / seam mobility floor

# Targeted chart-map regularisers (replace the local-isometry / uniform-stretch
# "distortion" term, which neither preserved reconstruction nor caught global
# folds).  ``distortion_weight`` still scales their sum.
_COMPRESSION_RATIO_FLOOR = 0.1     # r_min: floor on local dz/dp before L_compression bites
_SEPARATION_FRAC = 0.25           # d_latent_min = this * latent diameter
_SEPARATION_PHYS_QUANTILE = 0.7   # a batch pair is "physically separated" above this dp quantile

# Frontier fold guard (see ``_fold_score``).  A chart's encoder MUST fold
# somewhere -- no continuous map from a closed manifold to R^d is injective --
# so folding is never penalised in itself.  What is guarded is the fold entering
# the chart's own domain: a candidate pose whose code lands on top of the core
# while living somewhere else on the manifold is a false neighbour, and training
# on it is a direct contradiction with poses the chart already holds.  The
# physical bandwidth is a fraction of the core's own physical diameter, mirroring
# ``_SEPARATION_FRAC``'s use of the latent diameter, so the test is scale free.
_FOLD_PHYS_FRAC = 0.25            # h = this * the core's physical diameter


class _Progress:
    """Minimal single-line epoch progress bar for the chart-training loops.

    Writes to stderr with a carriage return so it never pollutes the JSON /
    ``log`` stream.  Uses ``tqdm`` when it is importable, otherwise a
    dependency-free fallback that refreshes ~100 times over the run.
    """

    def __init__(self, total, desc):
        self.total = max(1, int(total))
        self.desc = desc
        self.n = 0
        self.start = time.perf_counter()
        self._tqdm = None
        try:
            import tqdm as _t
            self._tqdm = _t.tqdm(total=self.total, desc=desc, dynamic_ncols=True,
                                 leave=True, file=sys.stderr)
        except Exception:
            self._render()

    def update(self, k=1):
        self.n += k
        if self._tqdm is not None:
            self._tqdm.update(k)
            return
        if self.n >= self.total or self.n % max(1, self.total // 100) == 0:
            self._render()

    def _render(self):
        frac = self.n / self.total
        elapsed = time.perf_counter() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        bar = ("#" * int(30 * frac)).ljust(30)
        sys.stderr.write(f"\r{self.desc} [{bar}] {self.n}/{self.total}  "
                         f"{elapsed:6.1f}s  eta {eta:6.1f}s  ({rate:5.1f} ep/s)")
        sys.stderr.flush()

    def close(self):
        if self._tqdm is not None:
            self._tqdm.close()
            return
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()


def _pose_features(q: np.ndarray) -> np.ndarray:
    """A declared-coordinate-free physical pose metric representation."""
    q = np.asarray(q, float)
    n_bodies = q.shape[1] // 12
    T = q.reshape(len(q), n_bodies, 3, 4)
    return np.concatenate((T[..., :3, :2].transpose(0, 1, 3, 2).reshape(len(q), -1),
                           T[..., 3].reshape(len(q), -1)), axis=1)


def physical_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_pose_features(x) - _pose_features(y), axis=-1)


def _bridge_filtered_neighbours(feature: np.ndarray, pairs: np.ndarray, c):
    """Undirected feature-kNN adjacency, optionally with gap-crossing edges cut.

    ``pairs`` is the fitter's own directed ``i -> nearest[i]`` kNN (never
    ``cover_edges``).  Symmetrise it.  When ``c`` is not None, an edge ``(i, j)``
    of length ``L`` is a bridge iff, with that edge removed, ``j`` is not
    reachable from ``i`` within ``c * L`` -- the only short route between its
    endpoints, so it almost certainly jumps a gap where two sheets pass close.
    B0 calibration found only ~5-8 such edges per genus-2/3 dataset, so the
    filter is a near no-op here and defaults OFF (``c=None``); turn it on only
    for a case that actually shows two sheets passing close.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(feature)
    und = {(int(min(a, b)), int(max(a, b))) for a, b in pairs if int(a) != int(b)}
    E = sorted(und)
    if not E:
        return [np.empty(0, dtype=int) for _ in range(n)], {
            "n_edges": 0, "n_dropped": 0, "n_components": n,
            "bridge_c": (None if c is None else float(c))}
    if c is None:
        neighbours = [[] for _ in range(n)]
        for (a, b) in E:
            neighbours[a].append(b); neighbours[b].append(a)
        neighbours = [np.asarray(sorted(v), dtype=int) for v in neighbours]
        kr = np.array([e[0] for e in E] + [e[1] for e in E])
        kc = np.array([e[1] for e in E] + [e[0] for e in E])
        ncomp = int(connected_components(
            csr_matrix((np.ones(len(kr)), (kr, kc)), shape=(n, n)), directed=False)[0])
        return neighbours, {"n_edges": len(E), "n_dropped": 0,
                            "n_components": ncomp, "bridge_c": None}
    from scipy.sparse.csgraph import dijkstra
    L = {e: float(np.linalg.norm(feature[e[0]] - feature[e[1]])) for e in E}
    rows = np.array([e[0] for e in E] + [e[1] for e in E])
    cols = np.array([e[1] for e in E] + [e[0] for e in E])
    w = np.array([L[e] for e in E] * 2, dtype=float)
    G = csr_matrix((w, (rows, cols)), shape=(n, n))
    pos = {}
    for u in range(n):
        for p in range(G.indptr[u], G.indptr[u + 1]):
            pos[(u, int(G.indices[p]))] = p
    kept = []
    for (a, b) in E:
        G.data[pos[(a, b)]] = np.inf
        G.data[pos[(b, a)]] = np.inf
        reachable = np.isfinite(dijkstra(G, indices=a, limit=c * L[(a, b)])[b])
        G.data[pos[(a, b)]] = L[(a, b)]
        G.data[pos[(b, a)]] = L[(a, b)]
        if reachable:
            kept.append((a, b))
    neighbours = [[] for _ in range(n)]
    for (a, b) in kept:
        neighbours[a].append(b)
        neighbours[b].append(a)
    neighbours = [np.asarray(sorted(set(v)), dtype=int) for v in neighbours]
    kr = np.array([e[0] for e in kept] + [e[1] for e in kept])
    kc = np.array([e[1] for e in kept] + [e[0] for e in kept])
    ncomp = int(connected_components(
        csr_matrix((np.ones(len(kr)), (kr, kc)), shape=(n, n)), directed=False)[0])
    return neighbours, {"n_edges": len(E), "n_dropped": len(E) - len(kept),
                        "n_components": ncomp, "bridge_c": float(c)}


def _responsibilities(competence: np.ndarray, temperature: float) -> np.ndarray:
    """Turn physical competence losses into a detached chart distribution."""
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("responsibility temperature must be positive")
    competence = np.asarray(competence, float)
    z = -(competence - competence.min(1, keepdims=True)) / temperature
    weights = np.exp(z - z.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


def _ownership_weight(loss: np.ndarray, threshold_topo: float) -> np.ndarray:
    """Per-chart, absolute, unnormalised ownership weight (two-phase phase 1).

    ``w_i(p) = sigmoid((threshold_topo - loss_i(p)) / tau)`` with
    ``tau = threshold_topo / 5``.  There is no softmax over charts and no
    background column: each chart's weight depends only on its own loss.  This
    is what makes chart geometry independent of K and stops an unreachable pose
    from being conscripted onto its least-bad chart.
    """
    threshold_topo = float(threshold_topo)
    tau = max(threshold_topo / 5.0, float(np.finfo(float).eps))
    z = np.clip((threshold_topo - np.asarray(loss, float)) / tau, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _projector_distance_sq(a, b, *, ridge_rel: float = 0.0, ridge_abs: float = 0.0):
    """``||P_a - P_b||_F^2`` per sample, for ``P_x = x (x^T x + lam I)^-1 x^T``.

    ``a``, ``b`` are ``[..., D, d]`` torch tensors; the result is ``[...]``.
    The ridge is ``lam = ridge_abs + ridge_rel * mean(diag(x^T x))`` per
    sample, matching the dense projector call sites.  Exact (not an
    approximation) but never forms the ``D x D`` projector: with
    ``G_x = x^T x``, ``R_x = G_x + lam I`` and ``C = a^T b``,

        ||P_a - P_b||^2 = tr((R_a^-1 G_a)^2) + tr((R_b^-1 G_b)^2)
                          - 2 tr(R_a^-1 C R_b^-1 C^T),

    so memory is O(D d) per sample instead of O(D^2).
    """
    import torch
    d = a.shape[-1]
    eye = torch.eye(d, dtype=a.dtype, device=a.device)

    def _reg(gram):
        lam = ridge_abs + ridge_rel * gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
        return gram + lam[..., None, None] * eye

    gram_a, gram_b = a.transpose(-1, -2) @ a, b.transpose(-1, -2) @ b
    reg_a, reg_b = _reg(gram_a), _reg(gram_b)
    s_a, s_b = torch.linalg.solve(reg_a, gram_a), torch.linalg.solve(reg_b, gram_b)
    cross = a.transpose(-1, -2) @ b
    m_a, m_b = torch.linalg.solve(reg_a, cross), torch.linalg.solve(reg_b, cross.transpose(-1, -2))
    return ((s_a * s_a.transpose(-1, -2)).sum((-2, -1)) + (s_b * s_b.transpose(-1, -2)).sum((-2, -1))
            - 2.0 * (m_a * m_b.transpose(-1, -2)).sum((-2, -1)))


def _principal_angle_deg(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Largest principal angle (degrees) between two ``[N, m, d]`` column spans.

    ``got`` are the decoder's ``d`` decoded-tangent columns per pose, ``want``
    the observed simulator frame.  Both are orthonormalised per pose, then the
    singular values of ``Qg^T Qw`` give ``cos`` of the principal angles; the
    reported value per pose is the worst (largest) angle.
    """
    got = np.asarray(got, float)
    want = np.asarray(want, float)

    def _ortho(mat):
        q, _ = np.linalg.qr(mat)
        return q[..., : mat.shape[-1]]

    s = np.linalg.svd(np.matmul(np.swapaxes(_ortho(got), -1, -2), _ortho(want)),
                      compute_uv=False)
    return np.degrees(np.arccos(np.clip(s.min(-1), -1.0, 1.0)))


def _jacobian_fold_ratio(got: np.ndarray) -> np.ndarray:
    """Per-pose ``sigma_min / sigma_max`` of the decoded ``[N, m, d]`` Jacobian.

    A value near zero is a local fold / rank collapse of the chart map; the
    phase-1 validity predicate rejects it regardless of reconstruction error.
    """
    s = np.linalg.svd(np.asarray(got, float), compute_uv=False)
    return s.min(-1) / np.clip(s.max(-1), 1e-12, None)


def _decoder_spectral_product(decoder, *, n_iter=2, warmup_iter=24):
    """Decoder Lipschitz bound from its affine layers' spectral norms.

    Each spectral norm is estimated by power iteration with a per-weight
    warm-started left singular vector (``layer._spectral_u``): after the first
    call the weights move slowly enough that one iteration tracks the top
    singular pair.  ``torch.linalg.matrix_norm(ord=2)`` -- a full SVD per layer,
    every chart, every epoch -- was ~13-16% of joint training at width 288.  The
    returned scalar is differentiable in the weights through the final
    ``u . W v`` (u, v detached), matching the SVD's gradient at convergence.
    """
    import torch

    factors = []
    for layer in decoder.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        weight = layer.weight
        u = getattr(layer, "_spectral_u", None)
        cold = u is None or u.shape != weight.shape[:1] or u.device != weight.device
        if cold:
            u = torch.randn(weight.shape[0], dtype=weight.dtype, device=weight.device)
        with torch.no_grad():
            for _ in range(warmup_iter if cold else max(1, int(n_iter))):
                v = torch.nn.functional.normalize(weight.t() @ u, dim=0, eps=1e-12)
                u = torch.nn.functional.normalize(weight @ v, dim=0, eps=1e-12)
        layer._spectral_u = u.detach()
        factors.append((u.detach() * (weight @ v.detach())).sum().clamp_min(1e-12))
    if not factors:
        raise ValueError("decoder has no affine layers")
    return torch.stack(factors).prod()


def _condition_gauge_covariance(covariance, max_condition: float):
    """Floor learned gauge covariance eigenvalues to bound triangular solves."""
    import torch

    if max_condition < 1.0:
        raise ValueError("gauge max condition must be at least one")
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.max().clamp_min(1e-12) / float(max_condition) ** 2
    return (vectors * values.clamp_min(floor)) @ vectors.T


def _patch_far_distance(feature: np.ndarray, patch: np.ndarray, rng=None) -> float:
    """Physical far-pair threshold without materializing a patch distance matrix."""
    patch = np.asarray(patch, dtype=int)
    if len(patch) < 2:
        return float("inf")
    if len(patch) <= 128:
        a, b = np.triu_indices(len(patch), 1)
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        a = rng.integers(len(patch), size=4096); b = rng.integers(len(patch), size=4096)
        keep = a != b; a, b = a[keep], b[keep]
    return float(np.quantile(np.linalg.norm(feature[patch[a]] - feature[patch[b]], axis=1), .75))


def _fps_warm_start_indices(feature: np.ndarray, seeds: list[int], slot: int) -> tuple[np.ndarray, float, bool]:
    """Use FPS separation, not ``n / K``, to define an initial chart patch."""
    feature = np.asarray(feature, float)
    seed = int(seeds[slot])
    other = [int(value) for j, value in enumerate(seeds) if j != slot]
    if not other:
        return np.arange(len(feature), dtype=int), float("inf"), False
    distance = np.linalg.norm(feature - feature[seed], axis=1)
    separation = float(np.min(distance[other]))
    local = np.flatnonzero(distance <= 0.5 * separation)
    # A sparse sample can leave an FPS seed with only itself inside the radius.
    # Preserve a trainable AE in that exceptional case and make it visible in
    # the report; it is not the ordinary, K-dependent patch definition.
    fallback = len(local) < min(2, len(feature))
    if fallback:
        local = np.argsort(distance)[:min(2, len(feature))]
    return local, separation, fallback


def _chart_quality_passes(*, usage: float, recon_ratio: float, spread_ratio: float,
                          usage_floor: float, recon_ratio_max: float,
                          spread_ratio_min: float) -> bool:
    return (usage >= usage_floor and recon_ratio < recon_ratio_max and
            spread_ratio > spread_ratio_min)


def _reconstruction_diagnostics(target: np.ndarray, predictions: np.ndarray, recon_scale: np.ndarray = None):
    """Absolute reconstruction and inter-chart residual diagnostics.

    ``recon_scale`` (n_bodies*12,), when given, applies the same group-
    standardized metric as every other reconstruction site in this fitter
    (see ``_recon``) to ``best_chart_reconstruction_mse`` and the baseline it
    is compared against.  The per-chart covariance-trace rows stay on the raw
    scale -- they are relative diagnostics (alpha, spread ratios), not
    absolute MSE gates.
    """
    target = np.asarray(target, float)
    predictions = np.asarray(predictions, float)
    centred = target - target.mean(0, keepdims=True)
    base_mse = float(np.square(centred).mean())
    denominator = float(np.square(centred).sum())
    rows = []
    for prediction in predictions:
        output = prediction - target.mean(0, keepdims=True)
        rows.append({"alpha": float((centred * output).sum() / max(denominator, 1e-12)),
                     "decoder_covariance_trace": float(np.square(output).mean(0).sum()),
                     "data_covariance_trace": float(np.square(centred).mean(0).sum()),
                     "decoder_over_data_covariance_trace": float(np.square(output).mean() / max(base_mse, 1e-12)),
                     "decoder_out_spread_over_data_spread": float(output.std() / max(centred.std(), 1e-12))})
    if recon_scale is not None:
        scale = np.asarray(recon_scale, float)
        base_mse = float((np.square(centred) / scale[None, :]).mean())
        residuals = (np.square(predictions - target[None]) / scale[None, None, :]).mean(-1)
    else:
        residuals = np.square(predictions - target[None]).mean(-1)
    best_mse = float(residuals.min(0).mean())
    per_pose_spread = residuals.std(0)
    return {"mean_pose_baseline_mse": base_mse,
            "best_chart_reconstruction_mse": best_mse,
            "best_chart_recon_over_mean_pose_baseline": best_mse / max(base_mse, 1e-12),
            "per_chart": rows,
            "residual_spread_across_charts": {"per_pose": per_pose_spread.tolist(),
                                                "mean": float(per_pose_spread.mean()),
                                                "p95": float(np.percentile(per_pose_spread, 95))}}


def _soft_multiplicity_weights(reconstruction: np.ndarray, validity: np.ndarray, *, threshold: float,
                               atlas_dimension: int, deficit_gain: float = 0.0, excess_decay: float = 0.0):
    """Chart-local ownership (``membership``) and its per-pose sum.

    Two-phase note: this no longer emits a coverage-deficit reweight.  The
    phase-1 objective weight is a plain unit for owned poses plus an additive
    expansion bonus on frontier poses (never a whole-set reweight, which
    migrates chart mass), so ``weights`` is returned as ones for callers that
    still unpack four values.  ``membership`` is the absolute, unnormalised
    ownership weight ``sigmoid((threshold - (recon + valid)) / (threshold/5))``
    -- each chart depends only on its own loss.
    """
    loss = np.asarray(reconstruction, float) + np.asarray(validity, float)
    n, k = loss.shape
    tau = np.full(k, max(float(threshold) / 5.0, np.finfo(float).eps), dtype=float)
    membership = _ownership_weight(loss, float(threshold))
    multiplicity = membership.sum(1)
    weights = np.ones(n, dtype=float)
    return membership, multiplicity, weights, tau


def _integer_multiplicity_histogram(multiplicity: np.ndarray, atlas_dimension: int):
    """Counts membership near each integer, with a final overflow bin."""
    bins = np.r_[-0.5, np.arange(0.5, int(atlas_dimension) + 2.5, 1.0), np.inf]
    return {"bins": bins.tolist(), "counts": np.histogram(multiplicity, bins=bins)[0].tolist()}


def _exclusive_support(competent: np.ndarray):
    """Hard certified poses uniquely carried by every current chart."""
    competent = np.asarray(competent, bool)
    return (competent & (competent.sum(1, keepdims=True) == 1)).sum(0)


def _largest_component(mask, edges, n):
    """``mask`` restricted to its largest connected component in the kNN
    witness graph ``edges``.  A chart's certified support is a single patch,
    never a scatter of disconnected islands -- the outliers are dropped."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return mask
    internal = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    keep = np.zeros(n, dtype=bool)
    if not len(internal):
        keep[idx[0]] = True  # every member is graph-isolated: keep one representative
        return keep
    _, labels = connected_components(
        csr_matrix((np.ones(len(internal)), (internal[:, 0], internal[:, 1])), shape=(n, n)),
        directed=False)
    sub = labels[idx]
    keep[idx[sub == np.bincount(sub).argmax()]] = True
    return keep


def _restrict_to_components(competent, edges, n):
    """Apply :func:`_largest_component` to every chart column of a bool matrix."""
    competent = np.asarray(competent, bool).copy()
    for i in range(competent.shape[1]):
        competent[:, i] = _largest_component(competent[:, i], edges, n)
    return competent


def _soft_connectivity(weight, edges, n, *, edge_bar, decay):
    """Continuous analogue of :func:`_restrict_to_components`.

    Per chart column: an edge is 'connected' when ``min(w_a, w_b) >= edge_bar``.
    The heaviest weight-connected component keeps full strength; every other
    entry is MULTIPLIED by ``decay`` (never zeroed).  A thinning neck fades a
    lobe out continuously instead of amputating it, and a lobe that regrows its
    neck is restored on the next E-step.  When the column has no connecting
    edge at all (still forming) it is left untouched.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    weight = np.array(weight, float, copy=True)
    for i in range(weight.shape[1]):
        w = weight[:, i]
        strong = w >= float(edge_bar)
        e = edges[strong[edges[:, 0]] & strong[edges[:, 1]]]
        if int((w > 0).sum()) < 2 or not len(e):
            continue
        _, labels = connected_components(
            csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
        comp_mass = np.bincount(labels, weights=w, minlength=int(labels.max()) + 1)
        off = labels != int(comp_mass.argmax())
        weight[off, i] = w[off] * float(decay)
    return weight


def _load_resume_checkpoint(path, work, *, case, n_bodies, d, encoder_kind):
    """Trained charts to restart from, or None.

    Accepts a shipped atlas (``patch_atlas*.pt``: chart weights plus the RBF
    selector) or a ``phase1_final.pt`` checkpoint (chart weights and selector
    centres).  A relative path is looked up in ``work`` first.  Refuses a
    checkpoint whose case, body count, latent dimension or encoder differs
    from the run it is resumed into.
    """
    if path is None:
        return None
    import hashlib
    import torch
    path = str(path)
    if not os.path.isabs(path) and os.path.exists(os.path.join(work, path)):
        path = os.path.join(work, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume_from: {path} not found")
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    atlas = ck.get("chart_atlas", ck)
    states = atlas.get("states")
    if not states:
        raise ValueError(f"resume_from: {path} holds no chart states")
    found = {"case": ck.get("case"), "n_bodies": atlas.get("n_bodies"),
             "encoder_kind": atlas.get("encoder_kind")}
    wanted = {"case": case, "n_bodies": n_bodies, "encoder_kind": encoder_kind}
    for key, value in found.items():
        if value is not None and value != wanted[key]:
            raise ValueError(f"resume_from: {path} has {key}={value!r}, this run has {wanted[key]!r}")
    if "d" in atlas and int(atlas["d"]) != int(d):
        raise ValueError(f"resume_from: {path} has latent d={atlas['d']}, this run has {d}")
    return {"path": os.path.abspath(path), "sha256": digest, "states": states,
            "selector": atlas.get("selector"), "centres": atlas.get("centres")}


@dataclass
class PhysicalRBFSelector:
    """Serialized RBF receiving-chart ranker with a small physical correction."""
    centres: np.ndarray
    log_sigma: np.ndarray
    correction_w: np.ndarray
    correction_b: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def score(self, q: np.ndarray) -> np.ndarray:
        f = (_pose_features(np.asarray(q).reshape(1, -1)) - self.feature_mean) / self.feature_scale
        d2 = np.square(_pose_features(np.asarray(q).reshape(1, -1)) - self.centres).sum(-1)
        return -d2 / np.exp(2 * self.log_sigma) + self.correction_w @ f[0] + self.correction_b

    def rank(self, q: np.ndarray, candidates):
        scores = self.score(q)
        return sorted(candidates, key=lambda i: float(scores[int(i)]), reverse=True)

    def payload(self):
        return {k: np.asarray(v) for k, v in vars(self).items()}

    @classmethod
    def from_payload(cls, payload):
        return cls(**{k: np.asarray(v, float) for k, v in payload.items()})


class CompetitiveCAEFitter:
    fitter_id = "competitive_cae"

    def _validate(self, evidence, config):
        if config.fitter_id != self.fitter_id:
            raise ValueError(f"config selects {config.fitter_id!r}, not {self.fitter_id!r}")
        if len(evidence.poses) < 2:
            raise ValueError("competitive_cae requires at least two simulator poses")

    def fit(self, evidence: ChartFitInput, config: ChartFitConfig, **kwargs):
        self._validate(evidence, config)
        opts = dict(config.options)
        # ``tangent_every`` is a first-class config field (its patch_atlas
        # precedent is too); options/kwargs still win if set explicitly.
        opts.setdefault("tangent_every", int(getattr(config, "tangent_every", 1)))
        opts.update({k: v for k, v in kwargs.items() if k not in {"work", "case", "log", "validity"}})
        return self._fit(evidence, config, work=kwargs["work"], case=kwargs.get("case", "five_bar"),
                         validity=kwargs.get("validity", evidence.physical_validity), log=kwargs.get("log", print), **opts)

    def _fit(self, evidence, config, *, work, case, validity=None, log=print,
             encoder_kind="shallow", n_charts=8,
             field_epochs=800,
             phase1_max_epochs=None, phase2_epochs=None,
             distortion_weight=None, compactness_weight=None, competence_tolerance=None,
             competence_baseline_fraction=.03,
             tangent_weight=None, alias_weight=None, selector_weight=1.0,
             projector_weight=None,
             threshold_topo_angle_deg=15.0, fold_ratio_floor=.02,
             pretrain_steps=500,
             max_distortion_spread=2.0, overlap_temperature=.15, lip_weight=None,
             weight_decay=None,
             gauge_max_condition=4.0,
             decoder_norm_floor=1e-3,
             usage_ema_decay=.95, usage_ema_floor=.01, min_certified_support_fraction=.03,
             quality_recon_ratio_max=.5, quality_spread_ratio_min=.5,
             distortion_tail_fraction=0.1,
             latent_gauge=None, paper_baseline=False, warm_start_gauge_timing="before",
             warm_start_full_evaluator_diagnostic=False,
             warm_start_pca_scaffold=False, warm_start_pca_weight=.25,
             warm_start_tangent_weight=0.0, warm_start_tangent_every=4,
             off_anchor_cycle_weight=None, freeze_estep_probes=False, compression_weight=None,
             selector_calibration_min=.8, device="cpu", e_step_interval=20,
             tangent_every=1,
             m_step_batch_size=256,
             prune_warmup_epochs=100, retire_hysteresis_steps=2, retire_max_per_step=None,
             soft_edge_bar=0.1, component_decay=0.25, cover_level=0.5, core_level=0.5,
             rho=0.5, lambda_protect=1.0, protect_margin_frac=0.1,
             shell_sample_fraction=0.4, shell_weight_cap=2.0, fold_weight=10.0,
             e_step_min_interval=8, e_step_max_interval=25, shell_absorb_fraction=0.6,
             policy_eta_absorb=1.0,
             multiplicity_stability_steps=2,
             monotone_frontier=False,
             inertial_frontier=False, inertial_alpha=0.2, inertial_alpha_loss=None,
             ema_ownership=False, ema_ownership_alpha=0.2, ema_ownership_alpha_loss=0.05,
             snapshot_prune_steps=False,
             epoch_trace=False,
             exploration_snapshot_every=None,
             phase2_exploration_snapshot_every=None,
             lr_step_epochs=None, lr_step_gamma=0.5, lr_step_min=1e-5,
             lbfgs_polish_iters=0, lbfgs_polish_closure_mult=10.0, lbfgs_polish_overlap_tol=0.01,
             lbfgs_polish_rounds=2,
             closure_standardize_mult=None, resume_from=None, **_ignored):
        """Two-phase competitive charts.

        Phase 1 (cover) grows a topological cover under an *absolute*,
        unnormalised per-chart ownership weight -- ``sigmoid((threshold_topo -
        loss_i) / (threshold_topo / 5))`` -- so chart geometry is independent of
        K and unreachable poses are never conscripted.  Training is symmetric:
        all charts jointly maintain one-cover everywhere and second-cover only
        on learned physical boundary collars.  A runt is retired only after its
        disappearance preserves both invariants.  Phase 2 keeps the same
        coverage frontier active while tightening reconstruction accuracy.
        """
        import torch
        torch.manual_seed(config.seed); rng = np.random.default_rng(config.seed)
        timing_start = time.perf_counter()
        timing = {}
        compute_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        # Never install torch's global default-device dispatch mode: it wraps
        # every Python tensor operation and is especially costly on CUDA.
        _steer_default_device = False
        # Every ``None`` knob below is a plain default unless ``paper_baseline``
        # is set, in which case it takes the size-control paper's value instead
        # (the bounded sigmoid chart domain, the paper's Lipschitz/weight-decay
        # weights, optional preference terms zeroed).  An explicit value wins.
        paper = bool(paper_baseline)
        def _pick(value, paper_value, base_value):
            if value is not None:
                return value
            return paper_value if paper else base_value
        if bool(monotone_frontier) and bool(inertial_frontier):
            raise ValueError("monotone_frontier and inertial_frontier are mutually exclusive")
        if bool(ema_ownership) and bool(inertial_frontier):
            raise ValueError("ema_ownership and inertial_frontier are mutually exclusive "
                             "(inertial_frontier already carries its own claims EMA)")
        latent_gauge = bool(_pick(latent_gauge, False, True))
        lip_weight = float(_pick(lip_weight, 1e-2, 1e-4))
        weight_decay = float(_pick(weight_decay, 1e-2, 0.0))
        distortion_weight = float(_pick(distortion_weight, 0.0, 1.0))
        compactness_weight = float(_pick(compactness_weight, 0.0, 0.0))
        tangent_weight = float(_pick(tangent_weight, 0.0, 0.1))
        alias_weight = float(_pick(alias_weight, 0.0, 1.0))
        # Phase 2 forces the orientation projector on -- the cheapest accuracy
        # lever -- so its base default is 1.0 (was a 0.0 regression).
        projector_weight = float(_pick(projector_weight, 0.0, 1.0))
        off_anchor_cycle_weight = float(_pick(off_anchor_cycle_weight, 0.0, 1.0))
        freeze_estep_probes = bool(freeze_estep_probes)
        compression_weight = float(_pick(compression_weight, 1.0, 1.0))
        resolved_config = {"paper_baseline": paper, "latent_gauge": latent_gauge,
                           "resume_from": (None if resume_from is None else str(resume_from)),
                           "freeze_estep_probes": freeze_estep_probes,
                           "compression_weight": compression_weight,
                           "lip_weight": lip_weight, "weight_decay": weight_decay,
                           "distortion_weight": distortion_weight,
                           "compactness_weight": compactness_weight,
                           "tangent_weight": tangent_weight, "alias_weight": alias_weight,
                           "projector_weight": projector_weight,
                           "off_anchor_cycle_weight": off_anchor_cycle_weight,
                           "prune_warmup_epochs": int(prune_warmup_epochs),
                           "retire_hysteresis_steps": int(retire_hysteresis_steps),
                           "retire_max_per_step": (None if retire_max_per_step is None
                                                   else int(retire_max_per_step)),
                           "soft_edge_bar": float(soft_edge_bar),
                           "component_decay": float(component_decay),
                           "cover_level": float(cover_level),
                           "core_level": float(core_level),
                           "rho": float(rho), "lambda_protect": float(lambda_protect),
                           "protect_margin_frac": float(protect_margin_frac),
                           "shell_sample_fraction": float(shell_sample_fraction),
                           "shell_weight_cap": float(shell_weight_cap),
                           "fold_weight": float(fold_weight),
                           "e_step_min_interval": int(e_step_min_interval),
                           "e_step_max_interval": int(e_step_max_interval),
                           "shell_absorb_fraction": float(shell_absorb_fraction),
                           "policy_eta_absorb": float(policy_eta_absorb),
                           "multiplicity_stability_steps": int(multiplicity_stability_steps),
                           "monotone_frontier": bool(monotone_frontier),
                           "inertial_frontier": bool(inertial_frontier),
                           "inertial_alpha": float(inertial_alpha),
                           "inertial_alpha_loss": (float(inertial_alpha) if inertial_alpha_loss is None
                                                   else float(inertial_alpha_loss)),
                           "ema_ownership": bool(ema_ownership),
                           "ema_ownership_alpha": float(ema_ownership_alpha),
                           "ema_ownership_alpha_loss": float(ema_ownership_alpha_loss),
                           "policy_constants": {"eta_attach": _POLICY_ETA_ATTACH,
                                                "eta_overlap": _POLICY_ETA_OVERLAP,
                                                "absorb_temperature": _POLICY_ABSORB_TEMP,
                                                "epsilon": _POLICY_EPS}}
        return self._fit_inner(
            evidence, config, work=work, case=case, validity=validity, log=log,
            encoder_kind=encoder_kind, n_charts=n_charts, field_epochs=field_epochs,
            phase1_max_epochs=(None if phase1_max_epochs is None else int(phase1_max_epochs)),
            phase2_epochs=(None if phase2_epochs is None else int(phase2_epochs)),
            distortion_weight=distortion_weight,
            tangent_weight=tangent_weight, alias_weight=alias_weight,
            selector_weight=selector_weight, projector_weight=projector_weight,
            compactness_weight=compactness_weight, competence_tolerance=competence_tolerance,
            competence_baseline_fraction=competence_baseline_fraction,
            threshold_topo_angle_deg=float(threshold_topo_angle_deg),
            fold_ratio_floor=float(fold_ratio_floor),
            pretrain_steps=pretrain_steps, max_distortion_spread=max_distortion_spread,
            overlap_temperature=overlap_temperature, lip_weight=lip_weight,
            weight_decay=weight_decay,
            gauge_max_condition=gauge_max_condition,
            decoder_norm_floor=decoder_norm_floor,
            usage_ema_decay=usage_ema_decay, usage_ema_floor=usage_ema_floor,
            min_certified_support_fraction=min_certified_support_fraction,
            quality_recon_ratio_max=quality_recon_ratio_max,
            quality_spread_ratio_min=quality_spread_ratio_min,
            distortion_tail_fraction=distortion_tail_fraction,
            latent_gauge=latent_gauge, warm_start_gauge_timing=str(warm_start_gauge_timing),
            warm_start_full_evaluator_diagnostic=bool(warm_start_full_evaluator_diagnostic),
            warm_start_pca_scaffold=bool(warm_start_pca_scaffold),
            warm_start_pca_weight=float(warm_start_pca_weight),
            warm_start_tangent_weight=warm_start_tangent_weight,
            warm_start_tangent_every=int(warm_start_tangent_every),
            resolved_config=resolved_config,
            off_anchor_cycle_weight=off_anchor_cycle_weight,
            freeze_estep_probes=freeze_estep_probes, compression_weight=compression_weight,
            selector_calibration_min=selector_calibration_min, compute_device=compute_device,
            e_step_interval=e_step_interval, tangent_every=tangent_every,
            m_step_batch_size=m_step_batch_size,
            prune_warmup_epochs=int(prune_warmup_epochs),
            retire_hysteresis_steps=int(retire_hysteresis_steps),
            retire_max_per_step=(None if retire_max_per_step is None else int(retire_max_per_step)),
            soft_edge_bar=float(soft_edge_bar), component_decay=float(component_decay),
            cover_level=float(cover_level), core_level=float(core_level),
            rho=float(rho), lambda_protect=float(lambda_protect),
            protect_margin_frac=float(protect_margin_frac),
            shell_sample_fraction=float(shell_sample_fraction),
            shell_weight_cap=float(shell_weight_cap), fold_weight=float(fold_weight),
            e_step_min_interval=int(e_step_min_interval),
            e_step_max_interval=int(e_step_max_interval),
            shell_absorb_fraction=float(shell_absorb_fraction),
            policy_eta_absorb=float(policy_eta_absorb),
            multiplicity_stability_steps=int(multiplicity_stability_steps),
            monotone_frontier=bool(monotone_frontier),
            inertial_frontier=bool(inertial_frontier), inertial_alpha=float(inertial_alpha),
            inertial_alpha_loss=(float(inertial_alpha) if inertial_alpha_loss is None
                                 else float(inertial_alpha_loss)),
            ema_ownership=bool(ema_ownership), ema_ownership_alpha=float(ema_ownership_alpha),
            ema_ownership_alpha_loss=float(ema_ownership_alpha_loss),
            snapshot_prune_steps=bool(snapshot_prune_steps),
            epoch_trace=bool(epoch_trace),
            exploration_snapshot_every=(None if exploration_snapshot_every is None
                                        else int(exploration_snapshot_every)),
            phase2_exploration_snapshot_every=(None if phase2_exploration_snapshot_every is None
                                               else int(phase2_exploration_snapshot_every)),
            lr_step_epochs=(None if lr_step_epochs is None else int(lr_step_epochs)),
            lr_step_gamma=float(lr_step_gamma), lr_step_min=float(lr_step_min),
            lbfgs_polish_iters=int(lbfgs_polish_iters), lbfgs_polish_closure_mult=float(lbfgs_polish_closure_mult),
            lbfgs_polish_overlap_tol=float(lbfgs_polish_overlap_tol), lbfgs_polish_rounds=int(lbfgs_polish_rounds),
            closure_standardize_mult=(None if closure_standardize_mult is None else float(closure_standardize_mult)),
            steer_default_device=bool(_steer_default_device),
            resume_from=resume_from,
            timing=timing, timing_start=timing_start)

    def _fit_inner(self, evidence, config, *, work, case, validity, log,
                   encoder_kind, n_charts, field_epochs, phase1_max_epochs, phase2_epochs,
                   distortion_weight, tangent_weight, alias_weight, selector_weight,
                   projector_weight, compactness_weight, competence_tolerance, pretrain_steps, max_distortion_spread,
                   competence_baseline_fraction,
                   threshold_topo_angle_deg, fold_ratio_floor,
                   overlap_temperature, lip_weight, gauge_max_condition, decoder_norm_floor,
                   weight_decay,
                   usage_ema_decay, usage_ema_floor, min_certified_support_fraction,
                   quality_recon_ratio_max, quality_spread_ratio_min,
                   distortion_tail_fraction,
                   latent_gauge, warm_start_gauge_timing, warm_start_full_evaluator_diagnostic,
                   warm_start_pca_scaffold, warm_start_pca_weight, resolved_config,
                   warm_start_tangent_weight, warm_start_tangent_every,
                   off_anchor_cycle_weight, freeze_estep_probes, compression_weight,
                   selector_calibration_min, compute_device, e_step_interval, tangent_every,
                   m_step_batch_size,
                   prune_warmup_epochs, retire_hysteresis_steps, retire_max_per_step,
                   soft_edge_bar, component_decay, cover_level, core_level,
                   rho, lambda_protect, protect_margin_frac, shell_sample_fraction,
                   shell_weight_cap, fold_weight,
                   e_step_min_interval, e_step_max_interval, shell_absorb_fraction,
                   policy_eta_absorb, multiplicity_stability_steps,
                   monotone_frontier,
                   inertial_frontier, inertial_alpha, inertial_alpha_loss,
                   ema_ownership, ema_ownership_alpha, ema_ownership_alpha_loss,
                   snapshot_prune_steps,
                   epoch_trace,
                   exploration_snapshot_every,
                   phase2_exploration_snapshot_every,
                   lr_step_epochs, lr_step_gamma, lr_step_min,
                   lbfgs_polish_iters, lbfgs_polish_closure_mult, lbfgs_polish_overlap_tol, lbfgs_polish_rounds,
                   closure_standardize_mult,
                   steer_default_device, resume_from,
                   timing, timing_start):
        import torch
        from atlas_rom.charts import physical_chart_autoencoder as pca
        from atlas_rom.charts.validity import _validity_residual
        torch.manual_seed(config.seed); rng = np.random.default_rng(config.seed)
        monotone_frontier = bool(monotone_frontier)
        inertial_frontier = bool(inertial_frontier)
        inertial_alpha = float(inertial_alpha)
        inertial_alpha_loss = float(inertial_alpha if inertial_alpha_loss is None else inertial_alpha_loss)
        ema_ownership = bool(ema_ownership)
        ema_ownership_alpha = float(ema_ownership_alpha)
        ema_ownership_alpha_loss = float(ema_ownership_alpha_loss)
        _memprofile = os.environ.get("CAE_MEMPROFILE")
        if _memprofile:
            import gc, ctypes

            def _rss_mb():
                try:
                    class _PMC(ctypes.Structure):
                        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
                    pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
                    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                    h = ctypes.windll.kernel32.GetCurrentProcess()
                    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(pmc), pmc.cb)
                    return pmc.WorkingSetSize / 1e6
                except Exception:
                    return -1.0
        # freeze_estep_probes caches (read by chart_loss, which is called
        # during warm-start below, well before _m_step's own cache init).
        _cached_probe_direction = None
        _cached_probe_radius_fraction = None
        _cached_far_pairs = {}
        q = np.asarray(evidence.poses, float); frames = np.asarray(evidence.frames, float)
        n, n_bodies, d = len(q), q.shape[1] // 12, frames.shape[-1]
        Q = torch.as_tensor(q, dtype=torch.float32, device=compute_device)
        # Group-standardized reconstruction scale: divide each body's rotation
        # block and translation block by that block's own variance over the
        # dataset, so every group (however different its physical units) is
        # held to the same RELATIVE accuracy.  Joint-free and derived only
        # from q -- no case geometry.  Without this, an unscaled flat MSE is
        # dominated by whichever block happens to have the larger raw units
        # (translation, on cases like rotate, at ~99% of the budget), leaving
        # rotation error essentially unchecked by the reconstruction gate.
        q_blocks = q.reshape(n, n_bodies, 3, 4)
        _rot_var = q_blocks[..., :3].reshape(n, n_bodies, -1).var(0).mean(-1)
        _trans_var = q_blocks[..., 3].var(0).mean(-1)
        # A body block that is EXACTLY constant (e.g. a piston that never
        # rotates -- zero physical DOF in that block, not just small motion)
        # has zero variance, not just small variance.  Flooring it near-zero
        # (e.g. an absolute 1e-12, or a tiny fraction of the largest block)
        # asks the decoder for near machine-precision accuracy there before
        # the group-standardized MSE clears the gate at all -- any ordinary
        # residual on a constant channel then dominates every other block's
        # contribution by many orders of magnitude, and the phase-1
        # competence gate never fires anywhere.  A degenerate block carries
        # no information, so it should cost the SAME as a typical real one,
        # not more: floor it to the median of the dataset's own nonzero
        # block variances (falling back to 1.0 if every block is constant).
        _all_var = np.concatenate([_rot_var, _trans_var])
        _nonzero = _all_var[_all_var > 1e-12]
        _scale_floor = float(np.median(_nonzero)) if _nonzero.size else 1.0
        _rot_var = np.where(_rot_var > 1e-12, _rot_var, _scale_floor)
        _trans_var = np.where(_trans_var > 1e-12, _trans_var, _scale_floor)
        recon_scale = np.empty((n_bodies, 3, 4), dtype=float)
        recon_scale[..., :3] = _rot_var[:, None, None]
        recon_scale[..., 3] = _trans_var[:, None]
        recon_scale = recon_scale.reshape(-1)
        recon_scale_t = torch.as_tensor(recon_scale, dtype=Q.dtype, device=Q.device)
        mean_pose_baseline_mse = float(
            (np.square(q - q.mean(0, keepdims=True)) / recon_scale[None, :]).mean())
        threshold = float(competence_tolerance if competence_tolerance is not None
                          else float(competence_baseline_fraction) * max(mean_pose_baseline_mse, 1e-12))
        feature = _pose_features(q)
        # These are immutable simulator evidence.  Keep device-local copies so
        # chart_loss does not repeatedly cross the NumPy/Torch boundary.
        feature_t = torch.as_tensor(feature, dtype=Q.dtype, device=compute_device)
        frames_t = torch.as_tensor(frames, dtype=Q.dtype, device=compute_device)
        # ``threshold_topo`` is the phase-1 validity bound -- a reconstruction
        # sanity ceiling, NOT an MSE tolerance calibrated against d_NN.  It
        # reuses the same formula the fitter always used for ``threshold``; the
        # load-bearing phase-1 gate is the decoder-tangent-vs-oracle-frame angle
        # (``threshold_topo_angle_deg``) and the Jacobian fold ratio
        # (``fold_ratio_floor``), applied on top as hard predicates.
        threshold_topo = float(threshold)
        frontier_trace = []

        def _recon(pred, target):
            """Group-standardized per-pose reconstruction MSE: the ONE metric
            every reconstruction site in this fitter must use (chart_loss,
            warm start, warm diagnostics, off-sample probes, coverage margin,
            pruning/fidelity diagnostics).  Accepts numpy or torch; both
            arguments' LAST axis must already be the flat n_bodies*12 pose
            (callers reshape/flatten to that themselves, as before -- this
            only centralizes the scale division).  Any leading batch/model
            axes broadcast through unchanged.
            """
            if isinstance(pred, torch.Tensor) or isinstance(target, torch.Tensor):
                p = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred, dtype=Q.dtype, device=Q.device)
                t = target if isinstance(target, torch.Tensor) else torch.as_tensor(target, dtype=Q.dtype, device=Q.device)
                return ((p - t).square() / recon_scale_t).mean(-1)
            p = np.asarray(pred, float)
            t = np.asarray(target, float)
            return ((p - t) ** 2 / recon_scale).mean(-1)

        def _owner_responsibilities(ownership):
            """Row-normalised ownership with a uniform fallback for poses no
            chart owns (used only as the selector/gauge target -- coverage logic
            keys on the raw ``ownership`` weights and the hard-competent mask)."""
            ownership = np.asarray(ownership, float)
            row = ownership.sum(1, keepdims=True)
            k = ownership.shape[1]
            return np.where(row < 1e-8, 1.0 / max(k, 1),
                            ownership / np.where(row < 1e-8, 1.0, row))

        centre = q.reshape(n, n_bodies, 3, 4)[..., 3].mean((0, 1))
        scale = q.reshape(n, n_bodies, 3, 4)[..., 3].std((0, 1)).clip(1e-6)

        def new_model(seed_index=None):
            model = pca._module(n_bodies, config.width, d=d, encoder_kind=encoder_kind)(centre, scale).float().to(compute_device)
            if seed_index is not None:
                # The birth is centered on its actual worst physical witness;
                # this is normalization, never a latent-coordinate shortcut.
                model.centre.copy_(torch.as_tensor(q[int(seed_index)].reshape(n_bodies, 3, 4)[..., 3].mean(0), device=compute_device))
            return model

        # Closure residuals are in RAW pose units (squared pivot distance) while ``_recon`` is group-standardized, so
        # in phase 2 (validity_weight=1) the closure term dominates the loss by the data's raw variance and drives
        # coverage to collapse.  ``closure_standardize_mult`` (None = legacy raw units) divides the TRAINING closure by
        # the mean translation variance (the same scale the recon metric uses for translations) and multiplies by this
        # weight.  Only the training / competence loss is rescaled; physics fields still see the raw validity.
        closure_unit_factor = (1.0 if closure_standardize_mult is None
                               else float(closure_standardize_mult) / max(float(np.mean(_trans_var)), 1e-30))

        def validity_loss_raw(model, pred):
            r = _validity_residual(validity, pred.reshape(len(pred), -1), pred)
            return torch.zeros(len(pred), dtype=Q.dtype, device=pred.device) if r is None else r.reshape(len(pred), -1).mean(-1)

        def validity_loss(model, pred):
            return validity_loss_raw(model, pred) * closure_unit_factor

        # ``models`` are construction/serialization shells.  After the local
        # warm start the trainable source of truth is ``ensemble_params``: one
        # persistent tensor per state entry, with chart slot as dimension zero.
        # Keeping those tensors as optimizer leaves is important -- rebuilding
        # stacks inside a vmap makes StackBackward own the graph instead.
        ensemble_params = None
        ensemble_buffers = None
        ensemble_template = None

        def _chart_state(chart):
            if ensemble_params is None:
                return None
            return ({name: value[chart] for name, value in ensemble_params.items()},
                    {name: value[chart] for name, value in ensemble_buffers.items()})

        def _decode(model, chart, codes):
            if ensemble_params is None:
                return model.decode(codes)
            from torch.func import functional_call
            params, buffers = _chart_state(chart)
            return functional_call(ensemble_template, (params, buffers), (codes,), {"decode_only": True})

        def _transforms(model, chart, codes):
            return _decode(model, chart, codes).reshape(-1, n_bodies, 3, 4)

        def encode(model, chart, poses):
            # The gauge is a detached buffer on the model itself, so this is
            # exactly the map every downstream consumer (field fitting, the
            # runtime loader, the invertibility probe) reaches through
            # ``model.encode``.
            if ensemble_params is None:
                return model.encode(poses)
            from torch.func import functional_call
            params, buffers = _chart_state(chart)
            return functional_call(ensemble_template, (params, buffers), (poses,), {"encode_only": True})

        def ensemble_encode(poses):
            """K-way encode with no decoder work and no chart-state slicing."""
            from torch.func import functional_call, vmap
            return vmap(lambda params, buffers, values: functional_call(
                ensemble_template, (params, buffers), (values,), {"encode_only": True}),
                in_dims=(0, 0, 0))(ensemble_params, ensemble_buffers, poses)

        def ensemble_decode(codes):
            """K-way decode directly from the persistent chart tensors."""
            from torch.func import functional_call, vmap
            return vmap(lambda params, buffers, values: functional_call(
                ensemble_template, (params, buffers), (values,), {"decode_only": True}),
                in_dims=(0, 0, 0))(ensemble_params, ensemble_buffers, codes)

        def chart_loss(model, chart, pair, pair_weights=None, pose_ids=None, compute_tangent=True,
                       precomputed=None, precomputed_cycles=None, compute_distortion=True,
                       validity_weight=1.0):
            """Chart loss on all evidence (E-step) or a local M-step batch.

            ``pair`` is indexed into ``pose_ids``.  Full E-steps omit it and
            retain the original global local-pair witnesses; stochastic M-steps
            supply only responsibility-plus-margin poses and their local pairs.
            """
            if pose_ids is None:
                ids_np = np.arange(n, dtype=int)
                ids_t = None
                poses = Q
                local_nearest = torch.as_tensor(nearest, dtype=torch.long, device=Q.device)
            else:
                ids_np = np.asarray(pose_ids, dtype=int)
                ids_t = torch.as_tensor(ids_np, dtype=torch.long, device=Q.device)
                poses = Q[ids_t]
            n_local = len(ids_np)
            # Encode once.  The detached copy is sufficient for neighbor search
            # and avoids two redundant encoder graphs on every local M-step.
            if precomputed is None:
                u = encode(model, chart, poses)
                pred = _transforms(model, chart, u)
            else:
                u, pred = precomputed
            if pose_ids is not None and precomputed_cycles is None:
                # A batch need not contain an anchor's original kNN witnesses;
                # use its own local cloud for the off-anchor radius instead.
                # Only needed for the local-radius probe below: when the batched
                # cycle is precomputed this B x B cdist (and its graph) is pure
                # waste, so it stays inside the fallback branch.
                count = min(4, len(ids_np))
                distance = torch.cdist(u.detach(), u.detach())
                local_nearest = distance.topk(count, largest=False).indices[:, 1:]
            recon = _recon(pred.reshape(n_local, -1), poses)
            on_anchor_cycle = ((precomputed_cycles[0] if precomputed_cycles is not None else
                                encode(model, chart, pred.reshape(n_local, -1))) - u).square().mean(-1)
            # The certificate probes E(D(u + eps)); train that same local
            # inverse relation instead of only the trivial encoded vertices.
            # Probe a local neighbourhood of each encoded physical pose.  A
            # global latent standard deviation silently fills annular holes.
            if precomputed_cycles is None:
                if local_nearest.shape[1]:
                    near = local_nearest
                    local_radius = torch.linalg.vector_norm(u[near] - u[:, None], dim=-1).mean(1, keepdim=True)
                else:
                    local_radius = u.detach().std(0, keepdim=True).mean().expand(len(u), 1)
                probe_count = max(1, d)
                direction = torch.randn((len(u), probe_count, d), dtype=Q.dtype, device=Q.device)
                direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
                radius_fraction = torch.rand((len(u), probe_count, 1), dtype=Q.dtype, device=Q.device)
                probe_u = u[:, None, :] + (radius_fraction * local_radius.detach()[:, None, :]
                                            .clamp_min(1e-8) * direction)
                probe_flat = probe_u.reshape(-1, d)
                off_anchor_cycle = (encode(model, chart, _decode(model, chart, probe_flat)) - probe_flat).square().mean(-1)
                off_anchor_cycle = off_anchor_cycle.reshape(len(u), probe_count).amax(1)
            else:
                off_anchor_cycle = precomputed_cycles[1]
            cycle = ((on_anchor_cycle + float(off_anchor_cycle_weight) * off_anchor_cycle) /
                     (1.0 + float(off_anchor_cycle_weight)))
            # Skip the closure-residual work entirely when its weight is zero
            # (Phase-1 M-steps): it is real per-pose matmul/einsum cost that
            # would otherwise run on every chart, every epoch, only to be
            # multiplied away in per_pose below.  Nothing downstream reads
            # ``valid``'s actual value when validity_weight is 0 -- ``rv`` in
            # the M-step is reconstruction-only (see _assign_from) and the
            # only other callers of chart_loss (E-step/certificate paths)
            # always pass the default weight of 1.0.
            valid = (validity_loss(model, pred) if float(validity_weight) != 0.0
                     else torch.zeros(n_local, dtype=Q.dtype, device=Q.device))
            a, b = pair[:, 0], pair[:, 1]
            a_t = torch.as_tensor(a, dtype=torch.long, device=Q.device)
            b_t = torch.as_tensor(b, dtype=torch.long, device=Q.device)
            if pose_ids is None and pair is pairs:
                raw_dp = pair_physical_distance_t
            else:
                pair_a = a_t if pose_ids is None else ids_t[a_t]
                pair_b = b_t if pose_ids is None else ids_t[b_t]
                raw_dp = torch.linalg.vector_norm(feature_t[pair_a] - feature_t[pair_b], dim=1)
            nondegenerate = raw_dp >= float(distortion_pair_floor)
            dp = raw_dp.clamp_min(1e-8)
            dz = torch.linalg.vector_norm(u[a] - u[b], dim=-1)
            ratio = dz / dp
            # Log stretch makes reciprocal compression/expansion symmetric.
            # Its mean and upper tail continuously pull every local pair
            # toward uniform chart stretch, rather than only penalising walls.
            log_r = torch.log(ratio.clamp_min(1e-12))
            weights = (torch.ones_like(log_r) if pair_weights is None else
                       pair_weights.to(dtype=Q.dtype).clamp_min(0))
            weight_sum = weights.sum().clamp_min(1e-8)
            distortion_weights = weights * nondegenerate
            # Per-pair local-compression violation [1 - (dz/dp) / r_min]_+ in
            # [0, 1]: fires only when a kNN pair's latent gap collapses below
            # r_min x its physical gap.  Feeds both L_compression and the
            # per-pose alias flag.  Bounded and linear in the ratio: the former
            # squared log was unbounded, and a newly admitted pinched pose drew
            # 2-10 (100-1000x reconstruction), whose gradient wrecked the chart.
            pinch = torch.relu(1.0 - ratio / max(_COMPRESSION_RATIO_FLOOR, 1e-9)) * nondegenerate
            if not compute_distortion:
                distortion = torch.zeros((), dtype=Q.dtype, device=Q.device)
            else:
                l_compression = (distortion_weights * pinch).sum() / distortion_weights.sum().clamp_min(1e-8)
                # L_separation: physically distant poses must stay latent-far --
                # [d_latent_min - ||u_p - u_q||]_+^2 over physically separated
                # batch pairs -- which catches a global fold / wraparound that
                # local stretch cannot see.  d_latent_min scales with the chart's
                # own latent diameter.
                feat_local = feature_t[ids_t] if ids_t is not None else feature_t
                l_separation = torch.zeros((), dtype=Q.dtype, device=Q.device)
                if n_local > 3:
                    cached = _cached_far_pairs.get(chart)
                    if freeze_estep_probes and cached is not None and cached[2] == n_local:
                        gi, gj = cached[0], cached[1]
                    else:
                        m_far = min(4096, n_local * 8)
                        gi = torch.randint(0, n_local, (m_far,), device=Q.device)
                        gj = torch.randint(0, n_local, (m_far,), device=Q.device)
                        if freeze_estep_probes:
                            _cached_far_pairs[chart] = (gi, gj, n_local)
                    keep = gi != gj
                    gi, gj = gi[keep], gj[keep]
                    dp_far = torch.linalg.vector_norm(feat_local[gi] - feat_local[gj], dim=1)
                    far = dp_far >= torch.quantile(dp_far, _SEPARATION_PHYS_QUANTILE)
                    if far.any():
                        u_diam = (u.detach().amax(0) - u.detach().amin(0)).norm().clamp_min(1e-6)
                        d_latent_min = _SEPARATION_FRAC * u_diam
                        dz_far = torch.linalg.vector_norm(u[gi[far]] - u[gj[far]], dim=-1)
                        l_separation = torch.relu(d_latent_min - dz_far).square().mean()
                distortion = float(compression_weight) * l_compression + l_separation
            # Observed simulator tangent frames supervise the decoder at every
            # pose this chart owns (per-pose responsibility weight), not just a
            # handful of top pairs; the oracle tangent is the dominant lever on
            # chart accuracy so it must not be starved.  This is physical
            # evidence, not a declared tangent coordinate.
            # The jvp tangent term is the dominant per-call cost.  E-steps and
            # diagnostics never read it, and an M-step can stale it out via
            # ``tangent_every`` (precedent: patch_atlas run_phase).  When skipped
            # the term is a detached zero so the objective assembly is unchanged.
            if not compute_tangent:
                tangent = torch.zeros((), dtype=Q.dtype, device=Q.device)
            else:
              pose_w = torch.zeros(n_local, dtype=Q.dtype, device=Q.device)
              pose_w.index_add_(0, a_t, weights); pose_w.index_add_(0, b_t, weights)
              owned = torch.nonzero(pose_w > 1e-8).flatten()
              if len(owned) == 0:
                  owned = torch.topk(pose_w, min(8, n_local)).indices
              if len(owned) > 256:  # cap the jvp cost, keep the most strongly owned
                  owned = owned[torch.topk(pose_w[owned], 256).indices]
              w_owned = pose_w[owned].detach()
              w_owned = w_owned / w_owned.sum().clamp_min(1e-8)
              u_local = u[owned]
              observed = frames_t[ids_t[owned]].reshape(len(owned), -1, d) if ids_t is not None else frames_t[owned].reshape(len(owned), -1, d)
              directions = []
              from torch.func import jvp
              for axis in range(d):
                  direction = torch.zeros_like(u_local); direction[:, axis] = 1.0
                  _, decoded_direction = jvp(lambda z: _decode(model, chart, z), (u_local,), (direction,))
                  directions.append(decoded_direction)
              got = torch.stack(directions, -1)
              # Compare tangent subspaces through their normalized Gram matrices;
              # arbitrary latent-axis rotations are therefore gauge-equivalent.
              got_gram = got.transpose(-1, -2) @ got
              want_gram = observed.transpose(-1, -2) @ observed
              got_gram = got_gram / torch.linalg.vector_norm(got_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
              want_gram = want_gram / torch.linalg.vector_norm(want_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
              # Normalized Gram pins the tangent *shape* (singular-value ratio /
              # conditioning) but is blind to subspace orientation -- QJ and J
              # share a Gram.  The orthogonal projector P = J (JtJ)^-1 Jt is the
              # complementary quantity: invariant under latent rotation J->JR
              # (the gauge freedom), but P->QPQt under an ambient rotation, so
              # ||P_got - P_want||_F^2 penalizes the subspace angle directly.
              # Differentiable through the same jvp columns.
              gram_term = (w_owned[:, None, None] * (got_gram - want_gram).square()).sum() / (got_gram.shape[-1] * got_gram.shape[-2])
              if float(projector_weight) != 0.0:
                  proj_err = (w_owned * _projector_distance_sq(got, observed, ridge_abs=1e-8)).sum()
                  tangent = gram_term + float(projector_weight) * proj_err
              else:
                  tangent = gram_term
            # Per-pose alias flag: a reconstruction-perfect but locally pinched
            # (near-collapsed latent direction) chart is not usable.  Aggregates
            # the per-pair compression violation onto its incident poses,
            # normalised by incident pair COUNT so the pair weights scale it:
            # dividing by the weight sum cancelled them, and a barely-owned
            # shell pose drew its full pinch (2-4, ~300x reconstruction).
            alias = torch.zeros(n_local, dtype=Q.dtype, device=Q.device)
            incident_count = torch.zeros(n_local, dtype=Q.dtype, device=Q.device)
            alias.index_add_(0, a_t, weights * pinch)
            alias.index_add_(0, b_t, weights * pinch)
            incident_count.index_add_(0, a_t, torch.ones_like(weights))
            incident_count.index_add_(0, b_t, torch.ones_like(weights))
            alias = alias / incident_count.clamp_min(1.0)
            per_pose = recon + .1 * cycle + float(validity_weight) * valid + alias_weight * alias
            return per_pose, distortion, (recon, cycle, valid, alias, u, pred, dp, dz, tangent, nondegenerate)

        # Farthest-point seeding in the permitted physical pose metric gives
        # every chart a distinct initial neighbourhood without topology or
        # membership information.
        resume = _load_resume_checkpoint(resume_from, work, case=case, n_bodies=n_bodies,
                                         d=d, encoder_kind=encoder_kind)
        if resume is not None:
            n_charts = len(resume["states"])
            log(f"competitive_cae: resuming {n_charts} charts from {resume['path']} "
                f"(pretraining, phase 1 and phase 2 skipped)")
        K = int(n_charts)
        if K < 1:
            raise ValueError("n_charts must be positive")
        # Pair only physically local witnesses; no chart is asked to flatten
        # the entire manifold into a single latent patch.
        feature_tree = cKDTree(feature)
        neighbour_distance, nearest = feature_tree.query(feature, k=min(5, n))
        if n == 2:
            neighbour_distance, nearest = neighbour_distance[:, None], nearest[:, None]
        nearest = nearest[:, 1:]
        neighbour_distance = neighbour_distance[:, 1:]
        pairs = np.asarray([(i, j) for i in range(n) for j in nearest[i]], dtype=int)
        # The fitter's physical witness graph is inferred solely from simulator
        # frames.  Edge coverage, rather than an artificial double-cover
        # collar, is the atlas-wide coverage invariant.
        physical_edges = np.unique(np.sort(pairs, axis=1), axis=0)
        # Shell budget == core budget (both ``m_step_batch_size``), independent
        # of each other: a core pinned at its cap cannot starve the shell.
        _shell_budget = int(m_step_batch_size)
        # A data-relative floor excludes duplicate/nearly duplicate simulator
        # frames from stretch ratios while preserving resolved neighbourhoods.
        positive = np.where(neighbour_distance > 1e-12, neighbour_distance, np.inf)
        spacing_rank = min(1, positive.shape[1] - 1)  # second nonzero neighbour ignores a duplicate witness
        local_physical_spacing = np.partition(positive, spacing_rank, axis=1)[:, spacing_rank]
        finite_spacing = local_physical_spacing[np.isfinite(local_physical_spacing)]
        typical_spacing = float(np.median(finite_spacing)) if len(finite_spacing) else 1.0
        local_physical_spacing = np.where(np.isfinite(local_physical_spacing), local_physical_spacing, typical_spacing)
        distortion_pair_floor = .05 * typical_spacing
        pair_physical_distance = np.linalg.norm(feature[pairs[:, 0]] - feature[pairs[:, 1]], axis=1)
        pair_physical_distance_t = torch.as_tensor(pair_physical_distance, dtype=Q.dtype, device=Q.device)
        distortion_pair_ok = pair_physical_distance >= distortion_pair_floor
        # Local attachment and frontier growth use an ADAPTIVE PHYSICAL-DISTANCE BAND, not a kNN hop: q is a
        # neighbour of p iff d_pq <= _BAND_BETA * s_pq, s_pq = max(s_p, s_q) with s the per-pose physical
        # spacing above.  A k-hop reaches a density-dependent physical distance (short where sampling is dense,
        # long where it is sparse); the band reaches the same number of local spacings everywhere.  Each pose
        # keeps its nearest neighbour so none is isolated.  The kNN ``pairs`` / ``physical_edges`` remain the
        # witness graph for components, distortion pairs, the collar and the fold score.
        _BAND_BETA = 2.0
        _band_set = set()
        for p_, row in enumerate(feature_tree.query_ball_point(feature, r=_BAND_BETA * local_physical_spacing)):
            for q_ in row:
                if q_ != p_:
                    _band_set.add((min(p_, int(q_)), max(p_, int(q_))))
        for p_ in range(n):
            if nearest.shape[1]:
                q_ = int(nearest[p_, 0])
                if q_ != p_:
                    _band_set.add((min(p_, q_), max(p_, q_)))
        band_pairs = (np.asarray(sorted(_band_set), dtype=int) if _band_set else np.empty((0, 2), dtype=int))
        frontier_neighbours, frontier_graph_stats = _bridge_filtered_neighbours(feature, band_pairs, None)
        frontier_graph_stats = {**frontier_graph_stats, "neighbourhood": "adaptive_physical_distance_band",
                                "band_beta": _BAND_BETA,
                                "mean_degree": float(2 * len(band_pairs) / max(n, 1))}
        # Symmetric physical-kernel adjacency k_pq = exp(-(d_pq / s_pq)^2) over the band, for the E-step
        # attachment term H_i(p) (kernel-weighted neighbour mean of the ownership weight w_i); the bandwidth is
        # the pair's own local spacing, not one global median.
        from scipy.sparse import csr_matrix as _csr
        _band_d = np.linalg.norm(feature[band_pairs[:, 0]] - feature[band_pairs[:, 1]], axis=1)
        _band_s = np.maximum(np.maximum(local_physical_spacing[band_pairs[:, 0]],
                                        local_physical_spacing[band_pairs[:, 1]]), 1e-12)
        _kw = np.exp(-(_band_d / _band_s) ** 2)
        _attach_kernel = _csr((np.r_[_kw, _kw],
                               (np.r_[band_pairs[:, 0], band_pairs[:, 1]], np.r_[band_pairs[:, 1], band_pairs[:, 0]])),
                              shape=(n, n))
        _attach_rowsum = np.asarray(_attach_kernel.sum(1)).ravel().clip(1e-12, None)
        fps = [int(rng.integers(n))]
        min_d = np.linalg.norm(feature - feature[fps[0]], axis=1)
        for _ in range(1, min(K, n)):
            nxt = int(np.argmax(min_d)); fps.append(nxt)
            min_d = np.minimum(min_d, np.linalg.norm(feature - feature[nxt], axis=1))
        models = [new_model(i) for i in fps]
        # Opt-in per-epoch diagnostic trace (``epoch_trace``): records, never steers.
        _trace_margin = {}
        _last_gauge_change = {}
        epoch_trace_rows = []

        def refresh_gauge(weights):
            # ``latent_gauge=False`` (the size-control paper's setting) keeps the
            # encoder's bounded sigmoid domain: every chart is (0, 1)^d and
            # contractible by construction, with size governed by ``lip_weight``.
            if not latent_gauge:
                return
            if ensemble_params is not None:
                sync_model_shells()
            was_gauge_enabled = [bool(model._gauge_enabled) for model in models]
            with torch.no_grad():
                for i, model in enumerate(models):
                    w = torch.as_tensor(weights[:, i], dtype=Q.dtype, device=Q.device)
                    raw = model.encode_raw(Q)
                    mean = (w[:, None] * raw).sum(0) / w.sum().clamp_min(1e-8)
                    centred = raw - mean
                    cov = (centred.T * w) @ centred / w.sum().clamp_min(1e-8)
                    cov = cov + 1e-4 * torch.eye(d, dtype=Q.dtype, device=Q.device)
                    # Orthogonal gauge: the polar/eigen rotation of the weighted
                    # code covariance.  It fixes the rotation+shear degeneracy
                    # and nothing else -- latent scale stays with the pinned
                    # log-stretch loss.  ||gauge_map|| == 1 always, so it can
                    # neither amplify triangular solves nor degenerate as a
                    # chart's support shrinks.
                    _, vectors = torch.linalg.eigh(cov)
                    # Sign-align each axis with the previous frame so the latent
                    # coordinate does not flip between refreshes (eigh's sign
                    # convention is arbitrary).
                    prev = (model.gauge_map.to(Q.dtype) if model._gauge_enabled
                            else torch.eye(d, dtype=Q.dtype, device=Q.device))
                    dots = (vectors * prev).sum(0)
                    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
                    if epoch_trace and model._gauge_enabled:
                        new_map = (vectors * signs).detach()
                        relative = (prev.T @ new_map).cpu()
                        _last_gauge_change.setdefault("rotation_deg", []).append(
                            float(torch.rad2deg(torch.linalg.eigvals(relative).angle().abs().max())))
                        _last_gauge_change.setdefault("mean_shift", []).append(
                            float((mean.detach() - model.gauge_mean.to(Q.dtype)).norm()))
                    model.set_gauge(mean.detach(), (vectors * signs).detach())
                # Gauge refresh is a coordinate reparameterization: its decoder
                # adjustment must be written back into the optimizer-owned
                # ensemble, never left on a separately owned shell.
                if ensemble_params is not None:
                    for chart, model in enumerate(models):
                        for name, value in model.named_parameters():
                            ensemble_params[name][chart].copy_(value)
                        for name, value in model.named_buffers():
                            ensemble_buffers[name][chart].copy_(value)
                    # ``set_gauge`` rewrites the decoder's FIRST layer as a true
                    # reparameterization (weight <- weight @ A^T, bias <- bias +
                    # weight @ b) for any chart that already had a gauge
                    # installed -- the first installation replaces the bounded
                    # sigmoid code outright, so there is nothing prior to
                    # invalidate.  Adam's elementwise second moment is not
                    # rotation-covariant: momentum accumulated in the
                    # PRE-rotation coordinates, applied to the POST-rotation
                    # weight, is a spurious update in a basis the moments know
                    # nothing about.  A renormalization over fewer columns right
                    # after a chart retirement can swing this rotation hard,
                    # which is exactly when this was observed to blow up
                    # (certified coverage collapsing a few M-steps after an
                    # otherwise-safe deletion).  Reset the affected slice.
                    reset_charts = [i for i, was in enumerate(was_gauge_enabled) if was]
                    if reset_charts:
                        reset_idx = torch.as_tensor(reset_charts, dtype=torch.long, device=Q.device)
                        for name in ("decoder.0.weight", "decoder.0.bias"):
                            param = ensemble_params.get(name)
                            if param is None:
                                continue
                            state = ensemble_opt.state.get(param)
                            if not state:
                                continue
                            for key in ("exp_avg", "exp_avg_sq"):
                                buf = state.get(key)
                                if torch.is_tensor(buf) and buf.ndim > 0:
                                    buf.index_fill_(0, reset_idx.to(buf.device), 0.0)
        if warm_start_gauge_timing not in {"before", "after"}:
            raise ValueError("warm_start_gauge_timing must be 'before' or 'after'")
        if warm_start_gauge_timing == "before":
            refresh_gauge(np.full((n, len(models)), 1.0 / len(models)))
        # No weight decay on the chart AEs: the atlas poses are exact, so decay
        # only trades reconstruction accuracy for a smoothness the evidence does
        # not ask for (see the mesh-free AE convergence finding).
        optimizers = [torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=float(weight_decay)) for m in models]
        feat_mean, feat_scale = feature.mean(0), feature.std(0).clip(1e-6)
        F = torch.as_tensor((feature - feat_mean) / feat_scale, dtype=Q.dtype, device=Q.device)
        X = torch.as_tensor(feature, dtype=Q.dtype, device=Q.device)
        centre_t = torch.as_tensor(feature[fps], dtype=Q.dtype, device=Q.device)
        log_sigma = torch.nn.Parameter(torch.full((len(models),), np.log(max(1e-6, typical_spacing)),
                                                  dtype=Q.dtype, device=Q.device))
        correction_w = torch.nn.Parameter(torch.zeros((len(models), feature.shape[1]), dtype=Q.dtype, device=Q.device))
        correction_b = torch.nn.Parameter(torch.zeros(len(models), dtype=Q.dtype, device=Q.device))
        selector_opt = torch.optim.Adam((log_sigma, correction_w, correction_b), lr=2e-3, weight_decay=1e-2)

        def _slice_selector_optimizer(old_opt, old_params, new_params, keep_live):
            """Rebind a selector Adam while retaining moments for live charts."""
            new_opt = torch.optim.Adam(new_params, lr=2e-3, weight_decay=1e-2)
            keep = torch.as_tensor(keep_live, dtype=torch.bool, device=new_params[0].device)
            for old_param, new_param in zip(old_params, new_params):
                old_state = old_opt.state.get(old_param, {})
                state = {}
                for name, value in old_state.items():
                    if torch.is_tensor(value):
                        copied = value.detach().clone()
                        if copied.ndim > 0 and copied.shape[0] == len(keep_live):
                            copied = copied[keep]
                        state[name] = copied
                    else:
                        state[name] = value
                if state:
                    new_opt.state[new_param] = state
            return new_opt

        def selector_scores():
            squared_distance = (X.square().sum(1, keepdim=True) - 2.0 * X @ centre_t.T +
                                centre_t.square().sum(1)[None, :]).clamp_min(0.0)
            return -(squared_distance / torch.exp(2 * log_sigma)[None, :]) + F @ correction_w.T + correction_b
        # Eq. (5) warm start: each FPS chart reconstructs its local physical
        # neighbourhood and claims its own predictor class before competition.
        warm_start_rows = []
        warm_start_local = [None] * len(fps)
        warm_tangent_weight = (0.25 * float(tangent_weight)
                               if warm_start_tangent_weight is None
                               else float(warm_start_tangent_weight))
        if warm_tangent_weight < 0.0:
            raise ValueError("warm_start_tangent_weight must be non-negative")
        if int(warm_start_tangent_every) < 1:
            raise ValueError("warm_start_tangent_every must be positive")

        def _warm_tangent_alignment(model, chart, code, observed):
            """Phase-1-equivalent decoder tangent evidence on one birth patch."""
            from torch.func import jvp

            columns = []
            for axis in range(d):
                direction = torch.zeros_like(code)
                direction[:, axis] = 1.0
                _, jacobian_column = jvp(
                    lambda z: _decode(model, chart, z), (code,), (direction,))
                columns.append(jacobian_column)
            got = torch.stack(columns, -1)
            got_gram = got.transpose(-1, -2) @ got
            want_gram = observed.transpose(-1, -2) @ observed
            got_gram = got_gram / torch.linalg.vector_norm(
                got_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            want_gram = want_gram / torch.linalg.vector_norm(
                want_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            gram = (got_gram - want_gram).square().mean()
            if float(projector_weight) == 0.0:
                return gram
            # mean over the [P, D, D] projector entries, as the dense form was
            proj_err = _projector_distance_sq(got, observed, ridge_rel=1e-6).sum() / (
                got.shape[0] * got.shape[-2] ** 2)
            return gram + float(projector_weight) * proj_err

        def warm_start_slot(i, seed_index, steps):
            """Local physical AE birth used only during initial K-slot setup."""
            model, opt = models[i], optimizers[i]
            local, seed_separation, used_fallback = _fps_warm_start_indices(feature, fps, i)
            warm_start_local[i] = np.asarray(local, dtype=int)
            # PCA supplies a data-derived, temporary local plane: it is never
            # serialized as a coordinate system, exposed to routing, or used
            # after warm start.  The spectrum selects the usable directions;
            # the chart's already evidence-derived latent width remains only a
            # capacity cap.  Whitening prevents a single ambient scale from
            # becoming a latent-coordinate prior.
            pca_target = None
            pca_rank = 0
            pca_explained = 0.0
            if warm_start_pca_scaffold and len(local) >= 2:
                centred_local = feature[local] - feature[local].mean(0, keepdims=True)
                _, singular, right = np.linalg.svd(centred_local, full_matrices=False)
                energy = np.square(singular)
                total_energy = float(energy.sum())
                if total_energy > np.finfo(float).eps:
                    natural_rank = int(np.searchsorted(np.cumsum(energy) / total_energy, .95) + 1)
                    pca_rank = min(int(d), natural_rank, len(singular))
                    pca_explained = float(energy[:pca_rank].sum() / total_energy)
                    if pca_rank:
                        score = centred_local @ right[:pca_rank].T
                        score /= score.std(0, keepdims=True).clip(1e-6)
                        # A small, fixed RMS keeps the scaffold inside the
                        # learned chart's initial coordinate scale; its
                        # orientation and all higher geometry remain learned.
                        target_np = np.zeros((len(local), d), dtype=np.float32)
                        target_np[:, :pca_rank] = .25 * score
                        pca_target = torch.as_tensor(target_np, dtype=Q.dtype, device=Q.device)
            warm_start_rows.append({"chart": int(i), "seed_index": int(seed_index),
                                    "seed_separation": (None if not np.isfinite(seed_separation)
                                                        else float(seed_separation)),
                                    "radius": (None if not np.isfinite(seed_separation)
                                               else float(0.5 * seed_separation)),
                                    "n_local": int(len(local)), "used_minimum_sample_fallback": bool(used_fallback),
                                    "pca_scaffold": bool(pca_target is not None),
                                    "pca_rank": int(pca_rank),
                                    "pca_explained_fraction": float(pca_explained),
                                    "tangent_weight": float(warm_tangent_weight),
                                    "tangent_start_step": int(max(1, int(steps) // 4))})
            local_t = torch.as_tensor(local, dtype=torch.long, device=Q.device)
            observed = frames_t[local_t].reshape(len(local), -1, d)
            tangent_start = max(1, int(steps) // 4)
            for step in range(int(steps)):
                opt.zero_grad(); selector_opt.zero_grad()
                u = encode(model, i, Q[local_t]); pred = _transforms(model, i, u)
                seed_u = encode(model, i, Q[int(seed_index):int(seed_index) + 1])
                ae = _recon(pred.reshape(len(local), -1), Q[local_t]).mean() + .1 * seed_u.square().mean()
                if pca_target is not None and warm_start_pca_weight > 0.0:
                    # Fade before Phase 1: this is an initialization aid, not
                    # an imposed chart coordinate or a persistent loss.
                    fade = 1.0 - (float(step) / max(int(steps) - 1, 1))
                    ae = ae + float(warm_start_pca_weight) * fade * (u - pca_target).square().mean()
                if (warm_tangent_weight > 0.0 and step >= tangent_start and
                        (step - tangent_start) % int(warm_start_tangent_every) == 0):
                    ramp = float(step - tangent_start + 1) / max(int(steps) - tangent_start, 1)
                    ae = ae + warm_tangent_weight * ramp * _warm_tangent_alignment(model, i, u, observed)
                target = torch.full((len(local),), i, dtype=torch.long, device=Q.device)
                ce = torch.nn.functional.cross_entropy(selector_scores()[local_t], target)
                (ae + ce).backward(); opt.step(); selector_opt.step()
        pretrain_start = time.perf_counter()
        for i in range(len(models)):
            warm_start_slot(i, fps[i], pretrain_steps)
        timing["fps_pretrain_seconds"] = time.perf_counter() - pretrain_start
        if warm_start_gauge_timing == "after":
            refresh_gauge(np.full((n, len(models)), 1.0 / len(models)))
        # Warm-start is a lifecycle invariant: the local modules must be
        # competent on their own birth neighbourhood before any batched-state
        # transfer or Phase-1 policy can be trusted.
        def _warm_direct_metrics():
            rows = []
            with torch.no_grad():
                for i, model in enumerate(models):
                    ids = np.asarray(warm_start_local[i], dtype=int)
                    target = Q[torch.as_tensor(ids, dtype=torch.long, device=Q.device)]
                    code = model.encode(target)
                    pred = model.decode(code)
                    rec = _recon(pred, target)
                    valid = validity_loss(model, pred.reshape(len(ids), n_bodies, 3, 4))
                    competence = (rec + valid) <= float(threshold_topo)
                    local_feature = feature[ids]
                    centroid_distance = np.linalg.norm(
                        local_feature - local_feature.mean(0, keepdims=True), axis=1)
                    centroid_variance = float(np.square(local_feature - local_feature.mean(0, keepdims=True)).mean())
                    code_np = code.detach().cpu().numpy()
                    latent_std = float(code_np.std(0).mean())
                    if len(ids) >= 2:
                        latent_distance = np.linalg.norm(code_np[:, None] - code_np[None, :], axis=-1)
                        physical_distance = np.linalg.norm(local_feature[:, None] - local_feature[None, :], axis=-1)
                        latent_distance[np.diag_indices(len(ids))] = np.inf
                        physical_distance[np.diag_indices(len(ids))] = np.inf
                        latent_nn = float(np.median(latent_distance.min(1)))
                        physical_nn = float(np.median(physical_distance.min(1)))
                    else:
                        latent_nn = physical_nn = float("nan")
                    competent_np = competence.detach().cpu().numpy().astype(float)
                    centroid_competence_correlation = (
                        float(np.corrcoef(competent_np, centroid_distance)[0, 1])
                        if competent_np.std() > 0 and centroid_distance.std() > 0 else float("nan"))
                    rows.append({"chart": int(i), "n_local": int(len(ids)),
                                 "competent": int(competence.sum()),
                                 "reconstruction_mean": float(rec.mean()),
                                 "validity_mean": float(valid.mean()),
                                 "patch_centroid_variance": centroid_variance,
                                 "reconstruction_over_centroid_variance": float(rec.mean()) / max(centroid_variance, 1e-12),
                                 "reconstruction_over_threshold": float(rec.mean()) / max(float(threshold_topo), 1e-12),
                                 "validity_over_threshold": float(valid.mean()) / max(float(threshold_topo), 1e-12),
                                 "latent_std_mean": latent_std,
                                 "latent_nn_spacing_median": latent_nn,
                                 "physical_nn_spacing_median": physical_nn,
                                 "latent_std_over_nn_spacing": latent_std / max(latent_nn, 1e-12),
                                 "latent_nn_over_physical_nn_spacing": latent_nn / max(physical_nn, 1e-12),
                                 "competence_centroid_distance_correlation": centroid_competence_correlation})
            return rows
        warm_start_diagnostics = {"direct": _warm_direct_metrics()}
        # Materialize the chart axis once.  `_module` deliberately returns a
        # fresh local class per construction, so torch.func.stack_module_state
        # rejects otherwise identical shells by type identity.  Stack their
        # state explicitly here, once after warm start; the detached copies are
        # new optimizer leaves, not StackBackward views.
        raw_params = {name: torch.stack([dict(model.named_parameters())[name].detach()
                                         for model in models]).clone()
                      for name in dict(models[0].named_parameters())}
        raw_buffers = {name: torch.stack([dict(model.named_buffers())[name].detach()
                                          for model in models]).clone()
                       for name in dict(models[0].named_buffers())}
        ensemble_template = models[0]
        ensemble_params = {name: torch.nn.Parameter(value.detach().clone())
                           for name, value in raw_params.items()}
        ensemble_buffers = {name: value.detach().clone() for name, value in raw_buffers.items()}
        ensemble_opt = torch.optim.Adam(tuple(ensemble_params.values()), lr=2e-3,
                                         weight_decay=float(weight_decay))
        with torch.no_grad():
            transferred = []
            for i in range(len(models)):
                ids = np.asarray(warm_start_local[i], dtype=int)
                target = Q[torch.as_tensor(ids, dtype=torch.long, device=Q.device)]
                direct = models[i].decode(models[i].encode(target))
                tiled = target.unsqueeze(0).expand(len(models), -1, -1)
                u = ensemble_encode(tiled)[i]
                batched = ensemble_decode(u.unsqueeze(0).expand(len(models), -1, -1))[i]
                rec = _recon(batched, target)
                valid = validity_loss(models[i], batched.reshape(len(ids), n_bodies, 3, 4))
                transferred.append({"chart": int(i), "n_local": int(len(ids)),
                                    "competent": int(((rec + valid) <= float(threshold_topo)).sum()),
                                    "reconstruction_mean": float(rec.mean()),
                                    "max_direct_batched_abs": float((direct - batched).abs().max())})
            warm_start_diagnostics["batched"] = transferred
        # Drop individual optimizer ownership before the joint phase.  The
        # model shells are synchronized only when an individual state dict is
        # required for diagnostics or serialization.
        optimizers = None

        def sync_model_shells():
            """Copy persistent ensemble slots into individual diagnostic shells."""
            with torch.no_grad():
                for chart, model in enumerate(models):
                    state = model.state_dict()
                    for name in state:
                        source = (ensemble_params.get(name, ensemble_buffers.get(name)))
                        if source is not None:
                            state[name].copy_(source[chart])
                    model._gauge_enabled = bool(state["gauge_use"])

        def _slice_ensemble_optimizer(old_opt, old_params, keep_live):
            """Slice chart axis and retain the matching Adam moments."""
            keep = torch.as_tensor(keep_live, dtype=torch.bool, device=next(iter(old_params.values())).device)
            new_params = {name: torch.nn.Parameter(value.detach()[keep].clone())
                          for name, value in old_params.items()}
            new_opt = torch.optim.Adam(tuple(new_params.values()), lr=2e-3,
                                        weight_decay=float(weight_decay))
            for name, old_param in old_params.items():
                state = old_opt.state.get(old_param, {})
                copied_state = {}
                for state_name, value in state.items():
                    copied_state[state_name] = (value.detach().clone()[keep]
                                                if torch.is_tensor(value) and value.ndim > 0 and
                                                value.shape[0] == len(keep_live) else
                                                (value.detach().clone() if torch.is_tensor(value) else value))
                if copied_state:
                    new_opt.state[new_params[name]] = copied_state
            return new_params, new_opt

        def ensemble_decoder_spectral(chart):
            """Differentiable decoder bound directly from batched weights."""
            factors = []
            for name, values in ensemble_params.items():
                if not (name.startswith("decoder.") and name.endswith(".weight")):
                    continue
                weight = values[chart]
                u = torch.nn.functional.normalize(torch.ones(weight.shape[0], dtype=weight.dtype,
                                                             device=weight.device), dim=0)
                for _ in range(2):
                    v = torch.nn.functional.normalize(weight.t() @ u, dim=0, eps=1e-12)
                    u = torch.nn.functional.normalize(weight @ v, dim=0, eps=1e-12)
                factors.append((u.detach() * (weight @ v.detach())).sum().clamp_min(1e-12))
            return torch.stack(factors).prod()
        if resume is not None:
            # Replace the warm-started shells with the checkpoint's trained
            # charts, then re-stack the optimizer-owned ensemble from them.
            with torch.no_grad():
                for model, state in zip(models, resume["states"]):
                    model.load_state_dict(state)
                for chart, model in enumerate(models):
                    for name, value in model.named_parameters():
                        ensemble_params[name][chart].copy_(value)
                    for name, value in model.named_buffers():
                        ensemble_buffers[name][chart].copy_(value)
            ensemble_opt = torch.optim.Adam(tuple(ensemble_params.values()), lr=2e-3,
                                             weight_decay=float(weight_decay))
            selector_payload = resume.get("selector")
            if selector_payload is not None:
                if not (np.allclose(selector_payload["feature_mean"], feat_mean)
                        and np.allclose(selector_payload["feature_scale"], feat_scale)):
                    raise ValueError(f"resume_from {resume['path']}: selector feature "
                                     f"normalisation differs -- not the same dataset")
                with torch.no_grad():
                    centre_t = torch.as_tensor(selector_payload["centres"], dtype=Q.dtype, device=Q.device)
                    log_sigma.copy_(torch.as_tensor(selector_payload["log_sigma"], dtype=Q.dtype))
                    correction_w.copy_(torch.as_tensor(selector_payload["correction_w"], dtype=Q.dtype))
                    correction_b.copy_(torch.as_tensor(selector_payload["correction_b"], dtype=Q.dtype))
            elif resume.get("centres") is not None:
                centre_t = torch.as_tensor(resume["centres"], dtype=Q.dtype, device=Q.device)
        # This is learned from simulator poses and the warm-start model, then
        # fixed for joint competition so collapse changes the loss instead of
        # redefining its zero.  It must be pinned ON-DOMAIN: the reference is
        # the median log-stretch over the pairs THIS chart owns after the warm
        # start, not over every pair in the cloud (a global median sits well
        # above a chart's in-patch pairs, so the tail term then reads the whole
        # chart as compressive).
        a_t = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=Q.device)
        b_t = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=Q.device)
        dp = torch.as_tensor(np.linalg.norm(feature[pairs[:, 0]] - feature[pairs[:, 1]], axis=1),
                             dtype=Q.dtype, device=Q.device).clamp_min(1e-8)
        with torch.no_grad():
            per_chart_logr = []
            stretch_reference = []
            for i, model in enumerate(models):
                u = encode(model, i, Q)
                logr = torch.log((torch.linalg.vector_norm(u[a_t] - u[b_t], dim=-1) / dp).clamp_min(1e-12))
                keep = torch.as_tensor(distortion_pair_ok, dtype=torch.bool, device=Q.device)
                logr = logr[keep]
                per_chart_logr.append(logr)
                stretch_reference.append(logr.median().detach())  # provisional (global)
        # Warm responsibilities off the pretrained models, then re-pin each
        # chart's reference on the pairs IT owns.
        with torch.no_grad():
            warm_C = np.stack([chart_loss(m, i, pairs, compute_tangent=False)[0].cpu().numpy()
                               for i, m in enumerate(models)], axis=1)
        warm_resp = _responsibilities(warm_C, overlap_temperature)
        warm_winner = warm_resp.argmax(1)
        own_floor = 1.5 / len(models)
        for i in range(len(models)):
            owned = ((np.minimum(warm_resp[pairs[:, 0], i], warm_resp[pairs[:, 1], i]) >= own_floor) &
                     distortion_pair_ok)
            if owned.sum() >= 8:
                # ``per_chart_logr[i]`` is already restricted to the
                # non-degenerate pairs, so mask ``owned`` down to that subset
                # (it is a subset of ``distortion_pair_ok`` by construction).
                owned_kept = owned[distortion_pair_ok]
                stretch_reference[i] = per_chart_logr[i][
                    torch.as_tensor(owned_kept, device=per_chart_logr[i].device)].median().detach()
        responsibilities = np.full((n, len(models)), 1.0 / len(models))
        deficit_weights = np.ones(n, dtype=float)
        multiplicity = np.ones(n, dtype=float)
        ownership = np.full((n, len(models)), 1.0 / len(models))
        membership = ownership
        e_step_reconstruction = np.zeros((n, len(models)), dtype=float)
        e_step_validity = np.zeros((n, len(models)), dtype=float)
        rim_temperature = np.ones(len(models), dtype=float)
        usage_ema = responsibilities.mean(0)
        # ``m_step_batch`` closes over ``hard_competent`` but the E-step that
        # assigns it does not run until epoch 0's tick.  Seed it empty (every
        # chart bootstraps to its warm patch) and keep it column-aligned
        # through retirement.
        hard_competent = np.zeros((n, len(models)), dtype=bool)
        # Ungated recon+validity competence (no angle/fold gate).  This -- not the
        # gated ``hard_competent`` -- drives the zero-exclusive-support retirement
        # test: a chart that reconstructs nothing uniquely must be removable even
        # if the 15 deg gate transiently certifies a few boundary poses only on
        # it.  The gate stays load-bearing in the final competence report.
        cover_competent = np.zeros((n, len(models)), dtype=bool)
        # Stable per-chart identity: assigned at seeding, carried through every
        # retirement (the models list is compacted, so column 5 after a
        # retirement is NOT the same chart as column 5 before it -- consumers
        # that need continuity across snapshots key on this instead).
        chart_ids = list(range(len(models)))
        # Per-chart count of consecutive prune-eligible E-steps with zero
        # exclusive support; a chart retires only once this reaches
        # ``retire_hysteresis_steps`` (guards against pruning a flickering mask).
        zero_exclusive_streak = np.zeros(len(models), dtype=int)
        # EMA-smoothed retirement gate (``ema_ownership``): unlike
        # ``inertial_frontier`` this never touches M-step batch composition
        # (core/shell stay the default ownership-thresholded, capped sets) --
        # it only smooths the per-(pose, chart) ``cover_competent`` mask fed
        # into ``_exclusive_support`` for the zero-exclusive-support retirement
        # test, so a chart's eligibility reflects its recent trend rather than
        # one noisy E-step.  Asymmetric like ``inertial_frontier``'s claims:
        # react fast when a chart gains a pose (``ema_ownership_alpha``), decay
        # slowly when it loses one (``ema_ownership_alpha_loss``), so genuine
        # redundancy still shows up but a transient dip does not.
        _ema_cover = None
        # Monotone-frontier persistent territory (option-gated, see
        # ``monotone_frontier``).  ``persistent_territory[i]`` is a python
        # ``set`` of pose indices this chart owns for good -- never emptied,
        # only grown, and sliced alongside every other per-chart array on
        # retirement.  Named distinctly from the unrelated per-epoch
        # ``territory`` drift-bookkeeping list further below.
        # ``persistent_territory is None`` marks "not yet bootstrapped".
        persistent_territory = None
        # ``born[i]`` -- has chart i ever certified a genuine (non-provisional)
        # territory?  Until then it trains on a fixed provisional density (see
        # ``provisional_source``) that carries no coverage/exclusivity/pruning
        # weight, so a chart whose warm start happens to start out entirely
        # incompetent still gets a training gradient instead of being
        # permanently stuck at an empty ``S_i``.
        born = np.zeros(len(models), dtype=bool)
        provisional_source = None          # fixed (n,) density per chart, set once
        provisional_pool = None            # warm_start_local, sliced alongside retirement
        monotone_g = np.ones(len(models), dtype=float)   # repair multiplier g_i
        maintenance_violation = np.zeros(len(models), dtype=bool)
        maintenance_violation_total = np.zeros(len(models), dtype=int)
        monotone_admitted_last = [0 for _ in models]
        monotone_frontier_last = [0 for _ in models]
        monotone_frontier_trace = []
        _monotone_kernel_h = max(float(np.median(pair_physical_distance)), 1e-12)

        def _one_hop_frontier(ids):
            """``N(ids) \\ ids`` over the fitter's own physical kNN graph."""
            ids = np.asarray(ids, dtype=int)
            if not len(ids):
                return np.empty(0, dtype=int)
            neigh = np.concatenate([frontier_neighbours[int(p)] for p in ids])
            return np.setdiff1d(np.unique(neigh.astype(int)), ids, assume_unique=False)

        def _territory_mask(k):
            mask = np.zeros((n, k), dtype=bool)
            for i, s in enumerate(persistent_territory):
                if s:
                    mask[np.fromiter(s, dtype=int, count=len(s)), i] = True
            return mask

        def _connected_admissible_components(pool_ids, admissible_mask):
            """Connected components of ``pool_ids`` restricted to poses with
            ``admissible_mask`` True, under the physical kNN graph."""
            pool_ids = [int(p) for p in pool_ids]
            adm = [p for p in pool_ids if admissible_mask[p]]
            if not adm:
                return []
            adm_set = set(adm)
            seen = set()
            comps = []
            for start in adm:
                if start in seen:
                    continue
                stack = [start]
                comp = set()
                while stack:
                    p = stack.pop()
                    if p in comp:
                        continue
                    comp.add(p)
                    seen.add(p)
                    for q in frontier_neighbours[p]:
                        q = int(q)
                        if q in adm_set and q not in comp:
                            stack.append(q)
                comps.append(comp)
            return comps

        def _bootstrap_monotone_state():
            nonlocal persistent_territory, born, provisional_source, provisional_pool
            nonlocal maintenance_violation, maintenance_violation_total
            nonlocal monotone_admitted_last, monotone_frontier_last, monotone_g
            k = len(models)
            persistent_territory = [set() for _ in range(k)]
            born = np.zeros(k, dtype=bool)
            provisional_pool = [np.asarray(warm_start_local[i], dtype=int) for i in range(k)]
            provisional_source = []
            for i in range(k):
                pool = provisional_pool[i]
                dist = np.linalg.norm(feature[pool] - feature[fps[i]], axis=1)
                w0 = np.exp(-(dist / _monotone_kernel_h) ** 2)
                w0 = w0 / np.clip(w0.sum(), 1e-12, None)
                vec = np.zeros(n, dtype=float)
                vec[pool] = w0
                provisional_source.append(vec)
            maintenance_violation = np.zeros(k, dtype=bool)
            maintenance_violation_total = np.zeros(k, dtype=int)
            monotone_admitted_last = [0 for _ in range(k)]
            monotone_frontier_last = [0 for _ in range(k)]
            monotone_g = np.ones(k, dtype=float)

        def _smoothmax(values, temperature):
            values = np.asarray(values, dtype=float)
            if not len(values):
                return -np.inf
            z = values / temperature
            m = z.max()
            return float(temperature * (m + np.log(np.mean(np.exp(z - m)))))

        def _apply_monotone_territory(epoch, phase, rv_now):
            """Grow persistent territory / birth provisional charts and return
            the certified-territory (n,K) boolean mask.  ``rv_now`` is the
            current (recon + validity) per (pose, chart) -- exactly the
            quantity ``threshold_topo`` and the 0.9*tau admission rule are both
            defined on.  Admission (birth or growth) only ever happens in
            phase 1; phase 2 leaves territory untouched."""
            nonlocal persistent_territory, born, maintenance_violation, maintenance_violation_total
            nonlocal monotone_admitted_last, monotone_frontier_last, monotone_g
            if persistent_territory is None:
                _bootstrap_monotone_state()
            k = len(models)
            maintenance_violation = np.zeros(k, dtype=bool)
            monotone_admitted_last = [0 for _ in range(k)]
            monotone_frontier_last = [0 for _ in range(k)]
            g = np.ones(k, dtype=float)
            admit_tau = 0.9 * float(threshold_topo)
            T = max(float(threshold_topo) / 5.0, np.finfo(float).eps)
            can_admit = (phase == 1)
            for i in range(k):
                if not born[i]:
                    if can_admit:
                        admissible = rv_now[:, i] <= admit_tau
                        comps = _connected_admissible_components(provisional_pool[i], admissible)
                        if comps:
                            best = max(comps, key=len)
                            persistent_territory[i] = set(int(p) for p in best)
                            born[i] = True
                            monotone_admitted_last[i] = int(len(best))
                    continue        # provisional source carries no maintenance signal
                s_ids = np.fromiter(persistent_territory[i], dtype=int, count=len(persistent_territory[i]))
                rv_s = rv_now[s_ids, i]
                if bool(np.any(rv_s > float(threshold_topo))):
                    # Maintenance violation: this chart's existing territory has
                    # regressed past tau.  Block admission/pruning for it this
                    # E-step and keep training it -- never redefine its
                    # territory to make the violation disappear.
                    maintenance_violation[i] = True
                    maintenance_violation_total[i] += 1
                g[i] = float(_ownership_weight(np.array([_smoothmax(rv_s, T)]), threshold_topo)[0])
                if maintenance_violation[i] or not can_admit:
                    continue
                frontier_ids = _one_hop_frontier(s_ids)
                monotone_frontier_last[i] = int(len(frontier_ids))
                if len(frontier_ids):
                    admitted = frontier_ids[rv_now[frontier_ids, i] <= admit_tau]
                    if len(admitted):
                        persistent_territory[i].update(int(p) for p in admitted)
                        monotone_admitted_last[i] = int(len(admitted))
            monotone_g = g
            return _territory_mask(k)

        def _monotone_shell_priority():
            """Smooth per-pose frontier weight f_i, (n,K): the physical kernel
            K_pq summed over the chart's territory (or, while unborn, its fixed
            provisional density), masked off the territory itself and
            normalized to unit mass per chart -- so a larger territory does not
            automatically pull a bigger share of training."""
            k = len(models)
            F = np.zeros((n, k), dtype=float)
            for i in range(k):
                if born[i] and persistent_territory[i]:
                    ids = np.fromiter(persistent_territory[i], dtype=int, count=len(persistent_territory[i]))
                    ind = np.zeros(n, dtype=float)
                    ind[ids] = 1.0
                    raw = np.asarray(_attach_kernel @ ind).ravel()
                    raw[ids] = 0.0
                else:
                    raw = np.asarray(_attach_kernel @ provisional_source[i]).ravel()
                total = raw.sum()
                F[:, i] = raw / total if total > 1e-15 else raw
            return F

        # Inertial-frontier smooth claims (option-gated, mutually exclusive
        # with ``monotone_frontier``).  ``claims[:, i]`` is a continuous [0, 1]
        # per-pose ownership-like claim that decays/grows by EMA every E-step
        # -- no hard admission, no born/unborn state, no permanent set, no
        # maintenance block.  ``claims is None`` marks "not yet bootstrapped".
        #
        # Everything downstream of the claim is deliberately dense: the
        # physical kernel ``_dense_physical_kernel`` is a full (n, n) Gaussian
        # of physical distance (no kNN graph, so no dynamic-cutoff support),
        # and every chart trains on the FULL pose set every M-step, weighted
        # continuously by its claim / frontier pressure -- there is no
        # core/shell pool membership threshold and no shell-weight cap.
        claims = None
        inertial_trace = []
        _inertial_kernel_h = max(float(np.median(pair_physical_distance)), 1e-12)
        _dense_physical_kernel = None
        if inertial_frontier:
            from scipy.spatial.distance import cdist as _cdist
            _dense_physical_kernel = np.exp(
                -(_cdist(feature, feature) / max(float(typical_spacing), 1e-12)) ** 2)
            np.fill_diagonal(_dense_physical_kernel, 0.0)
            # Row-normalize: (K s)(p) must stay a WEIGHTED AVERAGE of s over
            # p's physical neighbourhood (bounded to roughly [0, max(s)]), not
            # a raw sum that grows with however many of the n poses fall
            # within the kernel's support -- an unnormalized dense sum over
            # ~100 poses saturates exp(-K s) to ~0 almost everywhere, which
            # destroys the reachable field's locality signal and injects
            # near-uniform frontier weight over poses that are not actually
            # nearby.
            _dense_physical_kernel = (_dense_physical_kernel
                                      / _dense_physical_kernel.sum(1, keepdims=True).clip(1e-12, None))

        def _bootstrap_inertial_claims():
            nonlocal claims
            k = len(models)
            claims = np.zeros((n, k), dtype=float)
            for i in range(k):
                dist = np.linalg.norm(feature - feature[fps[i]], axis=1)
                claims[:, i] = np.exp(-(dist / _inertial_kernel_h) ** 2)

        def _soft_exclusive_mass(s):
            """E_i = sum_p [s_i(p) - max_{j != i} s_j(p)]_+ -- the soft analogue
            of ``_exclusive_support`` used only to gate inertial-frontier
            retirement."""
            s = np.asarray(s, float)
            k = s.shape[1]
            if k < 2:
                return s.clip(0.0, None).sum(0)
            part = np.partition(s, k - 2, axis=1)
            w2 = part[:, k - 2]
            wmax = s.max(1)
            is_argmax = s >= wmax[:, None] - 1e-12
            max_excl = np.where(is_argmax, w2[:, None], wmax[:, None])
            return np.clip(s - max_excl, 0.0, None).sum(0)

        def _update_inertial_claims(epoch, phase, rv_now):
            """Smooth competence c_i(p) = sigma((tau - e_i(p)) / T), gated by
            a smooth locally-reachable field r_i = 1 - (1-s_i)exp(-Ks_i)
            (dense K, so no kNN cutoff) before the EMA -- a pose far from any
            current claim gets r_i approx 0 regardless of how well the chart
            happens to reconstruct it there, so growth stays local instead of
            popping up wherever competence is merely high.  Claims are frozen
            (read-only) outside phase 1."""
            nonlocal claims
            if claims is None:
                _bootstrap_inertial_claims()
            if phase != 1:
                return
            c = _ownership_weight(rv_now, threshold_topo)          # (n,K)
            Ks = _dense_physical_kernel @ claims                   # (n,K)
            r = 1.0 - (1.0 - claims) * np.exp(-Ks)
            target = r * c
            # Asymmetric EMA: react fast to a claim gaining ground
            # (inertial_alpha) but decay slowly when it loses ground
            # (inertial_alpha_loss).  Coverage collapsed within ~100 epochs
            # under the symmetric EMA -- a chart's claim only has to dip
            # below target briefly (e.g. while another chart is mid-update)
            # for it to lose territory at the same rate it gained it, and
            # zero_exclusive_support retirement reads that dip as genuine
            # redundancy.  Slow loss decay keeps a claim's floor from
            # collapsing on a transient dip.
            alpha = np.where(target >= claims, float(inertial_alpha), float(inertial_alpha_loss))
            claims = np.clip((1.0 - alpha) * claims + alpha * target, 0.0, 1.0)

        def _update_ema_ownership(cover_mask):
            """Asymmetric EMA of the boolean ``cover_competent`` mask, used only
            to gate zero-exclusive-support retirement (see ``_ema_cover`` init).
            Batch composition is untouched -- this never feeds ``core``/``shell``."""
            nonlocal _ema_cover
            target = cover_mask.astype(float)
            if _ema_cover is None:
                _ema_cover = target.copy()
                return
            alpha = np.where(target >= _ema_cover, ema_ownership_alpha, ema_ownership_alpha_loss)
            _ema_cover = (1.0 - alpha) * _ema_cover + alpha * target

        def _inertial_frontier_weight():
            """f_i(p) = (1 - s_i(p)) * (K s_i)(p) * d(p), (n,K):
            (1-s_i)(Ks_i)  -- expand locally, past a dense physical kernel;
            d(p) = exp(-sum_j s_j(p)) -- smooth global demand, preferring
            empty or underrepresented regions over ones already well-claimed
            by ANY chart, not just this one."""
            Ks = _dense_physical_kernel @ claims                   # (n,K)
            d = np.exp(-claims.sum(1))                             # (n,)
            return (1.0 - claims) * Ks * d[:, None]

        _frontier_epoch_log = []
        _frontier_shell_global = {}
        _frontier_core_global = {}
        _frontier_full_shell = {}
        _frontier_shell_scale = {}
        _frontier_shell_fold = {}
        # Frontier is recomputed only at an E-step and reused for the whole
        # interval (spec: "keep the frontier fixed within each E-step interval").
        _cached_batches = None
        _rebatch_needed = True
        # When freeze_estep_probes is set, the off-anchor cycle probe noise
        # (direction/radius_fraction) and the l_separation far-pair indices
        # are drawn once per E-step interval and reused for every M-step in
        # it, instead of being redrawn every epoch -- isolates instability
        # caused by resampling noise from instability caused by the loss
        # weights themselves.  Cleared alongside _cached_batches.
        _cached_probe_direction = None
        _cached_probe_radius_fraction = None
        _cached_far_pairs = {}
        # Fraction of the current shell whose live loss has fallen below
        # threshold_topo -- the adaptive-E-step "substantially absorbed" signal.
        _last_shell_absorbed = 1.0
        _last_shell_absorbed_per_chart = np.ones(len(models), dtype=float)
        # Learned, physical boundary collar.  It is recomputed from the live
        # shared competence field; it is never an assigned donor/receiver map.
        boundary_collar = np.zeros(n, dtype=bool)
        phase1_stable_streak = 0
        core_protection_need = np.ones((n, len(models)), dtype=float)
        # Per-(pose, chart) recon+validity loss cached at the last E-step; the
        # core-protection hinge penalises the live loss rising above it + margin.
        core_loss_cache = np.full((n, len(models)), np.inf)
        # Detached E-step shell-priority policy Q_i(p), (n, K); recomputed each
        # E-step, consumed by m_step_batch (sampling) and the M-step shell mean.
        shell_priority = np.ones((n, len(models)), dtype=float)
        trace = []
        held = rng.choice(n, size=max(1, n // 5), replace=False)
        support_floor = max(1, int(np.ceil(n * float(min_certified_support_fraction))))
        territory_previous = None
        retirement_events, phase_trace, chart_count_trajectory = [], [], []
        retirement_rejection_log = []
        phase1_coverage_curve, phase2_coverage_curve = [], []
        # Commit-3 validation instrumentation.
        core_retention_curve = []
        shell_promotion_curve = []
        _prev_core_mask = None          # (n,) union: some chart held it as core
        _prev_shell_union = None        # (n,) union of last interval's shell poses
        # Phase-1 pruning is stability-gated, not scheduled: a runt is retired
        # only while the cover is already complete, so removing a
        # zero-exclusive-support chart cannot open a hole.
        prune_snapshot_dir = os.path.join(work, "prune_snapshots")
        prune_snapshot_index = []
        if snapshot_prune_steps:
            os.makedirs(prune_snapshot_dir, exist_ok=True)

        if exploration_snapshot_every is not None and exploration_snapshot_every <= 0:
            raise ValueError("exploration_snapshot_every must be a positive epoch interval")
        if (phase2_exploration_snapshot_every is not None
                and phase2_exploration_snapshot_every <= 0):
            raise ValueError("phase2_exploration_snapshot_every must be a positive epoch interval")
        exploration_snapshot_dir = os.path.join(work, "chart_exploration_snapshots")
        exploration_snapshot_index = []
        if exploration_snapshot_every is not None:
            os.makedirs(exploration_snapshot_dir, exist_ok=True)
        phase2_exploration_snapshot_dir = os.path.join(work, "phase2_chart_exploration_snapshots")
        phase2_exploration_snapshot_index = []
        if phase2_exploration_snapshot_every is not None:
            os.makedirs(phase2_exploration_snapshot_dir, exist_ok=True)

        def _write_prune_snapshot(epoch, phase):
            """Serialize the current ensemble AE state at a prune tick."""
            sync_model_shells()
            path = os.path.join(prune_snapshot_dir, f"prune_epoch{int(epoch):05d}_{phase}.pt")
            torch.save({"epoch": int(epoch), "phase": phase, "n_charts": len(models),
                        "case": case, "chart_ids": list(chart_ids),
                        "states": [m.state_dict() for m in models],
                        "encoder_kind": encoder_kind, "n_bodies": n_bodies,
                        "centres": centre_t.detach().cpu().numpy(),
                        "hard_competent": hard_competent.copy()}, path)
            prune_snapshot_index.append({"epoch": int(epoch), "phase": phase,
                                         "n_charts": len(models), "file": os.path.basename(path)})

        def _write_exploration_snapshot(epoch, *, phase, final=False):
            """Save an evidence-only replay frame for the Phase-1 chart viewer.

            The viewer re-encodes the simulator-produced sample poses with these
            saved learned fields.  We deliberately do not serialize declared
            coordinates, chart routing labels, or a membership lookup.
            """
            if phase == 1:
                directory, index = exploration_snapshot_dir, exploration_snapshot_index
                enabled = exploration_snapshot_every is not None
            else:
                directory, index = phase2_exploration_snapshot_dir, phase2_exploration_snapshot_index
                enabled = phase2_exploration_snapshot_every is not None
            if not enabled:
                return
            sync_model_shells()
            # Record the fitter's actual ownership rule, not a viewer-side
            # reconstruction proxy.  During phase 1 it is re-evaluated after
            # this epoch's M-step from recon+physical-validity and the physical
            # connectivity smoothing; phase 2 retains that final P1 ownership.
            if phase == 1:
                _, snapshot_rec, snapshot_val, snapshot_gate = _eval_all()
                snapshot_ownership, _, _ = _assign_from(
                    snapshot_rec, snapshot_val, snapshot_gate)
            else:
                snapshot_ownership = np.asarray(ownership, float).copy()
            # The M-step's owned core is deliberately overlapping: a pose is
            # in every chart whose P1 ownership weight clears ``core_level``.
            # Preserve its low-support fallback as well, so the saved support
            # exactly matches the fitter's definition of a chart's core.
            snapshot_core_owned = np.asarray(
                snapshot_ownership >= float(core_level), dtype=bool)
            for chart in range(snapshot_core_owned.shape[1]):
                if int(snapshot_core_owned[:, chart].sum()) < 2:
                    top = np.argsort(snapshot_ownership[:, chart])[::-1][:min(2, n)]
                    snapshot_core_owned[top, chart] = True
            snapshot_owner_chart_ids = np.asarray(chart_ids, dtype=int)[
                np.asarray(snapshot_ownership).argmax(axis=1)]
            name = f"epoch{int(epoch):05d}.pt"
            path = os.path.join(directory, name)
            states = []
            for model in models:
                states.append({key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()})
            torch.save({"format": "competitive_cae_exploration_v1",
                        "epoch": int(epoch), "phase": int(phase), "final": bool(final),
                        "case": case, "chart_ids": list(chart_ids),
                        "n_charts": int(len(models)), "encoder_kind": encoder_kind,
                        "n_bodies": int(n_bodies), "d": int(d), "states": states,
                        "ownership_weights": np.asarray(snapshot_ownership, np.float32),
                        "core_owned_mask": snapshot_core_owned,
                        "ownership_core_level": float(core_level),
                        "owner_chart_ids": snapshot_owner_chart_ids,
                        "ownership_phase": 1,
                        "centres": centre_t.detach().cpu().numpy()}, path)
            index.append({"epoch": int(epoch), "phase": int(phase), "final": bool(final),
                          "n_charts": int(len(models)), "file": name})

        joint_start = time.perf_counter()
        joint_timing = {"e_step_seconds": 0.0, "chart_loss_seconds": 0.0,
                        "objective_assembly_seconds": 0.0,
                        "spectral_penalty_seconds": 0.0,
                        "selector_seconds": 0.0,
                        "overlap_replay_seconds": 0.0, "backward_step_seconds": 0.0}
        epoch_measurements = []

        def sync_device():
            if compute_device.type == "cuda":
                torch.cuda.synchronize(compute_device)

        def sync_tick():
            """Drain queued device work, THEN read the clock.

            A bare ``time.perf_counter()`` taken before synchronizing lets GPU
            work queued by the previous stage finish inside this stage's
            interval, mis-attributing its cost.  Every timed interval below opens
            with this and closes with ``elapsed_since`` (which also syncs), so a
            bucket measures only the work dispatched within it.
            """
            sync_device()
            return time.perf_counter()

        def elapsed_since(tick):
            sync_device()
            return time.perf_counter() - tick

        # ``joint_breakdown_seconds`` buckets are NESTED, not a partition:
        # ``chart_loss_seconds`` already contains ``batch_selection`` /
        # ``dense_forward`` / ``batched_cycle`` / ``batched_tangent``, and the
        # E-step calls ``chart_loss`` too.  Do not sum them for a wall-clock
        # total -- use ``joint_training_seconds`` for that.
        joint_timing["breakdown_note"] = ("buckets are nested/overlapping; "
                                          "do not sum -- see joint_training_seconds")

        def dense_chart_forward(batch_ids):
            """Evaluate all chart AE forwards on one padded [K,B] batch.

            The parameter and buffer pytrees are persistent [K,...] ensemble
            state.  There is deliberately no per-epoch state-dict rebuild or
            parameter stack here.
            """
            from torch.func import functional_call, vmap
            k = len(models)
            max_batch = max(len(ids) for ids in batch_ids)
            padded = torch.zeros((k, max_batch, Q.shape[1]), dtype=Q.dtype, device=Q.device)
            mask = torch.zeros((k, max_batch), dtype=torch.bool, device=Q.device)
            for i, ids in enumerate(batch_ids):
                count = len(ids)
                padded[i, :count] = Q[torch.as_tensor(ids, dtype=torch.long, device=Q.device)]
                mask[i, :count] = True
            def call_one(params, buffers, q_batch):
                return functional_call(ensemble_template, (params, buffers), (q_batch,))

            encoded, predicted = vmap(call_one, in_dims=(0, 0, 0))(
                ensemble_params, ensemble_buffers, padded)
            return encoded, predicted, mask

        def _tangent_owned(selected_batches):
            """Reproduce chart_loss's per-chart owned-pose set and weights.

            Same rule as the scalar tangent block: pose weight is the incident
            sum of ``min`` responsibilities over the chart's local pairs; keep
            every pose with weight, else the 8 strongest, capped at 256.
            """
            local_idx, local_w, global_idx = [], [], []
            for chart, (ids, local_pairs) in enumerate(selected_batches):
                ids = np.asarray(ids, dtype=int)
                n_local = len(ids)
                if len(local_pairs):
                    lp = np.asarray(local_pairs, dtype=int)
                    w = np.minimum(responsibilities[ids[lp[:, 0]], chart],
                                   responsibilities[ids[lp[:, 1]], chart])
                else:
                    lp = np.zeros((1, 2), dtype=int)
                    w = np.ones(1)
                w = np.maximum(w, 0.0)
                pose_w = np.zeros(n_local)
                np.add.at(pose_w, lp[:, 0], w)
                np.add.at(pose_w, lp[:, 1], w)
                owned = np.flatnonzero(pose_w > 1e-8)
                if len(owned) == 0:
                    owned = np.argsort(pose_w)[::-1][:min(8, n_local)].copy()
                if len(owned) > 256:
                    owned = owned[np.argsort(pose_w[owned])[::-1][:256]]
                weight = pose_w[owned]
                weight = weight / max(weight.sum(), 1e-8)
                local_idx.append(np.asarray(owned, dtype=int))
                local_w.append(weight)
                global_idx.append(ids[owned])
            return local_idx, local_w, global_idx

        def batched_tangent(selected_batches, dense_u):
            """One ``jvp(ensemble_decode, .)`` per latent axis, across all charts.

            The scalar path sliced a single chart out of the canonical [K,...]
            parameter tensors on every call (select/scatter autograd on the big
            leaves) and ran ``d`` jvps per chart -- the largest remaining
            backward cost.  Here the owned samples are padded to ``[K, P, d]``
            and each chart's ``d`` decoded-tangent columns come from ``d``
            vmapped jvps on the persistent ensemble.  Padding carries weight 0,
            so the returned per-chart ``[K]`` vector's ``.mean()`` reproduces
            the scalar ``torch.stack([x[2][8] for x in losses]).mean()`` for
            identical parameters (gram term exactly; projector term likewise).
            """
            from torch.func import jvp
            k = len(models)
            local_idx, local_w, global_idx = _tangent_owned(selected_batches)
            P = max(1, max(len(o) for o in local_idx))
            f_dim = frames_t.shape[1]
            u_pad = torch.zeros((k, P, d), dtype=Q.dtype, device=Q.device)
            w_pad = torch.zeros((k, P), dtype=Q.dtype, device=Q.device)
            obs_pad = torch.zeros((k, P, f_dim, d), dtype=Q.dtype, device=Q.device)
            for chart in range(k):
                m = len(local_idx[chart])
                if not m:
                    continue
                rows = torch.as_tensor(local_idx[chart], dtype=torch.long, device=Q.device)
                grows = torch.as_tensor(global_idx[chart], dtype=torch.long, device=Q.device)
                u_pad[chart, :m] = dense_u[chart, rows]
                w_pad[chart, :m] = torch.as_tensor(local_w[chart], dtype=Q.dtype, device=Q.device)
                obs_pad[chart, :m] = frames_t[grows].reshape(m, -1, d)
            columns = []
            for axis in range(d):
                tang = torch.zeros_like(u_pad)
                tang[..., axis] = 1.0
                _, jac_col = jvp(ensemble_decode, (u_pad,), (tang,))
                columns.append(jac_col)
            got = torch.stack(columns, -1)                      # [k, P, f_dim, d]
            got_gram = got.transpose(-1, -2) @ got
            want_gram = obs_pad.transpose(-1, -2) @ obs_pad
            got_gram = got_gram / torch.linalg.vector_norm(got_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            want_gram = want_gram / torch.linalg.vector_norm(want_gram, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            gram_term = ((w_pad[:, :, None, None] * (got_gram - want_gram).square()).sum(dim=(1, 2, 3)) /
                         (got_gram.shape[-1] * got_gram.shape[-2]))
            if float(projector_weight) != 0.0:
                # A fixed 1e-8 jitter is negligible once a batch's local
                # tangent frame is genuinely rank-deficient (e.g. a fold,
                # or a claim/EMA setting that hands the M-step a
                # near-duplicate batch) -- torch.linalg.solve then raises
                # LinAlgError mid-training.  Scale the ridge to the
                # gram's own magnitude so it always restores invertibility,
                # not just when gram already happens to be near unit scale.
                proj_err = (w_pad * _projector_distance_sq(got, obs_pad, ridge_rel=1e-6)).sum(dim=1)
                tangent = gram_term + float(projector_weight) * proj_err
            else:
                tangent = gram_term
            return tangent, [int(len(o)) for o in local_idx]

        # Off-node seam neighbourhood (Step 1).  Displace a chart's owned codes
        # by a fraction of the data-derived local support radius and keep only
        # those still inside the union-of-balls support.  Probing at a fixed
        # small fraction (the legacy 0.05) reads encoder-pushed reconstruction
        # error -- a constant offset, not a local chart property -- so the
        # default is a per-point uniform draw over (0, 1] * r_local; a fixed
        # ``fraction`` is used only to reproduce the radius sweep.
        def seam_candidates(chart, owned_ids, fraction=None, cap=256):
            """Displaced in-support seam codes for one chart, WITHOUT a host sync.

            Returns ``(u0, u1, accepted)`` -- all detached device tensors, the
            acceptance kept as a bool tensor so a batched consumer can mask it
            in a reduction rather than forcing ``bool(accepted.any())`` (a CUDA
            synchronization).  ``None`` only for the host-decidable ``< 2``
            owned case.
            """
            m = models[chart]
            own = torch.as_tensor(np.asarray(owned_ids, dtype=int), dtype=torch.long, device=Q.device)
            if len(own) < 2:
                return None
            with torch.no_grad():
                support_u = encode(m, chart, Q[own])
                # Keep local-union support entirely on the active device.  This
                # is equivalent to latent_support.in_local_support: nearest
                # witness owns the candidate and contributes its kNN radius.
                support_distance = torch.cdist(support_u, support_u)
                support_k = min(max(1, max(8, 4 * d)), max(1, len(support_u) - 1))
                support_radii = support_distance.topk(support_k + 1, largest=False).values[:, -1].clamp_min(1e-12)
                probe_ids = (own if len(own) <= cap
                             else own[torch.randperm(len(own), device=own.device)[:cap]])
                u0 = encode(m, chart, Q[probe_ids])
                scale = support_radii[torch.cdist(u0, support_u).argmin(1), None]
                direction = torch.randn_like(u0)
                direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
                frac = (torch.rand(len(u0), 1, dtype=Q.dtype, device=u0.device) if fraction is None
                        else torch.full((len(u0), 1), float(fraction), dtype=Q.dtype, device=u0.device))
                u1 = u0 + frac * scale * direction
                nearest_distance, nearest = torch.cdist(u1, support_u).min(1)
                accepted = nearest_distance <= support_radii[nearest]
            return u0.detach(), u1.detach(), accepted.detach()

        def seam_neighbourhood(chart, owned_ids, fraction=None, cap=256):
            """Filtered ``(u0, u1)`` for the terminal diagnostics (a host sync
            here is fine -- the result is moved to numpy immediately after)."""
            cand = seam_candidates(chart, owned_ids, fraction=fraction, cap=cap)
            if cand is None:
                return None
            u0, u1, accepted = cand
            keep = torch.nonzero(accepted).flatten()
            if not len(keep):
                return None
            return u0[keep].detach(), u1[keep].detach()

        def off_node_redundancy(cover):
            """Physical-coverage probe for safe retirement.

            ``cover`` is the cover mask ``(recon + valid) <= threshold_topo`` --
            the same mask that defines loss, coverage and ownership.  A chart is
            removable only if (a) it has no exclusive anchor territory and (b)
            its own in-support, off-sample neighbourhood lies over physical
            data that at least one *other* live chart covers.  The probe is
            decoded by the candidate, mapped to the nearest real pose in the
            physical metric, and checked against the survivors' cover mask; the
            survivors are never asked to reproduce the candidate's decoder
            output (which would also demand its errors).
            """
            k = len(models)
            anchor_exclusive = _exclusive_support(cover)
            redundant = np.zeros(k, dtype=bool)
            rows = []
            _OFF_NODE_FRACTION = 0.95
            with torch.no_grad():
                for i, model in enumerate(models):
                    ids = np.flatnonzero(cover[:, i])
                    row = {"anchor_exclusive": int(anchor_exclusive[i]),
                           "off_node_candidates": int(len(ids)),
                           "off_node_certified": 0,
                           "off_node_covered_by_survivor": 0,
                           "off_node_ok_fraction": 0.0,
                           "anchor_redundant": bool(anchor_exclusive[i] == 0),
                           "off_node_redundant": False,
                           "rejection_reasons": []}
                    if not row["anchor_redundant"]:
                        row["rejection_reasons"].append("anchor_not_redundant")
                    if len(ids) < 2:
                        row["rejection_reasons"].append("too_few_off_node_candidates")
                        rows.append(row)
                        continue
                    source_q = Q[torch.as_tensor(ids, dtype=torch.long, device=Q.device)]
                    source_u = encode(model, i, source_q)
                    support_np = source_u.cpu().numpy()
                    radii = torch.as_tensor(local_support_radii(support_np), dtype=Q.dtype,
                                             device=Q.device)[:, None]
                    direction = torch.randn_like(source_u)
                    direction = direction / torch.linalg.vector_norm(
                        direction, dim=-1, keepdim=True).clamp_min(1e-8)
                    u1 = source_u + torch.rand((len(source_u), 1), dtype=Q.dtype,
                                                device=Q.device) * radii * direction
                    accepted = in_local_support(support_np, u1.cpu().numpy())
                    if not accepted.any():
                        row["rejection_reasons"].append("no_offsample_probes_accepted")
                        rows.append(row)
                        continue
                    take = torch.as_tensor(np.flatnonzero(accepted), dtype=torch.long, device=u1.device)
                    q1 = _decode(model, i, u1[take]).detach().cpu().numpy().astype(float)
                    _, nearest_pose = feature_tree.query(_pose_features(q1), k=1)
                    survivors = np.delete(cover, i, axis=1)
                    covered_by_other = survivors[nearest_pose].any(1)
                    frac_ok = float(covered_by_other.mean()) if len(covered_by_other) else 0.0
                    row["off_node_certified"] = int(len(covered_by_other))
                    row["off_node_covered_by_survivor"] = int(covered_by_other.sum())
                    row["off_node_ok_fraction"] = frac_ok
                    row["off_node_redundant"] = bool(frac_ok >= _OFF_NODE_FRACTION)
                    if not row["off_node_redundant"]:
                        row["rejection_reasons"].append(
                            f"offsample_fraction_{frac_ok:.3f}_below_{_OFF_NODE_FRACTION}")
                    redundant[i] = bool(row["anchor_redundant"] and row["off_node_redundant"])
                    rows.append(row)
            return redundant, rows

        def _fold_score(chart, core, cand):
            """False-neighbour score ``phi`` (detached) of candidate poses against
            a chart's core.

            A chart's encoder must fold somewhere -- no continuous map from a
            closed manifold into R^d is injective, so the code read on the WHOLE
            sample set always turns back on itself.  That fold is mandatory and
            is never penalised.  What must not happen is the fold entering the
            chart's own domain, and ``phi`` measures exactly that distance:

                phi(p) = sum_q exp(-(|u_p - u_q| / r)^2) * (1 - exp(-(|x_p - x_q| / h)^2))

            over core poses q, with ``r`` the core's own nearest-neighbour code
            spacing and ``h = _FOLD_PHYS_FRAC`` x the core's physical diameter.
            A term is large only when p is latent-NEAR q while physically FAR
            from it -- a false neighbour, i.e. the chart being asked to decode
            two different places on the manifold from the same code.  A pose one
            hop past the domain boundary is latent-near its neighbours but also
            physically adjacent to them, so it scores ~0 and growth proceeds;
            a pose on the fold's return path is latent-near poses half a
            manifold away and scores O(1) per collision.

            This is the growth-time form of the ``l_separation`` predicate.  That
            term cannot see this: its "physically far" test is a 0.7 QUANTILE
            taken inside the M-step batch, and the batch is an arc plus a 1-hop
            shell that never contains the collision, so it reads 0 at every
            epoch of a run that folds.  Here the bandwidth is absolute (set by
            the core's own extent) and the candidates are scored BEFORE they are
            trained on.
            """
            core = np.asarray(core, dtype=int)
            cand = np.atleast_1d(np.asarray(cand, dtype=int))
            if float(fold_weight) == 0.0 or len(core) < 2 or not len(cand):
                return np.zeros(len(cand), dtype=float)
            with torch.no_grad():
                uc = encode(models[chart], chart,
                            Q[torch.as_tensor(core, dtype=torch.long, device=Q.device)]).cpu().numpy()
                us = encode(models[chart], chart,
                            Q[torch.as_tensor(cand, dtype=torch.long, device=Q.device)]).cpu().numpy()
            code_gap = np.linalg.norm(uc[:, None] - uc[None], axis=-1)
            np.fill_diagonal(code_gap, np.inf)
            r = float(np.median(code_gap.min(1)))
            physical = np.linalg.norm(feature[core][:, None] - feature[core][None], axis=-1)
            h = float(_FOLD_PHYS_FRAC) * float(physical.max())
            if not (np.isfinite(r) and r > 0.0 and np.isfinite(h) and h > 0.0):
                return np.zeros(len(cand), dtype=float)
            latent_near = np.exp(-(np.linalg.norm(us[:, None] - uc[None], axis=-1) / r) ** 2)
            physical_near = np.exp(-(np.linalg.norm(feature[cand][:, None] - feature[core][None], axis=-1) / h) ** 2)
            return (latent_near * (1.0 - physical_near)).sum(1)

        def m_step_batch(chart):
            """One chart's detached M-step support = its owned core plus a 1-hop
            frontier/overlap shell, on INDEPENDENT budgets.

            core  = poses with ownership weight >= ``core_level`` (continuous, no
                    hard mask), capped at ``m_step_batch_size``; a nascent chart
                    with an empty core falls back to its two most-owned rows.
            shell = the 1-hop kNN neighbours of the core, capped at
                    ``_shell_budget``.  The full shell is used when it fits that
                    budget; otherwise a ``shell_sample_fraction`` Q_i-weighted
                    sample.  The shell budget is separate from the core budget so
                    a core at its cap can never shrink the shell to zero.

            Core and shell are normalised SEPARATELY in the M-step; this only
            controls which poses enter each set.
            """
            w_i = np.asarray(ownership[:, chart], float)
            core_cap = max(2, int(m_step_batch_size))
            shell_cap = max(1, int(_shell_budget))
            if monotone_frontier:
                # Core = (a training-budget sample of) the certified territory
                # S_i (empty while the chart is unborn -- all of its training
                # mass then comes from the provisional frontier below).  Shell
                # = the smooth frontier f_i (see ``_monotone_shell_priority``):
                # its SUPPORT is the graph-exact one-hop of S_i (or of the
                # fixed provisional density while unborn), but its per-pose
                # WEIGHT is the physical kernel K_pq, not a uniform coefficient
                # -- frontier poses gain/lose weight smoothly as admission
                # grows the territory, instead of appearing/disappearing as
                # binary shell members.  No fold-score reweighting.
                if born[chart] and persistent_territory[chart]:
                    full_core = np.fromiter(persistent_territory[chart], dtype=int,
                                             count=len(persistent_territory[chart]))
                    core = full_core
                    if len(core) > core_cap:
                        core = np.unique(rng.choice(core, size=core_cap, replace=False))
                else:
                    core = np.empty(0, dtype=int)
                shell_pool = np.flatnonzero(shell_priority[:, chart] > 1e-15)
                full_shell = len(shell_pool) <= shell_cap
                if full_shell:
                    shell_global = shell_pool
                elif len(shell_pool):
                    pw = np.clip(shell_priority[shell_pool, chart], 1e-12, None)
                    _frontier_shell_scale[chart] = float(pw.mean())
                    shell_global = rng.choice(shell_pool, size=shell_cap, replace=False, p=pw / pw.sum())
                else:
                    shell_global = np.empty(0, dtype=int)
                shell_global = np.asarray(shell_global, dtype=int)
                fold = np.zeros(len(shell_global), dtype=float)
            elif inertial_frontier:
                # No pool membership threshold and no budget cap: every chart
                # trains on the FULL pose set every M-step, weighted
                # continuously by its claim s_i (retained/"core" term) and its
                # frontier pressure f_i = (1-s_i)(Ks_i)d (``shell_priority``,
                # see ``_inertial_frontier_weight``) -- both dense over all n.
                core = np.arange(n, dtype=int)
                shell_global = np.arange(n, dtype=int)
                full_shell = True
                fold = np.zeros(len(shell_global), dtype=float)
            else:
                core = np.unique(np.flatnonzero(w_i >= float(core_level)).astype(int))
                if len(core) < 2:
                    core = np.unique(np.argsort(w_i)[::-1][:min(2, n)].astype(int))
                if len(core) > core_cap:
                    core = np.unique(rng.choice(core, size=core_cap, replace=False))
                shell_pool = (np.concatenate([frontier_neighbours[i] for i in core])
                              if len(core) else np.empty(0, dtype=int)).astype(int)
                shell_pool = np.setdiff1d(np.unique(shell_pool), core)
                frac = min(max(float(shell_sample_fraction), 0.0), 1.0)
                full_shell = len(shell_pool) <= shell_cap
                if full_shell:
                    # full one-hop shell: the M-step weights its mean by Q_i.
                    shell_global = shell_pool
                elif len(shell_pool):
                    # can't afford the full shell: Q_i-SAMPLE it (never both --
                    # the M-step then uses a uniform shell mean).
                    want = min(shell_cap, max(1, int(round(len(shell_pool) * frac))))
                    pw = np.clip(shell_priority[shell_pool, chart], 1e-12, None)
                    _frontier_shell_scale[chart] = float(pw.mean())
                    shell_global = rng.choice(shell_pool, size=want, replace=False, p=pw / pw.sum())
                else:
                    shell_global = np.empty(0, dtype=int)
                shell_global = np.asarray(shell_global, dtype=int)
                fold = _fold_score(chart, core, shell_global)
            ids = np.unique(np.concatenate([core, shell_global]).astype(int))
            if len(ids) < 2:
                ids = np.unique(np.r_[ids, int(w_i.argmax())])
            _frontier_core_global[chart] = set(int(x) for x in core)
            _frontier_shell_global[chart] = set(int(x) for x in np.atleast_1d(shell_global))
            _frontier_shell_fold[chart] = {int(p): float(v)
                                           for p, v in zip(np.atleast_1d(shell_global), fold)}
            _frontier_full_shell[chart] = bool(full_shell)
            _frontier_epoch_log.append({"chart": int(chart), "core": int(len(core)),
                                        "batch": int(len(ids)),
                                        "shell": int(len(np.atleast_1d(shell_global))),
                                        "full_shell": bool(full_shell),
                                        "core_at_cap": bool(len(core) >= core_cap),
                                        "shell_fold_mean": float(fold.mean()) if len(fold) else 0.0,
                                        "shell_admittance": float(np.mean(1.0 / (1.0 + float(fold_weight) * fold)))
                                        if len(fold) else 1.0})
            # Brute-force kNN on-device instead of a fresh cKDTree per chart per
            # E-step: at this batch size (<=512) the tree's own build/teardown
            # overhead dominates, and a scipy tree of a new size every call was
            # the main source of host-heap fragmentation under many charts.
            k_local = min(5, len(ids))
            if k_local == 1:
                pair = np.empty((0, 2), dtype=int)
            else:
                ids_t = torch.as_tensor(ids, dtype=torch.long, device=feature_t.device)
                local_feat = feature_t[ids_t].to(torch.float64)
                local_near = torch.cdist(local_feat, local_feat).topk(
                    k_local, largest=False).indices.cpu().numpy()
                pair = np.asarray([(a, b) for a in range(len(ids)) for b in np.atleast_1d(local_near[a])[1:]], dtype=int)
            return ids, pair
        # ================================================================
        # Two-phase training.  Phase 1 grows a topological cover under the
        # absolute per-chart ownership weight; phase 2 retains it as soft
        # inertia while continuing the shared frontier objective.
        # and fits reconstruction accuracy.
        # ================================================================
        base_lr = 2e-3
        p1_max = int(phase1_max_epochs) if phase1_max_epochs is not None else max(1, int(config.epochs))
        p2_max = int(phase2_epochs) if phase2_epochs is not None else max(1, int(config.epochs))
        p1_max = max(1, p1_max); p2_max = max(1, p2_max)
        if resume is not None:
            # The loops are skipped; the boundary E-step after phase 1 and the
            # final certifying E-step still run on the resumed charts.
            p1_max = p2_max = 0
        C = None
        frozen_owned = None
        # d_nn / threshold_fit must live in the SAME standardized units as
        # reconstruction (see _recon), or the threshold_fit hinge in the
        # M-step (compared against ``rv``, now standardized) is unit-
        # mismatched against a raw feature-space spacing.  This does NOT
        # touch the neighbour graph itself (``nearest``/``pairs``/
        # ``frontier_neighbours``, still built on the unscaled ``feature`` --
        # a separate, bigger change) -- only the distance VALUE at the same
        # rank-selected neighbour is recomputed in standardized units.
        _std_neighbour_distance = np.sqrt(
            (((q[:, None, :] - q[nearest, :]) ** 2) / recon_scale[None, None, :]).sum(-1))
        _std_positive = np.where(_std_neighbour_distance > 1e-12, _std_neighbour_distance, np.inf)
        _std_spacing = np.partition(_std_positive, spacing_rank, axis=1)[:, spacing_rank]
        _std_finite = _std_spacing[np.isfinite(_std_spacing)]
        standardized_typical_spacing = float(np.median(_std_finite)) if len(_std_finite) else 1.0
        d_nn = float(standardized_typical_spacing)
        n_components_pose = int(q.shape[1])
        threshold_fit = (0.05 * d_nn) ** 2 / max(n_components_pose, 1)

        def _principal_angle_deg_t(got, want):
            # torch mirror of _principal_angle_deg, kept on ``got``'s device so
            # the (n, f_dim, d) jvp tensor never round-trips through host RAM.
            def _ortho(mat):
                qm, _ = torch.linalg.qr(mat)
                return qm[..., : mat.shape[-1]]
            s = torch.linalg.svdvals(torch.matmul(_ortho(got).transpose(-1, -2), _ortho(want)))
            return torch.rad2deg(torch.arccos(torch.clamp(s.amin(-1), -1.0, 1.0)))

        def _jacobian_fold_ratio_t(got):
            s = torch.linalg.svdvals(got)
            return s.amin(-1) / s.amax(-1).clamp_min(1e-12)

        def _gate_angles_folds():
            """Per (pose, chart): worst decoder-tangent vs oracle-frame principal
            angle (deg) and Jacobian fold ratio sigma_min / sigma_max."""
            from torch.func import jvp
            # float64 to match the previous numpy(float) implementation's
            # precision exactly -- these feed a hard angle/fold gate threshold.
            want_r = torch.as_tensor(frames, dtype=torch.float64, device=Q.device).reshape(n, -1, d)
            angles = torch.empty((n, len(models)), dtype=torch.float64, device=Q.device)
            folds = torch.empty((n, len(models)), dtype=torch.float64, device=Q.device)
            with torch.no_grad():
                for i, m in enumerate(models):
                    u = encode(m, i, Q)
                    cols = []
                    for axis in range(d):
                        tang = torch.zeros_like(u); tang[:, axis] = 1.0
                        _, dc = jvp(lambda z, _m=m, _i=i: _decode(_m, _i, z), (u,), (tang,))
                        cols.append(dc)
                    got = torch.stack(cols, -1).to(torch.float64)
                    angles[:, i] = _principal_angle_deg_t(got, want_r)
                    folds[:, i] = _jacobian_fold_ratio_t(got)
            return angles.cpu().numpy(), folds.cpu().numpy()

        certificate_gate = np.ones((n, len(models)), dtype=bool)
        certificate_events = []

        def _eval_all_lean():
            """Batched full-atlas reconstruction and physical-validity pass.

            The routine E-step needs only reconstruction/validity to update
            ownership, multiplicity, the physical frontier, and retirement
            candidates.  Keeping this on the active device avoids the former
            per-chart ``chart_loss`` calls (off-anchor probes, distortion work,
            and host transfers) and deliberately leaves JVP/QR/SVD work to the
            checkpoint certificate below.
            """
            k = len(models)
            poses = Q.unsqueeze(0).expand(k, -1, -1)
            with torch.no_grad():
                codes = ensemble_encode(poses)
                predicted = ensemble_decode(codes)
                reconstruction_t = _recon(predicted, poses)
                # Validity can be simulator-specific, so retain its established
                # predicate while batching the expensive learned-field forward.
                validity_t = torch.stack([
                    validity_loss(models[i], predicted[i].reshape(n, n_bodies, 3, 4))
                    for i in range(k)], dim=1)
            reconstruction = reconstruction_t.transpose(0, 1).cpu().numpy()
            validity_component = validity_t.cpu().numpy()
            # C only ranks otherwise equally removable charts; recon+valid is
            # the same physical quantity that defines routine coverage.
            return reconstruction + validity_component, reconstruction, validity_component

        def _eval_all_composite():
            """Legacy full composite evaluator, reserved for retirement.

            This intentionally retains off-anchor inverse probes, local alias
            evidence, and separation/distortion work.  It is not a routine
            assignment path: invoking it only when a chart is actually
            retirement-eligible keeps lifecycle decisions on the full learned
            objective without charging every E-step for it.
            """
            with torch.no_grad():
                ev = [chart_loss(m, i, pairs, compute_tangent=False)
                      for i, m in enumerate(models)]
            composite = np.stack([x[0].cpu().numpy() for x in ev], axis=1)
            reconstruction = np.stack([x[2][0].cpu().numpy() for x in ev], axis=1)
            validity_component = np.stack([x[2][2].cpu().numpy() for x in ev], axis=1)
            return composite, reconstruction, validity_component

        if warm_start_full_evaluator_diagnostic:
            # Preserve an explicit warm-boundary parity record.  The original
            # 12->6 artifact predates the lean E-step, so its first-E support
            # must not be compared with a lean count unless the retained full
            # evaluator agrees on the same just-warmed chart states.
            lean_C_warm, lean_rec_warm, lean_val_warm = _eval_all_lean()
            composite_C_warm, composite_rec_warm, composite_val_warm = _eval_all_composite()
            warm_angles, warm_folds = _gate_angles_folds()
            warm_gate = ((warm_angles <= float(threshold_topo_angle_deg)) &
                         (warm_folds >= float(fold_ratio_floor)))
            lean_cover_warm = lean_C_warm <= float(threshold_topo)
            composite_rv_cover_warm = (composite_rec_warm + composite_val_warm) <= float(threshold_topo)
            warm_start_diagnostics["full_vs_lean"] = {
                "lean_covered_per_chart": lean_cover_warm.sum(0).astype(int).tolist(),
                "lean_exclusive_per_chart": _exclusive_support(lean_cover_warm).astype(int).tolist(),
                "full_reconstruction_validity_covered_per_chart": composite_rv_cover_warm.sum(0).astype(int).tolist(),
                "full_composite_covered_per_chart": (composite_C_warm <= float(threshold_topo)).sum(0).astype(int).tolist(),
                "full_tangent_fold_gated_per_chart": (composite_rv_cover_warm & warm_gate).sum(0).astype(int).tolist(),
                "reconstruction_max_abs_difference": float(np.abs(lean_rec_warm - composite_rec_warm).max()),
                "validity_max_abs_difference": float(np.abs(lean_val_warm - composite_val_warm).max()),
                "reconstruction_validity_predicate_disagreements": int(
                    np.count_nonzero(lean_cover_warm != composite_rv_cover_warm)),
            }

        def _eval_all(*, certify=False, certificate_reason=None):
            nonlocal certificate_gate
            C_, rec_, val_ = _eval_all_lean()
            if certify:
                angles_, folds_ = _gate_angles_folds()
                certificate_gate = ((angles_ <= float(threshold_topo_angle_deg)) &
                                    (folds_ >= float(fold_ratio_floor)))
                certificate_events.append({"reason": str(certificate_reason or "checkpoint"),
                                           "n_charts": int(len(models)),
                                           "n_certified": int(certificate_gate.sum())})
            return C_, rec_, val_, certificate_gate
            with torch.no_grad():
                ev = [chart_loss(m, i, pairs, compute_tangent=False) for i, m in enumerate(models)]
            C_ = np.stack([x[0].cpu().numpy() for x in ev], axis=1)
            rec_ = np.stack([x[2][0].cpu().numpy() for x in ev], axis=1)
            val_ = np.stack([x[2][2].cpu().numpy() for x in ev], axis=1)
            ang_, fold_ = _gate_angles_folds()
            gate_ = ((ang_ <= float(threshold_topo_angle_deg)) &
                     (fold_ >= float(fold_ratio_floor)))
            return C_, rec_, val_, gate_

        def _assign_from(rec_, val_, gate_):
            # Ownership (what a chart trains on and grows into) is pure
            # reconstruction/validity -- NO angle/fold gate.  The gate is a hard
            # local predicate that fails wherever the chart is not already
            # accurate, so gating ownership self-locks a chart out of curved
            # territory it has not trained on yet (the curvature deadlock, same
            # shape as the bootstrap bug).  The topological gate is load-bearing
            # only in ``hard`` -- the final competence certificate.  Binarize at
            # the decision, not in the pipeline feeding it.
            #
            # ``cover`` is THE cover mask: cover_competent[p,i] = (recon+valid) <=
            # threshold_topo -- exactly the argument of the ownership-weight
            # sigmoid.  Coverage, multiplicity, exclusive support, greedy cover
            # and zero-exclusive retirement all key on this one mask so they
            # cannot disagree.
            # Validity is dropped here by design: it is affine in the pose
            # entries (w = R.p + t), so any blend of valid poses -- a mean, a
            # kNN average, a decoder smearing across a fold -- scores ~0 and
            # is blind to exactly the failure mode routine ownership needs to
            # see.  The group-standardized ``rec_`` (see recon_scale above)
            # is joint-free and already holds rotation to the same relative
            # accuracy as translation, so it alone defines cover/ownership.
            # val_ is accepted for signature compatibility but unused.
            del val_
            own = _ownership_weight(rec_, threshold_topo)
            own = _soft_connectivity(own, pairs, n, edge_bar=soft_edge_bar, decay=component_decay)
            cover = np.asarray(rec_, float) <= threshold_topo
            hard = cover & gate_
            return own, hard, cover

        def _removable_chart(chart):
            """Observational retirement check: candidate-local one-cover only.

            The GLOBAL collar check (every boundary-collar pose keeps
            multiplicity >= 2 after removal) was removed: it blocked
            retirement on collar state anywhere in the atlas, not just near
            the candidate, which is a much stronger bar than "this chart is
            redundant."  One-cover survival on the candidate's own affected
            poses is the actual redundancy test.
            """
            keep = np.arange(len(models)) != int(chart)
            affected = cover_competent[:, int(chart)]
            one_cover_survives = bool((~affected).all() or
                                      cover_competent[affected][:, keep].any(1).all())
            return one_cover_survives

        def _receiver_certificate(chart):
            """Local full-rank / tangent check on the territory a retirement hands over.

            Every pose in the candidate's cover must keep at least one *other*
            chart that covers it AND passes the tangent-angle / fold-ratio
            gate there.  Evaluated with JVP on the affected poses of the
            receiving charts only -- not on all poses of all charts.
            """
            from torch.func import jvp
            chart = int(chart)
            affected = np.flatnonzero(cover_competent[:, chart])
            if not len(affected):
                return True, {"affected": 0, "uncertified": 0}
            want = torch.as_tensor(frames, dtype=torch.float64, device=Q.device).reshape(n, -1, d)[
                torch.as_tensor(affected, dtype=torch.long, device=Q.device)]
            certified = np.zeros(len(affected), dtype=bool)
            worst_angle, worst_fold = 0.0, float("inf")
            with torch.no_grad():
                for j, m in enumerate(models):
                    if j == chart:
                        continue
                    rows = np.flatnonzero(cover_competent[affected, j])
                    if not len(rows):
                        continue
                    pose_t = Q[torch.as_tensor(affected[rows], dtype=torch.long, device=Q.device)]
                    u = encode(m, j, pose_t)
                    cols = []
                    for axis in range(d):
                        tang = torch.zeros_like(u); tang[:, axis] = 1.0
                        _, dc = jvp(lambda z, _m=m, _j=j: _decode(_m, _j, z), (u,), (tang,))
                        cols.append(dc)
                    got = torch.stack(cols, -1).to(torch.float64)
                    ang = _principal_angle_deg_t(got, want[torch.as_tensor(rows, dtype=torch.long, device=Q.device)]).cpu().numpy()
                    fold = _jacobian_fold_ratio_t(got).cpu().numpy()
                    ok = (ang <= float(threshold_topo_angle_deg)) & (fold >= float(fold_ratio_floor))
                    certified[rows] |= ok
                    worst_angle = max(worst_angle, float(np.max(ang)))
                    worst_fold = min(worst_fold, float(np.min(fold)))
            return bool(certified.all()), {"affected": int(len(affected)),
                                           "uncertified": int((~certified).sum()),
                                           "worst_receiver_angle_deg": worst_angle,
                                           "worst_receiver_fold_ratio": worst_fold}

        def _learned_boundary_collar(own, cover, competence):
            """Physical collar around changing, weak learned chart boundaries.

            The collar is inferred from the current joint field: it contains
            only low-margin endpoints of a physical primary-chart boundary,
            plus immediately adjacent weak-margin poses.  It deliberately does
            not dilate through a whole graph hop: that turned a noisy boundary
            into a global multiplicity-two target.
            """
            if len(models) < 2:
                return np.zeros(n, dtype=bool)
            primary = np.asarray(own).argmax(1)
            edge_change = primary[physical_edges[:, 0]] != primary[physical_edges[:, 1]]
            seed = np.zeros(n, dtype=bool)
            if edge_change.any():
                seed[np.unique(physical_edges[edge_change].ravel())] = True
            top2 = np.partition(np.asarray(competence), -2, axis=1)[:, -2:]
            second, best = top2[:, 0], top2[:, 1]
            margin = best - second
            collar = seed & (margin <= 0.25)
            adjacent_to_collar = ((_attach_kernel @ collar.astype(float)) /
                                  _attach_rowsum >= 0.35)
            collar |= (adjacent_to_collar & (margin <= 0.15) &
                       (second < float(cover_level)))
            return collar

        def _double_cover_mask(phase):
            """Where two-cover is demanded: everywhere in phase 1 (exploration builds wide overlaps and
            strips redundant charts of exclusive territory), only the learned boundary collar in phase 2."""
            return np.ones(n, dtype=bool) if phase == 1 else boundary_collar

        def _shell_policy(w, phase):
            """Symmetric physical shell priority; never assigns receivers."""
            w = np.asarray(w, float)
            H = (_attach_kernel @ w) / _attach_rowsum[:, None]
            H = np.clip(H, 0.0, 1.0)
            count = cover_competent.sum(1) if cover_competent.shape[1] else np.zeros(n, int)
            demand = ((count == 0) | (_double_cover_mask(phase) & (count < 2))).astype(float)
            # Only a locally attached, not-already-competent chart receives
            # frontier mass.  This is a mask on a shared objective, not a
            # pose-to-chart allocation.
            eligible = (H >= 0.02) & ~cover_competent
            Q = _POLICY_EPS + demand[:, None] * H * eligible
            return np.clip(Q, 1e-9, None)

        def _protection_need(reconstruction):
            """(rv_competence, core_protection_need) for the current chart set.

            need(p, i) is high where no OTHER chart is competent at p, so the
            core-protection hinge guards poses that chart i alone holds.
            """
            _rv_logit = (np.asarray(reconstruction) -
                         float(threshold_topo)) / max(float(threshold_topo) / 5.0, 1e-12)
            competence = 1.0 / (1.0 + np.exp(np.clip(_rv_logit, -60.0, 60.0)))
            if competence.shape[1] > 1:
                top2 = np.partition(competence, -2, axis=1)[:, -2:]
                best = top2[:, 1]
                other = np.where(competence >= best[:, None] - 1e-12, top2[:, 0, None], best[:, None])
            else:
                other = np.zeros_like(competence)
            need = np.clip((float(cover_level) - other) / max(float(cover_level), 1e-12), 0.0, 1.0)
            return competence, need

        def _e_step(epoch, phase, *, certify=False, certificate_reason=None):
            nonlocal responsibilities, ownership, multiplicity, membership
            nonlocal deficit_weights, rim_temperature, hard_competent, cover_competent, usage_ema
            nonlocal C, e_step_reconstruction, e_step_validity, territory_previous
            nonlocal models, ensemble_params, ensemble_buffers, ensemble_opt, centre_t
            nonlocal stretch_reference, log_sigma, correction_w, correction_b, selector_opt
            nonlocal zero_exclusive_streak, chart_ids
            nonlocal core_loss_cache, _rebatch_needed, shell_priority
            nonlocal phase1_stable_streak, boundary_collar
            nonlocal core_protection_need
            nonlocal _last_shell_absorbed_per_chart
            nonlocal certificate_gate
            nonlocal _prev_core_mask, _prev_shell_union
            nonlocal persistent_territory, maintenance_violation, maintenance_violation_total
            nonlocal monotone_admitted_last, monotone_frontier_last
            nonlocal born, provisional_source, provisional_pool, monotone_g
            nonlocal claims
            nonlocal _ema_cover
            tick = sync_tick()
            C, e_step_reconstruction, e_step_validity, gate = _eval_all(
                certify=certify, certificate_reason=certificate_reason)
            ownership, hard_competent, cover_competent = _assign_from(e_step_reconstruction, e_step_validity, gate)
            membership = ownership
            # ``rv_now`` / ``_rv_logit`` drive routine ownership/territory --
            # reconstruction only (see _assign_from): validity is affine and
            # blind to the pose blends this machinery must actually detect.
            if monotone_frontier:
                rv_now = np.asarray(e_step_reconstruction, float)
                cover_competent = _apply_monotone_territory(epoch, phase, rv_now)
            if inertial_frontier:
                # Claims never redefine coverage -- final certificates stay on
                # the standard hard recon <= threshold_topo mask.
                rv_now = np.asarray(e_step_reconstruction, float)
                _update_inertial_claims(epoch, phase, rv_now)
            if ema_ownership and phase == 1:
                _update_ema_ownership(cover_competent)
            multiplicity = cover_competent.sum(1).astype(float)
            coverage_complete = bool(cover_competent.any(1).all())
            rv_competence, core_protection_need = _protection_need(e_step_reconstruction)
            boundary_collar = _learned_boundary_collar(ownership, cover_competent, rv_competence)
            locally_certified = bool(hard_competent.any(1).all())
            # Completion is a local-atlas condition: a feasible one-cover,
            # no removable runt, and passing local certificates.  Two-cover is
            # only a Phase-1 training incentive.
            removable_now = bool(np.any(_exclusive_support(cover_competent) == 0))
            collar_complete = bool((multiplicity[boundary_collar] >= 2).all()
                                   if boundary_collar.any() else True)
            phase1_stable_streak = (phase1_stable_streak + 1
                                    if phase == 1 and coverage_complete and collar_complete
                                    and locally_certified and not removable_now else 0)
            deficit_weights = np.ones(n, dtype=float)
            rim_temperature = np.full(len(models), max(threshold_topo / 5.0, np.finfo(float).eps))
            if phase == 2:
                # Strict, locally feasible competition.  Phase-1 ownership is
                # merely inertia: temperature anneals toward specialization,
                # while the shared frontier/collar objective remains active.
                local = np.clip((_attach_kernel @ ownership) / _attach_rowsum[:, None], 0.0, 1.0)
                temperature = max(float(threshold_fit), float(threshold_topo) *
                                  (0.35 * (1.0 - min(epoch / max(p2_max, 1), 1.0)) + 0.05))
                strict = _responsibilities(np.asarray(e_step_reconstruction), temperature)
                strict *= np.maximum(local, 1e-4)
                strict /= strict.sum(1, keepdims=True)
                responsibilities = 0.80 * responsibilities + 0.20 * strict
                responsibilities /= responsibilities.sum(1, keepdims=True)
                frozen_owned[:] = [np.flatnonzero(responsibilities[:, i] >=
                                                   max(1.0 / len(models), 0.10))
                                   for i in range(len(models))] if frozen_owned is not None else []
            else:
                responsibilities = _owner_responsibilities(ownership)
            usage_ema = (float(usage_ema_decay) * usage_ema +
                         (1.0 - float(usage_ema_decay)) * responsibilities.mean(0))
            covered = cover_competent.any(1)
            (phase1_coverage_curve if phase == 1 else phase2_coverage_curve).append(
                {"epoch": int(epoch), "n_charts": int(len(models)),
                 "n_uncovered": int((~covered).sum()),
                 "n_uncovered_gated": int((~hard_competent.any(1)).sum()),
                 "n_singleton": int((multiplicity == 1).sum()),
                 "n_removable": int((_exclusive_support(cover_competent) == 0).sum()),
                 "phase1_stable": bool(phase1_stable_streak >= int(retire_hysteresis_steps)),
                 "coverage_complete": coverage_complete})
            # Validation: fixed-core retention (K-independent -- union of poses
            # some chart holds as core) and shell-promotion (last interval's
            # shell poses that reached core this E-step).
            core_mask = (ownership >= float(core_level)).any(1)
            if _prev_core_mask is not None:
                kept = int((_prev_core_mask & core_mask).sum())
                core_retention_curve.append(
                    {"epoch": int(epoch), "retained": kept,
                     "prev_core": int(_prev_core_mask.sum()),
                     "retention": float(kept / max(int(_prev_core_mask.sum()), 1))})
            if _prev_shell_union is not None and _prev_shell_union.any():
                promoted = int((_prev_shell_union & core_mask).sum())
                shell_promotion_curve.append(
                    {"epoch": int(epoch), "promoted": promoted,
                     "shell": int(_prev_shell_union.sum()),
                     "promotion": float(promoted / max(int(_prev_shell_union.sum()), 1))})
            _prev_core_mask = core_mask
            _prev_shell_union = np.zeros(n, dtype=bool)
            for s in _frontier_shell_global.values():
                if s:
                    _prev_shell_union[np.fromiter((int(x) for x in s), dtype=int)] = True
            territory = []
            for i in range(len(models)):
                ids = np.flatnonzero(hard_competent[:, i])
                territory.append({"ids": ids, "centroid": (feature[ids].mean(0) if len(ids)
                                  else np.full(feature.shape[1], np.nan))})
            drift = []
            if territory_previous is not None and len(territory_previous) == len(territory):
                for before, after in zip(territory_previous, territory):
                    union = np.union1d(before["ids"], after["ids"])
                    inter = np.intersect1d(before["ids"], after["ids"])
                    drift.append({"jaccard": float(len(inter) / max(1, len(union))),
                                  "centroid_displacement": float(np.linalg.norm(before["centroid"] - after["centroid"]))
                                  if len(before["ids"]) and len(after["ids"]) else float("inf")})
            territory_previous = territory
            phase_events = []
            # Snapshot the atlas at EVERY phase-1 E-step (not only prune ticks)
            # so the whole cover-formation trajectory is replayable; the prune
            # block below adds a matching "post" only on E-steps that retire.
            if snapshot_prune_steps and phase == 1:
                _write_prune_snapshot(epoch, "pre")
            if (phase == 1 and coverage_complete and len(models) > 1 and
                    epoch >= int(prune_warmup_epochs)):
                _retirement_tick = sync_tick()
                joint_timing["retirement_checkpoints"] = joint_timing.get("retirement_checkpoints", 0) + 1
                # Retire a chart whose UNIQUE cover support is zero -- every pose
                # in its cover mask is also covered by another chart, so removing
                # it cannot open a hole.  The test keys on ``cover_competent``
                # ((recon+valid) <= threshold_topo) -- the SAME mask coverage,
                # multiplicity and greedy-cover use -- NOT the angle/fold-gated
                # ``hard_competent`` (a cover-redundant chart must not be kept
                # immortal because the 15 deg gate flickers a boundary pose onto
                # it at one E-step).  ``exclusive == 0`` must still hold for
                # ``retire_hysteresis_steps`` consecutive prune-eligible E-steps;
                # recompute after each retirement so mutually redundant charts do
                # not retire together.  No reseed.
                if inertial_frontier:
                    # Soft exclusive mass E_i = sum_p [s_i(p) - max_{j!=i}
                    # s_j(p)]_+, thresholded at 0.5 pose-equivalent -- a
                    # continuous analogue of "zero exclusive support" that
                    # never redefines coverage itself.
                    zero_exclusive_streak = np.where(
                        _soft_exclusive_mass(claims) < 0.5, zero_exclusive_streak + 1, 0)
                elif ema_ownership:
                    # Same hard exclusive-support test, but on the EMA-smoothed
                    # cover mask instead of the raw (noisy) per-epoch one; batch
                    # composition upstream is entirely unaffected.
                    zero_exclusive_streak = np.where(
                        _exclusive_support(_ema_cover > 0.5) == 0, zero_exclusive_streak + 1, 0)
                else:
                    zero_exclusive_streak = np.where(
                        _exclusive_support(cover_competent) == 0, zero_exclusive_streak + 1, 0)
                if monotone_frontier:
                    # An unborn chart trivially shows zero certified coverage
                    # (its territory is empty) -- that must never accrue a
                    # retirement streak, only a genuinely swallowed BORN
                    # chart can.
                    zero_exclusive_streak = np.where(born, zero_exclusive_streak, 0)
                _retired_this_estep = 0
                while len(models) > 1:
                    if (retire_max_per_step is not None
                            and _retired_this_estep >= int(retire_max_per_step)):
                        break
                    if inertial_frontier:
                        exclusive_now = _soft_exclusive_mass(claims)
                        prune_eligible = (zero_exclusive_streak >= int(retire_hysteresis_steps)) & (exclusive_now < 0.5)
                    elif ema_ownership:
                        exclusive_now = _exclusive_support(_ema_cover > 0.5)
                        prune_eligible = (zero_exclusive_streak >= int(retire_hysteresis_steps)) & (exclusive_now == 0)
                    else:
                        exclusive_now = _exclusive_support(cover_competent)
                        prune_eligible = (zero_exclusive_streak >= int(retire_hysteresis_steps)) & (exclusive_now == 0)
                    if monotone_frontier:
                        # A chart with a live maintenance violation, or one
                        # that has never been born, is blocked from pruning
                        # this E-step -- it keeps training, its territory is
                        # not redefined out from under it.
                        prune_eligible = prune_eligible & ~maintenance_violation & born
                    cand = np.flatnonzero(prune_eligible)
                    _pre_removable = cand
                    cand = np.asarray([i for i in cand if _removable_chart(i)], dtype=int)
                    for _i in np.setdiff1d(_pre_removable, cand):
                        retirement_rejection_log.append({
                            "epoch": int(epoch), "chart": int(chart_ids[_i]) if _i < len(chart_ids) else int(_i),
                            "stage": "prescreen", "reasons": ["not_removable_local_one_cover"]})
                    if not len(cand):
                        break
                    # Retirement is decided on the cover mask alone -- the same
                    # ``(recon + valid) <= threshold_topo`` mask that defines
                    # loss, coverage and ownership.  The full composite /
                    # tangent-fold re-evaluation is gone; the extra evidence is
                    # (1) a physical-coverage probe against the data and (2) a
                    # receiver tangent/fold check restricted to the territory
                    # being handed over.
                    off_node_ok, _off_node_rows = off_node_redundancy(cover_competent)
                    _pre_offnode = cand
                    cand = np.asarray([i for i in cand if off_node_ok[i]], dtype=int)
                    for _i in np.setdiff1d(_pre_offnode, cand):
                        _reasons = (_off_node_rows[_i]["rejection_reasons"] if _i < len(_off_node_rows)
                                    else ["off_node_not_redundant"])
                        retirement_rejection_log.append({
                            "epoch": int(epoch), "chart": int(chart_ids[_i]) if _i < len(chart_ids) else int(_i),
                            "stage": "off_sample_probe", "reasons": list(_reasons)})
                    _pre_receiver = cand
                    _receiver_ok = []
                    for _i in cand:
                        _ok, _stats = _receiver_certificate(_i)
                        if _ok:
                            _receiver_ok.append(_i)
                        else:
                            retirement_rejection_log.append({
                                "epoch": int(epoch), "chart": int(chart_ids[_i]) if _i < len(chart_ids) else int(_i),
                                "stage": "receiver_certificate",
                                "reasons": ["inherited_pose_without_certified_receiver"], **_stats})
                    cand = np.asarray(_receiver_ok, dtype=int)
                    if not len(cand):
                        break
                    _retired_this_estep += 1
                    retired = int(min(cand, key=lambda i: (-float(C[:, i].mean()), int(i))))
                    covered_before = cover_competent.any(1)
                    runt_territory = np.flatnonzero(cover_competent[:, retired])
                    event = {"epoch": int(epoch), "retired_chart": int(retired),
                             "reasons": ["zero_exclusive_support", "collar_and_offsample_preserved"],
                             "exclusive_certified": 0,
                             "chart_count_before": len(models),
                             "uncovered_before": int((~covered_before).sum()),
                             "epochs_to_recover": 0}
                    keep_live = np.ones(len(models), dtype=bool); keep_live[retired] = False
                    keep_mask = torch.as_tensor(keep_live, dtype=torch.bool, device=Q.device)
                    models = [m for m, k in zip(models, keep_live) if k]
                    ensemble_params, ensemble_opt = _slice_ensemble_optimizer(
                        ensemble_opt, ensemble_params, keep_live)
                    ensemble_buffers = {name: value[keep_mask.to(value.device)].clone()
                                        for name, value in ensemble_buffers.items()}
                    centre_t = centre_t[keep_mask.to(centre_t.device)]
                    stretch_reference = [v for v, k in zip(stretch_reference, keep_live) if k]
                    usage_ema = usage_ema[keep_live]
                    old_selector_params = (log_sigma, correction_w, correction_b)
                    log_sigma = torch.nn.Parameter(log_sigma.detach()[keep_mask.to(log_sigma.device)])
                    correction_w = torch.nn.Parameter(correction_w.detach()[keep_mask.to(correction_w.device)])
                    correction_b = torch.nn.Parameter(correction_b.detach()[keep_mask.to(correction_b.device)])
                    selector_opt = _slice_selector_optimizer(
                        selector_opt, old_selector_params,
                        (log_sigma, correction_w, correction_b), keep_live)
                    territory_previous = [v for v, k in zip(territory_previous, keep_live) if k]
                    zero_exclusive_streak = zero_exclusive_streak[keep_live]
                    _last_shell_absorbed_per_chart = _last_shell_absorbed_per_chart[keep_live]
                    certificate_gate = certificate_gate[:, keep_live]
                    chart_ids = [c for c, k in zip(chart_ids, keep_live) if k]
                    if monotone_frontier:
                        persistent_territory = [v for v, k in zip(persistent_territory, keep_live) if k]
                        born = born[keep_live]
                        provisional_source = [v for v, k in zip(provisional_source, keep_live) if k]
                        provisional_pool = [v for v, k in zip(provisional_pool, keep_live) if k]
                        maintenance_violation = maintenance_violation[keep_live]
                        maintenance_violation_total = maintenance_violation_total[keep_live]
                        monotone_admitted_last = [v for v, k in zip(monotone_admitted_last, keep_live) if k]
                        monotone_frontier_last = [v for v, k in zip(monotone_frontier_last, keep_live) if k]
                    if inertial_frontier:
                        claims = claims[:, keep_live]
                    if ema_ownership and _ema_cover is not None:
                        _ema_cover = _ema_cover[:, keep_live]
                    C, e_step_reconstruction, e_step_validity, gate = _eval_all()
                    ownership, hard_competent, cover_competent = _assign_from(
                        e_step_reconstruction, e_step_validity, gate)
                    # The pre-retirement need counted the retired chart as an
                    # "other" cover (need ~0 on its territory) and has its column.
                    _, core_protection_need = _protection_need(e_step_reconstruction)
                    membership = ownership
                    if monotone_frontier:
                        # No new admission here -- only the chart set shrank;
                        # rebuild the mask from the (already sliced) territory.
                        cover_competent = _territory_mask(len(models))
                    multiplicity = cover_competent.sum(1).astype(float)
                    responsibilities = _owner_responsibilities(ownership)
                    event.update(chart_count_after=len(models),
                                 uncovered_after=int((~cover_competent.any(1)).sum()),
                                 newly_uncovered=int((covered_before & ~cover_competent.any(1)).sum()),
                                 territory_lost=int((~cover_competent[runt_territory].any(1)).sum())
                                 if len(runt_territory) else 0)
                    retirement_events.append(event); phase_events.append(event)
                    # Retirement is observational and strictly one-at-a-time;
                    # resume the same symmetric objective before another check.
                    phase1_stable_streak = 0
                    break
                joint_timing["retirement_checkpoint_seconds"] = (
                    joint_timing.get("retirement_checkpoint_seconds", 0.0) + elapsed_since(_retirement_tick))
                if snapshot_prune_steps:
                    _write_prune_snapshot(epoch, "post")
            chart_count_trajectory.append({"epoch": int(epoch), "phase": int(phase), "n_charts": int(len(models))})
            phase_trace.append({"epoch": int(epoch), "phase": int(phase), "n_charts": int(len(models)),
                                "multiplicity_histogram": np.histogram(multiplicity, bins=[0.0, 0.5, 1.0, float(d + 1), float("inf")])[0].tolist(),
                                "hard_multiplicity_counts": {"0": int((multiplicity == 0).sum()),
                                                            "1": int((multiplicity == 1).sum()),
                                                            "2": int((multiplicity == 2).sum()),
                                                            "3+": int((multiplicity >= 3).sum())},
                                "n_singleton": int((multiplicity == 1).sum()),
                                "n_removable": int((_exclusive_support(cover_competent) == 0).sum()),
                                "phase1_stable": bool(phase1_stable_streak >= int(retire_hysteresis_steps)),
                                "integer_multiplicity_histogram": _integer_multiplicity_histogram(multiplicity, d),
                                "exclusive_certified": _exclusive_support(cover_competent).tolist(),
                                "boundary_collar_size": int(boundary_collar.sum()),
                                "raw_coverage_complete": coverage_complete,
                                "exclusive_certified_gated": _exclusive_support(hard_competent).tolist(),
                                "territory_drift": drift, "retirements": phase_events})
            refresh_gauge(responsibilities)
            n_unc = int((~cover_competent.any(1)).sum())
            for rec in _frontier_epoch_log:
                frontier_trace.append({"epoch": int(epoch), "phase": int(phase), "n_uncovered": n_unc, **rec})
            # Cache the consolidated per-pose loss and force a frontier rebuild:
            # the M-step core-protection hinge penalises the live loss rising
            # above this snapshot; the frontier is fixed until the next E-step.
            core_loss_cache = np.asarray(e_step_reconstruction, float).copy()
            # Recompute post-retirement so the absorption gate reflects the
            # current chart set, not the pre-prune coverage snapshot.
            if phase == 2 and monotone_frontier:
                # Retained only as a normalized diagnostic; frozen Phase-2
                # batches never consume it for frontier expansion.
                shell_priority = _monotone_shell_priority()
            elif phase == 2 and inertial_frontier:
                shell_priority = _inertial_frontier_weight()
            elif phase == 2:
                shell_priority = _shell_policy(ownership, phase)
            elif monotone_frontier:
                # Pre-coverage D/H/X/A shell prioritization is disabled: the
                # frontier weight in the M-step is the smooth physical-kernel
                # density f_i (see the loss-assembly fold guard below, which
                # is skipped under this flag too).
                shell_priority = _monotone_shell_priority()
            elif inertial_frontier:
                shell_priority = _inertial_frontier_weight()
            else:
                shell_priority = _shell_policy(ownership, phase)
            if inertial_frontier:
                soft_exclusive_final = _soft_exclusive_mass(claims)
                for i in range(len(models)):
                    inertial_trace.append({
                        "epoch": int(epoch), "phase": int(phase),
                        "chart": int(chart_ids[i]),
                        "claim_mass": float(claims[:, i].sum()),
                        "claim_max": float(claims[:, i].max()),
                        "n_core_pool": int(np.count_nonzero(claims[:, i] > 1e-3)),
                        "frontier_mass": float(shell_priority[:, i].sum()),
                        "soft_exclusive_mass": float(soft_exclusive_final[i]),
                        "n_uncovered": int((~cover_competent.any(1)).sum()),
                        "retirements_this_step": int(len(phase_events))})
            if monotone_frontier:
                exclusive_now_final = _exclusive_support(cover_competent)
                for i in range(len(models)):
                    monotone_frontier_trace.append({
                        "epoch": int(epoch), "phase": int(phase),
                        "chart": int(chart_ids[i]),
                        "born": bool(born[i]),
                        "territory_size": int(len(persistent_territory[i])),
                        "live_competent_in_territory": int(np.count_nonzero(
                            np.asarray(e_step_reconstruction, float)[:, i][
                                np.fromiter(persistent_territory[i], dtype=int,
                                            count=len(persistent_territory[i]))]
                            <= float(threshold_topo)) if persistent_territory[i] else 0),
                        "maintenance_violation": bool(maintenance_violation[i]),
                        "maintenance_violations_total": int(maintenance_violation_total[i]),
                        "frontier_size": int(monotone_frontier_last[i]),
                        "frontier_mass": float(shell_priority[:, i].sum()),
                        "repair_multiplier": float(monotone_g[i]),
                        "admitted": int(monotone_admitted_last[i]),
                        "n_uncovered": int((~cover_competent.any(1)).sum()),
                        "exclusive_support": int(exclusive_now_final[i]),
                        "retirements_this_step": int(len(phase_events))})
            _rebatch_needed = True
            joint_timing["e_step_seconds"] += elapsed_since(tick)
            return coverage_complete

        def _frozen_batch(chart):
            own = np.asarray(frozen_owned[chart], dtype=int)
            if len(own) > int(m_step_batch_size):
                own = rng.choice(own, size=int(m_step_batch_size), replace=False)
            own = np.unique(own)
            if len(own) < 2:
                own = np.unique(np.r_[own, np.argsort(-responsibilities[:, chart])[:2]])
            k_local = min(5, len(own))
            if k_local == 1:
                pr = np.empty((0, 2), dtype=int)
            else:
                own_t = torch.as_tensor(own, dtype=torch.long, device=feature_t.device)
                local_feat = feature_t[own_t].to(torch.float64)
                ln = torch.cdist(local_feat, local_feat).topk(k_local, largest=False).indices.cpu().numpy()
                pr = np.asarray([(a, b) for a in range(len(own)) for b in np.atleast_1d(ln[a])[1:]], dtype=int)
            return own, pr

        def _coverage_margin_loss(target_threshold, phase):
            """Shared one-cover and two-cover competence margins (two-cover global in phase 1, collar in phase 2)."""
            count = min(n, max(32, int(m_step_batch_size)))
            ids = rng.choice(n, size=count, replace=False)
            if not len(ids):
                return torch.zeros((), dtype=Q.dtype, device=Q.device)
            k = len(models)
            poses = Q[torch.as_tensor(ids, dtype=torch.long, device=Q.device)]
            tiled = poses.unsqueeze(0).expand(k, -1, -1)
            codes = ensemble_encode(tiled)
            predicted = ensemble_decode(codes)
            reconstruction = _recon(predicted, tiled)
            # Reconstruction-only competence, the same quantity _assign_from
            # covers on.  Including validity held competence ~0 everywhere
            # (one_cover pinned at 0.7^2) so the margin was dead except for
            # rare 1/tau-steep pulses carrying validity's oversized gradient.
            competence = torch.sigmoid((float(target_threshold) - reconstruction) /
                                       max(float(target_threshold) / 5.0, 1e-12))
            # A chart can answer a pose only from its physically attached
            # domain.  Always retain the most attached chart to avoid a cold
            # start with an empty mask.
            locality = (_attach_kernel @ ownership) / _attach_rowsum[:, None]
            local = locality[ids]
            eligible = local >= 0.02
            eligible[np.arange(len(ids)), local.argmax(1)] = True
            eligible_t = torch.as_tensor(eligible.T, dtype=torch.bool, device=Q.device)
            masked = competence.masked_fill(~eligible_t, -1.0)
            top = torch.topk(masked, k=min(3, len(models)), dim=0).values
            best = top[0]
            one_cover = torch.relu(0.70 - best).square().mean()
            if len(models) < 2:
                return one_cover
            collar_t = torch.as_tensor(_double_cover_mask(phase)[ids], dtype=Q.dtype, device=Q.device)
            second = top[1]
            second_cover = ((collar_t * torch.relu(0.60 - second).square()).sum() /
                            collar_t.sum().clamp_min(1.0))
            # Keep the existing triple-overlap guard: a third competent chart
            # is useful nowhere in this objective and should yield the region
            # to the locally best two.
            triple_overlap = (torch.relu(top[2] - 0.35).square().mean()
                              if len(models) >= 3 else torch.zeros_like(one_cover))
            if epoch_trace:
                _trace_margin.update(one_cover=float(one_cover), second_cover=float(second_cover),
                                     triple_overlap=float(triple_overlap))
            return one_cover + 0.75 * second_cover + 0.20 * triple_overlap

        def _m_step(epoch, phase, *, freeze):
            nonlocal _cached_batches, _rebatch_needed, _last_shell_absorbed
            nonlocal _last_shell_absorbed_per_chart
            nonlocal _cached_probe_direction, _cached_probe_radius_fraction
            losses = []
            tick = sync_tick()
            tangent_on = (float(tangent_weight) != 0.0 and epoch % max(1, int(tangent_every)) == 0)
            batch_tick = sync_tick()
            if freeze:
                _frontier_epoch_log.clear()
                selected_batches = [_frozen_batch(chart) for chart in range(len(models))]
            else:
                # Frontier fixed within the E-step interval: rebuild only when an
                # E-step (or retirement) has flagged it, then reuse the same
                # batches every M-step until the next E-step.
                if (_rebatch_needed or _cached_batches is None
                        or len(_cached_batches) != len(models)):
                    _frontier_epoch_log.clear()
                    _frontier_shell_global.clear(); _frontier_core_global.clear()
                    _frontier_full_shell.clear(); _frontier_shell_fold.clear()
                    _frontier_shell_scale.clear()
                    _cached_batches = [m_step_batch(chart) for chart in range(len(models))]
                    _rebatch_needed = False
                    _cached_probe_direction = None
                    _cached_probe_radius_fraction = None
                    _cached_far_pairs.clear()
                    n_unc = int((~cover_competent.any(1)).sum())
                    for rec in _frontier_epoch_log:
                        frontier_trace.append({"epoch": int(epoch), "phase": int(phase),
                                               "n_uncovered": n_unc, **rec})
                selected_batches = _cached_batches
            batch_selection_seconds = elapsed_since(batch_tick)
            batch_ids = [item[0] for item in selected_batches]
            dense_tick = sync_tick()
            dense_u, dense_pred, dense_mask = dense_chart_forward(batch_ids)
            joint_timing.setdefault("dense_forward_seconds", 0.0)
            joint_timing["dense_forward_seconds"] += elapsed_since(dense_tick)
            k_live, max_batch = dense_u.shape[:2]
            dense_cycle_tick = sync_tick()
            on_cycle_codes = ensemble_encode(dense_pred.reshape(k_live, max_batch, -1))
            # Bound the O(max_batch^2) cost of this latent-neighbour search: it
            # only calibrates the off-anchor probe radius (an approximate local
            # scale, not an exact kNN topology), but a full self-cdist over the
            # WHOLE padded batch grows quadratically as core/shell batches climb
            # toward their per-chart cap -- with 12 charts and batches nearing
            # 512, that pairwise matrix dominates the epoch.  Above the cap,
            # probe against a random reference subsample of the same batch
            # instead of every pose in it (cost -> O(max_batch * cap)).
            _NEIGHBOR_REF_CAP = 128
            if max_batch <= _NEIGHBOR_REF_CAP:
                distance = torch.cdist(dense_u.detach(), dense_u.detach())
                distance = distance.masked_fill(~dense_mask[:, None, :], float("inf"))
                near_count = min(4, max_batch)
                near = distance.topk(near_count, largest=False).indices
                near_rest = near[:, :, 1:]  # drop the trivial self-match (distance 0)
            else:
                ref_idx = torch.randint(0, max_batch, (_NEIGHBOR_REF_CAP,), device=Q.device)
                ref_u = dense_u.detach()[:, ref_idx, :]
                ref_mask = dense_mask[:, ref_idx]
                distance = torch.cdist(dense_u.detach(), ref_u)
                distance = distance.masked_fill(~ref_mask[:, None, :], float("inf"))
                near_count = min(4, _NEIGHBOR_REF_CAP)
                near_local = distance.topk(near_count, largest=False).indices
                # A random reference subsample essentially never contains a
                # query's own index, so there is no trivial self-match to drop.
                near_rest = ref_idx[near_local]
            # Per-chart fallback radius, matching the scalar path's
            # ``u.std(0).mean()``: the std is taken over that chart's REAL rows
            # only (padding excluded), so it never dilutes toward zero.
            real_per_chart = dense_mask.sum(1, keepdim=True).clamp_min(1)
            chart_mean = (dense_u.detach() * dense_mask[..., None]).sum(1, keepdim=True) / real_per_chart[..., None]
            chart_var = (((dense_u.detach() - chart_mean) * dense_mask[..., None]).square().sum(1, keepdim=True) /
                         real_per_chart[..., None])
            fallback_radius = chart_var.sqrt().mean(-1, keepdim=True).expand(-1, max_batch, 1)
            if near_rest.shape[-1] > 0:
                near_codes = dense_u[:, None, :, :].expand(-1, max_batch, -1, -1).gather(
                    2, near_rest[:, :, :, None].expand(-1, -1, -1, d))
                near_valid = dense_mask[:, None, :].expand(-1, max_batch, -1).gather(2, near_rest)
                near_distance = torch.linalg.vector_norm(near_codes - dense_u[:, :, None, :], dim=-1)
                neighbour_count = near_valid.sum(2, keepdim=True)
                averaged = (near_distance * near_valid).sum(2, keepdim=True) / neighbour_count.clamp_min(1)
                # A singleton chart (or any row whose only neighbours are
                # padding) has neighbour_count 0 -> that would be a zero radius,
                # not the scalar fallback.  Substitute the chart std there.
                local_radius = torch.where(neighbour_count > 0, averaged, fallback_radius)
            else:
                local_radius = fallback_radius
            probe_count = max(1, d)
            reuse_probe = (bool(freeze_estep_probes) and _cached_probe_direction is not None
                           and _cached_probe_direction.shape[:2] == (k_live, max_batch))
            if reuse_probe:
                direction = _cached_probe_direction
                radius_fraction = _cached_probe_radius_fraction
            else:
                direction = torch.randn((k_live, max_batch, probe_count, d), dtype=Q.dtype, device=Q.device)
                direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
                radius_fraction = torch.rand((k_live, max_batch, probe_count, 1), dtype=Q.dtype, device=Q.device)
                if freeze_estep_probes:
                    _cached_probe_direction = direction
                    _cached_probe_radius_fraction = radius_fraction
            probe_u = dense_u[:, :, None, :] + (radius_fraction * local_radius.detach()[:, :, None, :]
                                           .clamp_min(1e-8) * direction)
            probe_flat = probe_u.reshape(k_live, max_batch * probe_count, d)
            probe_cycle = (ensemble_encode(ensemble_decode(probe_flat)) - probe_flat).square().mean(-1)
            probe_cycle = probe_cycle.reshape(k_live, max_batch, probe_count).amax(-1)
            joint_timing.setdefault("batched_cycle_seconds", 0.0)
            joint_timing["batched_cycle_seconds"] += elapsed_since(dense_cycle_tick)
            if tangent_on:
                tangent_tick = sync_tick()
                dense_tangent, tangent_owned_counts = batched_tangent(selected_batches, dense_u)
                joint_timing.setdefault("batched_tangent_seconds", 0.0)
                joint_timing["batched_tangent_seconds"] += elapsed_since(tangent_tick)
            else:
                dense_tangent, tangent_owned_counts = None, [0 for _ in models]
            for chart, m in enumerate(models):
                ids, local_pairs = selected_batches[chart]
                count = len(ids)
                if len(local_pairs):
                    # Live, detached competence weight (the E-step ownership
                    # sigmoid on THIS M-step's reconstruction): a pair's alias /
                    # compression counts only as far as the chart already
                    # reconstructs both poses, so a shell pose enters ~0 and
                    # ramps in as it is learned -- no step change at the E-step.
                    with torch.no_grad():
                        ids_t = torch.as_tensor(ids, dtype=torch.long, device=Q.device)
                        rv_live = _recon(dense_pred[chart, :count].reshape(count, -1), Q[ids_t])
                        live_w = torch.sigmoid((float(threshold_topo) - rv_live) /
                                               max(float(threshold_topo) / 5.0, 1e-12))
                        lp_t = torch.as_tensor(local_pairs, dtype=torch.long, device=Q.device)
                        pair_w = (live_w[lp_t[:, 0]] * live_w[lp_t[:, 1]]).to(Q.dtype)
                else:
                    local_pairs = np.asarray([[0, 0]], dtype=int)
                    pair_w = torch.ones(1, dtype=Q.dtype, device=Q.device)
                per, dist, extra = chart_loss(
                    m, chart, local_pairs, pair_w, pose_ids=ids, compute_tangent=False,
                    precomputed=(dense_u[chart, :count], dense_pred[chart, :count]),
                    precomputed_cycles=(on_cycle_codes[chart, :count], probe_cycle[chart, :count]),
                    compute_distortion=bool(distortion_weight),
                    # Phase-1 M-step: validity is affine (blind to pose blends)
                    # and at weight 1 its gradient outweighs reconstruction
                    # 20-150x, so it is dropped from the training loss here.
                    # Phase 2 keeps it, unchanged, as the closure training term.
                    validity_weight=(1.0 if phase == 2 else 0.0))
                losses.append((per, dist, extra))
            joint_timing["chart_loss_seconds"] += elapsed_since(tick)
            # Per-chart two-part objective (spec).  Core and shell are normalised
            # SEPARATELY so a larger shell cannot dilute core training:
            #   L_i = mean_{core_i} l_i
            #       + rho * ( sum_{shell_i} Q_i(p) l_i(p) / sum_{shell_i} Q_i(p) )   [full shell]
            #       + lambda_protect * mean_{core_i} relu(rv_i(p) - (cached_i(p) + margin))
            # Q_i is the detached E-step shell-priority policy; it weights the
            # mean only for a FULL shell (a Q-sampled shell already carries the
            # priority, so its mean is uniform).  Q_i, the cache and the margin
            # are detached -- gradient flows only through the reconstruction /
            # validity losses and the protection hinge.
            objective_tick = sync_tick()
            protect_margin = float(protect_margin_frac) * float(threshold_topo)
            chart_L = []
            absorbed = []
            _trace_chart = []
            _trace_margin.clear()
            for chart, ((per, _dist, extra), ids) in enumerate(zip(losses, batch_ids)):
                # ``rv`` = reconstruction ONLY -- the exact quantity the E-step
                # ownership weight, ``core_loss_cache`` and ``threshold_topo`` are
                # all defined on (validity is dropped from routine ownership;
                # see _assign_from).  The protection hinge and the absorption
                # signal must use it (not ``per``, which also carries cycle,
                # validity and alias).
                rv = extra[0]
                ids = np.asarray(ids, dtype=int)
                slot = {int(v): k for k, v in enumerate(ids)}
                core_g = _frontier_core_global.get(chart)
                if core_g is None:                       # frozen / phase 2: whole batch is core
                    core_loc = np.arange(len(ids), dtype=int)
                    shell_loc = np.empty(0, dtype=int)
                else:
                    core_loc = np.array(sorted(slot[g] for g in core_g if g in slot), dtype=int)
                    shell_loc = np.array(sorted(slot[g] for g in (_frontier_shell_global.get(chart) or ())
                                                if g in slot), dtype=int)
                    if not len(core_loc):
                        core_loc = np.array([int(np.asarray(ownership[ids, chart]).argmax())])
                core_t = torch.as_tensor(core_loc, dtype=torch.long, device=per.device)
                if inertial_frontier and core_g is not None:
                    # Core loss mass is normalized by the retained claim s_i
                    # itself -- a pose barely inside the claim contributes
                    # little, one deep inside contributes close to its full
                    # share, instead of every core pose counting equally.
                    core_w = torch.as_tensor(np.clip(claims[ids[core_loc], chart], 1e-9, None),
                                             dtype=per.dtype, device=per.device)
                    L = (core_w * per[core_t]).sum() / core_w.sum()
                else:
                    L = per[core_t].mean()
                _L_core = float(L.detach()) if epoch_trace else 0.0
                # Maintenance smoothly suppresses expansion (monotone_frontier
                # only): g_i in (0, 1] is the repair multiplier from the
                # E-step's smoothmax-over-territory sigmoid.  g_i scales the
                # frontier term down and (1 - g_i) adds extra weight back onto
                # the territory itself, so training mass shifts from frontier
                # to repair smoothly as the territory degrades -- no abrupt
                # mode switch at the hard tau admission-blocking gate.
                repair_scale = 0.0
                if monotone_frontier and core_g is not None:
                    frontier_scale = float(monotone_g[chart]) if chart < len(monotone_g) else 1.0
                    repair_scale = 1.0 - frontier_scale
                    if repair_scale > 0.0:
                        L = L + float(rho) * repair_scale * per[core_t].mean()
                else:
                    frontier_scale = 1.0
                if len(shell_loc):
                    sg = ids[shell_loc]
                    # Q_i weights the mean only for a FULL shell; a Q-sampled
                    # shell already carries the priority in its membership, so
                    # its mean is uniform ("use Q_i to sample OR to weight the
                    # full-shell mean -- never both").
                    if _frontier_full_shell.get(chart, False):
                        g_np = np.clip(shell_priority[sg, chart], 1e-9, None)
                    else:
                        g_np = np.ones(len(sg), dtype=float)
                    g_t = torch.as_tensor(g_np, dtype=per.dtype, device=per.device).clamp_min(1e-9)
                    shell_t = torch.as_tensor(shell_loc, dtype=torch.long, device=per.device)
                    if not (monotone_frontier or inertial_frontier):
                        # Demand-scaled shell mean: weight = Q_i itself averaged
                        # over the shell (a Q-sampled shell carries its pool's
                        # mean Q).  Self-normalising Q divided out the demand, so
                        # a shell with NO demand still drew the full rho weight
                        # (constant outward pressure), and renormalising after the
                        # cap let one demanded pose take ~18x the uniform share.
                        # Q <= 1 + eps, so no pose exceeds the uniform share.
                        if not _frontier_full_shell.get(chart, False):
                            g_t = torch.full_like(g_t, _frontier_shell_scale.get(chart, 1.0))
                        if float(fold_weight) != 0.0:
                            fold_np = np.array([_frontier_shell_fold.get(chart, {}).get(int(p), 0.0)
                                                for p in sg], dtype=float)
                            g_t = g_t / (1.0 + float(fold_weight) *
                                         torch.as_tensor(fold_np, dtype=per.dtype, device=per.device))
                        L = L + float(rho) * frontier_scale * (g_t * per[shell_t]).sum() / len(g_t)
                    else:
                        # monotone / inertial frontiers keep their own self-
                        # normalised shell mean (inertial skips the cap).
                        g_t = g_t / g_t.sum()
                        if not inertial_frontier:
                            g_t = torch.clamp(g_t, max=float(shell_weight_cap) / len(g_t))
                        L = L + float(rho) * frontier_scale * (g_t * per[shell_t]).sum() / g_t.sum()
                _L_shell = float(L.detach()) if epoch_trace else 0.0
                if (not freeze) and float(lambda_protect) != 0.0 and core_g is not None:
                    cached = torch.as_tensor(core_loss_cache[ids[core_loc], chart],
                                             dtype=rv.dtype, device=rv.device)
                    need = torch.as_tensor(core_protection_need[ids[core_loc], chart],
                                           dtype=rv.dtype, device=rv.device)
                    L = L + float(lambda_protect) * torch.relu(
                        rv[core_t] - (cached + protect_margin)).mul(need).mean()
                if phase == 2:
                    # ``threshold_fit`` is the actual final accuracy target,
                    # not merely a reported diagnostic.
                    L = L + 2.0 * torch.relu(rv[core_t] - float(threshold_fit)).mean()
                _L_protect = float(L.detach()) if epoch_trace else 0.0
                chart_L.append(L)
                if epoch_trace:
                    # Trace-only decomposition of L: core term (recon + .1 cycle
                    # + alias_weight * alias), shell term, protection hinge.
                    cycle_d, alias_d = extra[1].detach(), extra[3].detach()
                    row = {"chart_loss": float(L.detach()),
                           "core_recon_mean": float(rv[core_t].detach().mean()),
                           "core_recon_max": float(rv[core_t].detach().max()),
                           "core_cycle_mean": float(cycle_d[core_t].mean()),
                           "core_alias_mean": float(alias_d[core_t].mean()),
                           "core_alias_max": float(alias_d[core_t].max()),
                           "core_term": _L_core,
                           "shell_term": _L_shell - _L_core,
                           "protect_term": _L_protect - _L_shell,
                           "core_size": int(len(core_loc)),
                           "shell_size": int(len(shell_loc))}
                    if len(shell_loc):
                        _st = torch.as_tensor(shell_loc, dtype=torch.long, device=per.device)
                        row.update(shell_recon_mean=float(rv[_st].detach().mean()),
                                   shell_alias_mean=float(alias_d[_st].mean()))
                    _trace_chart.append(row)
                if not freeze:
                    probe_t = torch.as_tensor(shell_loc if len(shell_loc) else core_loc,
                                              dtype=torch.long, device=rv.device)
                    with torch.no_grad():
                        absorbed.append(float((rv[probe_t] < float(threshold_topo)).double().mean()))
            objective = torch.stack(chart_L).mean()
            _obj_terms = {"chart": objective} if epoch_trace else None   # trace-only references
            cover_target = float(threshold_topo) if phase == 1 else float(threshold_fit)
            _cov_term = 0.5 * _coverage_margin_loss(cover_target, phase)
            objective = objective + _cov_term
            if epoch_trace:
                _obj_terms["coverage_margin"] = _cov_term
            if not freeze:
                _last_shell_absorbed = float(np.mean(absorbed)) if absorbed else 1.0
                _last_shell_absorbed_per_chart = np.asarray(absorbed, dtype=float)
            if float(distortion_weight) != 0.0:
                _dist_term = distortion_weight * torch.stack([x[1] for x in losses]).mean()
                objective = objective + _dist_term
                if epoch_trace:
                    _obj_terms["distortion"] = _dist_term
            if tangent_on:
                _tan_term = float(tangent_weight) * dense_tangent.mean()
                objective = objective + _tan_term
                if epoch_trace:
                    _obj_terms["tangent"] = _tan_term
            joint_timing["objective_assembly_seconds"] += elapsed_since(objective_tick)
            spectral_tick = sync_tick()
            if float(lip_weight) != 0.0:
                decoder_lip = torch.stack([ensemble_decoder_spectral(i) for i in range(len(models))])
                _lip_term = float(lip_weight) * (decoder_lip.max() + decoder_lip.mean())
                objective = objective + _lip_term
                if epoch_trace:
                    _obj_terms["lip"] = _lip_term
            joint_timing["spectral_penalty_seconds"] += elapsed_since(spectral_tick)
            selector_tick = sync_tick()
            if (not freeze) and float(selector_weight) != 0.0:
                selector_target = torch.as_tensor(responsibilities, dtype=Q.dtype, device=Q.device)
                selector_loss = -(selector_target * torch.log_softmax(selector_scores(), dim=1)).sum(1).mean()
                objective = objective + float(selector_weight) * selector_loss
            joint_timing["selector_seconds"] += elapsed_since(selector_tick)
            if (not freeze) and compactness_weight:
                objective = objective + compactness_weight * torch.stack([x[2][4].square().mean() for x in losses]).mean()
            step_tick = sync_tick()
            ensemble_opt.zero_grad(); selector_opt.zero_grad()
            _term_grad_norms = {}
            if epoch_trace:
                # Trace-only: per-chart gradient norm of each objective term on
                # the ensemble parameters (extra backward passes, no update).
                _plist = list(ensemble_params.values())
                for _tname, _tval in _obj_terms.items():
                    if not torch.is_tensor(_tval) or not _tval.requires_grad:
                        continue
                    _gs = torch.autograd.grad(_tval, _plist, retain_graph=True, allow_unused=True)
                    _sq = torch.zeros(len(models), dtype=torch.float64, device=Q.device)
                    for _g in _gs:
                        if _g is not None:
                            _sq += _g.detach().double().reshape(len(models), -1).square().sum(1)
                    _term_grad_norms[_tname] = _sq.sqrt().cpu().tolist()
            objective.backward()
            if epoch_trace:
                # Adam amplification: total grad norm vs the RMS its second
                # moment expects (state from the previous step, pre-update).
                with torch.no_grad():
                    _gsq = torch.zeros(len(models), dtype=torch.float64, device=Q.device)
                    _vsq = torch.zeros(len(models), dtype=torch.float64, device=Q.device)
                    for _p in ensemble_params.values():
                        if _p.grad is not None:
                            _gsq += _p.grad.detach().double().reshape(len(models), -1).square().sum(1)
                        _st = ensemble_opt.state.get(_p, {})
                        if "exp_avg_sq" in _st:
                            _vsq += _st["exp_avg_sq"].detach().double().reshape(len(models), -1).sum(1)
                    _term_grad_norms["total"] = _gsq.sqrt().cpu().tolist()
                    _term_grad_norms["adam_rms"] = _vsq.sqrt().cpu().tolist()
            lr_now = base_lr
            if phase == 2 and lr_step_epochs:
                lr_now = max(float(lr_step_min),
                             base_lr * (float(lr_step_gamma) ** (epoch // int(lr_step_epochs))))
                for group in ensemble_opt.param_groups:
                    group["lr"] = lr_now
            if epoch_trace:
                _before = {name: value.detach().clone() for name, value in ensemble_params.items()}
            ensemble_opt.step()
            if epoch_trace:
                with torch.no_grad():
                    _step_sq = torch.zeros(len(models), dtype=torch.float64, device=Q.device)
                    _param_sq = torch.zeros(len(models), dtype=torch.float64, device=Q.device)
                    for name, value in ensemble_params.items():
                        delta = (value.detach() - _before[name]).double().reshape(len(models), -1)
                        _step_sq += delta.square().sum(1)
                        _param_sq += _before[name].double().reshape(len(models), -1).square().sum(1)
                    _rel_step = (_step_sq.sqrt() / _param_sq.sqrt().clamp_min(1e-12)).cpu().tolist()
                for row, rel in zip(_trace_chart, _rel_step):
                    row["rel_param_step"] = float(rel)
                for _ci, row in enumerate(_trace_chart):
                    row["grad_norm"] = {k: float(v[_ci]) for k, v in _term_grad_norms.items()}
                    row["distortion"] = float(losses[_ci][1].detach())
                    if tangent_on and dense_tangent is not None:
                        row["tangent"] = float(dense_tangent[_ci].detach())
                epoch_trace_rows.append({
                    "epoch": int(epoch), "phase": int(phase), "chart_ids": [int(c) for c in chart_ids],
                    "objective": float(objective.detach()), "margin": dict(_trace_margin),
                    "charts": _trace_chart,
                    "gauge_change": {k: list(v) for k, v in _last_gauge_change.items()},
                    "n_uncovered_last_estep": int((~cover_competent.any(1)).sum())
                    if cover_competent is not None and cover_competent.shape[1] else None})
                _last_gauge_change.clear()
            if not freeze:
                selector_opt.step()
            joint_timing["backward_step_seconds"] += elapsed_since(step_tick)
            epoch_measurements.append({
                "epoch": int(epoch), "phase": int(phase), "n_charts": int(len(models)),
                "batch_sizes": [int(len(ids)) for ids in batch_ids],
                "pair_counts": [int(len(item[1])) for item in selected_batches],
                "tangent_samples": (list(tangent_owned_counts) if tangent_on else [0 for _ in batch_ids]),
                "lr": float(lr_now),
                "batch_selection_seconds": float(batch_selection_seconds)})
            if C is not None and epoch % 10 == 0:
                trace.append({"epoch": int(epoch), "phase": int(phase), "n_charts": len(models),
                              "worst_competence": float(C.min(1).max())})

        # ---- PHASE 1: cover ----------------------------------------------
        # Runs the full budget: the early-exit "stability" check fired before
        # pruning could start (its no-retirement-this-step clause is vacuously
        # true for the first ``prune_warmup_epochs``), so easy manifolds left
        # phase 1 with every seeded chart intact.
        phase1_start = time.perf_counter()
        p1_run = 0
        phase1_exit = "budget"
        # Adaptive E-step timing (inverted trigger): re-derive assignments early
        # for a *struggling* chart -- one whose shell absorption has fallen below
        # shell_absorb_fraction has lost its territory, and the M-step keeps it
        # training only on its own stale subset (no replay), so the drift
        # compounds: low absorption -> long gap -> more forgetting -> lower
        # absorption.  Below the fraction we now fire at e_min instead of letting
        # the chart fall through to e_max.  A well-absorbed chart still holds its
        # ownership, so it rides the e_max safety net rather than churning the
        # assignment every e_min epochs.  Bounded by [e_min, e_max].
        e_min = max(1, int(e_step_min_interval))
        e_max = max(e_min, int(e_step_max_interval))
        since_e = e_max  # force an E-step on epoch 0
        e_step_epochs = []
        _p1_bar = _Progress(p1_max, "competitive_cae phase 1 (cover)")
        for epoch in range(p1_max):
            p1_run = epoch + 1
            ran_estep = False
            if since_e >= e_max or (since_e >= e_min and np.any(
                    _last_shell_absorbed_per_chart < float(shell_absorb_fraction))):
                _e_step(epoch, 1)
                ran_estep = True
                e_step_epochs.append({"epoch": int(epoch), "gap": int(since_e),
                                      "shell_absorbed": float(_last_shell_absorbed),
                                      "shell_absorbed_per_chart": _last_shell_absorbed_per_chart.tolist(),
                                      "n_charts": int(len(models))})
                since_e = 0
            _m_step(epoch, 1, freeze=False)
            if _memprofile and epoch % 5 == 0:
                gc.collect()
                print(f"[memprofile] epoch={epoch} rss={_rss_mb():.1f}MB n_charts={len(models)}",
                      file=sys.stderr, flush=True)
            if (exploration_snapshot_every is not None
                    and (epoch == 0 or epoch % exploration_snapshot_every == 0)):
                _write_exploration_snapshot(epoch, phase=1)
            since_e += 1
            _p1_bar.update()
            # Phase-1 exits on budget only now -- the early "converged" break
            # is disabled.  It tested ``phase1_stable_streak`` against the
            # tangent/fold certificate gate only via its cached (usually
            # all-True placeholder) snapshot -- it never actually re-ran the
            # real certificate during the loop -- so it could (and, on the
            # standardized recon metric, did) declare victory well before the
            # real geometry caught up.  ``phase1_exit`` stays "budget"; the
            # boundary certificate below still runs and is still logged, now
            # purely as a report (see "boundary_certificate_full_cover").
        _p1_bar.close()
        _e_step(p1_run, 1, certify=True, certificate_reason="phase1_boundary")
        boundary_certificate_full_cover = bool(cover_competent.any(1).all() and
                                               hard_competent.any(1).all())
        if not boundary_certificate_full_cover:
            phase1_exit = "budget_boundary_certificate_incomplete"
        if exploration_snapshot_every is not None:
            _write_exploration_snapshot(p1_run, phase=1, final=True)
        timing["e_step_schedule"] = e_step_epochs
        timing["phase1_seconds"] = time.perf_counter() - phase1_start

        # ---- phase-1 checkpoint (resumable into phase 2) ----------------
        sync_model_shells()
        # Phase 2 trains each chart on its full cover territory -- the UNGATED
        # ``cover_competent`` ((recon+valid) <= threshold_topo), NOT the
        # angle/fold-gated ``hard_competent``.  Gating this set leaves the
        # gate-excluded poses (~20% on genus surfaces) out of every phase-2
        # batch, so 60 fresh-LR epochs drift the decoder off them -- exactly the
        # slop visible in a D(E(q)) overlay and fatal for a mechanism's joints.
        frozen_owned = [np.flatnonzero(cover_competent[:, i]) if cover_competent[:, i].any()
                        else np.flatnonzero(responsibilities.argmax(1) == i)
                        for i in range(len(models))]
        phase1_checkpoint = os.path.join(work, "phase1_final.pt")
        if resume is None:
            torch.save({"format": "competitive_cae_phase1_v1", "case": case,
                        "chart_ids": list(chart_ids),
                        "n_charts": int(len(models)), "encoder_kind": encoder_kind,
                        "n_bodies": int(n_bodies), "d": int(d),
                        "states": [m.state_dict() for m in models],
                        "centres": centre_t.detach().cpu().numpy(),
                        "hard_competent": hard_competent.copy(),
                        "cover_competent": cover_competent.copy(),
                        "frozen_owned": [x.tolist() for x in frozen_owned],
                        "stretch_reference": [float(x) for x in stretch_reference],
                        "resolved_config": resolved_config,
                        "phase1_epochs_run": int(p1_run), "phase1_exit": phase1_exit,
                        "threshold_topo": float(threshold_topo)}, phase1_checkpoint)

        # ---- PHASE 2: strict competitive specialization ------------------
        # Freeze chart count, not ownership or coverage pressure.  The same
        # learned boundary collar remains active while fitting tightens.
        _cached_batches = None; _rebatch_needed = True
        phase2_start = time.perf_counter()
        _p2_bar = _Progress(p2_max, "competitive_cae phase 2 (fit) ")
        for epoch in range(p2_max):
            if epoch % max(1, e_max) == 0:
                _e_step(epoch, 2)
            _m_step(epoch, 2, freeze=False)
            if (phase2_exploration_snapshot_every is not None
                    and (epoch == 0 or epoch % phase2_exploration_snapshot_every == 0)):
                _write_exploration_snapshot(epoch, phase=2)
            _p2_bar.update()
        _p2_bar.close()
        p2_run = p2_max
        if phase2_exploration_snapshot_every is not None:
            _write_exploration_snapshot(p2_run, phase=2, final=True)
        lbfgs_polish_report = None
        if int(lbfgs_polish_iters) > 0:
            # Optional final stage: full-batch L-BFGS on ALL charts at once (stacked ensemble; the charts are
            # independent so the summed loss is block-diagonal).  Adam plateaus ~10x above the accuracy the physics
            # needs on exact data; a quasi-Newton step does not.  Per-chart training set = the chart's competent
            # domain (the terminal ``competent`` rule: standardized recon + training closure <= threshold_topo,
            # largest physical component), overlaps included -- a polish restricted to owned poses bends the
            # chart's surface in the seam it never sees, which is exactly where handoffs happen.
            # Loss = recon + mult * closure / closure_scale, where closure_scale (mean translation variance)
            # puts the raw-unit closure residual on the standardized recon scale.
            # Run in ``lbfgs_polish_rounds`` rounds, recomputing each chart's competent domain before every round:
            # the domain of the charts coming out of phase 2 is smaller than the domain of the polished charts, and
            # a single round leaves the poses that only become competent during the polish untrained (measured on
            # stephenson: 549/677 trained vs 734/810 final competent, chart tangent off the frame by up to 81 deg there).
            sync_device(); polish_tick = time.perf_counter()
            n_k, polish_scale = len(models), float(np.mean(_trans_var))
            # Projector only: it pins the tangent PLANE, which is what velocity transport needs.  The normalized
            # Gram term additionally demands a conformal chart (J^T J ~ I), a metric constraint the physics does
            # not need (the mass field absorbs latent anisotropy); in the node-exact L-BFGS polish it cost
            # fresh-pose accuracy (five_bar: fresh rms p95 0.045 -> 0.054) for no angle gain.
            polish_tangent = float(tangent_weight) != 0.0 and float(projector_weight) != 0.0

            def _polish_round(round_iters):
                with torch.no_grad():
                    pred_all = ensemble_decode(ensemble_encode(Q.unsqueeze(0).expand(n_k, -1, -1)))
                    err_all = _recon(pred_all, Q.unsqueeze(0))
                    closure_all = validity_loss(models[0], pred_all.reshape(n_k * n, n_bodies, 3, 4)).reshape(n_k, n)
                polish_angles, polish_folds = _gate_angles_folds()
                polish_competent = _restrict_to_components(
                    ((err_all + closure_all) <= float(threshold_topo)).T.cpu().numpy()
                    & (polish_angles <= float(threshold_topo_angle_deg)) & (polish_folds >= float(fold_ratio_floor)),
                    pairs, n)
                sels = []
                for k in range(n_k):
                    sel = torch.as_tensor(np.flatnonzero(polish_competent[:, k]), dtype=torch.long, device=Q.device)
                    sels.append(sel if len(sel) >= 40 else err_all[k].argsort()[:120])
                n_pad = max(len(x) for x in sels)
                pad_idx = torch.stack([torch.cat([x, x[:1].repeat(n_pad - len(x))]) for x in sels])
                pad_mask = torch.stack([torch.cat([torch.ones(len(x)), torch.zeros(n_pad - len(x))])
                                        for x in sels]).to(dtype=Q.dtype, device=Q.device)
                X_pad, pad_count = Q[pad_idx], pad_mask.sum(1)
                # The polish must also hold the decoder tangent plane to the observed simulator frame: recon +
                # closure alone drive pose agreement in the seam while leaving the tangent free to bend there.
                # Projector term at the M-step tangent weight, one forward jvp per latent axis.
                obs_pad = frames_t[pad_idx].reshape(n_k, n_pad, -1, d) if polish_tangent else None

                def _polish_terms():
                    """Per-chart (recon + closure, tangent) means over the polish set."""
                    from torch.func import jvp
                    codes = ensemble_encode(X_pad)
                    tangent = torch.zeros(n_k, dtype=Q.dtype, device=Q.device)
                    if polish_tangent:
                        columns = []
                        for axis in range(d):
                            direction = torch.zeros_like(codes); direction[..., axis] = 1.0
                            pred_pad, column = jvp(ensemble_decode, (codes,), (direction,))
                            columns.append(column)
                        got = torch.stack(columns, -1)                               # [K, P, f_dim, d]
                        per_tangent = float(projector_weight) * _projector_distance_sq(
                            got, obs_pad, ridge_rel=1e-6)
                        tangent = (per_tangent * pad_mask).sum(1) / pad_count
                    else:
                        pred_pad = ensemble_decode(codes)
                    valid_pad = validity_loss_raw(models[0], pred_pad.reshape(n_k * n_pad, n_bodies, 3, 4)
                                              ).reshape(n_k, n_pad) / polish_scale
                    per_pose = _recon(pred_pad, X_pad) + float(lbfgs_polish_closure_mult) * valid_pad
                    return (per_pose * pad_mask).sum(1) / pad_count, tangent

                def _polish_losses():
                    fit, tangent = _polish_terms()
                    return fit + float(tangent_weight) * tangent

                round_tick = time.perf_counter()
                backup = {name: value.detach().clone() for name, value in ensemble_params.items()}
                terms_before = [x.detach() for x in _polish_terms()]
                loss_before = _polish_losses().detach()
                polish_opt = torch.optim.LBFGS(list(ensemble_params.values()), lr=1.0, max_iter=50, history_size=50,
                                               line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)

                def _polish_closure():
                    polish_opt.zero_grad(); total = _polish_losses().sum(); total.backward(); return total
                for _ in range(max(1, -(-int(round_iters) // 50))):
                    polish_opt.step(_polish_closure)
                loss_after = _polish_losses().detach()
                terms_after = [x.detach() for x in _polish_terms()]
                kept = bool(torch.isfinite(loss_after).all() and (loss_after <= loss_before).all())
                if not kept:                                # never keep a round that diverged or made any chart worse
                    with torch.no_grad():
                        for name, value in ensemble_params.items():
                            value.copy_(backup[name])
                sync_device()
                return {"iters": int(round_iters), "train_sizes": [int(len(x)) for x in sels], "kept": kept,
                        "loss_before": loss_before.tolist(), "loss_after": loss_after.tolist(),
                        "fit_before": terms_before[0].tolist(), "fit_after": terms_after[0].tolist(),
                        "tangent_before": terms_before[1].tolist(), "tangent_after": terms_after[1].tolist(),
                        "seconds": time.perf_counter() - round_tick}

            n_rounds = max(1, int(lbfgs_polish_rounds))
            round_iters = [int(lbfgs_polish_iters) // n_rounds + (1 if r < int(lbfgs_polish_iters) % n_rounds else 0)
                           for r in range(n_rounds)]
            polish_rounds = [_polish_round(it) for it in round_iters if it > 0]
            lbfgs_polish_report = {"iters": int(lbfgs_polish_iters), "rounds": polish_rounds,
                                   "closure_mult": float(lbfgs_polish_closure_mult), "closure_scale": polish_scale,
                                   "overlap_tol": float(lbfgs_polish_overlap_tol),
                                   "tangent_weight": float(tangent_weight), "projector_weight": float(projector_weight),
                                   # flat summary: last round's training set, first round's before / last round's after
                                   "train_sizes": polish_rounds[-1]["train_sizes"],
                                   "kept": all(r["kept"] for r in polish_rounds),
                                   "loss_before": polish_rounds[0]["loss_before"],
                                   "loss_after": polish_rounds[-1]["loss_after"],
                                   "seconds": time.perf_counter() - polish_tick}
            timing["lbfgs_polish_seconds"] = lbfgs_polish_report["seconds"]
        _e_step(p2_run, 2, certify=True, certificate_reason="final")
        timing["phase2_seconds"] = time.perf_counter() - phase2_start
        timing["joint_training_seconds"] = time.perf_counter() - joint_start
        timing["joint_breakdown_seconds"] = joint_timing
        timing["per_epoch"] = epoch_measurements

        diagnostics_start = time.perf_counter()
        with torch.no_grad():
            evaluations = [chart_loss(m, i, pairs, compute_tangent=False) for i, m in enumerate(models)]
            C = np.stack([item[0].cpu().numpy() for item in evaluations], axis=1)
            reconstruction = np.stack([item[2][0].cpu().numpy() for item in evaluations], axis=1)
            validity_component = np.stack([item[2][2].cpu().numpy() for item in evaluations], axis=1)
            alias_component = np.stack([item[2][3].cpu().numpy() for item in evaluations], axis=1)
        # The competence threshold is fixed, off both the models and the
        # competence matrix it gates.  Competence C is a mean-square pose
        # residual; CONSTRAINT_TOL is the sqrt-residual ceiling used across the
        # atlas (patch_atlas.py), so its square is the matching MSE-unit bar.
        # The encode/decode round-trip noise floor is measured further down,
        # once each chart's own patch is known (it must be probed on-domain,
        # with noise scaled to the chart's latent extent); it is reported and
        # gated on its own p95, never folded into the competence threshold.
        _, multiplicity, deficit_weights, rim_temperature = _soft_multiplicity_weights(
            reconstruction, validity_component, threshold=threshold_topo, atlas_dimension=d)
        responsibilities = _owner_responsibilities(
            _ownership_weight(reconstruction + validity_component, threshold_topo))
        usage_ema = (float(usage_ema_decay) * usage_ema +
                     (1.0 - float(usage_ema_decay)) * responsibilities.mean(0))
        # THE cover mask: (recon + valid) <= threshold_topo -- the same quantity
        # the E-step coverage / multiplicity / exclusive-support / retirement all
        # key on, and the argument of the ownership-weight sigmoid.
        competent = (reconstruction + validity_component) <= threshold_topo
        winners = responsibilities.argmax(1)
        decoder_norm = np.asarray([float(ensemble_decoder_spectral(i).detach()) for i in range(len(models))])
        initial_exclusive = _exclusive_support(competent)
        pruning_rows = []
        with torch.no_grad():
            for i, model in enumerate(models):
                ids = torch.as_tensor(np.flatnonzero(winners == i), dtype=torch.long, device=Q.device)
                reconstruction_support_defined = len(ids) >= support_floor
                if not reconstruction_support_defined:
                    recon_ratio, spread_ratio = float("inf"), 0.0
                else:
                    target = Q[ids]
                    rebuilt = _decode(model, i, encode(model, i, target))
                    recon = float(_recon(rebuilt, target).mean())
                    baseline = float(_recon(target, target.mean(0, keepdim=True)).mean())
                    spread_ratio = float(rebuilt.std(0).mean() / target.std(0).mean().clamp_min(1e-12))
                    recon_ratio = recon / max(baseline, 1e-12)
                quality_ok = _chart_quality_passes(
                    usage=float(usage_ema[i]), recon_ratio=recon_ratio, spread_ratio=spread_ratio,
                    usage_floor=float(usage_ema_floor), recon_ratio_max=float(quality_recon_ratio_max),
                    spread_ratio_min=float(quality_spread_ratio_min))
                spectral_ok = bool(decoder_norm[i] >= float(decoder_norm_floor))
                # This is terminal diagnosis only.  The live lifecycle, not a
                # post-training selector, owns chart retirement.
                stretch_defined = bool(np.any(
                    np.minimum(responsibilities[pairs[:, 0], i],
                               responsibilities[pairs[:, 1], i]) >= 1.2 / len(models)))
                n_certified = int(competent[:, i].sum())
                support_ok = bool(stretch_defined and n_certified >= support_floor)
                pruning_rows.append({"ema_usage": float(usage_ema[i]),
                                     "recon_over_mean_baseline": recon_ratio,
                                     "decoder_out_spread_over_data_spread": spread_ratio,
                                     "spectral_product": float(decoder_norm[i]),
                                     "n_certified": n_certified,
                                     "reconstruction_support_defined": bool(reconstruction_support_defined),
                                     "reconstruction_support": int(len(ids)),
                                     "exclusive_certified_initial": int(initial_exclusive[i]),
                                     "exclusive_certified": 0,
                                     "stretch_defined": stretch_defined,
                                     "keep": bool(quality_ok and spectral_ok and support_ok),
                                     "reasons": (([] if reconstruction_support_defined else ["insufficient_reconstruction_support"]) +
                                                ([] if quality_ok else ["usage_or_quality"])) +
                                                ([] if spectral_ok else ["spectral_floor"]) +
                                                ([] if support_ok else ["support_floor"])})
        # Terminal evaluation may never change the slot set.  In particular,
        # a nonzero exclusive count is a proof that deleting the chart makes a
        # hole; report that lifecycle failure rather than concealing it.
        exclusive_certified = _exclusive_support(competent)
        retired_with_exclusive_support = []
        for i, row in enumerate(pruning_rows):
            row["exclusive_certified"] = int(exclusive_certified[i])
            terminal_failure = bool(row["reasons"])
            row["keep"] = True
            if terminal_failure and exclusive_certified[i] > 0:
                retired_with_exclusive_support.append({"chart": i,
                                                      "exclusive_certified": int(exclusive_certified[i]),
                                                      "reasons": list(row["reasons"])})
        # Refresh one last time after the training tail; no model, selector,
        # or responsibility label is removed or refit at this terminal step.
        refresh_gauge(responsibilities)
        with torch.no_grad():
            evaluations = [chart_loss(m, i, pairs, compute_tangent=False) for i, m in enumerate(models)]
            C = np.stack([item[0].cpu().numpy() for item in evaluations], axis=1)
            reconstruction = np.stack([item[2][0].cpu().numpy() for item in evaluations], axis=1)
            validity_component = np.stack([item[2][2].cpu().numpy() for item in evaluations], axis=1)
            alias_component = np.stack([item[2][3].cpu().numpy() for item in evaluations], axis=1)
        # A chart is competent on a pose only if it reconstructs it AND represents the manifold there: its decoder
        # tangent within ``threshold_topo_angle_deg`` of the observed simulator frame and unfolded.  Recon +
        # closure alone admitted poses a chart passes a fold of its surface through (pantograph: ~8% of
        # "competent" poses at recon ~1e-2 with the chart tangent 80-90 deg off), and every downstream consumer
        # (patches, physics fields, the polish set) trained on them.  Training ownership stays ungated.
        terminal_angles, terminal_folds = _gate_angles_folds()
        terminal_tangent_ok = ((terminal_angles <= float(threshold_topo_angle_deg)) &
                               (terminal_folds >= float(fold_ratio_floor)))
        competent_reconstruction_only = _restrict_to_components(
            (reconstruction + validity_component) <= threshold_topo, pairs, n)
        competent = _restrict_to_components(
            ((reconstruction + validity_component) <= threshold_topo) & terminal_tangent_ok, pairs, n)
        responsibilities = _owner_responsibilities(
            _ownership_weight(reconstruction + validity_component, threshold_topo))
        _, multiplicity, deficit_weights, rim_temperature = _soft_multiplicity_weights(
            reconstruction, validity_component, threshold=threshold_topo, atlas_dimension=d)
        predictions = np.stack([item[2][5].reshape(n, -1).cpu().numpy() for item in evaluations])
        reconstruction_diagnostics = _reconstruction_diagnostics(q, predictions, recon_scale)
        absolute_reconstruction_ok = bool(
            reconstruction_diagnostics["best_chart_recon_over_mean_pose_baseline"] <= .05)
        # Recompute the uniqueness mass against the pruned, re-evaluated state;
        # the pre-prune value describes charts that no longer exist.
        winners = responsibilities.argmax(1)
        unique = ((winners[:, None] == np.arange(len(models))) & competent).sum(0) / n
        responsibility_entropy = float((-(responsibilities * np.log(np.clip(responsibilities, 1e-12, None))).sum(1)).mean())
        primary_responsibility_mass = responsibilities.sum(0).tolist()
        natural_boundary_overlap = int((responsibilities >= (0.25 / max(len(models), 1))).sum(1).clip(0, 2).mean() * n)
        cores = [np.flatnonzero((responsibilities[:, i] == responsibilities.max(1)) & competent[:, i]) for i in range(len(models))]
        # A chart's patch is its full competent domain, overlaps included:
        # handoffs happen in the overlap, so the learned fields and the
        # certificates must cover it.  Ownership (the winner partition) is
        # ``cores``.  Routing is the learned advantage router, not this set.
        patches_from_fallback = [not bool(np.any(competent[:, i])) for i in range(len(models))]
        patches = [(np.flatnonzero(competent[:, i]) if np.any(competent[:, i])
                    else np.flatnonzero(winners == i)) for i in range(len(models))]
        anchors = [encode(m, i, Q[patch]).detach().cpu().numpy() if len(patch) else np.empty((0, d)) for i, (m, patch) in enumerate(zip(models, patches))]
        latent_extent = [encode(m, i, Q[patch]).detach().cpu().numpy().std(0).tolist() if len(patch) else [0.0] * d
                         for i, (m, patch) in enumerate(zip(models, patches))]
        # Terminal selector refit (Step 4).  The target is the warm physical
        # posterior, but zeroed wherever a pose's code falls OUTSIDE that
        # chart's own local support: the predictor then physically cannot route
        # into a chart whose learned domain excludes the query, however well it
        # reconstructs.  Rows with no in-support chart keep the unmasked
        # posterior.  Refitting only the retained RBF outputs also removes
        # labels for retired, interchangeable charts.
        # (The in-loop analogue is blocked on the support-radius dedup:
        # fit_local_support_sdf fits from a scalar radius while certification
        # uses per-point local_support_radii, so an in-loop mask would gate
        # against a definition the shipped package does not use.)
        support_mask = np.ones_like(responsibilities)
        with torch.no_grad():
            for i, m in enumerate(models):
                if len(anchors[i]) >= 2:
                    support_mask[:, i] = in_local_support(
                        anchors[i], encode(m, i, Q).cpu().numpy()).astype(float)
        masked_target = responsibilities * support_mask
        empty_row = masked_target.sum(1) < 1e-8
        masked_target[empty_row] = responsibilities[empty_row]
        masked_target = masked_target / masked_target.sum(1, keepdims=True)
        selector_opt = torch.optim.Adam((log_sigma, correction_w, correction_b), lr=2e-3, weight_decay=1e-2)
        selector_target = torch.as_tensor(masked_target, dtype=Q.dtype, device=Q.device)
        for _ in range(max(1, int(pretrain_steps))):
            selector_opt.zero_grad()
            selector_loss = -(selector_target * torch.log_softmax(selector_scores(), dim=1)).sum(1).mean()
            selector_loss.backward()
            selector_opt.step()
        selector_support_masking = {
            "rows_with_no_in_support_chart": int(empty_row.sum()),
            "mean_charts_in_support_per_pose": float(support_mask.sum(1).mean())}
        # Encode/decode round-trip invertibility, measured two ways.
        #  * on-domain: each chart perturbed only on its competent domain, with
        #    the noise scaled to that chart's own latent extent -- comparable
        #    across charts and across K.  This is the number the atlas is
        #    gated on.
        #  * off-domain: the legacy global measure -- every chart probed on
        #    one shared held-out set at fixed absolute noise.  Ungated but
        #    reported: a large value is real information about how the encoder
        #    behaves when the runtime steps outside a chart (at a handoff).
        with torch.no_grad():
            on_domain_probe, off_domain_probe = [], []
            on_support_counts, off_support_counts = [], []
            held_q = Q[torch.as_tensor(held, dtype=torch.long, device=Q.device)]
            for i, (m, patch) in enumerate(zip(models, patches)):
                if len(patch):
                    own = torch.as_tensor(patch, dtype=torch.long, device=Q.device)
                    support_u = encode(m, i, Q[own])
                    if len(own) > 256:
                        own = own[torch.randperm(len(own))[:256]]
                    u0 = encode(m, i, Q[own])
                    support_np = support_u.cpu().numpy()
                    radii = local_support_radii(support_np)
                    distance = torch.cdist(u0, support_u)
                    nearest_support = distance.argmin(1)
                    scale = torch.as_tensor(radii, dtype=Q.dtype, device=Q.device)[nearest_support, None]
                    probe_count = max(1, d)
                    direction = torch.randn((len(u0), probe_count, d), dtype=Q.dtype, device=Q.device)
                    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
                    u1 = (u0[:, None, :] + scale[:, None, :] * direction).reshape(-1, d)
                    accepted = in_local_support(support_np, u1.cpu().numpy())
                    displacement = torch.linalg.vector_norm(u1 - u0[:, None, :].expand(-1, probe_count, -1).reshape(-1, d), dim=-1).cpu().numpy()
                    on_support_counts.append({"candidates": int(len(u1)), "accepted": int(accepted.sum()),
                                              "local_radius_p50": float(np.percentile(scale.cpu().numpy(), 50)),
                                              "local_radius_p95": float(np.percentile(scale.cpu().numpy(), 95)),
                                              "displacement_p50": float(np.percentile(displacement, 50)),
                                              "displacement_p95": float(np.percentile(displacement, 95))})
                    if accepted.any():
                        ids = torch.as_tensor(np.flatnonzero(accepted), dtype=torch.long, device=Q.device)
                        on_domain_probe.append(((encode(m, i, _decode(m, i, u1[ids])) - u1[ids]).square().mean(-1)).cpu().numpy())
                g0 = encode(m, i, held_q)
                if len(patch):
                    distance = torch.cdist(g0, support_u)
                    nearest_support = distance.argmin(1)
                    scale = torch.as_tensor(radii, dtype=Q.dtype, device=Q.device)[nearest_support, None]
                    probe_count = max(1, d)
                    direction = torch.randn((len(g0), probe_count, d), dtype=Q.dtype, device=Q.device)
                    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
                    g1 = (g0[:, None, :] + scale[:, None, :] * direction).reshape(-1, d)
                    accepted = in_local_support(support_np, g1.cpu().numpy())
                    displacement = torch.linalg.vector_norm(g1 - g0[:, None, :].expand(-1, probe_count, -1).reshape(-1, d), dim=-1).cpu().numpy()
                    off_support_counts.append({"candidates": int(len(g1)), "accepted": int(accepted.sum()),
                                               "local_radius_p50": float(np.percentile(scale.cpu().numpy(), 50)),
                                               "local_radius_p95": float(np.percentile(scale.cpu().numpy(), 95)),
                                               "displacement_p50": float(np.percentile(displacement, 50)),
                                               "displacement_p95": float(np.percentile(displacement, 95))})
                    if accepted.any():
                        ids = torch.as_tensor(np.flatnonzero(accepted), dtype=torch.long, device=Q.device)
                        off_domain_probe.append(((encode(m, i, _decode(m, i, g1[ids])) - g1[ids]).square().mean(-1)).cpu().numpy())
        on_all = np.concatenate(on_domain_probe) if on_domain_probe else np.zeros(1)
        off_all = np.concatenate(off_domain_probe) if off_domain_probe else np.zeros(1)
        inverse_noise_floor = float(np.percentile(on_all, 95))
        off_domain_noise_floor = float(np.percentile(off_all, 95))
        # This is a latent-space round-trip diagnostic, not a pose-MSE
        # certificate.  Injectivity is certified below from physical far pairs.
        inverse_noise_floor_ok = bool(inverse_noise_floor <= threshold)
        footprint_connectivity, far_pair_injectivity = [], []
        alias_certificate = []
        for i, (model, patch) in enumerate(zip(models, patches)):
            u = anchors[i]
            if len(u) < 2:
                footprint_connectivity.append({"components": int(len(u)), "threshold": 0.0})
                far_pair_injectivity.append({"tested": 0, "aliases": 0, "rate": 0.0,
                                              "physical_far_threshold": float("inf")})
                alias_certificate.append({"tested": 0, "aliases": 0, "rate": 0.0, "pass": False})
                continue
            latent_tree = cKDTree(u)
            nearest_du = latent_tree.query(u, k=min(3, len(u)))[0]
            local_scale = float(np.median(nearest_du[:, -1]))
            threshold_u = 2.5 * local_scale
            parent = np.arange(len(u))
            def find(x):
                while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for a, b in latent_tree.query_pairs(threshold_u, output_type="ndarray"):
                ra, rb = find(int(a)), find(int(b))
                if ra != rb: parent[ra] = rb
            footprint_connectivity.append({"components": int(len({find(j) for j in range(len(u))})),
                                           "threshold": threshold_u})
            count = min(2048, len(patch) * max(1, len(patch) - 1) // 2)
            a = rng.integers(len(patch), size=count); b = rng.integers(len(patch), size=count)
            far_dp = _patch_far_distance(feature, patch, rng)
            valid = (a != b) & (np.linalg.norm(feature[patch[a]] - feature[patch[b]], axis=1) >= far_dp)
            dz = np.linalg.norm(u[a[valid]] - u[b[valid]], axis=1)
            close_sample = min(4096, len(u) * max(1, len(u) - 1) // 2)
            ca = rng.integers(len(u), size=close_sample); cb = rng.integers(len(u), size=close_sample)
            close_pairs = np.linalg.norm(u[ca[ca != cb]] - u[cb[ca != cb]], axis=1)
            close = float(np.quantile(close_pairs, .05)) if len(close_pairs) else 0.0
            aliases = int((dz <= close).sum())
            alias_row = {"tested": int(valid.sum()), "aliases": aliases,
                         "rate": float(aliases / max(1, valid.sum())),
                         "physical_far_threshold": far_dp, "latent_close_threshold": close}
            alias_row["pass"] = bool(alias_row["tested"] > 0 and alias_row["aliases"] == 0)
            far_pair_injectivity.append(alias_row)
            alias_certificate.append({key: alias_row[key] for key in ("tested", "aliases", "rate", "pass")})

        # Absolute reconstruction fidelity per chart.  ``recon_over_mean_baseline``
        # is the reconstruction MSE relative to predicting the covered set's mean
        # pose (1.0 == no better than a constant); the spread ratio is near 0 for
        # a collapsed decoder and near 1 for one that spans the data.  Two charts
        # can match on every other reported field yet differ here by 40x.
        chart_fidelity = []
        with torch.no_grad():
            for i, (model, patch) in enumerate(zip(models, patches)):
                ids = (torch.as_tensor(patch, dtype=torch.long, device=Q.device) if len(patch)
                       else torch.arange(n, device=Q.device))
                target = Q[ids]
                rebuilt = _decode(model, i, encode(model, i, target))
                recon = float(_recon(rebuilt, target).mean())
                mean_baseline = float(_recon(target, target.mean(0, keepdim=True)).mean())
                out_spread = float(rebuilt.std(0).mean())
                data_spread = float(target.std(0).mean())
                fidelity_defined = len(ids) >= support_floor
                chart_fidelity.append({
                    "covered": int(len(ids)),
                    "recon_mse": recon, "mean_baseline_mse": mean_baseline,
                    "fidelity_defined": bool(fidelity_defined),
                    "recon_over_mean_baseline": (recon / max(mean_baseline, 1e-12)
                                                 if fidelity_defined else None),
                    "decoder_out_spread": out_spread, "data_spread": data_spread,
                    "decoder_out_spread_over_data_spread": (out_spread / max(data_spread, 1e-12)
                                                               if fidelity_defined else None)})

        # ||J_ji . J_ij - I||_F on off-node overlap points.  J_ij = d(E_j o D_i);
        # a working transition has J_ji . J_ij ~ I on an open neighbourhood, not
        # only at shared vertices.  Invariant under reparametrization of either
        # chart, so this is the handoff-transport number to trust.  Reported,
        # not gated (baseline diagnostic).
        from torch.func import jacrev, vmap
        overlap_jacobian_roundtrip = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    continue
                shared = np.flatnonzero(competent[:, i] & competent[:, j])
                sample = seam_neighbourhood(i, shared, fraction=1.0)
                if sample is None:
                    continue
                _, u1 = sample
                def fwd(u_vec, a=i, b=j):
                    return encode(models[b], b, _decode(models[a], a, u_vec[None]))[0]
                def bwd(v_vec, a=i, b=j):
                    return encode(models[a], a, _decode(models[b], b, v_vec[None]))[0]
                J_ij = vmap(jacrev(fwd))(u1)
                J_ji = vmap(jacrev(bwd))(vmap(fwd)(u1))
                resid = torch.linalg.matrix_norm(
                    J_ji @ J_ij - torch.eye(d, dtype=J_ij.dtype, device=J_ij.device), ord="fro").detach().cpu().numpy()
                overlap_jacobian_roundtrip.append({
                    "charts": [int(i), int(j)], "points": int(len(u1)),
                    "roundtrip_minus_identity_fro_p50": float(np.percentile(resid, 50)),
                    "roundtrip_minus_identity_fro_p95": float(np.percentile(resid, 95)),
                    "roundtrip_minus_identity_fro_max": float(resid.max())})

        selector = PhysicalRBFSelector(centre_t.detach().cpu().numpy(), log_sigma.detach().cpu().numpy(),
                                       correction_w.detach().cpu().numpy(),
                                       correction_b.detach().cpu().numpy(), feat_mean, feat_scale)
        selector_argmax = np.argmax(np.vstack([selector.score(x) for x in q]), 1)
        # The selector is trained to the support-masked posterior (Step 4), so
        # calibration is measured against that target; the raw-posterior
        # agreement is still reported for continuity.
        calibration = float(np.mean(selector_argmax == np.argmax(masked_target, 1)))
        calibration_vs_raw_posterior = float(np.mean(selector_argmax == np.argmax(responsibilities, 1)))
        calibration_ok = bool(calibration >= float(selector_calibration_min))

        distortion_rows = []
        for i, model in enumerate(models):
            local = ((np.minimum(responsibilities[pairs[:, 0], i], responsibilities[pairs[:, 1], i]) >=
                     1.2 / K) & distortion_pair_ok)
            if not np.any(local):
                distortion_rows.append({"lower": 0.0, "upper": float("inf"), "pass": False,
                                        "tested_pairs": 0, "reason": "no nondegenerate owned pairs"}); continue
            with torch.no_grad():
                extra = chart_loss(model, i, pairs[local], compute_tangent=False)[2]
                ratio = (extra[7][extra[9]] / extra[6][extra[9]]).cpu().numpy()
            lower, upper = float(ratio.min()), float(ratio.max())
            distortion_rows.append({"lower": lower, "upper": upper,
                                    "pass": bool(upper / max(lower, 1e-12) <= max_distortion_spread ** 2),
                                    "tested_pairs": int(len(ratio))})
        coverage_ok = bool(competent.any(1).all())
        distortion_ok = bool(all(x["pass"] for x in distortion_rows))
        own = responsibilities.argmax(1)
        coherence = float(np.mean([np.mean(own[nearest[i]] == own[i]) for i in range(n)]))
        # Measured bidirectional overlap transport: wherever two charts are both
        # competent on the same poses, their decoded poses (and cross-encodings)
        # must agree to within the competence threshold.  No declared overlap is
        # used; this is verified, not assumed.
        overlap_rows = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                shared = np.flatnonzero(competent[:, i] & competent[:, j])
                if len(shared) == 0:
                    continue
                with torch.no_grad():
                    ids = torch.as_tensor(shared, dtype=torch.long, device=Q.device)
                    ui, uj = encode(models[i], i, Q[ids]), encode(models[j], j, Q[ids])
                    qi, qj = _decode(models[i], i, ui), _decode(models[j], j, uj)
                    # Group-standardized like every other reconstruction quantity: ``threshold`` and
                    # ``chart_fidelity[*]["recon_mse"]`` below are in these units, so the raw-pose MSE would
                    # make both the ``agree`` gate and the handoff ratio scale with the data's raw variance.
                    pose_gap_mse = float(_recon(qi, qj).mean())
                    # Cross-encode residual is still reported (latent units,
                    # ungated) as information about handoff behaviour; the old
                    # latent-energy normalizer and its 0.05 bar are gone -- once
                    # the seam is trained in pose units the pose gap carries the
                    # same signal without a gauge-dependent denominator.
                    cross_mse = float(.5 * ((encode(models[i], i, qj) - ui).square().mean()
                                              + (encode(models[j], j, qi) - uj).square().mean()))
                agree = bool(pose_gap_mse <= threshold)
                # Headline field: the seam gap measured against sqrt(2) times the
                # WORSE of the two charts' own reconstruction rms (the per-pair
                # bound -- a global denominator smears the sqrt(2) prediction).
                # This is the one quantity the seam training term must move.
                rms_i = float(np.sqrt(max(chart_fidelity[i]["recon_mse"], 1e-30)))
                rms_j = float(np.sqrt(max(chart_fidelity[j]["recon_mse"], 1e-30)))
                handoff_ratio = (pose_gap_mse ** 0.5) / (np.sqrt(2.0) * max(rms_i, rms_j, 1e-30))
                overlap_rows.append({"charts": [i, j], "shared": int(len(shared)),
                                     "pose_gap_rms": pose_gap_mse ** 0.5,
                                     "handoff_over_sqrt2_recon": float(handoff_ratio),
                                     "cross_encode_rms": cross_mse ** 0.5,
                                     "pass": agree})
        overlap_evaluated = len(overlap_rows) > 0
        # A singleton atlas has no transition to certify; absence of a pair is
        # therefore vacuously sufficient, while multi-chart atlases still need
        # measured bidirectional overlap transport.
        overlap_required = len(models) > 1
        overlap_ok = bool(all(x["pass"] for x in overlap_rows)) if overlap_evaluated else not overlap_required
        # A provisional split is not a chart replacement.  It remains
        # incomplete until a subsequent atomic-admission pass can prove both
        # children have local distortion/off-anchor competence, unique support,
        # parent-domain union coverage, and bidirectional overlap transport.
        alias_ok = bool(all(item["pass"] for item in alias_certificate))
        injectivity_lower_bound_ok = alias_ok
        all_pass = (coverage_ok and overlap_ok and calibration_ok and injectivity_lower_bound_ok
                    and absolute_reconstruction_ok)
        timing["post_training_diagnostics_seconds"] = time.perf_counter() - diagnostics_start
        timing["total_before_fields_seconds"] = time.perf_counter() - timing_start

        # ---- commit-3 validation summary ------------------------------------
        # Greedy / exclusive / uncovered on THE cover mask, unrestricted (a
        # chart's full redundancy, not split by component cleanup).
        _cov = (reconstruction + validity_component) <= threshold_topo
        _greedy, _remaining = [], set(np.flatnonzero(_cov.any(1)).tolist())
        _cols = [set(np.flatnonzero(_cov[:, i]).tolist()) for i in range(_cov.shape[1])]
        while _remaining:
            _b = max(range(len(_cols)), key=lambda i: len(_cols[i] & _remaining))
            if not (_cols[_b] & _remaining):
                break
            _greedy.append(_b); _remaining -= _cols[_b]
        _drift_last = (phase_trace[-1]["territory_drift"] if phase_trace and
                       phase_trace[-1].get("territory_drift") else [])
        validation = {
            "phase1_final_charts": int(chart_count_trajectory[-1]["n_charts"])
            if chart_count_trajectory else int(len(models)),
            "phase2_final_charts": int(len(models)),
            "greedy_cover_charts": int(len(_greedy)),
            "phase1_over_greedy": (float(chart_count_trajectory[-1]["n_charts"] / max(len(_greedy), 1))
                                   if chart_count_trajectory else None),
            "uncovered_curve": [c["n_uncovered"] for c in phase1_coverage_curve],
            "uncovered_final": int((~_cov.any(1)).sum()),
            "cover_exclusive_support": _exclusive_support(_cov).tolist(),
            "cover_multiplicity_histogram": np.bincount(
                _cov.sum(1), minlength=len(models) + 1).tolist(),
            "hard_multiplicity_counts": {"0": int((_cov.sum(1) == 0).sum()),
                                        "1": int((_cov.sum(1) == 1).sum()),
                                        "2": int((_cov.sum(1) == 2).sum()),
                                        "3+": int((_cov.sum(1) >= 3).sum())},
            "core_retention_curve": core_retention_curve,
            "core_retention_min": (min((r["retention"] for r in core_retention_curve), default=None)),
            "shell_promotion_curve": shell_promotion_curve,
            "shell_promotion_mean": (float(np.mean([r["promotion"] for r in shell_promotion_curve]))
                                     if shell_promotion_curve else None),
            "territory_drift_last_mean_jaccard": (
                float(np.mean([d["jaccard"] for d in _drift_last])) if _drift_last else None),
            "e_step_schedule": timing.get("e_step_schedule", []),
        }

        report = {"status": "pass" if all_pass else "incomplete",
                  "validation": validation,
                  "case": case,
                  "all_pass": all_pass, "n_patches": len(models),
                  "uses_cover_edges": False, "uses_declared_coordinates": False,
                  "uses_memberships": False, "uses_prescribed_chart_count": False,
                  "n_charts_seeded": int(K), "n_charts_final": int(len(models)),
                  "chart_ids": list(chart_ids),
                  "competence_threshold": threshold,
                  "mean_pose_baseline_mse": mean_pose_baseline_mse,
                  "competence_baseline_fraction": (None if competence_tolerance is not None
                                                    else float(competence_baseline_fraction)),
                  "resumed_from": (None if resume is None else
                                   {"path": resume["path"], "sha256": resume["sha256"],
                                    "n_charts": len(resume["states"])}),
                  "warm_start": warm_start_rows,
                  "warm_start_diagnostics": warm_start_diagnostics,
                  "n_uncovered": int((~competent.any(1)).sum()),
                  "uncovered_pose_indices": np.flatnonzero(~competent.any(1)).tolist(),
                  "competence_tangent_certificate": {
                      "angle_deg": float(threshold_topo_angle_deg), "fold_ratio_floor": float(fold_ratio_floor),
                      "competent_reconstruction_only": competent_reconstruction_only.sum(0).astype(int).tolist(),
                      "competent": competent.sum(0).astype(int).tolist(),
                      "uncovered_reconstruction_only": int((~competent_reconstruction_only.any(1)).sum())},
                  "n_competition_uncovered": int((C.min(1) > threshold).sum()),
                  "absolute_reconstruction_ok": absolute_reconstruction_ok,
                  "reconstruction_diagnostics": reconstruction_diagnostics,
                  "coverage_ok": coverage_ok, "distortion_ok": distortion_ok, "overlap_ok": overlap_ok,
                  "overlap_required": overlap_required,
                  "overlap_evaluated": overlap_evaluated, "overlap_transport": overlap_rows,
                  "overlap_jacobian_roundtrip": overlap_jacobian_roundtrip,
                  "handoff_over_sqrt2_recon": {
                      "per_pair": [r["handoff_over_sqrt2_recon"] for r in overlap_rows],
                      "p50": (float(np.percentile([r["handoff_over_sqrt2_recon"] for r in overlap_rows], 50))
                              if overlap_rows else None),
                      "max": (float(np.max([r["handoff_over_sqrt2_recon"] for r in overlap_rows]))
                              if overlap_rows else None)},
                  "inverse_noise_floor": inverse_noise_floor,
                  "inverse_noise_floor_ok": inverse_noise_floor_ok,
                  "inverse_noise_floor_gated": False,
                  "injectivity_lower_bound_ok": injectivity_lower_bound_ok,
                  "off_domain_noise_floor": off_domain_noise_floor,
                  "inverse_noise_probe_support": {"on_domain": on_support_counts,
                                                   "off_domain": off_support_counts},
                  "chart_fidelity": chart_fidelity,
                  "rbf_assignment_calibration_ok": calibration_ok,
                  "responsibility_coherence": coherence,
                  "latent_footprint_connectivity": footprint_connectivity,
                  "far_pair_injectivity": far_pair_injectivity,
                  "distortion": {"per_chart": distortion_rows,
                                 "lower_ratio_min": min(x["lower"] for x in distortion_rows),
                                 "upper_ratio_max": max(x["upper"] for x in distortion_rows),
                                 "physical_pair_floor_fraction": .05,
                                 "physical_pair_floor": float(distortion_pair_floor),
                                 "excluded_degenerate_pairs": int((~distortion_pair_ok).sum())},
                  "rbf_assignment_calibration": calibration,
                  "rbf_assignment_calibration_vs_raw_posterior": calibration_vs_raw_posterior,
                  "selector_support_masking": selector_support_masking,
                  "unique_responsibility_mass": unique.tolist(),
                  "data_assignment": "ownership_weight",
                  "m_step_assignment": "absolute_per_chart_ownership",
                  "overlap_temperature": float(overlap_temperature),
                  "two_phase": {
                      "phase1": {"max_epochs": int(p1_max), "epochs_run": int(p1_run),
                                 "exited_on": phase1_exit,
                                 "converged": bool(phase1_stable_streak >= int(retire_hysteresis_steps)),
                                 "boundary_certificate_full_cover": boundary_certificate_full_cover,
                                 "stability_streak": int(phase1_stable_streak),
                                 "threshold_topo": float(threshold_topo),
                                 "threshold_topo_angle_deg": float(threshold_topo_angle_deg),
                                 "fold_ratio_floor": float(fold_ratio_floor),
                                 "coverage_curve": phase1_coverage_curve,
                                 "checkpoint": "phase1_final.pt",
                                 "seconds": float(timing.get("phase1_seconds", 0.0))},
                      "phase2": {"max_epochs": int(p2_max), "epochs_run": int(p2_run),
                                 "threshold_fit": float(threshold_fit),
                                 "responsibility_entropy": responsibility_entropy,
                                 "primary_responsibility_mass": primary_responsibility_mass,
                                 "natural_boundary_overlap_size": natural_boundary_overlap,
                                 "d_nn": float(d_nn), "n_components": int(n_components_pose),
                                 "recon_rms_over_d_nn": float(np.sqrt(max(
                                     reconstruction.min(1).mean(), 0.0) * n_components_pose) / max(d_nn, 1e-12)),
                                 "best_chart_recon_mse": float(reconstruction.min(1).mean()),
                                 "coverage_curve": phase2_coverage_curve,
                                 "seconds": float(timing.get("phase2_seconds", 0.0))}},
                  "soft_multiplicity": {
                      "margin_temperature": rim_temperature.tolist(),
                      "per_pose": multiplicity.tolist(), "deficit_weight_per_pose": deficit_weights.tolist(),
                      "histogram": {"bins": [0.0, 0.5, 1.0, float(d + 1), float("inf")],
                                    "counts": np.histogram(multiplicity, bins=[0.0, 0.5, 1.0, float(d + 1), float("inf")])[0].tolist()},
                      "integer_centered_histogram": _integer_multiplicity_histogram(multiplicity, d),
                      "policy": "coverage_only"},
                  "phase1_redundancy": {
                      "shell_incentive": "one_cover_global_second_cover_global_p1_collar_p2",
                      "collective_loss": "locality_masked_best_and_second_best_margins",
                      "boundary_collar_poses": int(boundary_collar.sum()),
                      "singleton_neighbourhoods": int((_cov.sum(1) == 1).sum()),
                      "uncovered_neighbourhoods": int((~_cov.any(1)).sum()),
                      "stability_streak": int(phase1_stable_streak),
                      "retirement_mode": "candidate_local_zero_exclusive_one_at_a_time"},
                  "certification_schedule": {
                      "routine_estep": "batched_reconstruction_and_physical_validity",
                      "full_composite": "before_retirement",
                      "tangent_fold": "pre_retirement_phase1_boundary_final",
                      "events": certificate_events},
                  "lifecycle_config": {"retirement_mode": "phase1_zero_exclusive_support",
                                "runt_support_floor": 0,
                                "prune_warmup_epochs": int(prune_warmup_epochs),
                                "e_step_interval": int(e_step_interval),
                                "retirements": retirement_events,
                                "off_node_probe_radius": "uniform_(0,1]*local_support_radius"},
                  "lip_weight": float(lip_weight),
                  "weight_decay": float(weight_decay),
                  "distortion_weight": float(distortion_weight),
                  "latent_gauge": bool(latent_gauge),
                  "paper_baseline": bool(resolved_config["paper_baseline"]),
                  "resolved_config": resolved_config,
                  "frontier": {
                      "shell_sample_fraction": float(shell_sample_fraction),
                      "rho": float(rho), "lambda_protect": float(lambda_protect),
                      "e_step_schedule": timing.get("e_step_schedule", []),
                      "graph": frontier_graph_stats,
                      "trace": frontier_trace},
                  "monotone_frontier": {
                      "enabled": bool(monotone_frontier),
                      "final_territory_size": ([int(len(s)) for s in persistent_territory]
                                                if monotone_frontier and persistent_territory is not None
                                                else []),
                      "final_born": ([bool(v) for v in born]
                                     if monotone_frontier and persistent_territory is not None else []),
                      "maintenance_violations_total": ([int(v) for v in maintenance_violation_total]
                                                        if monotone_frontier else []),
                      "trace": monotone_frontier_trace},
                  "inertial_frontier": {
                      "enabled": bool(inertial_frontier),
                      "alpha": float(inertial_alpha),
                      "alpha_loss": float(inertial_alpha_loss),
                      "final_claim_mass": ([float(claims[:, i].sum()) for i in range(claims.shape[1])]
                                           if inertial_frontier and claims is not None else []),
                      "trace": inertial_trace},
                  "ema_ownership": {
                      "enabled": bool(ema_ownership),
                      "alpha": float(ema_ownership_alpha),
                      "alpha_loss": float(ema_ownership_alpha_loss),
                      "final_ema_cover_mass": ([float(_ema_cover[:, i].sum()) for i in range(_ema_cover.shape[1])]
                                               if ema_ownership and _ema_cover is not None else [])},
                  "lbfgs_polish": lbfgs_polish_report,
                  "closure_units": {"standardize_mult": closure_standardize_mult, "factor": float(closure_unit_factor)},
                  "lr_schedule": {"base_lr": 2e-3, "phase": "phase2_only",
                                  "step_epochs": (int(lr_step_epochs) if lr_step_epochs else None),
                                  "gamma": float(lr_step_gamma), "min_lr": float(lr_step_min),
                                  "final_lr": (max(float(lr_step_min),
                                                   2e-3 * (float(lr_step_gamma) ** (max(0, int(p2_max) - 1) // int(lr_step_epochs))))
                                               if lr_step_epochs else 2e-3)},
                  "epoch_trace": (epoch_trace_rows if epoch_trace else None),
                  "prune_snapshots": {"enabled": bool(snapshot_prune_steps),
                                      "dir": ("prune_snapshots" if snapshot_prune_steps else None),
                                      "index": prune_snapshot_index},
                  "exploration_snapshots": {
                      "enabled": exploration_snapshot_every is not None,
                      "every_epochs": exploration_snapshot_every,
                      "dir": ("chart_exploration_snapshots"
                              if exploration_snapshot_every is not None else None),
                      "index": exploration_snapshot_index},
                  "phase2_exploration_snapshots": {
                      "enabled": phase2_exploration_snapshot_every is not None,
                      "every_epochs": phase2_exploration_snapshot_every,
                      "dir": ("phase2_chart_exploration_snapshots"
                              if phase2_exploration_snapshot_every is not None else None),
                      "index": phase2_exploration_snapshot_index},
                  "tangent_weight": float(tangent_weight),
                  "projector_weight": float(projector_weight),
                  "alias_weight": float(alias_weight),
                  "selector_weight": float(selector_weight),
                  "off_anchor_cycle_weight": float(off_anchor_cycle_weight),
                  "m_step": {"mode": "core_shell_with_shared_coverage_in_both_phases",
                             "core_batch_size_per_chart": int(m_step_batch_size),
                             "shell_batch_size_per_chart": int(_shell_budget),
                             "shell_sample_fraction": float(shell_sample_fraction)},
                  "gauge_max_condition": float(gauge_max_condition),
                  "decoder_norm_floor": float(decoder_norm_floor),
                  "decoder_spectral_product": decoder_norm.tolist(),
                  "pruning": {"usage_ema_decay": float(usage_ema_decay),
                              "usage_ema_floor": float(usage_ema_floor),
                              "min_certified_support_fraction": float(min_certified_support_fraction),
                              "min_certified_support": int(support_floor),
                              "quality_recon_ratio_max": float(quality_recon_ratio_max),
                              "quality_spread_ratio_min": float(quality_spread_ratio_min),
                              "per_chart": pruning_rows,
                              "retired_with_exclusive_support": retired_with_exclusive_support},
                  "pinned_log_stretch_reference": [float(x) for x in stretch_reference],
                  "component_mean_loss": {"reconstruction": reconstruction.mean(0).tolist(),
                                          "validity": validity_component.mean(0).tolist(),
                                          "alias": alias_component.mean(0).tolist()},
                  "component_certificate": {
                      "reconstruction_covered": (reconstruction <= threshold).sum(0).tolist(),
                      "validity_covered": (validity_component <= threshold).sum(0).tolist(),
                      "certified": competent.sum(0).tolist(),
                      "alias": alias_certificate,
                      "alias_ok": alias_ok},
                  "patches_from_fallback": patches_from_fallback,
                  "latent_extent_std": latent_extent,
                  "timing_seconds": timing,
                  "encoder_kind": encoder_kind, "encoder_ablation": ["linear", "shallow", "current"],
                  "lifecycle_trace": trace, "lifecycle_phases": phase_trace,
                  "chart_count_trajectory": chart_count_trajectory,
                  "retirement_events": retirement_events,
                  "retirement_rejection_log": retirement_rejection_log}
        with open(os.path.join(work, config.out_report or "competitive_cae_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_json_safe)
        # Individual modules are diagnostic/serialization views only.  Export
        # their state after the final copy from the optimizer-owned ensemble.
        sync_model_shells()
        # Construction-only is intentionally artifact-free, matching the
        # established fitter contract for cheap structural validation.
        if config.construction_only:
            if steer_default_device:
                torch.set_default_device("cpu")
            return report, models, patches, anchors

        # Preserve the latest fitted atlas even when diagnostics are incomplete.
        # Its report remains authoritative; consumers can inspect the status
        # rather than silently losing the most recent learned state.

        # Runtime authority is learned-field based.  The fields are distilled
        # from the chart models and physical validity evidence only; neither
        # graph edges nor stored memberships enter the serialized route.
        from atlas_rom.cases import load_spec
        from atlas_rom.spec import geometry
        from atlas_rom.rom import learned as learned_fields
        from atlas_rom.charts.physics_fitting import make_physics_fitter
        if steer_default_device:
            torch.set_default_device("cpu")
        case_spec = load_spec(case)
        bodies = geometry.build_bodies(case_spec.mechanism)
        bodies = {**bodies, "up": case_spec.mechanism.up_vector, "gravity": case_spec.mechanism.gravity}
        physics = make_physics_fitter(learned_fields)
        mass_fields, potential_fields, support_fields, boundary_fields = [], [], [], []
        for i, (model, patch, anchor) in enumerate(zip(models, patches, anchors)):
            carries_edge = (competent[physical_edges[:, 0], i] &
                            competent[physical_edges[:, 1], i])
            edge_vertices = (np.unique(physical_edges[carries_edge].ravel())
                             if carries_edge.any() else np.empty(0, dtype=int))
            reference_ids = np.unique(np.r_[patch, edge_vertices]).astype(int)
            reference = q[reference_ids] if len(reference_ids) else q
            # Positive runtime support evidence at genuine transition vertices:
            # these are inferred from independently competent charts, never
            # from a declared seam or stored routing membership.
            transition_ids = np.flatnonzero(competent[:, i] & (competent.sum(1) >= 2))
            negative = q[np.flatnonzero(~competent[:, i])] if competent.shape[1] else q[:0]
            with torch.no_grad():
                negative_u = (encode(model, i, torch.as_tensor(negative, dtype=Q.dtype, device=compute_device))
                              .detach().cpu().numpy() if len(negative) else np.empty((0, d)))
                transition_u = (encode(model, i, Q[torch.as_tensor(transition_ids, dtype=torch.long,
                                                                     device=Q.device)]).detach().cpu().numpy()
                                if len(transition_ids) else np.empty((0, d)))
            model = model.to("cpu").double()
            fitted = physics.fit_chart(model, anchor, bodies=bodies, validity=validity, seed=config.seed + i,
                                       transition_positive=transition_u, local_map_negative=negative_u, support_reference_q=reference,
                                       support_interior_q=reference, epochs=int(field_epochs), validity_tol=CONSTRAINT_TOL)
            mass_fields.append(fitted["mass"]); potential_fields.append(fitted["potential"])
            support_fields.append(fitted["support"]); boundary_fields.append(fitted["boundary"])
        state = {"format": ARTIFACT_FORMAT, "case": case,
                 "chart_fitter": {"id": self.fitter_id, "config": {"encoder_kind": encoder_kind, "latent_gauge": bool(latent_gauge)}, "payload_format": 1},
                 "chart_atlas": {"patches": [x.tolist() for x in patches], "cores": [x.tolist() for x in cores],
                                 "states": [m.state_dict() for m in models], "anchors": [x.tolist() for x in anchors],
                                 "n_bodies": n_bodies, "encoder_kind": encoder_kind, "selector": selector.payload()},
                 "physics_fields": {"mass_fields": [learned_fields.state(x) for x in mass_fields],
                                    "potential_fields": [learned_fields.state(x) for x in potential_fields],
                                    "support_fields": [learned_fields.state(x) for x in support_fields],
                                    "boundary_fields": [learned_fields.sdf_state(x) for x in boundary_fields]}}
        import torch as _torch
        _torch.save(state, os.path.join(work, config.out_artifact or ARTIFACT))
        return report, models, patches, anchors

    def load_payload(self, chart_payload, physics_payload, *, config=None):
        required = {"states", "patches", "cores", "anchors", "n_bodies", "encoder_kind", "selector"}
        missing = required - set(chart_payload)
        if missing:
            raise ValueError(f"competitive_cae chart payload missing {sorted(missing)}")
        return {**chart_payload, **physics_payload}


def encoder_ablation(evidence: ChartFitInput, config: ChartFitConfig, *, work: str, **kwargs):
    """Run the approved encoder comparison on identical physical evidence.

    This is construction-only by design: it compares the chart map and its
    distortion/competence metrics without writing three competing runtime
    artifacts.  No graph, coordinate declaration, or membership label is
    supplied to any arm.
    """
    fitter = CompetitiveCAEFitter()
    reports = {}
    for kind in ("linear", "shallow", "current"):
        options = {**dict(config.options), "encoder_kind": kind}
        arm = replace(config, construction_only=True, options=options,
                      out_report=f"competitive_cae_{kind}_ablation.json")
        reports[kind] = fitter.fit(evidence, arm, work=work, **kwargs)[0]
    return reports


register(CompetitiveCAEFitter)
