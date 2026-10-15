"""Learned per-chart physical fields for the shipped patch-atlas ROM.

The pose decoder is *not* the dynamics model.  During the build, its Jacobian
and the declared rigid-body data supply M and V teachers; the runtime then uses
these compact learned M/V fields.  The same checkpoint also carries a learned
support classifier and an SDF boundary for each chart, so membership and
physical limits do not silently fall back to anchor-cloud heuristics.
"""
from __future__ import annotations

import numpy as np
import torch

from atlas_rom.charts.latent_support import local_support_radii
import torch.nn as nn


def _mlp(d, out, width=96, layers=3):
    blocks = []
    for a, b in zip([d] + [width] * layers, [width] * layers + [out]):
        blocks.append(nn.Linear(a, b))
        if b != out:
            blocks.append(nn.SiLU())
    return nn.Sequential(*blocks).double()


class MassField(nn.Module):
    """SPD matrix field, parameterized by a Cholesky factor."""
    def __init__(self, d, width=96, layers=3, eps=1e-8):
        super().__init__(); self.d = int(d); self.eps = float(eps)
        self.net = _mlp(d, d * (d + 1) // 2, width, layers)
        # Physical scale of M: the net learns M / scale (O(1)); runtime sees physical units.
        self.register_buffer("scale", torch.ones((), dtype=torch.float64))

    def forward(self, u):
        x = self.net(u if u.ndim == 2 else u.unsqueeze(0))
        L = torch.zeros(len(x), self.d, self.d, dtype=x.dtype, device=x.device)
        tri = torch.tril_indices(self.d, self.d, device=x.device)
        L[:, tri[0], tri[1]] = x
        diag = torch.arange(self.d, device=x.device)
        L[:, diag, diag] = torch.nn.functional.softplus(L[:, diag, diag]) + self.eps
        return self.scale * (L @ L.transpose(-1, -2))

    def normalized(self, u):
        return self(u) / self.scale


class ScalarField(nn.Module):
    def __init__(self, d, width=96, layers=3):
        super().__init__(); self.d = int(d); self.net = _mlp(d, 1, width, layers)
        # value = net(u) * scale + shift.  Identity by default; fit_dynamics sets the physical scale so the
        # net itself learns an O(1) target (V is ~1e6 for heavy bodies) and the force -grad V stays conservative.
        self.register_buffer("shift", torch.zeros((), dtype=torch.float64))
        self.register_buffer("scale", torch.ones((), dtype=torch.float64))
    def forward(self, u): return self.normalized(u) * self.scale + self.shift
    def normalized(self, u): return self.net(u if u.ndim == 2 else u.unsqueeze(0)).squeeze(-1)


def _fit(model, u, target, *, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(int(epochs)):
        opt.zero_grad(); loss = (model(u) - target).square().mean(); loss.backward(); opt.step()
    return float(loss.detach())


def _lbfgs(params, loss_fn, iters):
    """Full-batch float64 L-BFGS finish (the targets are tiny: a few hundred anchors)."""
    if int(iters) <= 0:
        return
    opt = torch.optim.LBFGS(list(params), lr=1.0, max_iter=int(iters), history_size=50,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad(); loss = loss_fn(); loss.backward(); return loss
    opt.step(closure)


def fit_dynamics(model, anchors, masses, inertia, com, up, gravity, *, epochs=800, grad_weight=1.0,
                 lbfgs_iters=300):
    """Distill Jacobian-derived M and gravitational V into learned fields.

    Targets are normalized (the fields carry their physical scale as buffers) so the nets learn O(1)
    quantities whatever the body masses are.  V is trained on its value AND on its gradient, where the
    gradient is ``autograd`` of the SAME scalar network: the learned force ``-grad V`` is therefore exactly
    conservative -- no separate force head.  Adam (``epochs``) is followed by a full-batch L-BFGS finish.
    """
    from atlas_rom.rom import ops
    U = np.asarray(anchors, float); d = U.shape[1]
    M = np.asarray([ops.mass_matrix(model, u, masses, inertia, com) for u in U])
    VG = [ops.potential_and_grad(model, u, masses, com, up, gravity) for u in U]
    V = np.asarray([v for v, _ in VG]); G = np.asarray([g for _, g in VG])
    ut = torch.as_tensor(U, dtype=torch.float64)
    mass, potential = MassField(d), ScalarField(d)
    m_scale = float(np.mean(np.trace(M, axis1=1, axis2=2)) / d)
    v_shift, v_scale = float(V.mean()), float(V.std())
    if not (np.isfinite(m_scale) and m_scale > 0): m_scale = 1.0
    if not (np.isfinite(v_scale) and v_scale > 1e-12): v_scale = 1.0
    with torch.no_grad():
        mass.scale.fill_(m_scale); potential.shift.fill_(v_shift); potential.scale.fill_(v_scale)
    mt = torch.as_tensor(M / m_scale, dtype=torch.float64)
    vt = torch.as_tensor((V - v_shift) / v_scale, dtype=torch.float64)
    gt = torch.as_tensor(G / v_scale, dtype=torch.float64)
    g_rms2 = float(gt.square().sum(-1).mean().clamp_min(1e-12))
    m_den = mt.square().sum((-2, -1)).clamp_min(1e-12)

    def mass_loss():
        return ((mass.normalized(ut) - mt).square().sum((-2, -1)) / m_den).mean()

    def pot_terms():
        x = ut.clone().requires_grad_(True)
        n = potential.normalized(x)
        (gn,) = torch.autograd.grad(n.sum(), x, create_graph=True)
        return (n - vt).square().mean(), (gn - gt).square().sum(-1).mean() / g_rms2

    def pot_loss():
        value, grad = pot_terms()
        return value + float(grad_weight) * grad

    for net, fn in ((mass, mass_loss), (potential, pot_loss)):
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
        for _ in range(int(epochs)):
            opt.zero_grad(); loss = fn(); loss.backward(); opt.step()
        _lbfgs(net.parameters(), fn, lbfgs_iters)
    with torch.enable_grad():
        v_err, g_err = pot_terms()
    return mass.eval(), potential.eval(), {"mass_mse": float(mass_loss().detach()),
                                           "potential_mse": float(v_err.detach()),
                                           "potential_grad_rel_mse": float(g_err.detach())}


def _cloud_samples(anchors, rng):
    U = np.asarray(anchors, float); d = U.shape[1]
    scale = np.maximum(U.std(0), 1e-4)
    inside = np.r_[U, U + rng.normal(size=U.shape) * (0.10 * scale)]
    outside = U.mean(0) + rng.uniform(-2.0, 2.0, size=(max(4 * len(U), 128), d)) * (2.5 * scale)
    return inside, outside


def _neutral_boundary_sdf(d, anchors, margin):
    """A learned field for a sampled manifold with no boundary witnesses.

    A chart's finite encoded cloud is never evidence of a physical boundary:
    another chart may continue the same configuration-space direction.  With no
    *physical* failed-continuation observations, the only defensible SDF is
    strictly interior everywhere.  Keep it in the normal SDFNet format so the
    runtime has one field interface; zero weights make its gradient exactly
    zero, hence it cannot manufacture a chart-edge force.
    """
    from atlas_rom.rom.sdf import SDFNet
    U = np.asarray(anchors, float)
    sdf = SDFNet(d, width=96, layers=3, center=U.mean(0),
                 scale=float(np.std(U) + 1e-9)).double()
    with torch.no_grad():
        for parameter in sdf.parameters():
            parameter.zero_()
        sdf.net[-1].bias.fill_(-float(max(margin, 1e-3)))
    return sdf.eval()


def _fit_margin_sdf(points, margin, *, epochs=800, seed=0, near_band=0.05, device=None):
    """Regress a latent SDF onto the decoded poses' normalized physical margin: ``phi(u) ~= -margin``.

    A smooth continuous target, rather than inside/outside witness classification: witness labels
    interleave in latent space (a chart decodes limit-violating poses between and around its valid codes),
    and a classifier fit to them put the wall inside the legal range.  Points near the zero level set are
    up-weighted so the wall's location is what the fit gets right first.
    """
    from atlas_rom.rom.sdf import SDFNet
    points = np.asarray(points, float); margin = np.asarray(margin, float)
    keep = np.isfinite(margin)
    points, margin = points[keep], np.clip(margin[keep], -1.0, 1.0)
    device = torch.device("cuda" if device is None and torch.cuda.is_available()
                          else ("cpu" if device is None else device))
    sdf = SDFNet(points.shape[1], width=96, layers=3, center=points.mean(0),
                 scale=float(np.maximum(points.std(), 1e-6))).double().to(device)
    torch.manual_seed(int(seed))
    x = torch.as_tensor(points, dtype=torch.float64, device=device)
    target = torch.as_tensor(-margin, dtype=torch.float64, device=device)
    weight = torch.as_tensor(1.0 + 4.0 * np.exp(-(margin / float(near_band)) ** 2), dtype=torch.float64, device=device)
    opt = torch.optim.Adam(sdf.parameters(), lr=2e-3, weight_decay=1e-6)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = (weight * torch.nn.functional.smooth_l1_loss(sdf(x), target, reduction="none", beta=0.01)).sum() / weight.sum()
        loss.backward(); opt.step()
    _lbfgs(sdf.parameters(), lambda: (weight * torch.nn.functional.smooth_l1_loss(
        sdf(x), target, reduction="none", beta=0.01)).sum() / weight.sum(), 200)
    return sdf.eval()


def _fit_witness_sdf(inside, outside, margin, *, epochs=800, seed=0, inside_weight=4.0,
                     outside_weight=1.0, sign_weight=1.0, eikonal_weight=0.25,
                     batch_size=4096, device=None):
    """Fit a signed field from physical inside/outside witnesses.

    The labels establish only the side of the wall; the Eikonal term makes the
    resulting level set behave as a distance field instead of a classifier.
    """
    from atlas_rom.rom.sdf import SDFNet

    inside = np.asarray(inside, float)
    outside = np.asarray(outside, float)
    points = np.vstack([inside, outside])
    device = torch.device("cuda" if device is None and torch.cuda.is_available()
                          else ("cpu" if device is None else device))
    center = points.mean(0)
    scale = float(np.maximum(points.std(), 1e-6))
    sdf = SDFNet(points.shape[1], width=96, layers=3,
                 center=center, scale=scale).double().to(device)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(points, dtype=torch.float64, device=device)
    target = torch.as_tensor(
        np.r_[-float(margin) * np.ones(len(inside)),
              2.0 * float(margin) * np.ones(len(outside))], dtype=torch.float64,
        device=device)
    inside_weight = np.asarray(inside_weight, float)
    if inside_weight.ndim == 0:
        inside_weight = np.full(len(inside), float(inside_weight))
    if inside_weight.shape != (len(inside),):
        raise ValueError("inside_weight must be scalar or have one value per inside witness")
    outside_weight = np.asarray(outside_weight, float)
    if outside_weight.ndim == 0:
        outside_weight = np.full(len(outside), float(outside_weight))
    if outside_weight.shape != (len(outside),):
        raise ValueError("outside_weight must be scalar or have one value per outside witness")
    weights = torch.as_tensor(np.r_[inside_weight, outside_weight], dtype=torch.float64,
                              device=device)
    lo, hi = points.min(0), points.max(0)
    opt = torch.optim.Adam(sdf.parameters(), lr=2e-3, weight_decay=1e-5)
    for _ in range(int(epochs)):
        opt.zero_grad()
        if len(x) > int(batch_size):
            take = torch.randint(len(x), (int(batch_size),), device=device)
            xb, target_b, weights_b = x[take], target[take], weights[take]
        else:
            xb, target_b, weights_b = x, target, weights
        value = sdf(xb)
        sign_loss = (weights_b * torch.nn.functional.smooth_l1_loss(
            value, target_b, reduction="none")).sum() / weights_b.sum()
        eik = torch.as_tensor(
            rng.uniform(lo, hi, size=(min(512, max(64, len(points))), points.shape[1])),
            dtype=torch.float64, device=device).requires_grad_(True)
        phi = sdf(eik)
        (grad,) = torch.autograd.grad(phi.sum(), eik, create_graph=True)
        eikonal_loss = (grad.norm(dim=-1) - 1.0).square().mean()
        loss = float(sign_weight) * sign_loss + float(eikonal_weight) * eikonal_loss
        loss.backward(); opt.step()
    return sdf.eval()


def fit_constraint_radius_sdf(model, anchors, validity, *, radius, constraint_tol,
                              probes_per_anchor=32, epochs=800, seed=0):
    """Fit an SDF from local decoded-constraint witnesses only.

    Every supplied anchor is an inside witness, irrespective of its measured
    residual.  Around each anchor we draw a ball of radius ``radius`` in that
    chart's latent coordinates, decode the probe, and evaluate the mechanism's
    physical residual.  A probe is outside exactly when its largest RMS
    constraint error exceeds ``constraint_tol``.  The anchor cloud is used
    only while creating the offline witnesses; the returned SDF is the sole
    runtime representation and performs no nearest-anchor lookup.

    ``validity`` deliberately has the same tiny interface used by the patch
    trainer: it can therefore cover arbitrary mechanisms with a differentiable
    residual evaluator, not merely Sarrus linkages.
    """
    U = np.asarray(anchors, float)
    if U.ndim != 2 or len(U) == 0:
        raise ValueError("anchors must have shape (n, latent_dimension)")
    if validity is None:
        raise ValueError("constraint-radius SDF requires a physical residual evaluator")
    radius = float(radius)
    constraint_tol = float(constraint_tol)
    if radius <= 0.0 or constraint_tol < 0.0:
        raise ValueError("radius must be positive and constraint_tol non-negative")

    rng = np.random.default_rng(seed)
    d = U.shape[1]
    # Uniform-volume ball samples.  Independent directions make this a local
    # physical test instead of a topology or sample-membership declaration.
    direction = rng.normal(size=(len(U), int(probes_per_anchor), d))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-12)
    radial = rng.random((len(U), int(probes_per_anchor), 1)) ** (1.0 / d)
    probes = (U[:, None, :] + radius * radial * direction).reshape(-1, d)
    device = next(model.parameters()).device
    x = torch.as_tensor(probes, dtype=torch.float64, device=device)
    with torch.no_grad():
        packed = model.decode(x)
        transforms = packed.reshape(len(x), model.n_bodies, 3, 4)
        residual = validity.residual(packed, transforms)
    if residual is None:
        raise ValueError("physical residual evaluator returned None")
    residual = residual.detach().sqrt().amax(-1).cpu().numpy()
    probe_inside = probes[residual <= constraint_tol]
    probe_outside = probes[residual > constraint_tol]
    if len(probe_outside) < max(8, 4 * d):
        raise RuntimeError("constraint-radius SDF found too few violating probes; "
                           "increase radius, probes_per_anchor, or lower constraint_tol")
    # The contract intentionally starts with all anchors *and all locally
    # constraint-valid probes* inside.  Do not discard valid probes: they are
    # exactly the evidence needed to prevent an interior false-positive force.
    inside = np.r_[U, probe_inside]
    margin = max(1e-6, 0.05 * radius)
    # Anchors are the explicit contract: all are inside even if the existing
    # decoder has a nonzero residual there.  Give them sufficient weight to
    # preserve that contract while fitting the sampled local classification.
    weights = np.r_[12.0 * np.ones(len(U)), np.ones(len(probe_inside))]
    outside_weight = float(weights.sum() / len(probe_outside))
    sdf = _fit_witness_sdf(inside, probe_outside, margin, epochs=epochs, seed=seed,
                           inside_weight=weights, outside_weight=outside_weight,
                           sign_weight=50.0, eikonal_weight=0.02, device=device)
    sdf._witness_points = np.vstack([inside, probe_outside])
    sdf._witness_kind = np.r_[np.zeros(len(inside), dtype=int),
                               np.ones(len(probe_outside), dtype=int)]
    return sdf, {"boundary_mode": "constraint_radius_witnesses",
                 "radius": radius, "constraint_tol": constraint_tol,
                 "anchors_forced_inside": int(len(U)),
                 "probe_inside": int(len(probe_inside)),
                 "probe_outside": int(len(probe_outside)),
                 "max_probe_constraint_error": float(residual.max()),
                 "boundary_margin": margin}


def fit_constraint_energy_sdf(model, accepted_u, validity, *, constraint_tol=0.05,
                              probes_per_anchor=16, epochs=1200, seed=0):
    """Regress decoded physical constraint energy directly in latent space.

    Unlike a cloud-support classifier, this field is supervised by the
    decoded mechanism: closure error is the continuous energy and a declared
    joint-limit violation receives a positive penalty.  The accepted states
    set the zero-level calibration, but are never queried at runtime.
    """
    U = np.asarray(accepted_u, float)
    if U.ndim != 2 or len(U) == 0:
        raise ValueError("accepted_u must have shape (n, latent_dimension)")
    if validity is None:
        raise ValueError("constraint-energy SDF requires a physical residual evaluator")
    rng = np.random.default_rng(seed)
    d = U.shape[1]
    span = np.maximum(U.max(0) - U.min(0), 0.1)
    # Mix local probes with an enclosing box so the learned energy has a
    # meaningful off-manifold gradient instead of merely memorising anchors.
    local = U[:, None, :] + rng.normal(size=(len(U), int(probes_per_anchor), d)) * (0.20 * span)
    n_box = max(4096, len(U) * int(probes_per_anchor))
    box = rng.uniform(U.min(0) - 0.35 * span, U.max(0) + 0.35 * span, size=(n_box, d))
    X = np.vstack([U, local.reshape(-1, d), box])
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float64, device=device)
        packed = model.decode(x)
        transforms = packed.reshape(len(x), model.n_bodies, 3, 4)
        closure = validity.residual(packed, transforms).sqrt().amax(-1).cpu().numpy()
        limit_bad = (validity.joint_limit_invalid(transforms)
                     if hasattr(validity, "joint_limit_invalid") else np.zeros(len(X), bool))
    # A violated physical stop is outside even if a closed-loop decoder still
    # has negligible attachment residual.  Give it a finite energy above the
    # tolerated closure level so it defines a usable smooth wall.
    energy = np.maximum(closure, np.where(limit_bad, 2.0 * float(constraint_tol), 0.0))
    n_accepted = len(U)
    threshold = float(energy[:n_accepted].max() + max(1e-3, 0.05 * constraint_tol))
    target = energy - threshold
    from atlas_rom.rom.sdf import SDFNet
    sdf = SDFNet(d, width=96, layers=3, center=U.mean(0),
                 scale=float(np.std(U) + 1e-9)).double().to(device)
    xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(target, dtype=torch.float64, device=device)
    weight = torch.ones(len(X), dtype=torch.float64, device=device)
    weight[:n_accepted] = 40.0
    opt = torch.optim.Adam(sdf.parameters(), lr=2e-3, weight_decay=1e-5)
    for _ in range(int(epochs)):
        opt.zero_grad()
        pred = sdf(xt)
        loss = (weight * torch.nn.functional.smooth_l1_loss(pred, yt, reduction="none")).sum() / weight.sum()
        loss.backward(); opt.step()
    # Finite regression error must never put an accepted configuration on the
    # force-producing side of the learned energy threshold.
    with torch.no_grad():
        max_accepted = float(sdf(xt[:n_accepted]).max())
        sdf.net[-1].bias.sub_(max_accepted + max(1e-3, 0.02 * constraint_tol))
    return sdf.eval(), {"boundary_mode": "learned_constraint_energy",
                        "accepted_samples": int(n_accepted),
                        "probes": int(len(X) - n_accepted),
                        "constraint_tol": float(constraint_tol),
                        "energy_threshold": threshold,
                        "limit_violations": int(np.count_nonzero(limit_bad)),
                        "max_constraint_energy": float(energy.max())}


def fit_local_support_sdf(anchors, *, radii=None, probes_per_anchor=32, epochs=1200, seed=0):
    """Fit a conservative learned support SDF around locally valid samples.

    This is intentionally a support field, not a physical-limit classifier:
    anchors and a small ball around every anchor are inside; each ball uses the
    same per-anchor 8-NN radius as atlas certification.  A point is outside
    when it lies beyond the radius of its nearest anchor.  The cloud is used
    only to create offline supervision.  The returned SDF is a smooth learned
    runtime field and contains no nearest-neighbour or membership query.
    """
    from scipy.spatial import cKDTree

    U = np.asarray(anchors, float)
    if U.ndim != 2 or len(U) == 0:
        raise ValueError("anchors must have shape (n, latent_dimension)")
    radii = local_support_radii(U) if radii is None else np.asarray(radii, float)
    if radii.shape != (len(U),):
        raise ValueError("radii must have one positive value per anchor")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii must be finite and positive")
    rng = np.random.default_rng(seed)
    d = U.shape[1]
    direction = rng.normal(size=(len(U), int(probes_per_anchor), d))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-12)
    # Positive witnesses stay strictly inside each anchor's certified local
    # support ball.  A global radius fills latent holes in dense regions.
    radial = (0.5 * radii[:, None, None] *
              rng.random((len(U), int(probes_per_anchor), 1)) ** (1.0 / d))
    inside = np.r_[U, (U[:, None, :] + radial * direction).reshape(-1, d)]
    tree = cKDTree(U)
    extent = float(radii.max())
    lo, hi = U.min(0) - 1.25 * extent, U.max(0) + 1.25 * extent
    outside_parts, need = [], len(inside)
    while need > 0:
        candidates = rng.uniform(lo, hi, size=(max(4 * need, 256), d))
        distance, nearest = tree.query(candidates, k=1)
        far = candidates[distance > radii[nearest]]
        outside_parts.append(far)
        need -= len(far)
    outside = np.vstack(outside_parts)[:len(inside)]
    # The provided valid samples are a hard no-force contract.  The jitter
    # fills their local support, but cannot outweigh the samples themselves.
    weights = np.r_[100.0 * np.ones(len(U)), np.ones(len(inside) - len(U))]
    typical_radius = float(np.median(radii))
    margin = max(1e-6, 0.10 * typical_radius)
    sdf = _fit_witness_sdf(inside, outside, margin, epochs=epochs, seed=seed,
                           inside_weight=weights, outside_weight=1.0,
                           sign_weight=30.0, eikonal_weight=0.02)
    # Calibrate the learned zero level to the supplied valid samples.  This is
    # a field-level offset, not a runtime lookup: no accepted sample may begin
    # on the force-producing side merely from finite fitting error.
    with torch.no_grad():
        device = next(sdf.parameters()).device
        worst_inside = float(sdf(torch.as_tensor(U, dtype=torch.float64, device=device)).max())
        slack = 0.01 * typical_radius
        sdf.net[-1].bias.sub_(worst_inside + slack)
    return sdf, {"boundary_mode": "local_support_witnesses",
                 "support_radius_min": float(radii.min()),
                 "support_radius_median": typical_radius,
                 "support_radius_max": float(radii.max()),
                 "anchors_forced_inside": int(len(U)), "inside_witnesses": int(len(inside)),
                 "outside_witnesses": int(len(outside)), "boundary_margin": margin}


def _limit_crossing_pairs(model, anchors, candidates, validity, *, seed=0, max_pairs=1024):
    """Bisect physical joint-limit witnesses into local valid/invalid pairs."""
    if not hasattr(validity, "joint_limit_invalid") or len(candidates) == 0:
        return np.empty((0, anchors.shape[1])), np.empty((0, anchors.shape[1]))
    rng = np.random.default_rng(seed)
    take = min(len(candidates), int(max_pairs))
    ids = rng.choice(len(candidates), size=take, replace=False)
    inside, outside = [], []
    for i in ids:
        hi = np.asarray(candidates[i], float)
        j = int(np.argmin(np.linalg.norm(anchors - hi, axis=1)))
        lo = np.asarray(anchors[j], float)
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            with torch.no_grad():
                q_mid = model.decode(torch.as_tensor(mid, dtype=torch.float64).unsqueeze(0))
            T_mid = q_mid.reshape(1, model.n_bodies, 3, 4).detach().cpu().numpy()
            bad = bool(np.asarray(validity.joint_limit_invalid(T_mid), dtype=bool)[0])
            if bad:
                hi = mid
            else:
                lo = mid
        inside.append(lo); outside.append(hi)
    return np.asarray(inside), np.asarray(outside)


def fit_support_and_boundary(model, anchors, validity, *, epochs=800, seed=0,
                             inverse_tol=0.05, validity_tol=0.1,
                             transition_positive=None, transition_negative=None,
                             local_map_negative=None,
                             support_reference_q=None, support_interior_q=None):
    """Fit chart-local support and a physical-boundary SDF.

    The support classifier learns a calibrated inverse of decoded full-space
    reconstruction error over the sampler's generic configuration set, plus
    transition replay trust.  ``local_map_negative`` contains encoded physical
    poses from the nearest witnessed graph rings that chart construction kept
    outside this certified support.  They bound runtime authority even when a
    locally accurate decoder produces no reconstruction failures.  All labels
    are training supervision only; runtime uses the learned field.

    Physical boundary observations must come from failed continuation/closure
    attempts.  That witness contract has not yet been emitted by the sampler,
    so an atlas without such evidence receives a neutral, strictly-negative
    SDF rather than an invented wall.  The result is still a learned field with
    the normal runtime interface, but it has zero force until real boundary
    witnesses are available.
    """
    rng = np.random.default_rng(seed); U = np.asarray(anchors, float); d = U.shape[1]
    _inside, candidates = _cloud_samples(U, rng)
    with torch.no_grad():
        x_probe = torch.as_tensor(candidates, dtype=torch.float64)
        q_probe = model.decode(x_probe)
        inv = torch.linalg.vector_norm(model.encode(q_probe) - x_probe, dim=-1).cpu().numpy()
        invalid = inv > float(inverse_tol)
        if validity is not None:
            transforms_probe = q_probe.reshape(len(q_probe), model.n_bodies, 3, 4)
            residual = validity.residual(q_probe, transforms_probe)
            if residual is not None:
                invalid |= residual.detach().sqrt().amax(-1).cpu().numpy() > float(validity_tol)
            if hasattr(validity, "joint_limit_invalid"):
                limit_invalid = np.asarray(validity.joint_limit_invalid(transforms_probe), dtype=bool)
                invalid |= limit_invalid
            else:
                limit_invalid = np.zeros(len(candidates), dtype=bool)
        else:
            limit_invalid = np.zeros(len(candidates), dtype=bool)
    probe_outside = candidates[invalid]
    limit_witnesses = int(np.count_nonzero(limit_invalid & invalid))
    replay_positive = (np.empty((0, d)) if transition_positive is None else
                       np.asarray(transition_positive, float).reshape(-1, d))
    replay_negative = (np.empty((0, d)) if transition_negative is None else
                       np.asarray(transition_negative, float).reshape(-1, d))
    map_negative = (np.empty((0, d)) if local_map_negative is None else
                    np.asarray(local_map_negative, float).reshape(-1, d))
    outside = np.r_[probe_outside, replay_negative, map_negative]
    if len(outside) < max(8, U.shape[1] * 4):
        raise RuntimeError("cannot train a learned domain/boundary field: probe set contains "
                           "too few reconstruction/validity failures")
    inside = np.r_[U, replay_positive]
    x_parts = [inside, outside]
    y_parts = [np.ones(len(inside)), np.zeros(len(outside))]
    # Anchor-cloud and synthetic-invalid labels remain useful regularizers,
    # but must not override measured full-space reliability.  Replay labels
    # and the all-sample reconstruction targets are the support definition.
    w_parts = [np.r_[.05 * np.ones(len(U)), 5.0 * np.ones(len(replay_positive))],
               np.r_[.05 * np.ones(len(probe_outside)),
                     5.0 * np.ones(len(replay_negative)),
                     5.0 * np.ones(len(map_negative))]]
    reconstruction_tol = None
    if support_reference_q is not None:
        Q_ref = torch.as_tensor(np.asarray(support_reference_q, float), dtype=torch.float64)
        Q_interior = (Q_ref if support_interior_q is None else
                      torch.as_tensor(np.asarray(support_interior_q, float), dtype=torch.float64))
        with torch.no_grad():
            U_ref = model.encode(Q_ref)
            ref_err = torch.sqrt(((model.decode(U_ref) - Q_ref) ** 2).mean(-1)).cpu().numpy()
            interior_err = torch.sqrt(((model.decode(model.encode(Q_interior)) - Q_interior) ** 2).mean(-1)).cpu().numpy()
        # The chart's own well-reconstructed samples establish the zero-logit
        # boundary.  Soft targets preserve error ordering instead of turning a
        # continuous reliability signal into an arbitrary hard membership cut.
        reconstruction_tol = max(1e-8, float(np.quantile(interior_err, .95)))
        softness = max(1e-8, .25 * reconstruction_tol)
        y_ref = 1.0 / (1.0 + np.exp(np.clip((ref_err - reconstruction_tol) / softness, -60, 60)))
        x_parts.append(U_ref.cpu().numpy()); y_parts.append(y_ref)
        w_parts.append(10.0 * np.ones(len(y_ref)))
    x = torch.as_tensor(np.concatenate(x_parts, axis=0), dtype=torch.float64)
    y = torch.as_tensor(np.concatenate(y_parts, axis=0), dtype=torch.float64)
    weight = torch.as_tensor(np.concatenate(w_parts, axis=0), dtype=torch.float64)
    support = ScalarField(d); opt = torch.optim.Adam(support.parameters(), lr=2e-3)
    support_loss = None
    for _ in range(int(epochs)):
        opt.zero_grad()
        per_example = torch.nn.functional.binary_cross_entropy_with_logits(support(x), y, reduction="none")
        loss = (weight * per_example).sum() / weight.sum()
        loss.backward(); opt.step(); support_loss = float(loss.detach())
    margin = 0.05 * float(np.median(np.linalg.norm(U[:, None] - U[None, :], axis=-1)[np.triu_indices(len(U), 1)]))
    if limit_witnesses:
        # The physical boundary is a declared joint stop / workspace bound.  Its SDF is distilled by
        # REGRESSION onto the decoded poses' normalized signed physical margin (a build-time label from the
        # mechanism's declared bounds; the runtime sees only the learned field).  Witness classification put
        # the wall inside the legal range: stephenson's ROM came to rest at die_slot -0.05 against a stop
        # declared at -0.086, with phi > 0 on 16-23% of valid sampler poses.
        from atlas_rom.spec.physical_domain import signed_margin
        interior_u = np.empty((0, d))
        if support_interior_q is not None and len(support_interior_q):
            with torch.no_grad():
                interior_u = model.encode(torch.as_tensor(np.asarray(support_interior_q, float),
                                                          dtype=torch.float64)).cpu().numpy()
        sdf_points = np.r_[U, _inside, candidates, interior_u]
        with torch.no_grad():
            decoded = model.decode(torch.as_tensor(sdf_points, dtype=torch.float64))
            decoded = decoded.reshape(len(sdf_points), model.n_bodies, 3, 4).cpu().numpy()
        spec = validity.case_spec
        sdf_margin = np.asarray([signed_margin(spec, T) for T in decoded], float)
        sdf = _fit_margin_sdf(sdf_points, sdf_margin, epochs=int(epochs), seed=seed)
        sdf._witness_points = sdf_points
        sdf._witness_kind = (sdf_margin < 0).astype(int)
    else:
        sdf = _neutral_boundary_sdf(d, U, margin)
    return support.eval(), sdf, {"support_bce": support_loss,
                                 "boundary_margin": float(margin),
                                  "transition_support_positive": int(len(replay_positive)),
                                  "transition_support_negative": int(len(replay_negative)),
                                  "local_map_support_negative": int(len(map_negative)),
                                 "support_reconstruction_tol": reconstruction_tol,
                                 "boundary_witnesses": int(len(outside)),
                                 "joint_limit_witnesses": limit_witnesses,
                                 "boundary_mode": ("joint_limit_witnesses"
                                                   if limit_witnesses else
                                                   "neutral_no_physical_witness")}


def enforce_support_coverage(models, supports, anchors, reference_q, *, epochs=800,
                             owner_margin=0.5, owner_weight=12.0,
                             retention_weight=0.25):
    """Jointly enforce the learned-domain contract on valid sampler samples.

    Every chart must positively claim its own encoded anchors.  The union
    condition is a *gate*, not a loss: the certified patch cover already gives
    every valid sample an owner.  Training a global max/union objective lets a
    foreign chart claim a state it cannot receive through a low-error
    transition, which creates exactly the wrong-branch handoffs this field is
    meant to prevent.  A small score-retention term preserves the individual
    reconstruction/replay calibration while owner margins are repaired.
    """
    if len(models) != len(supports) or len(models) != len(anchors):
        raise ValueError("models, supports, and anchors must have matching chart counts")
    Q = torch.as_tensor(np.asarray(reference_q, float), dtype=torch.float64)
    with torch.no_grad():
        reference_u = [model.encode(Q) for model in models]
        owner_u = [torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
                   for a in anchors]

    def margins():
        owner = torch.cat([field(u) for field, u in zip(supports, owner_u)])
        scores = torch.stack([field(u) for field, u in zip(supports, reference_u)])
        # A small-temperature logsumexp is a differentiable max.  Unlike an
        # ordinary logsumexp it cannot report a positive union when every
        # chart score is still negative.
        union = 0.05 * torch.logsumexp(scores / 0.05, dim=0)
        return owner, union

    with torch.no_grad():
        owner_before, union_before = margins()
        reference_scores_before = torch.stack([
            field(u) for field, u in zip(supports, reference_u)]).detach()
    opt = torch.optim.Adam([p for field in supports for p in field.parameters()], lr=1e-3)
    for _ in range(int(epochs)):
        opt.zero_grad()
        owner, union = margins()
        scores = torch.stack([field(u) for field, u in zip(supports, reference_u)])
        owner_loss = torch.nn.functional.softplus(owner_margin - owner).square().mean()
        retention_loss = (scores - reference_scores_before).square().mean()
        loss = owner_weight * owner_loss + retention_weight * retention_loss
        loss.backward(); opt.step()
    with torch.no_grad():
        owner_after, union_after = margins()
    return {"owner_support_min_before": float(owner_before.min()),
            "owner_support_min_after": float(owner_after.min()),
            "owner_support_nonpositive_after": int((owner_after <= 0).sum()),
            "union_support_nonpositive_before": int((union_before <= 0).sum()),
            "union_support_nonpositive_after": int((union_after <= 0).sum()),
            "union_support_min_after": float(union_after.min())}


def state(model):
    return {"d": model.d, "state_dict": model.state_dict()}


def _with_defaults(state_dict, **defaults):
    """Artifacts written before the fields carried normalization buffers lack them: identity."""
    out = dict(state_dict)
    for k, v in defaults.items():
        out.setdefault(k, torch.as_tensor(v, dtype=torch.float64))
    return out
def load_mass(entry):
    m = MassField(entry["d"]); m.load_state_dict(_with_defaults(entry["state_dict"], scale=1.0)); return m.eval()
def load_scalar(entry):
    m = ScalarField(entry["d"]); m.load_state_dict(_with_defaults(entry["state_dict"], shift=0.0, scale=1.0)); return m.eval()
def load_sdf(entry):
    from atlas_rom.rom.sdf import SDFNet
    e = entry; m = SDFNet(e["d"], width=e["width"], layers=e["layers"], center=e["center"], scale=e["scale"]).double()
    m.load_state_dict(e["state_dict"]); return m.eval()
def sdf_state(model):
    weights = [v for k, v in model.state_dict().items() if k.endswith(".weight")]
    return {"d": model.d, "width": int(weights[0].shape[0]), "layers": len(weights) - 1,
            "center": model.center.detach().cpu().numpy(), "scale": float(1 / model.iscale),
            "state_dict": model.state_dict()}
