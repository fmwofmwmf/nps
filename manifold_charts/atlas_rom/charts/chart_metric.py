"""Reference-relative physical-pose metric and its Jacobian/gauge diagnostics.

Any chart decoder -- something with a ``transforms(u) -> (..., n_bodies, 3, 4)``
method -- can be scored on the SAME footing with these.  Translation deltas
and a chart-relative rotation vector are commensurate physical units (world
length, radians), unlike raw Gram-Schmidt columns or an internal ambient
feature vector, so a condition number computed here means "how
anisotropically does this chart stretch the physical mechanism" and not "how
are this particular architecture's internal coordinates scaled" -- and it is
computed identically for every chart construction, so numbers are comparable
across them.

The Jacobian alone still carries an ungauged latent affine ambiguity: any
invertible linear change of ``u`` changes the condition number without
touching reconstruction, closure, or injectivity.  :func:`gauge_fixed`
removes exactly that ambiguity by whitening against the average pullback
metric, so what is left is the chart's own anisotropy.
"""
from __future__ import annotations

import numpy as np


def rotation_log(R):
    """Principal ``SO(3) -> so(3)``, vectorised, valid for ``||omega|| < pi``.

    Same construction as ``invertible_chart_ae.rotation_log`` -- kept here,
    not imported from there, so this module has no dependency on any one
    chart architecture and can score all of them the same way.
    """
    import torch
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1) / 2).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    v = torch.stack((R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    sin_theta = torch.sin(theta).clamp_min(1e-8)
    general = v * (theta / (2 * sin_theta)).unsqueeze(-1)
    small = v / 2
    return torch.where((theta < 1e-4).unsqueeze(-1), small, general)


def physical_twists(transforms, reference):
    """``(..., n_bodies, 3, 4)`` torch transforms -> ``(..., 6*n_bodies)`` twists.

    Reference-relative: translation delta in world units, rotation as the
    chart-relative principal log.  Same convention as
    ``physical_pose.pose_twists`` uses for its numpy targets, so torch
    predictions and numpy labels are directly comparable.
    """
    import torch
    dt = transforms[..., 3] - reference[..., 3]
    relative = reference[..., :3].transpose(-1, -2) @ transforms[..., :3]
    omega = rotation_log(relative)
    return torch.cat((dt, omega), dim=-1).reshape(*transforms.shape[:-3], -1)


def twist_jacobian(decode_transforms, u, reference):
    """``d(physical_twists)/du``, shape ``(n, 6*n_bodies, d)``.

    Exact forward-mode autodiff of whatever ``decode_transforms(u) ->
    (n_bodies, 3, 4)`` the caller's chart provides -- ``d`` is 2 here, so this
    is two tangent propagations through the whole decoder, not a finite
    difference.
    """
    from torch.func import jacfwd, vmap

    def one(uu):
        return physical_twists(decode_transforms(uu.unsqueeze(0)), reference).squeeze(0)

    flat = u.reshape(-1, u.shape[-1])
    J = vmap(jacfwd(one))(flat)
    return J.reshape(*u.shape[:-1], -1, u.shape[-1])


def gauge_fixed(J):
    """Whiten the latent's affine gauge by the mean pullback metric ``G``.

    ``G = mean_n J_n^T J_n``; with ``G = L L^T`` (Cholesky), ``J' = J L^{-T}``
    satisfies ``mean_n J'^T_n J'_n = I`` exactly.  A condition number computed
    on ``J'`` cannot be shrunk by re-scaling or rotating ``u`` -- that freedom
    has already been used to make the AVERAGE isometric, so what remains is
    how much any one sample's chart deviates from that average.
    """
    import torch
    # NOT "nij,nik->njk" -- that keeps the sample axis on the output, i.e. it
    # whitens every sample by ITS OWN Gram matrix, which makes every
    # per-sample condition trivially exactly 1.0 regardless of the chart's
    # actual geometry.  Summing `n` away is what makes `G` the AVERAGE
    # pullback metric that a single shared affine gauge can whiten against.
    G = torch.einsum("nij,nik->jk", J, J) / J.shape[0]
    L = torch.linalg.cholesky(G)
    return J @ torch.linalg.inv(L).transpose(-1, -2)


def condition_report(J):
    """median / p95 / max condition, raw and gauge-fixed, plus where the
    worst case occurs -- a single max hides whether a chart is uniformly
    stiff or has one pathological vertex, and the raw number alone cannot be
    told apart from an ungauged latent rescaling."""
    import torch
    singular = torch.linalg.svdvals(J)
    sigma_min = singular[..., -1].clamp_min(1e-30)
    condition = (singular[..., 0] / sigma_min).numpy()
    gauge_singular = torch.linalg.svdvals(gauge_fixed(J))
    gauge_condition = (gauge_singular[..., 0]
                       / gauge_singular[..., -1].clamp_min(1e-30)).numpy()
    return {
        "sigma_min_median": float(singular[..., -1].median()),
        "sigma_min_min": float(singular[..., -1].min()),
        "condition_median": float(np.median(condition)),
        "condition_p95": float(np.percentile(condition, 95)),
        "condition_max": float(condition.max()),
        "condition_argmax": int(np.argmax(condition)),
        "gauge_condition_median": float(np.median(gauge_condition)),
        "gauge_condition_p95": float(np.percentile(gauge_condition, 95)),
        "gauge_condition_max": float(gauge_condition.max()),
        "gauge_condition_argmax": int(np.argmax(gauge_condition)),
    }


def medoid_reference(q, n_bodies):
    """The chart's medoid sample, as a ``(n_bodies, 3, 4)`` reference pose.

    The medoid (not sample 0) minimises the worst-case relative rotation any
    other sample in the chart has to be expressed against, which is what
    keeps ``rotation_log`` away from its ``pi`` cut for as much of the chart
    as the data allows.
    """
    from atlas_rom.spec.physical_pose import physical_distance_matrix
    q = np.asarray(q, float)
    medoid = int(np.argmin(physical_distance_matrix(q).sum(axis=1)))
    return q[medoid].reshape(n_bodies, 3, 4), medoid
