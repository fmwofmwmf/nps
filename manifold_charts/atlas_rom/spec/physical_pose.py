"""Physical SE(3) coordinates and distances used before atlas learning.

This module deliberately has no dependency on chart labels, UVs, or declared
coordinates.  It is the only representation admitted to the prior-free
five-bar neighbourhood discovery path.
"""
from __future__ import annotations

import numpy as np


def _so3_log(rotation):
    """Principal rotation vector; stable also for planar five-bar poses."""
    theta = float(np.arccos(np.clip((np.trace(rotation) - 1.0) * .5, -1.0, 1.0)))
    vee = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
    return .5 * vee if theta < 1e-8 else theta * vee / max(2.0 * np.sin(theta), 1e-12)


def se3_lift(reference, twists, translation_scale=1.0):
    """SE3Lift(reference pose + body twists) with rotations valid by construction."""
    ref = np.asarray(reference, float).reshape(-1, 3, 4)
    xi = np.asarray(twists, float).reshape(-1, len(ref), 6)
    out = np.empty((len(xi), len(ref), 3, 4), float)
    for row, state in enumerate(xi):
        for body, (linear, angular) in enumerate(zip(state[:, :3], state[:, 3:])):
            theta = np.linalg.norm(angular)
            skew = np.array(((0., -angular[2], angular[1]),
                             (angular[2], 0., -angular[0]),
                             (-angular[1], angular[0], 0.)))
            rotation = (np.eye(3) + skew if theta < 1e-8 else
                        np.eye(3) + np.sin(theta) / theta * skew
                        + (1. - np.cos(theta)) / theta**2 * (skew @ skew))
            out[row, body, :, :3] = ref[body, :, :3] @ rotation
            out[row, body, :, 3] = ref[body, :, 3] + float(translation_scale) * linear
    return out.reshape(len(xi), -1)


def pose_twists(q, reference=None, translation_scale=1.0):
    """Map packed transforms to reference-relative, length-normalised twists."""
    q = np.asarray(q, float)
    if q.ndim == 1: q = q[None]
    if q.shape[1] % 12: raise ValueError("packed poses must contain 12 values per body")
    poses = q.reshape(len(q), -1, 3, 4)
    ref = poses[0] if reference is None else np.asarray(reference, float).reshape(-1, 3, 4)
    out = np.empty((len(q), poses.shape[1], 6), float)
    scale = max(float(translation_scale), 1e-12)
    for row, state in enumerate(poses):
        for body, transform in enumerate(state):
            base = ref[body]
            out[row, body, :3] = (transform[:, 3] - base[:, 3]) / scale
            out[row, body, 3:] = _so3_log(base[:, :3].T @ transform[:, :3])
    return out.reshape(len(q), -1)


def physical_distance_matrix(q, translation_scale=1.0):
    """RMS body-twist distance under the physical pose metric.

    Computed via the norm-expansion identity (``||a-b||^2 = ||a||^2 +
    ||b||^2 - 2 a.b``, one matmul) rather than broadcasting an explicit
    ``(n, n, 6*n_bodies)`` delta array: at a few thousand samples and
    n_bodies in the tens, that intermediate is tens of gigabytes even
    though the final result is a single (n, n) matrix.
    """
    x = pose_twists(q, translation_scale=translation_scale)
    sq = np.einsum("ij,ij->i", x, x)
    sqdist = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(sqdist, 0.0, out=sqdist)
    d = np.sqrt(sqdist / x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def pose_distance_matrix(q, rotation_radius=1.0):
    """Dense (N, N) translation-plus-rotation-geodesic pose distance.

    ``rotation_radius`` converts a rotation angle (radians) to an equivalent
    translation length, so the two contributions are commensurate; 1.0 is the
    right order of magnitude for these packed body transforms (translations
    are raw, unnormalized world units of the same scale as the mechanism).
    """
    q = np.asarray(q, float)
    n, n_bodies = len(q), q.shape[1] // 12
    T = q.reshape(n, n_bodies, 3, 4)
    R = T[..., :3]              # (n, b, 3, 3)
    t = T[..., 3]                # (n, b, 3)
    # Accumulate over the (small, fixed) body axis instead of broadcasting
    # it into an (n, n, b) intermediate -- at a few thousand samples and
    # n_bodies in the tens that intermediate is tens of gigabytes, times
    # three (dt2/trace/ang), even though the final result is one (n, n)
    # matrix.
    sq2 = np.zeros((n, n))
    for body in range(n_bodies):
        tb = t[:, body, :]
        dt2 = ((tb[:, None] - tb[None, :]) ** 2).sum(-1)        # (n, n)
        Rb = R[:, body]                                          # (n, 3, 3)
        trace = np.einsum("aoi,coi->ac", Rb, Rb)                 # tr(Ra^T Rc)
        ang = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))     # (n, n)
        sq2 += dt2 + (rotation_radius * ang) ** 2
    d = np.sqrt(sq2)
    np.fill_diagonal(d, 0.0)
    return d
