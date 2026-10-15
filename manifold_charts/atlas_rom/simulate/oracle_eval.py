"""oracle_eval.py -- fresh on-manifold evaluation poses from the physical oracle.

Build the case's projection oracle (analytic ideal-joint when supported, else libuipc), draw fresh
configurations by perturbing training samples and projecting them back, and measure the worst
revolute-joint closure gap of a pose set.  Moved unchanged from ``viewers/check_data_validity.py``
so the build pipeline does not depend on viewer code."""
from __future__ import annotations

import numpy as np

from atlas_rom.simulate import ideal_oracle


def _build_oracle(spec):
    if ideal_oracle.supports(spec):
        return ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None)
    from atlas_rom.simulate.oracle import build_oracle
    return build_oracle(spec)


def _oracle_test_poses(q_train, oracle, n, *, seed, sigma, tol=1e-6):
    """``n`` fresh on-manifold configs: perturb a random training sample, project."""
    rng = np.random.default_rng(seed)
    base = q_train[rng.integers(len(q_train), size=max(n * 4, 16))]
    poses = []
    for row in base:
        candidate = row + sigma * rng.standard_normal(row.shape)
        try:
            projected = np.asarray(oracle.project(candidate), float)
        except Exception:
            continue
        if np.linalg.norm(oracle.residual(projected), np.inf) > tol:
            continue
        poses.append(projected)
        if len(poses) >= n:
            break
    if not poses:
        raise RuntimeError("oracle produced no converged poses; lower --sigma")
    return np.asarray(poses, float)


def _worst_joint_gap(spec, q_flat, n_bodies):
    """Max revolute-joint closure gap ``||T_a p_a - T_b p_b||`` per pose.

    Joint pivots are given in the rest world frame; each is pulled into its
    body's local frame with that body's ``initial_transform`` first (identity for
    planar cases, but a real offset for a stacked mechanism), so the gap is 0 on
    exact on-manifold poses regardless of how the bodies are laid out at rest.
    """
    driven = spec.mechanism.driven_bodies
    order = {b.name: i for i, b in enumerate(driven)}
    t0 = []
    for b in driven:
        it = np.asarray(getattr(b, "initial_transform", np.eye(4)), float).reshape(4, 4)
        t0.append(it)
    joints = []
    for j in spec.mechanism.joints:
        if j.kind != "revolute":
            continue
        a = None if j.body_a == "__ground" else order.get(j.body_a)
        b = None if j.body_b == "__ground" else order.get(j.body_b)
        if (a is not None and a >= n_bodies) or (b is not None and b >= n_bodies):
            continue
        piv_w = np.asarray(j.pivot, float)
        pa = piv_w if a is None else t0[a][:3, :3].T @ (piv_w - t0[a][:3, 3])
        pb = piv_w if b is None else t0[b][:3, :3].T @ (piv_w - t0[b][:3, 3])
        joints.append((a, b, pa, pb, piv_w))
    if not joints:
        return np.zeros(len(q_flat))
    T = q_flat.reshape(len(q_flat), n_bodies, 3, 4)
    R, t = T[..., :3, :3], T[..., :3, 3]
    gaps = []
    for a, b, pa, pb, piv_w in joints:
        sa = np.broadcast_to(piv_w, (len(q_flat), 3)) if a is None else np.einsum("nij,j->ni", R[:, a], pa) + t[:, a]
        sb = np.broadcast_to(piv_w, (len(q_flat), 3)) if b is None else np.einsum("nij,j->ni", R[:, b], pb) + t[:, b]
        gaps.append(np.linalg.norm(sa - sb, axis=1))
    return np.max(gaps, axis=0)
