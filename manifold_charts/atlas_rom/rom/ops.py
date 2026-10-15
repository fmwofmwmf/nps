"""ops.py — reduced mass/potential/actuator operators for a patch-atlas chart.

Reuses ``patch_atlas.py``'s JVP-per-basis-direction pattern (``d`` forward
JVPs of the decoder, never a dense per-sample Jacobian) for the mass matrix,
and reverse-mode autodiff (a VJP in spirit -- one scalar-output backward pass
per quantity) for potential and actuator forces.  The mass matrix is NOT the
naive ``J^T M_flat J`` pullback of a fixed ambient metric: the decoder's flat
output packs rotation-matrix COLUMNS, whose velocity relates to angular
velocity by a pose-dependent linear map (``dR = [w]x R``), so the physically
correct per-body rotational metric has to be rebuilt from the current pose
every call.  This mirrors ``dynamics.py::ReducedDynamics.mass`` exactly,
differentiating with JVPs instead of the per-component autograd loop there.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp


def joint_angle_torch(transforms, joint, body_order):
    """Signed joint angle from a torch (..., n_bodies, 3, 4) transform stack.

    Differentiable, so ``dtheta/du`` for an actuator's generalized force comes
    straight from autodiff through the decoder -- no per-case kinematics.
    Uses ``atan2(axis.w, trace-1)`` rather than ``arccos``, which is
    numerically flat (and gradient-singular) exactly at theta=0, where a rest
    pose sits.  Ported in from the archived ``atlas_rom_experiments/
    coordinates.py`` -- this is the one piece of it still load-bearing for a
    production actuator map, so it lives here rather than reviving the whole
    (much larger) archived module.
    """
    sample = transforms[..., 0, :, :3]
    identity = torch.eye(3, dtype=sample.dtype,
                         device=sample.device).expand_as(sample)

    def rotation(name):
        if name is None or name not in body_order:
            return identity
        return transforms[..., body_order.index(name), :, :3]

    Ra, Rb = rotation(joint.body_a), rotation(joint.body_b)
    R = Ra.transpose(-1, -2) @ Rb
    axis = torch.as_tensor(np.asarray(joint.axis, float), dtype=Ra.dtype,
                           device=Ra.device)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin2 = (w * axis).sum(-1)
    cos2 = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0
    return torch.atan2(sin2, cos2)


def transforms(model, u) -> torch.Tensor:
    """Decoded (n_bodies, 3, 4) pose at ``u``, no grad."""
    u_t = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
    with torch.no_grad():
        return model.transforms(u_t).squeeze(0)


def mass_matrix(model, u, masses, inertia, com) -> np.ndarray:
    """M(u) = Jp^T diag(m) Jp + Jw^T (R I R^T) Jw, Jacobians via d forward JVPs."""
    u_t = torch.as_tensor(np.asarray(u, float))
    d = u_t.shape[-1]

    def transforms_flat(uu):
        return model.transforms(uu.unsqueeze(0)).squeeze(0)      # (n_bodies,3,4)

    with torch.no_grad():
        T = transforms_flat(u_t)
        cols = []
        for k in range(d):
            basis = torch.zeros(d, dtype=u_t.dtype)
            basis[k] = 1.0
            _, col = jvp(transforms_flat, (u_t,), (basis,))
            cols.append(col)
        dT = torch.stack(cols, dim=-1)                            # (n_bodies,3,4,d)

        R = T[..., :3]
        dR = dT[..., :3, :]
        dt = dT[..., 3, :]
        com_t = torch.as_tensor(np.asarray(com, float), dtype=T.dtype)
        m_t = torch.as_tensor(np.asarray(masses, float), dtype=T.dtype)
        I_t = torch.as_tensor(np.asarray(inertia, float), dtype=T.dtype)

        Jp = torch.einsum("nrck,nc->nrk", dR, com_t) + dt          # (n,3,d)
        M = torch.einsum("n,nrk,nrl->kl", m_t, Jp, Jp)
        Omega = torch.einsum("nrck,nsc->nrsk", dR, R)               # (n,3,3,d)
        Jw = torch.stack([Omega[:, 2, 1, :], Omega[:, 0, 2, :],
                          Omega[:, 1, 0, :]], dim=1)                # (n,3,d)
        I_world = torch.einsum("nij,njk,nlk->nil", R, I_t, R)
        M = M + torch.einsum("nik,nij,njl->kl", Jw, I_world, Jw)
        M = 0.5 * (M + M.T)
    return M.numpy()


def potential_and_grad(model, u, masses, com, up, gravity) -> tuple[float, np.ndarray]:
    """(V, dV/du) for V = sum_b m_b * g * (com_b_world . up)."""
    u_t = torch.as_tensor(np.asarray(u, float), dtype=torch.float64).requires_grad_(True)
    T = model.transforms(u_t.unsqueeze(0)).squeeze(0)
    com_t = torch.as_tensor(np.asarray(com, float), dtype=T.dtype)
    m_t = torch.as_tensor(np.asarray(masses, float), dtype=T.dtype)
    up_t = torch.as_tensor(np.asarray(up, float), dtype=T.dtype)
    world_com = torch.einsum("nij,nj->ni", T[..., :3], com_t) + T[..., 3]
    V = float(gravity) * (m_t * (world_com @ up_t)).sum()
    (grad,) = torch.autograd.grad(V, u_t)
    return float(V.detach()), grad.detach().numpy()


def actuator_jacobian(model, u, joint, body_order) -> np.ndarray:
    """d(theta_joint)/du, for the generalized force F = J^T tau of a joint-torque
    actuator -- one scalar-output backward pass, VJP in spirit."""
    u_t = torch.as_tensor(np.asarray(u, float), dtype=torch.float64).requires_grad_(True)
    T = model.transforms(u_t.unsqueeze(0)).squeeze(0)
    theta = joint_angle_torch(T, joint, body_order)
    (grad,) = torch.autograd.grad(theta, u_t)
    return grad.detach().numpy()
