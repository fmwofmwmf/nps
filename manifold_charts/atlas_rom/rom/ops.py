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
    # Measured from the rest pose in body_a's rest frame (`MechanismSpec`
    # attaches `rest_local`); a bare JointDef means identity rest poses.
    rest = getattr(joint, "rest_local", None) or {}
    if "rel" in rest:
        R = R @ torch.as_tensor(np.asarray(rest["rel"], float).T, dtype=R.dtype,
                                device=R.device)
    axis = torch.as_tensor(np.asarray(rest.get("axis_a", joint.axis), float), dtype=Ra.dtype,
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


_TEACHER_CHUNK = 4096


def mass_matrix_batch(model, U, masses, inertia, com) -> torch.Tensor:
    """(N, d, d) ``mass_matrix`` at every row of ``U`` (a float64 tensor on the model's device).

    Same formula as :func:`mass_matrix`; each forward JVP carries the basis direction for the whole batch, which is
    exact because the decoder acts row-wise.  Processed in chunks of ``_TEACHER_CHUNK`` rows."""
    d = U.shape[-1]
    com_t = torch.as_tensor(np.asarray(com, float), dtype=U.dtype, device=U.device)
    m_t = torch.as_tensor(np.asarray(masses, float), dtype=U.dtype, device=U.device)
    I_t = torch.as_tensor(np.asarray(inertia, float), dtype=U.dtype, device=U.device)
    out = []
    with torch.no_grad():
        for lo in range(0, len(U), _TEACHER_CHUNK):
            Uc = U[lo:lo + _TEACHER_CHUNK]
            T = model.transforms(Uc)                                   # (N,n,3,4)
            cols = []
            for k in range(d):
                basis = torch.zeros_like(Uc)
                basis[:, k] = 1.0
                _, col = jvp(model.transforms, (Uc,), (basis,))
                cols.append(col)
            dT = torch.stack(cols, dim=-1)                             # (N,n,3,4,d)
            R, dR, dt = T[..., :3], dT[..., :3, :], dT[..., 3, :]
            Jp = torch.einsum("Nnrck,nc->Nnrk", dR, com_t) + dt        # (N,n,3,d)
            M = torch.einsum("n,Nnrk,Nnrl->Nkl", m_t, Jp, Jp)
            Omega = torch.einsum("Nnrck,Nnsc->Nnrsk", dR, R)           # (N,n,3,3,d)
            Jw = torch.stack([Omega[:, :, 2, 1, :], Omega[:, :, 0, 2, :],
                              Omega[:, :, 1, 0, :]], dim=2)            # (N,n,3,d)
            I_world = torch.einsum("Nnij,njk,Nnlk->Nnil", R, I_t, R)
            M = M + torch.einsum("Nnik,Nnij,Nnjl->Nkl", Jw, I_world, Jw)
            out.append(0.5 * (M + M.transpose(-1, -2)))
    return torch.cat(out, 0)


def potential_and_grad_batch(model, U, masses, com, up, gravity) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,) V and (N, d) dV/du of :func:`potential_and_grad` at every row of ``U``.  One backward pass per chunk is
    exact: each row's V depends only on its own u."""
    com_t = torch.as_tensor(np.asarray(com, float), dtype=U.dtype, device=U.device)
    m_t = torch.as_tensor(np.asarray(masses, float), dtype=U.dtype, device=U.device)
    up_t = torch.as_tensor(np.asarray(up, float), dtype=U.dtype, device=U.device)
    Vs, Gs = [], []
    for lo in range(0, len(U), _TEACHER_CHUNK):
        x = U[lo:lo + _TEACHER_CHUNK].detach().clone().requires_grad_(True)
        T = model.transforms(x)
        world_com = torch.einsum("Nnij,nj->Nni", T[..., :3], com_t) + T[..., 3]
        V = float(gravity) * (m_t * (world_com @ up_t)).sum(-1)
        (grad,) = torch.autograd.grad(V.sum(), x)
        Vs.append(V.detach()); Gs.append(grad.detach())
    return torch.cat(Vs, 0), torch.cat(Gs, 0)


def actuator_jacobian(model, u, joint, body_order) -> np.ndarray:
    """d(theta_joint)/du, for the generalized force F = J^T tau of a joint-torque
    actuator -- one scalar-output backward pass, VJP in spirit."""
    u_t = torch.as_tensor(np.asarray(u, float), dtype=torch.float64).requires_grad_(True)
    T = model.transforms(u_t.unsqueeze(0)).squeeze(0)
    theta = joint_angle_torch(T, joint, body_order)
    (grad,) = torch.autograd.grad(theta, u_t)
    return grad.detach().numpy()
