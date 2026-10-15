"""handoff.py -- decoder-space projection and handoff initialisation (shipped path).

``ProjectedHandoff`` is what the advantage router uses to start a receiver chart: the destination coordinate is found
by decoder-space projection of the source pose and the velocity is transferred through the decoder tangents.
``EncoderHandoff`` is the plain ``E_j(D_i(u))`` map (comparison only).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp


def decode_flat(rom, chart, u) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
        return rom.models[int(chart)].decode(x).squeeze(0).numpy()

def decoder_jacobian(rom, chart, u) -> np.ndarray:
    """``d decode / du`` as a (D, d) matrix."""
    model = rom.models[int(chart)]
    x = torch.as_tensor(np.asarray(u, float), dtype=torch.float64)
    # Forward mode: d (latent) is small, D = 12*n_bodies outputs is not.
    J = torch.func.jacfwd(lambda z: model.decode(z.unsqueeze(0)).squeeze(0))(x)
    return J.detach().numpy()

def project_pose(rom, chart, q_target, u_init, *, iters=30, tol=1e-6):
    """Levenberg-Marquardt ``argmin_u ||decode_chart(u) - q_target||^2``.

    Returns ``(u, mse)``.  Started from ``u_init`` (the encoder's guess); the
    decoder, not the encoder, is the authority on what the receiver draws."""
    q_t = np.asarray(q_target, float).reshape(-1)
    u = np.asarray(u_init, float).copy()
    r = decode_flat(rom, chart, u) - q_t
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        J = decoder_jacobian(rom, chart, u)
        A, g = J.T @ J, J.T @ r
        improved = False
        for _try in range(8):
            step = np.linalg.solve(A + lam * np.eye(len(u)) * max(np.trace(A) / len(u), 1e-12), -g)
            u_try = rom.topology.wrap(chart, u + step)
            r_try = decode_flat(rom, chart, u_try) - q_t
            c_try = float(r_try @ r_try)
            if c_try < cost:
                u, r, improved = u_try, r_try, True
                done = cost - c_try < tol * max(cost, 1.0)
                cost, lam = c_try, max(lam * 0.3, 1e-9)
                break
            lam *= 10.0
        if not improved or done:
            break
    return u, cost / q_t.size

def velocity_from_pose_rate(rom, chart, u, qdot) -> np.ndarray:
    """Latent velocity whose decoder tangent best matches a packed-pose rate."""
    J = decoder_jacobian(rom, chart, u)
    return np.linalg.lstsq(J, np.asarray(qdot, float).reshape(-1), rcond=None)[0]

class EncoderHandoff:
    """Production map: ``u_j = E_j(D_i(u_i))``, velocity by the JVP of that map."""
    name = "encoder"

    def initialize(self, rt, source, target, u, p):
        u_dest, p_dest = rt._transition_state(source, target, u, p)
        q_src = decode_flat(rt.rom, source, u)
        q_dst = decode_flat(rt.rom, target, u_dest)
        return u_dest, p_dest, float(np.mean((q_dst - q_src) ** 2))

class ProjectedHandoff:
    """Destination coordinate by decoder-space projection of the source pose;
    velocity transferred through decoder tangents (``J_j^+ J_i v_i``)."""
    name = "projected"

    def initialize(self, rt, source, target, u, p):
        rom = rt.rom
        q_src = decode_flat(rom, source, u)
        u_guess = rom.encode(target, q_src)
        u_dest, residual = project_pose(rom, target, q_src, u_guess)
        v_src = np.linalg.solve(rt._mass(source, u), np.asarray(p, float))
        x = torch.as_tensor(np.asarray(u, float))
        _, qdot = jvp(lambda z: rom.models[source].decode(z.unsqueeze(0)).squeeze(0),
                      (x,), (torch.as_tensor(v_src),))
        v_dest = velocity_from_pose_rate(rom, target, u_dest, qdot.detach().numpy())
        return u_dest, rt._mass(target, u_dest) @ v_dest, residual
