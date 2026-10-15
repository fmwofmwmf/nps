"""position_runtime.py — PositionAtlasRuntime: a velocityless (two-position) variational stepper.

State is ``(chart, u_prev, u)``; there is no momentum.  Each step solves the forced discrete Euler-Lagrange
equation of the midpoint discrete Lagrangian built only from the learned fields

    L_d(a, b) = h [ ½ ((b-a)/h)ᵀ M(m) ((b-a)/h) - W(m) ],   m = (a+b)/2,
    W = V + ½ k relu(φ)²                                     (learned potential + learned SDF penalty)

    D2 L_d(u_prev, u) + D1 L_d(u, u_next) + h F_ctrl(u) - α M(u)(u_next - u) = 0

for ``u_next`` by Newton.  The ½ vᵀ∂M v term and the SDF wall come out of the discrete Lagrangian, so neither
``_hamiltonian_mass_force`` nor ``_project_boundary`` is used on this path.

Chart transitions (``transport``):
  ``"none"``       both positions of the pair are decoded in the source chart and projected into the receiver's
                   decoder (``handoff.project_pose``); velocity is never formed or transported.
  ``"projected"``  control arm: the finite-difference velocity is transported by ``ProjectedHandoff`` (decoder
                   tangents) and the previous position rebuilt as ``u' - h v'``.
Routing (advantage router, margin, bisection) is identical to ``PatchAtlasRuntime.step``.
"""
from __future__ import annotations

import numpy as np
import torch

from atlas_rom.rom.handoff import ProjectedHandoff, decode_flat, project_pose
from atlas_rom.rom.runtime import PatchAtlasRuntime


class PositionAtlasRuntime(PatchAtlasRuntime):
    def __init__(self, rom, *, transport="none", newton_iters=20, newton_tol=1e-10, **kw):
        kw.setdefault("project_boundary", False)
        super().__init__(rom, **kw)
        if transport not in ("none", "projected"):
            raise ValueError(f"transport must be 'none' or 'projected', got {transport!r}")
        self.transport = transport
        self.newton_iters = int(newton_iters)
        self.newton_tol = float(newton_tol)
        self.last_newton = None
        self.handoff_log = []

    # -- states -------------------------------------------------------------
    def initial_state(self, chart: int = 0) -> dict:
        state = super().initial_state(chart)
        del state["p"]
        state["u_prev"] = state["u"].copy()
        return state

    def set_velocity(self, state: dict, udot) -> None:
        state["u_prev"] = np.asarray(state["u"], float) - self.dt * np.asarray(udot, float)

    def velocity(self, state: dict) -> np.ndarray:
        """Readout only: the backward difference ``(u - u_prev) / h``."""
        return (np.asarray(state["u"], float) - np.asarray(state["u_prev"], float)) / self.dt

    def energy(self, state: dict) -> dict:
        c, u, up = int(state["chart"]), state["u"], state["u_prev"]
        v = self.velocity(state)
        ke = 0.5 * float(v @ self._mass(c, 0.5 * (u + up)) @ v)
        V, _ = self._potential(c, u)
        return {"kinetic": ke, "potential": V, "total": ke + V}

    # -- discrete Lagrangian --------------------------------------------------
    def _W(self, chart, x):
        factor = 1.0 if self.rom.mechanism.gravity == 0 else self.gravity / self.rom.mechanism.gravity
        w = factor * self.rom.potential_fields[chart](x).squeeze()
        if self.sdf_enabled:
            w = w + .5 * 1e5 * torch.relu(self.rom.boundary_fields[chart](x).squeeze()).square()
        return w

    def _Ld(self, chart, a, b):
        h = self.dt
        m, d = 0.5 * (a + b), b - a
        M = self.rom.mass_fields[chart](m.unsqueeze(0)).squeeze(0)
        return 0.5 / h * (d @ M @ d) - h * self._W(chart, m)

    def _del_step(self, chart, u_prev, u, controls):
        """Newton solve of the forced DEL for ``u_next``.  Returns ``(u_next, converged)``."""
        chart = int(chart)
        tt = lambda z: torch.as_tensor(np.asarray(z, float), dtype=torch.float64)
        up_t, u_t = tt(u_prev), tt(u)
        b = u_t.clone().requires_grad_(True)
        (D2,) = torch.autograd.grad(self._Ld(chart, up_t, b), b)
        rhs = D2.detach()
        F_ext = self._control_force(self.rom.models[chart], np.asarray(u, float), controls)
        if F_ext is not None:
            rhs = rhs + self.dt * tt(F_ext)
        alpha = self.velocity_damping
        Mu = tt(self._mass(chart, u)) if alpha > 0.0 else None

        def G(x):
            a = u_t.clone().requires_grad_(True)
            (D1,) = torch.autograd.grad(self._Ld(chart, a, x), a, create_graph=True)
            g = rhs + D1
            if Mu is not None:
                g = g - alpha * (Mu @ (x - u_t))
            return g

        scale = 1.0 + float(rhs.norm())
        x = (2.0 * u_t - up_t).detach()
        g = G(x.clone().requires_grad_(True)).detach()
        it = 0
        for it in range(1, self.newton_iters + 1):
            if float(g.norm()) <= self.newton_tol * scale:
                it -= 1
                break
            J = torch.autograd.functional.jacobian(G, x.detach())
            dx = torch.linalg.solve(J, -g)
            t, f0 = 1.0, float(g @ g)
            for _ls in range(12):
                x_try = (x.detach() + t * dx)
                g_try = G(x_try.clone().requires_grad_(True)).detach()
                if np.isfinite(float(g_try @ g_try)) and float(g_try @ g_try) < f0:
                    break
                t *= 0.5
            x, g = x_try.detach(), g_try
        res = float(g.norm()) / scale
        self.last_newton = {"iters": it, "rel_residual": res}
        u_next = self.rom.topology.wrap(chart, x.numpy())
        return u_next, bool(np.isfinite(res) and res <= max(self.newton_tol, 1e-8)
                            and np.all(np.isfinite(u_next)))

    # -- handoff ----------------------------------------------------------------
    def _handoff_pair(self, source, target, u_prev, u):
        """Map the position pair into ``target``.  Returns ``(u_prev', u', pose_mse)``."""
        q = decode_flat(self.rom, source, u)
        u_dst, res = project_pose(self.rom, target, q, self.rom.encode(target, q))
        if self.transport == "none":
            q_prev = decode_flat(self.rom, source, u_prev)
            u_dst_prev, _ = project_pose(self.rom, target, q_prev, self.rom.encode(target, q_prev))
            return u_dst_prev, u_dst, res
        v = (np.asarray(u, float) - np.asarray(u_prev, float)) / self.dt
        ud, pd, res = ProjectedHandoff().initialize(self, source, target, u, self._mass(source, u) @ v)
        v_dst = np.linalg.solve(self._mass(target, ud), pd)
        return ud - self.dt * v_dst, ud, res

    # -- stepping -------------------------------------------------------------------
    def _fail(self, state):
        state["u_prev"] = state["u"].copy(); state["time"] += self.dt
        state["stalls"] += 1; state["blocked"] = True

    def _step_single_chart(self, state: dict, controls) -> bool:
        c = int(state["chart"])
        try:
            u1, ok = self._del_step(c, state["u_prev"], state["u"], controls)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            ok = False
        if not ok:
            self._fail(state)
            return True
        state["u_prev"], state["u"] = state["u"], u1
        state["time"] += self.dt; state["blocked"] = False
        return True

    def step(self, state: dict, controls=None) -> bool:
        router = self.rom.router
        if router is None:
            return self._step_single_chart(state, controls)
        c, u_prev, u = int(state["chart"]), state["u_prev"], state["u"]
        margin = -float(np.log(max(self.switch_ratio, 1e-6)))
        try:
            u1, ok = self._del_step(c, u_prev, u, controls)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            ok = False
        if not ok:
            self._fail(state)
            self._note_route({}, c, "NUMERICAL FAILURE in the DEL solve: state held, velocity zeroed",
                             margin, c, True)
            self._since_handoff += 1
            return True
        jb, ab, adv = router.best(c, u1)
        if not (ab > margin) or self._since_handoff < self.dwell:
            why = (f"STAY: best advantage A={ab:+.2f} (c{jb}) is not above the margin {margin:.2f}"
                   if ab <= margin else f"STAY: dwell ({self._since_handoff}/{self.dwell})")
            self._note_route(adv, c, why, margin, c)
            state["u_prev"], state["u"] = u, u1
            state["time"] += self.dt; state["blocked"] = False
            self._since_handoff += 1
            return True
        # bisect to the crossing along the step; the pair at the crossing keeps the step's secant
        s_star = 1.0
        if self.bisect and router.adv(c, u, jb) <= margin:
            lo, hi = 0.0, 1.0
            for _ in range(self.bisect_iters):
                mid = 0.5 * (lo + hi)
                if router.adv(c, u + mid * (u1 - u), jb) > margin:
                    hi = mid
                else:
                    lo = mid
            s_star = hi
        u_s = u + s_star * (u1 - u)
        u_ps = u_s - (u1 - u)
        e_src = self._pair_energy(c, u_ps, u_s)
        ud_prev, ud, res = self._handoff_pair(c, jb, u_ps, u_s)
        e_dst = self._pair_energy(jb, ud_prev, ud)
        self.last_pop = float(np.sqrt(res * self.rom.n_bodies * 12))
        self.last_handoff = {"source": c, "target": int(jb), "residual": float(res), "s_star": float(s_star),
                             "q_pre": self.rom.decode(c, u_s)}
        self.last_dH = e_src + e_dst
        self.handoff_log.append({"time": state["time"], "source": c, "target": int(jb), "s_star": float(s_star),
                                 "pose_l2": self.last_pop, "E_src": sum(e_src), "E_dst": sum(e_dst)})
        self._note_route(adv, jb, f"SWITCH c{c} -> c{jb}: advantage A={ab:+.2f} > margin {margin:.2f}; "
                         f"handoff at {s_star*100:.0f}% of the step (bisected), transport={self.transport}",
                         margin, c)
        state["chart"], state["u_prev"], state["u"] = int(jb), ud_prev, ud
        state["time"] += s_star * self.dt
        state["handoffs"] += 1; state["blocked"] = False
        self._since_handoff = 0
        return True

    def _pair_energy(self, chart, u_prev, u):
        v = (np.asarray(u, float) - np.asarray(u_prev, float)) / self.dt
        ke = 0.5 * float(v @ self._mass(chart, 0.5 * (u + u_prev)) @ v)
        return ke, self._potential(chart, u)[0]
