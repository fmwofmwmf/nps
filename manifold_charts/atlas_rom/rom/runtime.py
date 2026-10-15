"""runtime.py — PatchAtlasRuntime: passive + scripted-actuator stepping.

v1 scope, deliberately: PASSIVE dynamics (gravity) plus scripted
``joint_torque`` actuator forces (``dtheta/du`` by autodiff through the
decoder, generic per ``ActuatorDef``).  ``coordinate_force`` actuators are
declared against the case's OWN coordinate, which the patch atlas's learned
latent ``u`` has no guaranteed relationship to.  Passive dynamics remains
valid in their presence; a non-zero command is refused loudly rather than
silently applying a wrong force.

Chart routing is the learned advantage router (``adv_router`` / ``rom.router``): the net predicts
A(i, u, j) = log((e_i + eps) / (e_j + eps)), how much better chart j reproduces the true pose than staying in
chart i, and the runtime hands off to argmax_j A once A exceeds a dimensionless margin ``-log(switch_ratio)``
(bisected so the handoff lands on the boundary).  The receiver starts from the decoder-space projection of the
source pose (``handoff.ProjectedHandoff``).  No support field, tolerance, closure readout or oracle is involved.
The previous support-field routing runtime was retired.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp

from collections import deque

from atlas_rom.rom import ops
from atlas_rom.rom.handoff import ProjectedHandoff


class PatchAtlasRuntime:
    def __init__(self, rom, *, dt=1e-3, gravity=None,
                 stall_guard=True, velocity_damping=4.0,
                 project_boundary=True, euclidean_retraction=False,
                 switch_ratio=1.0 / 3.0, dwell=0, bisect=True, bisect_iters=8):
        self.rom = rom
        self.dt = float(dt)
        self.gravity = float(rom.mechanism.gravity if gravity is None else gravity)
        # Router: switch to argmax_j A when A > margin = -log(switch_ratio) (dimensionless hysteresis band).
        self.switch_ratio = float(switch_ratio)
        self.dwell = int(dwell)
        self.bisect = bool(bisect)
        self.bisect_iters = int(bisect_iters)
        self._since_handoff = 10 ** 9
        self.router_info = None
        self.router_ratio_hist = deque(maxlen=200)
        self.last_handoff = None
        self.last_dH = None
        self.last_pop = None
        self.stall_guard = bool(stall_guard)
        # Rayleigh damping in momentum form.  The full implicit solver loses
        # energy through its constrained solve; without a reduced analogue the
        # ROM is conservative and never settles under gravity.
        self.velocity_damping = max(0.0, float(velocity_damping))
        self.sdf_enabled = True
        # The learned SDF penalty is always independent from this switch.
        # Disable projection to make the boundary a pure penalty spring.
        self.project_boundary = bool(project_boundary)
        # Diagnostic only: retain the legacy coordinate-Euclidean retraction
        # direction.  The default is the kinetic mass-metric retraction.
        self.euclidean_retraction = bool(euclidean_retraction)
        from atlas_rom.charts.validity import validity_from_case
        # include_grounded=False: keep this readout (and every threshold tuned
        # against it, e.g. the router's oracle_tol) on the two-body joints only,
        # independent of whether the ROM was trained with grounded closure.
        self._validity = validity_from_case(rom.case, include_grounded=False)
        self.up = rom.mechanism.up_vector
        self.masses = rom.masses
        self.inertia = rom.inertia
        self.com = rom.com
        self.body_order = rom.body_order
        self.actuators = list(rom.case.actuators)
        self._joints = {}
        self._unsupported_coordinate = set()
        for a in self.actuators:
            if a.kind == "joint_torque":
                self._joints[a.name] = rom.case.mechanism.joint(a.target)
            elif a.kind == "coordinate_force":
                self._unsupported_coordinate.add(a.name)

    # -- states -------------------------------------------------------------
    def initial_state(self, chart: int = 0) -> dict:
        u, _cell = self.rom.topology.initial_u(chart)
        u = np.asarray(u, float)
        return {"chart": int(chart), "u": u, "p": np.zeros_like(u), "time": 0.0,
                "handoffs": 0, "stalls": 0, "blocked": False}

    def set_velocity(self, state: dict, udot) -> None:
        M = self._mass(state["chart"], state["u"])
        state["p"] = M @ np.asarray(udot, float)

    def set_sdf_enabled(self, enabled: bool) -> None:
        self.sdf_enabled = bool(enabled)

    def set_boundary_projection(self, enabled: bool) -> None:
        """Enable/disable post-step SDF retraction and velocity projection."""
        self.project_boundary = bool(enabled)

    def set_euclidean_retraction(self, enabled: bool) -> None:
        """Use coordinate-Euclidean rather than mass-metric SDF retraction."""
        self.euclidean_retraction = bool(enabled)



    # -- energy readouts ------------------------------------------------------
    def energy(self, state: dict) -> dict:
        chart = state["chart"]
        M = self._mass(chart, state["u"])
        ke = 0.5 * float(state["p"] @ np.linalg.solve(M, state["p"]))
        V, _ = self._potential(chart, state["u"])
        return {"kinetic": ke, "potential": V, "total": ke + V}

    def pose(self, state: dict) -> np.ndarray:
        """Decoded (n_bodies, 3, 4) pose at the current state."""
        return ops.transforms(self.rom.models[state["chart"]], state["u"]).numpy()

    def boundary_diagnostics(self, state: dict) -> dict:
        """Live learned-boundary and support values for viewer/debug output."""
        chart = int(state["chart"])
        x = torch.as_tensor(np.asarray(state["u"], float), dtype=torch.float64)
        x = x.requires_grad_(True)
        phi = self.rom.boundary_fields[chart](x).squeeze()
        (grad,) = torch.autograd.grad(phi, x)
        value = float(phi.detach())
        g = grad.detach().numpy()
        force = (-1e5 * max(value, 0.0) * g if self.sdf_enabled
                 else np.zeros_like(g))
        return {"sdf": value, "sdf_grad_norm": float(np.linalg.norm(g)),
                "sdf_force_norm": float(np.linalg.norm(force)),
                "support": self.rom.support_margin(chart, state["u"]),
                "sdf_enabled": self.sdf_enabled}

    # -- forcing ----------------------------------------------------------------
    def _mass(self, chart, u):
        """Runtime M is the learned SPD field, never a decoder Jacobian."""
        import torch
        with torch.no_grad():
            return self.rom.mass_fields[int(chart)](
                torch.as_tensor(np.asarray(u, float)).unsqueeze(0)).squeeze(0).numpy()

    def _potential(self, chart, u):
        """Runtime V and force come from the learned scalar field."""
        import torch
        x = torch.as_tensor(np.asarray(u, float)).requires_grad_(True)
        value = self.rom.potential_fields[int(chart)](x).squeeze()
        (grad,) = torch.autograd.grad(value, x)
        # The distilled teacher used the declared gravity. Preserve the existing
        # live gravity control without ever reintroducing the analytic teacher.
        factor = 1.0 if self.rom.mechanism.gravity == 0 else self.gravity / self.rom.mechanism.gravity
        return float((factor * value).detach()), (factor * grad.detach().numpy())

    def _hamiltonian_mass_force(self, chart, u, p):
        """``+½ vᵀ(∂M/∂u)v`` in Hamilton's momentum equation."""
        x = torch.as_tensor(np.asarray(u, float)).requires_grad_(True)
        covector = torch.as_tensor(np.asarray(p, float))
        M = self.rom.mass_fields[int(chart)](x).squeeze(0)
        Hk = .5 * covector @ torch.linalg.solve(M, covector)
        (grad,) = torch.autograd.grad(Hk, x)
        return -grad.detach().numpy()

    def _boundary_force(self, chart, u):
        if not self.sdf_enabled:
            return np.zeros_like(np.asarray(u, float))
        """Conservative learned-SDF penalty force ``-∇½k relu(φ)^2``."""
        x = torch.as_tensor(np.asarray(u, float)).requires_grad_(True)
        phi = self.rom.boundary_fields[int(chart)](x).squeeze()
        energy = .5 * 1e5 * torch.relu(phi).square()
        (grad,) = torch.autograd.grad(energy, x)
        return -grad.detach().numpy()

    def _project_boundary(self, chart, u, p):
        """Retract an overshot state to the learned SDF boundary.

        The SDF is authoritative at runtime: a finite time step may cross the
        learned wall, but it must not turn that crossing into a chart stall or
        an extrapolated pose.  Both retraction and velocity clipping use the
        kinetic mass metric, so tangential momentum remains available along the
        boundary independent of the chart's latent coordinate scaling.
        """
        if not self.sdf_enabled or not self.project_boundary:
            return np.asarray(u, float), np.asarray(p, float), True
        x = torch.as_tensor(np.asarray(u, float), dtype=torch.float64)
        # The SDF is inert in its interior.  Projection is an overshoot
        # recovery only; applying its normal-velocity correction for phi < 0
        # would make the field alter otherwise valid dynamics.
        with torch.no_grad():
            if float(self.rom.boundary_fields[int(chart)](x).squeeze()) <= 0.0:
                return np.asarray(u, float), np.asarray(p, float), True
        for _ in range(8):
            x = x.detach().requires_grad_(True)
            phi = self.rom.boundary_fields[int(chart)](x).squeeze()
            if float(phi.detach()) <= 1e-6:
                break
            (grad,) = torch.autograd.grad(phi, x)
            g = grad.detach().numpy()
            if self.euclidean_retraction:
                normal = g
            else:
                M = self._mass(chart, x.detach().numpy())
                normal = np.linalg.solve(M, g)
            gg = float(g @ normal)
            if not np.isfinite(gg) or gg < 1e-12:
                return np.asarray(u, float), np.asarray(p, float), False
            x = x.detach() - float(phi.detach()) * torch.as_tensor(
                normal / gg, dtype=x.dtype)
        else:
            return np.asarray(u, float), np.asarray(p, float), False

        safe_u = x.detach().numpy()
        if float(phi.detach()) > 1e-4:
            return np.asarray(u, float), np.asarray(p, float), False
        # Project outward velocity away from the wall in the mass metric.
        x = torch.as_tensor(safe_u, dtype=torch.float64).requires_grad_(True)
        phi = self.rom.boundary_fields[int(chart)](x).squeeze()
        (grad,) = torch.autograd.grad(phi, x)
        g = grad.detach().numpy()
        M = self._mass(chart, safe_u)
        velocity = np.linalg.solve(M, np.asarray(p, float))
        outward = float(g @ velocity)
        if outward > 0.0:
            metric_normal = np.linalg.solve(M, g)
            gg = float(g @ metric_normal)
            if not np.isfinite(gg) or gg < 1e-12:
                return np.asarray(u, float), np.asarray(p, float), False
            velocity = velocity - (outward / gg) * metric_normal
        return safe_u, M @ velocity, True

    def _decode_flat(self, chart, u) -> np.ndarray:
        """Flat packed-pose decode, the same representation ``model.encode``
        consumes and ``check_data_validity._best_roundtrip`` compares in."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
            return self.rom.models[int(chart)].decode(x).squeeze(0).numpy()

    def _closure_residual(self, q_flat) -> float:
        """Mean squared joint-pivot gap for a flat packed pose, ``0.0`` when
        this case has no revolute/spherical/fixed two-body pivots to check
        (a validity-free case, e.g. purely prismatic or single-body)."""
        if self._validity is None:
            return 0.0
        T = np.asarray(q_flat, float).reshape(self.rom.n_bodies, 3, 4)
        r = self._validity.residual(None, T)
        return float(np.mean(r)) if r is not None else 0.0


    def _transition_state(self, source, target, u, p):
        """Map coordinates and velocity through ``Dτ``; rebuild destination p."""
        x = torch.as_tensor(np.asarray(u, float))
        v = torch.as_tensor(np.linalg.solve(self._mass(source, u), np.asarray(p, float)))
        def tau(z):
            return self.rom.models[target].encode(
                self.rom.models[source].decode(z.unsqueeze(0))).squeeze(0)
        dest, v_dest = jvp(tau, (x,), (v,))
        u_dest, v_dest = dest.detach().numpy(), v_dest.detach().numpy()
        return u_dest, self._mass(target, u_dest) @ v_dest

    def _control_force(self, model, u, controls) -> np.ndarray | None:
        if controls is None or not self.actuators:
            return None
        controls = np.asarray(controls, float).reshape(-1)
        if controls.shape[0] != len(self.actuators):
            raise ValueError(f"controls has {controls.shape[0]} entries, "
                             f"expected {len(self.actuators)} (one per actuator)")
        F = np.zeros(u.shape, dtype=float)
        any_nonzero = False
        for a, tau in zip(self.actuators, controls):
            if tau == 0.0:
                continue
            if a.name in self._unsupported_coordinate:
                raise NotImplementedError(
                    f"actuator {a.name!r} is a coordinate_force targeting "
                    f"{a.target!r}; passive patch-ROM dynamics is available, "
                    "but non-zero coordinate controls require a learned actuator-force field")
            J = ops.actuator_jacobian(model, u, self._joints[a.name], self.body_order)
            F = F + a.gain * float(tau) * J
            any_nonzero = True
        return F if any_nonzero else None


    def _integrate_chart(self, chart, u, p, controls):
        """One semi-implicit-Euler step's worth of dynamics using ``chart``'s
        own learned mass/potential/actuator fields, before boundary handling
        or routing.  Factored out so every candidate chart can be integrated
        independently and compared (brute-force / GT-matched routing)."""
        model = self.rom.models[chart]
        _V, gradV = self._potential(chart, u)
        F_ext = self._control_force(model, u, controls)
        F = (-gradV + self._hamiltonian_mass_force(chart, u, p)
             + self._boundary_force(chart, u) - self.velocity_damping * p)
        if F_ext is not None:
            F = F + F_ext
        M = self._mass(chart, u)
        p_half = p + self.dt * F
        udot = np.linalg.solve(M, p_half)
        u_new = self.rom.topology.wrap(chart, u + self.dt * udot)
        return u_new, p_half



    # -- stepping (semi-implicit / symplectic Euler) -----------------------------


    # -- stepping (semi-implicit / symplectic Euler) + learned advantage routing ---
    def _note_route(self, adv, picked, reason, margin, c, boundary_fail=False):
        """Record the router's readout for viewers / diagnostics (readout only, never a gate)."""
        jb = max(adv, key=adv.get) if adv else None
        ab = adv[jb] if jb is not None else float("nan")
        ranked = [dict(j=j, cost=float(np.exp(-a)), rms=float(a), angle=float("nan"), closure=float(np.exp(-a)))
                  for j, a in sorted(adv.items(), key=lambda kv: -kv[1])]
        self.router_info = {"mode": "adv", "chart": c, "cur": float("nan"),
                            "scores": {j: float(np.exp(-a)) for j, a in adv.items()}, "adv": dict(adv), "best": jb,
                            "ratio": float(np.exp(-ab)) if jb is not None else float("inf"),
                            "switch_at": self.switch_ratio, "margin": margin, "tol": float("nan"),
                            "picked": int(picked), "supported": {}, "acceptable": {}, "ranked": ranked,
                            "reason": reason, "boundary_fail": boundary_fail}
        self.router_ratio_hist.append(min(self.router_info["ratio"], 10.0))

    def _step_single_chart(self, state: dict, controls) -> bool:
        """One step of a single-chart atlas: integrate and retract to the learned SDF boundary; no routing."""
        c, u, p = int(state["chart"]), state["u"], state["p"]
        try:
            u1, p1 = self._integrate_chart(c, u, p, controls)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            state["p"] = np.zeros_like(p); state["time"] += self.dt; state["stalls"] += 1; state["blocked"] = True
            return True
        u1p, p1p, ok = self._project_boundary(c, u1, p1)
        state["time"] += self.dt
        if ok and np.all(np.isfinite(u1p)) and np.all(np.isfinite(p1p)):
            state["u"], state["p"] = u1p, p1p
        else:
            state["p"] = np.zeros_like(p)
        state["blocked"] = False
        return True

    def step(self, state: dict, controls=None) -> bool:
        """Advance one ``dt``: integrate the active chart, retract to the learned SDF boundary, then let the
        advantage router decide whether to stay or hand off (``rom.router``).

        A singular learned mass matrix is a numerical failure: the state is held, momentum zeroed and
        ``stalls`` incremented (a counter, not a gate).  Returns True unless the router has no state to keep."""
        router = self.rom.router
        c, u, p = int(state["chart"]), state["u"], state["p"]
        margin = -float(np.log(max(self.switch_ratio, 1e-6)))
        if router is None:                    # single-chart atlas: nothing to route to, never a handoff
            return self._step_single_chart(state, controls)
        try:
            u1, p1 = self._integrate_chart(c, u, p, controls)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
            state["p"] = np.zeros_like(p); state["time"] += self.dt; state["stalls"] += 1; state["blocked"] = True
            self._note_route({}, c, "NUMERICAL FAILURE in the learned dynamics (singular mass matrix): "
                             "state held, momentum zeroed", margin, c, True)
            self._since_handoff += 1
            return True
        u1p, p1p, ok = self._project_boundary(c, u1, p1)
        good = bool(ok and np.all(np.isfinite(u1p)) and np.all(np.isfinite(p1p)))
        u_src, p_src = (u1p, p1p) if good else (u1, p1)
        jb, ab, adv = router.best(c, u_src)
        state["time"] += self.dt
        if not (ab > margin) or self._since_handoff < self.dwell:
            why = (f"STAY: best advantage A={ab:+.2f} (c{jb}, receiver error x{np.exp(-ab):.2f} of staying) is not "
                   f"above the margin {margin:.2f} (error ratio < {self.switch_ratio:.2f})"
                   if ab <= margin else f"STAY: dwell ({self._since_handoff}/{self.dwell})")
            self._note_route(adv, c, why, margin, c, not good)
            if good:
                state["u"], state["p"] = u1p, p1p
            else:
                state["p"] = np.zeros_like(p)
            state["blocked"] = False
            self._since_handoff += 1
            return True
        # the step crossed the band: bisect to the crossing so the handoff lands on the boundary
        s_star = 1.0
        if self.bisect and np.all(np.isfinite(u1)) and router.adv(c, u, jb) <= margin:
            lo, hi = 0.0, 1.0
            for _ in range(self.bisect_iters):
                mid = 0.5 * (lo + hi)
                if router.adv(c, u + mid * (u_src - u), jb) > margin:
                    hi = mid
                else:
                    lo = mid
            s_star = hi
        u_s, p_s = u + s_star * (u_src - u), p + s_star * (p_src - p)
        ud, pd, res = ProjectedHandoff().initialize(self, c, jb, u_s, p_s)

        def _ke_v(ch, uu, pp):
            M_ = self._mass(ch, uu)
            return 0.5 * float(pp @ np.linalg.solve(M_, pp)), self._potential(ch, uu)[0]
        # (KE_src, V_src, KE_dst, V_dst) right at the crossing, before the rest of the step
        self.last_dH = _ke_v(c, u_s, p_s) + _ke_v(jb, ud, pd)
        rest = 1.0 - s_star
        if rest > 1e-3:
            dt0 = self.dt
            try:
                self.dt = dt0 * rest
                un, pn = self._integrate_chart(jb, ud, pd, controls)
                un, pn, okn = self._project_boundary(jb, un, pn)
                if okn and np.all(np.isfinite(un)) and np.all(np.isfinite(pn)):
                    ud, pd = un, pn
            except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
                pass
            finally:
                self.dt = dt0

        self.last_pop = float(np.sqrt(res * self.rom.n_bodies * 12))   # L2 pose error of the receiver's reproduction
        self.last_handoff = {"source": c, "target": int(jb), "residual": float(res), "s_star": float(s_star),
                             "q_pre": self.rom.decode(c, u_s)}
        self._note_route(adv, jb, f"SWITCH c{c} -> c{jb}: advantage A={ab:+.2f} > margin {margin:.2f} "
                         f"(receiver error x{np.exp(-ab):.2f}); handoff at {s_star*100:.0f}% of the step (bisected)",
                         margin, c)
        state["chart"], state["u"], state["p"] = int(jb), ud, pd
        state["handoffs"] += 1; state["blocked"] = False
        self._since_handoff = 0
        return True
