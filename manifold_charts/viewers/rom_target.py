"""Interactive "click a target, the ROM's real dynamics drive the pen there" -- pantograph_2d and five_bar, ROM-based inverse kinematics (no analytic kinematics).

Click / drag in the viewport: that point is the target for the pen.  An operational-space (task-space) computed-torque controller drives the pen there using only the ROM's decoder
(pen Jacobian by autodiff), mass field, potential field (gravity feedforward) and actuator map -- no inverse kinematics, no joint angles, so assembly branches cannot be confused.  The ROM
integrates its learned dynamics with the net-only advantage router `srcsup`.  The only per-case constants are which body / local point is the pen.

    python -m viewers.rom_target --case pantograph_2d --rom pantograph_2d_polished --roms-dir <dir>      (from testing/manifold_charts)
    python -m viewers.rom_target --case five_bar      --rom five_bar_polished      --roms-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from atlas_rom.rom.loader import PatchAtlasROM  # noqa: E402
from atlas_rom.rom import routing_test_tracks as T, ops  # noqa: E402
from atlas_rom.charts.validity import validity_from_case  # noqa: E402
from viewers.view_routing_test import joint_pd_controls  # noqa: E402
from atlas_rom.rom.adv_router import AdvRouterTrack  # noqa: E402

def pen_target(case):
    """Per-case pen constants come from the case module's ``PEN_TARGET`` hook (pen name -> (body, rest point);
    the FIRST pen is the draggable one)."""
    import importlib
    hook = getattr(importlib.import_module(f"atlas_rom.cases.{case}"), "PEN_TARGET", None)
    if hook is None:
        raise SystemExit(f"case {case!r} defines no PEN_TARGET hook, so it has no pen to drive to a target")
    return hook



def pen_pd_controls(rom, track, body_idx, local_pt, target_xy, w0, zeta, lim_mult, gcomp, a_max=40.0):
    """Operational-space (task-space) computed torque on the PEN, using only the ROM's decoder, mass field, potential field and actuator map:
        pen = x(u),  G = dx/du (autodiff through the decoder),  J = d theta/du (the actuators' own map),  x_ddot = G M^-1 (J^T tau - grad V)
        tau = (G M^-1 J^T)^-1 (a_des + G M^-1 grad V),   a_des = w0^2 (x* - x) - 2 zeta w0 x_dot      (Coriolis ignored).
    No inverse kinematics and no joint angles in the loop, so assembly branches cannot be confused.  Returns (controls, rows)."""
    acts = list(rom.case.actuators)
    idx = [k for k, a in enumerate(acts) if a.kind == "joint_torque"]
    ch, u, p = int(track.state["chart"]), track.state["u"], track.state["p"]
    lp = torch.as_tensor(np.asarray(local_pt, float))

    def pen(z):
        Tm = rom.models[ch].transforms(z.unsqueeze(0)).squeeze(0)
        return (Tm[body_idx, :, :3] @ lp + Tm[body_idx, :, 3])[:2]
    ut = torch.as_tensor(np.asarray(u, float))
    G = torch.autograd.functional.jacobian(pen, ut).detach().numpy()                       # (2, d)
    x = pen(ut).detach().numpy()
    M = track.rt._mass(ch, u)
    J = np.stack([np.asarray(ops.actuator_jacobian(rom.models[ch], u, rom.case.mechanism.joint(acts[k].target), rom.body_order), float) for k in idx])   # (n, d)
    xdot = G @ np.linalg.solve(M, p)
    a_des = w0 ** 2 * (np.asarray(target_xy, float)[:2] - x) - 2.0 * zeta * w0 * xdot
    n = np.linalg.norm(a_des)
    if n > a_max:
        a_des = a_des * a_max / n
    ff = G @ np.linalg.solve(M, track.rt._potential(ch, u)[1]) if gcomp else 0.0
    A = G @ np.linalg.solve(M, J.T)                                                         # (2, n)
    lam = 1e-3 * max(np.linalg.norm(A, 2), 1e-9)
    tau = A.T @ np.linalg.solve(A @ A.T + lam ** 2 * np.eye(2), a_des + ff)
    ctl, rows = [0.0] * len(acts), []
    for r, k in enumerate(idx):
        a = acts[k]
        lim = lim_mult * float(a.hi)
        raw = float(tau[r]) / max(float(a.gain), 1e-12)
        cmd = float(np.clip(raw, -lim, lim))
        ctl[k] = cmd
        rows.append((a.name, float(np.linalg.norm(target_xy[:2] - x)), 0.0, cmd, abs(raw) > lim))
    return ctl, rows


class Rain:
    """Falling balls (2D discs in the mechanism's xy plane) that hit EVERY link.  Each link is the convex hull of its posed mesh projected to xy.
    A contact is an impulse exchange with the ROM: the contact point's Jacobian G (autodiff through the decoder) gives the body-point velocity G M^-1 p,
    the effective mass n^T G M^-1 G^T n, and the impulse enters the ROM as p += G^T (j n).  Static bodies are ignored; balls leaving the region are dropped."""
    def __init__(self, sim):
        self.sim, self.rng = sim, np.random.default_rng(0)
        self.on, self.rate, self.radius, self.mass, self.restitution, self.max_balls = False, 8.0, 0.04, 0.05, 0.5, 300
        self._hull_idx = None
        self.x, self.v, self.hits, self._acc = np.zeros((0, 2)), np.zeros((0, 2)), 0, 0.0
        r = max([rr for _n, rr in sim.cfg["circles"]] or [1.0])
        self.span, self.top, self.floor = r * 1.2, r * 3.0, -2.0 * r

    def clear(self):
        self.x, self.v, self.hits = np.zeros((0, 2)), np.zeros((0, 2)), 0

    @staticmethod
    def _hull(P):
        from scipy.spatial import ConvexHull
        return P[ConvexHull(P).vertices]                                  # counter-clockwise

    @staticmethod
    def _closest(poly, x):
        """Closest boundary point of the CCW convex polygon to x: (point, outward normal, signed gap) -- gap < 0 when x is inside."""
        a, b = poly, np.roll(poly, -1, 0)
        e = b - a
        t = np.clip(((x - a) * e).sum(1) / np.maximum((e * e).sum(1), 1e-18), 0.0, 1.0)
        c = a + t[:, None] * e
        d = np.linalg.norm(x - c, axis=1)
        k = int(np.argmin(d))
        inside = bool(np.all(e[:, 0] * (x[1] - a[:, 1]) - e[:, 1] * (x[0] - a[:, 0]) >= 0.0))
        n = (x - c[k]) / max(d[k], 1e-12)
        if inside:
            n = -n
        return c[k], n, (-d[k] if inside else d[k])

    def step(self, dt):
        sim = self.sim
        if self.on:
            self._acc += self.rate * dt
            while self._acc >= 1.0 and len(self.x) < self.max_balls:
                self._acc -= 1.0
                self.x = np.vstack([self.x, [self.rng.uniform(-self.span, self.span), self.top]])
                self.v = np.vstack([self.v, [0.0, 0.0]])
            self._acc = min(self._acc, 1.0)
        if not len(self.x):
            return
        g = getattr(sim.settings, "gravity", None)
        g = float(sim.rom.mechanism.gravity) if g is None else float(g)
        self.v[:, 1] -= g * dt
        self.x = self.x + self.v * dt
        Tm = sim.pose()
        st = sim.trk.state
        ch, u = int(st["chart"]), np.asarray(st["u"], float)
        model = sim.rom.models[ch]
        if self._hull_idx is None:                                      # links are rigid: the hull's vertex set is fixed, find it once in the local frame
            from scipy.spatial import ConvexHull
            self._hull_idx = [ConvexHull(np.asarray(V, float)[:, :2]).vertices for V, _F in sim.mesh_local]
            self._local = [np.asarray(V, float)[idx] for (V, _F), idx in zip(sim.mesh_local, self._hull_idx)]
        polys = [(Tm[i, :2, :3] @ Vl.T).T + Tm[i, :2, 3] for i, Vl in enumerate(self._local)]
        lo = np.min([p.min(0) for p in polys], 0) - self.radius
        hi = np.max([p.max(0) for p in polys], 0) + self.radius
        near = np.where(np.all((self.x >= lo) & (self.x <= hi), axis=1))[0]
        if len(near):
            M = Jt = None
            for k in near:
                for i, poly in enumerate(polys):
                    c, n, gap = self._closest(poly, self.x[k])
                    if gap >= self.radius:
                        continue
                    if M is None:
                        M = sim.trk.rt._mass(ch, u)
                    st = sim.trk.state
                    p_loc = Tm[i, :, :3].T @ (np.array([c[0], c[1], 0.0]) - Tm[i, :, 3])
                    if Jt is None:                                                                           # ONE Jacobian of all link transforms per step; any contact point is linear in it
                        Jt = torch.autograd.functional.jacobian(lambda z: model.transforms(z.unsqueeze(0)).squeeze(0), torch.as_tensor(u), vectorize=True, strategy="forward-mode").detach().numpy()   # (nb, 3, 4, d)
                    G = np.einsum("rcd,c->rd", Jt[i, :2, :3], p_loc) + Jt[i, :2, 3]                           # (2, d)
                    vb = G @ np.linalg.solve(M, st["p"])
                    vn = float((self.v[k] - vb) @ n)
                    self.x[k] = c + n * self.radius                                                          # resolve the overlap
                    if vn < 0.0:
                        w = float(n @ G @ np.linalg.solve(M, G.T @ n))
                        j = -(1.0 + self.restitution) * vn / (1.0 / self.mass + w)
                        self.v[k] = self.v[k] + j * n / self.mass
                        st["p"] = np.asarray(st["p"], float) - G.T @ (j * n)                     # Newton III: the link gets -j n
                        self.hits += 1
                    break
        keep = (self.x[:, 1] > self.floor) & (np.abs(self.x[:, 0]) < 4.0 * self.span)
        self.x, self.v = self.x[keep], self.v[keep]


class RomSim:
    def __init__(self, rom, case, dt=0.01, damping=0.0):
        self.rom, self.cfg = rom, pen_target(case)
        self.pens = self.cfg["pens"]
        self.drag_pen = next(iter(self.pens))
        self.acts = list(rom.case.actuators)
        self.joints = [rom.case.mechanism.joint(a.target) for a in self.acts]
        self.body_idx = {n: list(rom.body_order).index(b) for n, (b, _p) in self.pens.items()}
        self.validity = validity_from_case(rom.case, include_grounded=False)
        self.anchor_trees = [cKDTree(np.asarray(a, float) / np.maximum(np.asarray(a, float).std(0), 1e-4)) for a in rom.anchors]
        self.anchor_std = [np.maximum(np.asarray(a, float).std(0), 1e-4) for a in rom.anchors]
        self.settings = T.Settings(dt=dt, lookahead=1, damping=damping)
        self.trk = AdvRouterTrack(rom, self.settings)
        self.w0, self.zeta, self.lim, self.gcomp = 4.0, 1.0, float(self.cfg.get("torque_limit", 20.0)), True
        self.control_on = True
        self.wind = np.zeros(2)                                     # external world force on the drag pen (like wind), set by the viewer
        self._install_wind()
        self.mesh_local = list(zip(*[__import__('atlas_rom.spec.geometry', fromlist=['x']).build_bodies(rom.mechanism)[k] for k in ('V', 'F')]))
        self.rain = Rain(self)
        self.curve = None                                           # closed curve to trace: dict(pts, s, length); s = arclength of each sample
        self.smooth, self.raw_pts = 1e-5, []                          # spline smoothing per point (0 = interpolate the drawn points exactly)
        self.trace_on, self.trace_s, self.trace_speed, self.laps = False, 0.0, 0.3, 0.0
        self.reset()

    # ---- external force ("wind"): world force on the pen -> generalized force G^T F in whichever chart is integrating --------------------------------------------
    def _install_wind(self):
        rt, rom = self.trk.rt, self.rom
        orig = rt._control_force
        body, lp = self.body_idx[self.drag_pen], torch.as_tensor(np.asarray(self.pens[self.drag_pen][1], float))

        def with_wind(model, u, controls):
            F = orig(model, u, controls)
            if not np.any(self.wind):
                return F
            ch = next(i for i, m in enumerate(rom.models) if m is model)

            def pen(z):
                Tm = rom.models[ch].transforms(z.unsqueeze(0)).squeeze(0)
                return (Tm[body, :, :3] @ lp + Tm[body, :, 3])[:2]
            G = torch.autograd.functional.jacobian(pen, torch.as_tensor(np.asarray(u, float))).detach().numpy()
            Fw = G.T @ self.wind
            return Fw if F is None else F + Fw
        rt._control_force = with_wind

    # ---- closed curve ----------------------------------------------------------------------------------------------------------------------------------
    def set_curve(self, pts, n=400, smooth=None):
        """Smooth periodic spline through the drawn points, resampled uniformly in arclength."""
        from scipy.interpolate import splprep, splev
        self.raw_pts = [np.asarray(p, float)[:2].copy() for p in pts]
        smooth = self.smooth if smooth is None else smooth
        P = [np.asarray(pts[0], float)[:2]]
        for p in pts[1:]:
            if np.linalg.norm(np.asarray(p, float)[:2] - P[-1]) > 1e-3:
                P.append(np.asarray(p, float)[:2])
        if len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 1e-3:
            P.pop()
        if len(P) < 4:
            return False
        P = np.array(P + [P[0]])
        tck, _ = splprep(P.T, s=smooth * len(P), per=True)
        xy = np.array(splev(np.linspace(0.0, 1.0, n * 4, endpoint=False), tck)).T
        seg = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(s[-1])
        sg = np.linspace(0.0, L, n, endpoint=False)
        self.curve = {"pts": np.column_stack([np.interp(sg, s, np.append(xy[:, 0], xy[0, 0])), np.interp(sg, s, np.append(xy[:, 1], xy[0, 1]))]), "s": sg, "length": L}
        self.trace_s, self.laps = float(sg[int(np.argmin(np.linalg.norm(self.curve["pts"] - self.pen_points()[self.drag_pen][:2], axis=1)))]), 0.0
        return True

    def set_circle(self, center, radius, n=400):
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        self.set_curve([center[:2] + radius * np.array([np.cos(t), np.sin(t)]) for t in a[::10]], n)

    def curve_point(self, s):
        c = self.curve
        s = s % c["length"]
        x = np.interp(s, np.append(c["s"], c["length"]), np.append(c["pts"][:, 0], c["pts"][0, 0]))
        y = np.interp(s, np.append(c["s"], c["length"]), np.append(c["pts"][:, 1], c["pts"][0, 1]))
        return np.array([x, y])

    def advance_trace(self, dt):
        """Move the reference point along the curve at the constant trace speed."""
        if not (self.trace_on and self.curve is not None):
            return
        ds = self.trace_speed * dt
        self.laps += ds / self.curve["length"]
        self.trace_s = (self.trace_s + ds) % self.curve["length"]
        self.target_pt = self.curve_point(self.trace_s)

    # ---- helpers ----------------------------------------------------------------------------------------------------------------------------------------
    def pen_np(self, Tm, name):
        i, lp = self.body_idx[name], np.asarray(self.pens[name][1], float)
        return Tm[i, :, :3] @ lp + Tm[i, :, 3]

    def pose(self):
        return self.trk.pose()

    def pen_points(self):
        Tm = self.pose()
        return {n: self.pen_np(Tm, n) for n in self.pens}

    def angles_of(self, Tm):
        pose = torch.as_tensor(Tm, dtype=torch.float64)
        return [float(ops.joint_angle_torch(pose, j, self.rom.body_order)) for j in self.joints]

    def closure(self, Tm):
        if self.validity is None:
            return 0.0
        return float(np.mean(self.validity.residual(None, np.asarray(Tm, float))))

    # ---- state ----------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
        if hasattr(self, "rain"):
            self.rain.clear()
        nb = self.rom.n_bodies
        Tm = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (nb, 1, 1))          # every body starts at its rest transform (identity)
        q = Tm.reshape(-1)
        t = torch.as_tensor(q).unsqueeze(0)
        with torch.no_grad():
            err = [float(((m.decode(m.encode(t)) - t) ** 2).sum()) for m in self.rom.models]
        c = int(np.argmin(err))
        u0 = np.asarray(self.rom.encode(c, q), float)
        self.trk.reset()
        self.trk.rt.velocity_damping = self.settings.damping or 0.0
        self.trk.state = {"chart": c, "u": u0, "p": np.zeros_like(u0), "time": 0.0, "handoffs": 0, "stalls": 0, "blocked": False}
        self.target_pt = None
        self.trace_on = False
        self.error = ""
        self.rows = []

    # ---- target: the clicked point itself (task-space control; no inverse kinematics) ----------------------------------------------------------------------
    def set_target_point(self, hit_xy):
        self.target_pt = np.asarray(hit_xy, float)[:2]

    # ---- dynamics ----------------------------------------------------------------------------------------------------------------------------------------
    def step(self, n=1):
        rows = self.rows
        for _ in range(n):
            self.advance_trace(self.settings.dt)
            if not self.control_on:                                # controller OFF: no torque, the ROM's free dynamics (gravity, learned fields, router)
                ctl, rows = [0.0] * len(self.acts), []
            elif self.target_pt is None:
                ctl, rows = [0.0] * len(self.acts), []
                tgt_hold = self.pen_points()[self.drag_pen][:2]
                ctl, rows = pen_pd_controls(self.rom, self.trk, self.body_idx[self.drag_pen], self.pens[self.drag_pen][1], tgt_hold, self.w0, self.zeta, self.lim, self.gcomp)
            else:
                ctl, rows = pen_pd_controls(self.rom, self.trk, self.body_idx[self.drag_pen], self.pens[self.drag_pen][1], self.target_pt, self.w0, self.zeta, self.lim, self.gcomp)
            try:
                self.trk.step(ctl)
            except (RuntimeError, np.linalg.LinAlgError, ValueError) as ex:
                self.error = f"{type(ex).__name__}: {ex}"
                break
            self.rain.step(self.settings.dt)
        self.rows = rows
        return rows


def _ray_plane_hit(origin, direction, plane_z=0.0):
    if abs(direction[2]) < 1e-8:
        return None
    t = (plane_z - origin[2]) / direction[2]
    return None if t <= 0.0 else (origin + t * direction)[:2]


def run(args):
    import polyscope as ps
    from polyscope import imgui
    from viewers.mesh_view import MeshAdapter

    rom = PatchAtlasROM.load(os.path.join(args.roms_dir, args.rom))
    sim = RomSim(rom, args.case, dt=args.dt, damping=args.damping)
    adapter = MeshAdapter(rom.mechanism)
    ps.init()
    ps.set_up_dir(adapter.up_dir)
    ps.set_ground_plane_mode("shadow_only")
    adapter.register(ps, sim.pose())
    colors = [(0.20, 0.55, 0.90), (0.90, 0.35, 0.15), (0.35, 0.75, 0.35)]

    def circle(name, radius, color, n=96):
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        nodes = np.column_stack((radius * np.cos(a), radius * np.sin(a), np.zeros(n))).astype(np.float32)
        net = ps.register_curve_network(name, nodes, np.column_stack((np.arange(n), (np.arange(n) + 1) % n)).astype(np.int32))
        net.set_color(color)
        net.set_radius(0.003, relative=False)

    for name, r in sim.cfg["circles"]:
        circle(name, r, (0.55, 0.25, 0.85))

    def post(name, pt, color, h=0.15, r=0.012):
        net = ps.register_curve_network(name, np.array([[pt[0], pt[1], -h], [pt[0], pt[1], h]], dtype=np.float32), np.array([[0, 1]], dtype=np.int32))
        net.set_color(color)
        net.set_radius(r, relative=False)

    def seg(name, a, b, color):
        net = ps.register_curve_network(name, np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int32))
        net.set_color(color)
        net.set_radius(0.003, relative=False)

    def trail(name, pts, color):
        if len(pts) < 2:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)
            return
        nodes = np.asarray(pts, dtype=np.float32)
        net = ps.register_curve_network(name, nodes, np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))]).astype(np.int32))
        net.set_color(color)
        net.set_radius(0.004, relative=False)

    ui = {"clicked_off": False, "running": True, "spf": args.spf, "mouse": True, "dragging": False, "trails": {n: [] for n in sim.pens}, "gravity": float(rom.mechanism.gravity),
          "drawing": False, "draw_pts": [], "msg": "", "wind_mag": 0.0, "wind_dir": 0.0, "gust": False}

    def callback():
        io = imgui.GetIO()
        if imgui.Button("Pause" if ui["running"] else "Play"):
            ui["running"] = not ui["running"]
        imgui.SameLine()
        if imgui.Button("Controller: ON  (click to switch OFF)" if sim.control_on else "Controller: OFF  (click to switch ON)"):
            sim.control_on = not sim.control_on
            if sim.control_on and not ui["clicked_off"]:
                sim.target_pt = sim.pen_points()[sim.drag_pen][:2].copy()      # switching on: hold where the pen is now (unless a target was clicked while off)
            ui["clicked_off"] = False
        imgui.SameLine()
        if imgui.Button("Reset"):
            sim.reset()
            ui["trails"] = {n: [] for n in sim.pens}
        imgui.SameLine()
        if imgui.Button("Clear trails"):
            ui["trails"] = {n: [] for n in sim.pens}
        _, ui["mouse"] = imgui.Checkbox("mouse click / drag sets the target (off: orbit the camera)", ui["mouse"])

        imgui.Separator()
        if imgui.Button("Stop drawing" if ui["drawing"] else "Draw closed curve (drag in the viewport)"):
            ui["drawing"] = not ui["drawing"]
            if ui["drawing"]:
                ui["draw_pts"], sim.trace_on = [], False
            elif not sim.set_curve(ui["draw_pts"]) and ui["draw_pts"]:
                ui["msg"] = "curve needs at least 4 distinct points"
        imgui.SameLine()
        if imgui.Button("Circle preset"):
            c0 = sim.pen_points()[sim.drag_pen][:2]
            rr = [r for _n, r in sim.cfg["circles"]]
            sim.set_circle(np.zeros(2) if rr else c0, 0.5 * (min(rr) + max(rr)) if rr else 0.15)
        imgui.SameLine()
        if imgui.Button("Clear curve"):
            sim.curve, sim.trace_on, ui["draw_pts"], ui["drawing"] = None, False, [], False
        if sim.curve is not None:
            if imgui.Button("Trace: ON  (click to stop)" if sim.trace_on else "Trace: OFF  (click to start)"):
                sim.trace_on = not sim.trace_on
                sim.control_on = sim.control_on or sim.trace_on
                if sim.trace_on:
                    sim.trace_s = float(sim.curve["s"][int(np.argmin(np.linalg.norm(sim.curve["pts"] - sim.pen_points()[sim.drag_pen][:2], axis=1)))])
                    sim.target_pt = sim.curve_point(sim.trace_s)
            ch, v = imgui.SliderFloat("trace speed (length/s)", sim.trace_speed, 0.02, 2.0)
            if ch:
                sim.trace_speed = float(v)
            ch, v = imgui.SliderFloat("curve smoothing (0 = follow drawn points)", float(sim.smooth) * 1e4, 0.0, 20.0)
            if ch and sim.raw_pts:
                sim.smooth = float(v) * 1e-4
                keep = sim.trace_s
                sim.set_curve(sim.raw_pts)
                sim.trace_s = keep
            imgui.Text(f"curve length {sim.curve['length']:.2f}   laps {sim.laps:.2f}")
        if ui["msg"]:
            imgui.TextColored((1.0, 0.7, 0.2, 1.0), ui["msg"])
        wmag, wdir = ui["wind_mag"], ui["wind_dir"]
        ch1, wmag = imgui.SliderFloat("external force on the pen (wind)", wmag, 0.0, 100.0)
        ch2, wdir = imgui.SliderFloat("wind direction (deg, 0 = +x)", wdir, -180.0, 180.0)
        ch3, ui["gust"] = imgui.Checkbox("gusting (+-50% at 0.5 Hz)", ui["gust"])
        ui["wind_mag"], ui["wind_dir"] = float(wmag), float(wdir)
        imgui.Separator()

        origin = np.asarray(ps.get_view_camera_parameters().get_position(), float)
        direction = np.asarray(ps.screen_coords_to_world_ray(imgui.GetMousePos()), float)
        hit = _ray_plane_hit(origin, direction)
        if ui["drawing"]:
            if not io.WantCaptureMouse and hit is not None and imgui.IsMouseDown(0):
                ps.set_do_default_mouse_interaction(False)
                if not ui["draw_pts"] or np.linalg.norm(hit - ui["draw_pts"][-1]) > 0.02:
                    ui["draw_pts"].append(hit.copy())
            elif not imgui.IsMouseDown(0):
                ps.set_do_default_mouse_interaction(True)
            ui["msg"] = ""
        elif ui["mouse"] and not io.WantCaptureMouse and hit is not None:
            if imgui.IsMouseClicked(0):
                ui["dragging"] = True
                ps.set_do_default_mouse_interaction(False)
                sim.trace_on = False                                    # a manual click overrides the trace
            if ui["dragging"] and imgui.IsMouseDown(0):
                sim.set_target_point(hit)
                ui["clicked_off"] = ui["clicked_off"] or (not sim.control_on)
        if ui["dragging"] and imgui.IsMouseReleased(0):
            ui["dragging"] = False
            ps.set_do_default_mouse_interaction(True)
        if not ui["mouse"] and not ui["drawing"]:
            ps.set_do_default_mouse_interaction(True)

        gust = 1.0 + 0.5 * np.sin(2.0 * np.pi * 0.5 * sim.trk.state["time"]) if ui["gust"] else 1.0
        a = np.deg2rad(ui["wind_dir"])
        sim.wind = ui["wind_mag"] * gust * np.array([np.cos(a), np.sin(a)])

        rows = sim.rows
        if ui["running"] and not sim.error:
            rows = sim.step(ui["spf"])
        if sim.error:
            imgui.TextColored((1.0, 0.4, 0.3, 1.0), f"stopped: {sim.error}")
        if not sim.control_on:
            imgui.TextColored((1.0, 0.7, 0.2, 1.0), "controller OFF: no torque, free dynamics under gravity (the router still hands off).  Clicks still place the target.")

        adapter.update(sim.pose())
        pens = sim.pen_points()
        for i, (n, pt) in enumerate(pens.items()):
            post(f"pen_{n}", pt, colors[i % len(colors)])
            tr = ui["trails"][n]
            if not tr or np.linalg.norm(pt - tr[-1]) > 0.004:
                tr.append(pt.copy())
                if len(tr) > 2500:
                    tr.pop(0)
            trail(f"pen_{n} trail", tr, colors[i % len(colors)])
        if len(sim.rain.x):
            bp = ps.register_point_cloud("balls", np.column_stack((sim.rain.x, np.zeros(len(sim.rain.x)))).astype(np.float32))
            bp.set_radius(sim.rain.radius, relative=False)
            bp.set_point_render_mode("sphere")
            bp.set_color((0.25, 0.55, 0.95))
        elif ps.has_point_cloud("balls"):
            ps.remove_point_cloud("balls")
        if sim.curve is not None:
            cp = sim.curve["pts"]
            net = ps.register_curve_network("curve", np.column_stack((cp, np.zeros(len(cp)))).astype(np.float32),
                                            np.column_stack((np.arange(len(cp)), (np.arange(len(cp)) + 1) % len(cp))).astype(np.int32))
            net.set_color((0.95, 0.80, 0.15))
            net.set_radius(0.004, relative=False)
        elif ps.has_curve_network("curve"):
            ps.remove_curve_network("curve")
        dp = ui["draw_pts"] if ui["drawing"] else []
        if len(dp) >= 2:
            net = ps.register_curve_network("drawing", np.column_stack((np.array(dp), np.zeros(len(dp)))).astype(np.float32),
                                            np.column_stack((np.arange(len(dp) - 1), np.arange(1, len(dp)))).astype(np.int32))
            net.set_color((0.95, 0.80, 0.15))
            net.set_radius(0.004, relative=False)
        elif ps.has_curve_network("drawing"):
            ps.remove_curve_network("drawing")
        if np.any(sim.wind):                                           # wind arrow from the pen (length ~ force)
            p0 = pens[sim.drag_pen]
            tip = p0[:2] + sim.wind * 0.01
            seg("wind", p0, tip, (0.30, 0.85, 0.95))
        elif ps.has_curve_network("wind"):
            ps.remove_curve_network("wind")
        if sim.target_pt is not None:
            post("target",sim.target_pt, (0.20, 0.80, 0.30), h=0.25, r=0.02)
            seg("pen -> target", pens[sim.drag_pen], sim.target_pt, (0.20, 0.80, 0.30))

        imgui.Separator()
        ch, v = imgui.SliderInt("steps per frame", ui["spf"], 1, 8)
        if ch:
            ui["spf"] = int(v)
        ch, v = imgui.SliderFloat("gravity", ui["gravity"], 0.0, 20.0)
        if ch:
            ui["gravity"] = float(v)
            sim.settings.gravity = float(v)
        ch, v = imgui.SliderFloat("velocity damping (1/s)", float(sim.settings.damping or 0.0), 0.0, 10.0)
        if ch:
            sim.settings.damping = float(v)
            sim.trk.rt.velocity_damping = float(v)
        for key, lab, lo, hi in (("w0", "controller natural frequency w0 (rad/s)", 0.5, 15.0), ("zeta", "controller damping ratio", 0.3, 3.0), ("lim", "torque limit (x declared)", 0.5, 150.0)):
            ch, v = imgui.SliderFloat(lab, float(getattr(sim, key)), lo, hi)
            if ch:
                setattr(sim, key, float(v))
        ch, v = imgui.Checkbox("gravity compensation (from the learned potential)", sim.gcomp)
        if ch:
            sim.gcomp = bool(v)
        imgui.Separator()
        ra = sim.rain
        _, ra.on = imgui.Checkbox("rain balls", ra.on)
        imgui.SameLine()
        if imgui.Button("Clear balls"):
            ra.clear()
        for key, lab, lo, hi in (("rate", "balls per second", 0.5, 40.0), ("radius", "ball radius", 0.01, 0.15), ("mass", "ball mass", 0.005, 2.0), ("restitution", "bounciness", 0.0, 1.0)):
            ch, v = imgui.SliderFloat(lab, float(getattr(ra, key)), lo, hi)
            if ch:
                setattr(ra, key, float(v))
        imgui.Text(f"balls {len(ra.x)}   hits {ra.hits}")
        imgui.Separator()
        st, info = sim.trk.state, (sim.trk.info or {})
        imgui.Text(f"t={st['time']:.2f}s  chart {st['chart']}/{rom.n_patches}  handoffs {st['handoffs']}  stalls {st['stalls']}")
        for (name, _e, _z, cmd, sat) in rows:
            imgui.Text(f"  {name}: torque {cmd:+.1f}{'  SATURATED' if sat else ''}")
        if info.get("reason"):
            imgui.TextWrapped(f"router: {info['reason'][:170]}")
        if sim.target_pt is not None:
            imgui.Text(f"pen {sim.drag_pen} distance to the clicked target: {np.linalg.norm(pens[sim.drag_pen][:2] - sim.target_pt[:2]):.4f}")
        imgui.Text(f"joint closure of the decoded pose (diagnostic): {sim.closure(sim.pose()):.2e}")

    ps.set_user_callback(callback)
    print("click or drag in the viewport to set the target for the pen; the ROM's dynamics drive it there.")
    ps.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--roms-dir", default=os.path.join(ROOT, "roms"))
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--spf", type=int, default=2, help="ROM steps per rendered frame")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
