"""viewers/rom_drivers.py -- what moves the mechanism during an export.  A driver decides how the ROM is controlled and when the clip ends; it knows nothing about files, Blender or recording.

Interface (every driver)::

    Driver.add_args(ap)                register this driver's command-line flags
    Driver(args)
    d.make_sim(rom, args)              the simulator it needs (`RomBase`, or the pen-carrying `RomSim`)
    d.prepare(sim, spf)  -> dict       unrecorded setup before the clip starts (returned dict goes into the report)
    d.control(sim, t)    -> bool       called before every recorded frame; True ends the clip.  A clip of one frame is a driver that returns True at t = 0
    d.advance(sim, spf, t)             move one exported frame forward (default: `sim.step(spf)`)
    d.extras(sim)        -> dict       optional per-frame channels (name -> array) the recorder stores next to the pose
    d.static(sim)        -> dict       optional once-per-clip arrays stored in frames.npz
    d.report(sim)        -> dict       driver-specific fields for report.json

  free    no control at all: the ROM's own dynamics from the rest pose.
  rest    no ROM: the case's declared meshes at the rest pose, a single frame.  A ground-truth reference still for cases without a trained ROM (`--rom` not needed).
  pose    kinematic: solve the learned decoder for the requested actuator angles (`--joint NAME=START[:END]`, ramped over `--seconds`), no dynamics.  The way to get a still or a prescribed sweep.
  joints  dynamic: joint-space computed-torque PD on the ROM's own dynamics (`joint_pd_controls`).  Only as good as the learned fields; it saturates and stalls where they are wrong.
  pen     task-space PD drives a pen along a closed curve while balls rain on the mechanism (the pantograph clip).  Needs the case's `PEN_TARGET` hook.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from viewers.rom_record import Ragged  # noqa: E402
from viewers.rom_sim import ReferenceSim, RomBase  # noqa: E402


class Driver:
    name = ""
    needs_rom = True

    @staticmethod
    def add_args(ap):
        pass

    def __init__(self, args):
        self.args = args

    def make_sim(self, rom, args):
        return RomBase(rom, dt=args.dt, damping=args.damping)

    def prepare(self, sim, spf):
        return {}

    def control(self, sim, t):
        return t >= self.args.seconds

    def advance(self, sim, spf, t):
        sim.step(spf)

    def extras(self, sim):
        return {}

    def static(self, sim):
        return {}

    def report(self, sim):
        return {}


class FreeDriver(Driver):
    name = "free"


class RestDriver(Driver):
    """The case's declared geometry at its rest pose, one frame, no ROM.  A ground-truth reference still, not a ROM result."""
    name = "rest"
    needs_rom = False

    def make_sim(self, rom, args):
        return ReferenceSim(args.case)

    def control(self, sim, t):
        return True

    def report(self, sim):
        return {"reference": True, "note": "declared case geometry at the rest pose; no ROM, no learning, no simulation"}


class JointTargets(Driver):
    """Shared by the joint-space drivers: parses `--joint`, resolves names against the case's actuators, evaluates the (possibly ramped) targets."""

    @staticmethod
    def add_args(ap):
        g = ap.add_argument_group("joint targets (pose / joints drivers)")
        g.add_argument("--joint", action="append", default=[], metavar="NAME=START[:END]",
                       help="NAME = an actuator's name or the joint it drives; its target angle in radians; START:END ramps linearly over --seconds.  Repeat per actuator.  Unnamed actuators hold 0")

    def __init__(self, args):
        super().__init__(args)
        self.spec = {}
        for item in args.joint:
            name, _, val = item.partition("=")
            a, _, b = val.partition(":")
            self.spec[name.strip()] = (float(a), float(b) if b else float(a))
        self.names = None
        self.t = 0.0

    def targets(self, sim, t):
        if self.names is None:
            keys = [{a.name, a.target} for a in sim.acts]                 # an actuator is addressed by its own name or by the joint it drives
            bad = sorted(n for n in self.spec if not any(n in k for k in keys))
            if bad:
                raise SystemExit(f"--joint names {bad} match no actuator; this case has {[(a.name, a.target) for a in sim.acts]} as (name, joint)")
            self.names = [a.name for a in sim.acts]
            self.spec = {a.name: next(v for n, v in self.spec.items() if n in (a.name, a.target)) for a in sim.acts if any(n in (a.name, a.target) for n in self.spec)}
        T = self.args.seconds
        s = 0.0 if T <= 0 else min(max(t / T, 0.0), 1.0)
        return [(lambda ab: ab[0] + s * (ab[1] - ab[0]))(self.spec.get(n, (0.0, 0.0))) for n in self.names]

    def joint_idx(self, sim):
        return [k for k, ac in enumerate(sim.acts) if ac.kind == "joint_torque"]

    def angle_error(self, sim, t=0.0):
        tg, th = self.targets(sim, t), sim.angles()
        return float(max(abs(((tg[k] - th[k] + np.pi) % (2 * np.pi)) - np.pi) for k in self.joint_idx(sim)))

    def control(self, sim, t):
        self.t = t
        return t >= self.args.seconds

    def extras(self, sim):
        return {"joint_angles": np.asarray(sim.angles(), float)}


class JointDriver(JointTargets):
    name = "joints"

    @staticmethod
    def add_args(ap):
        JointTargets.add_args(ap)
        g = ap.add_argument_group("joints driver")
        g.add_argument("--settle-tol", type=float, default=1e-3, help="the settle ends once every angle error is below this (rad) for 10 consecutive frames")
        g.add_argument("--settle-max", type=float, default=20.0, help="give up settling after this many simulated seconds")
        g.add_argument("--w0", type=float, default=4.0, help="PD natural frequency (rad/s)")
        g.add_argument("--zeta", type=float, default=1.0, help="PD damping ratio")
        g.add_argument("--torque-limit", type=float, default=20.0, help="actuator command limit as a multiple of each actuator's `hi`")

    def make_sim(self, rom, args):
        from viewers.view_routing_test import joint_pd_controls          # lazy: keeps this module importable without the viewer's dependencies
        sim = super().make_sim(rom, args)
        a = self.args
        sim.controller = lambda s: joint_pd_controls(s.rom, s.trk, self.targets(s, self.t), a.w0, a.zeta, a.torque_limit, True, [0.0] * len(s.acts))
        return sim

    def prepare(self, sim, spf):
        a, dt = self.args, sim.settings.dt
        info, ok, n = {"seconds": 0.0, "final_error": None}, 0, int(a.settle_max / (spf * dt))
        self.t = 0.0                                                    # settle onto the START targets, unrecorded
        for _ in range(n):
            sim.step(spf)
            if sim.error:
                break
            info["seconds"] += spf * dt
            err = self.angle_error(sim)
            info["final_error"] = err
            ok = ok + 1 if err < a.settle_tol else 0
            if ok >= 10:
                break
        info["converged"] = bool(ok >= 10)
        return info

    def report(self, sim):
        return {"actuators": self.names, "targets": self.spec, "final_error": self.angle_error(sim, self.t)}


class PoseDriver(JointTargets):
    """Levenberg-Marquardt on the latent coordinate so the decoded pose has the requested actuator angles; warm-started from the previous frame and handed between charts through
    `rom.encode`.  Nothing is integrated, so the pose is exactly what the decoder says; the residual is reported and a miss beyond `--solve-tol` aborts rather than render a wrong pose."""
    name = "pose"

    @staticmethod
    def add_args(ap):
        JointTargets.add_args(ap)
        g = ap.add_argument_group("pose driver")
        g.add_argument("--solve-tol", type=float, default=1e-3, help="largest acceptable joint-angle residual in radians")

    def _angles_fn(self, sim, ch, idx):
        import torch
        from atlas_rom.rom import ops
        rom = sim.rom
        joints = [sim.joints[k] for k in idx]

        def f(u):
            Tm = rom.models[ch].transforms(u.unsqueeze(0)).squeeze(0)
            return torch.stack([ops.joint_angle_torch(Tm, j, rom.body_order) for j in joints])
        return f

    def _solve(self, sim, ch, u0, tg, idx, iters=60):
        import torch
        f = self._angles_fn(sim, ch, idx)
        u = torch.as_tensor(np.asarray(u0, float)).clone()
        tgt = torch.as_tensor([tg[k] for k in idx], dtype=torch.float64)
        wrap = lambda e: torch.remainder(e + np.pi, 2 * np.pi) - np.pi
        lam = 1e-3
        fv = lambda z: f(z).detach()
        err = wrap(tgt - fv(u))
        for _ in range(iters):
            if float(err.abs().max()) < 1e-6:
                break
            J = torch.autograd.functional.jacobian(f, u)                 # (n, d)
            step = torch.linalg.solve(J.T @ J + lam * torch.eye(len(u), dtype=torch.float64), J.T @ err)
            new = wrap(tgt - fv(u + step))
            if float(new.abs().max()) < float(err.abs().max()):
                u, err, lam = u + step, new, max(lam * 0.3, 1e-9)
            else:
                lam *= 10.0
        return u.detach().numpy(), float(err.abs().max())

    def _pose_to(self, sim, t):
        idx, tg = self.joint_idx(sim), self.targets(sim, t)
        st = sim.trk.state
        cur, q = int(st["chart"]), np.asarray(sim.pose(), float).reshape(-1)
        best = None
        for c in [cur] + [c for c in range(len(sim.rom.models)) if c != cur]:
            u0 = st["u"] if c == cur else np.asarray(sim.rom.encode(c, q), float)
            u, res = self._solve(sim, c, u0, tg, idx)
            if best is None or res < best[2]:
                best = (c, u, res)
            if res < self.args.solve_tol:
                break
        c, u, res = best
        self.worst = max(self.worst, res)
        if res > self.args.solve_tol:
            want = {sim.acts[k].name: round(tg[k], 4) for k in idx}
            raise SystemExit(f"pose driver: no chart reaches {want} (best residual {res:.4f} rad > --solve-tol {self.args.solve_tol}); the targets are probably outside the mechanism's reachable range")
        st.update(chart=c, u=np.asarray(u, float), p=np.zeros_like(u), time=float(t))

    def prepare(self, sim, spf):
        self.worst = 0.0
        self._pose_to(sim, 0.0)
        return {"kinematic": True, "solve_residual": self.worst}

    def advance(self, sim, spf, t):
        self._pose_to(sim, t + spf * sim.settings.dt)

    def report(self, sim):
        return {"actuators": self.names, "targets": self.spec, "worst_solve_residual": self.worst, "closure_residual": sim.closure(sim.pose()), "chart": int(sim.trk.state["chart"])}


class PenDriver(Driver):
    name = "pen"

    @staticmethod
    def add_args(ap):
        g = ap.add_argument_group("pen driver")
        g.add_argument("--curve", help="JSON list of [x, y] control points (closed curve)")
        g.add_argument("--save-curve", default=None, help="write the resampled closed curve as JSON (reloadable with --curve)")
        g.add_argument("--rotate", type=float, default=0.0, help="rotate the curve by this many degrees (counter-clockwise) about --pivot before tracing")
        g.add_argument("--pivot", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"), help="rotation centre in the mechanism's xy plane (default: the origin)")
        g.add_argument("--no-preroll", action="store_true", help="skip the unrecorded settle that brings the pen onto the curve's nearest point before the clip starts")
        g.add_argument("--smooth", type=float, default=None, help="spline smoothing per point (default: RomSim default; 0 = interpolate exactly)")
        g.add_argument("--trace-speed", type=float, default=0.3, help="reference-point speed along the curve (world units / s)")
        g.add_argument("--laps", type=float, default=1.0, help="number of times to trace the whole curve")
        g.add_argument("--wait", type=float, default=1.0, help="seconds to keep simulating after the curve is fully traced")
        g.add_argument("--rain-rate", type=float, default=8.0, help="balls per second")
        g.add_argument("--rain-delay", type=float, default=1.0, help="seconds before the rain starts")
        g.add_argument("--ball-radius", type=float, default=0.04)
        g.add_argument("--ball-mass", type=float, default=0.05)
        g.add_argument("--restitution", type=float, default=0.5)

    def __init__(self, args):
        super().__init__(args)
        if not args.curve:
            raise SystemExit("--drive pen needs --curve")
        with open(args.curve, encoding="utf-8") as f:
            data = json.load(f)
        pts = np.asarray(data["points"] if isinstance(data, dict) else data, float)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 4:
            raise SystemExit(f"{args.curve}: need a list of at least 4 [x, y] points")
        ctrl = pts[:, :2]
        if args.rotate:
            a, c = np.radians(args.rotate), np.asarray(args.pivot, float)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            ctrl = (ctrl - c) @ R.T + c
        self.ctrl, self.t_done, self.preroll = ctrl, None, {"seconds": 0.0, "final_error": None}
        self.track_err = []

    def make_sim(self, rom, args):
        from viewers.rom_target import RomSim                           # lazy: rom_target pulls in the interactive viewer's modules
        sim = RomSim(rom, args.case, dt=args.dt, damping=args.damping)
        if not sim.set_curve(self.ctrl, smooth=args.smooth):
            raise SystemExit("curve needs at least 4 distinct points")
        sim.trace_speed, sim.trace_on = args.trace_speed, True
        ra = sim.rain
        ra.radius, ra.mass, ra.restitution, ra.rate = args.ball_radius, args.ball_mass, args.restitution, args.rain_rate
        self.t_trace = args.laps * sim.curve["length"] / args.trace_speed
        if args.save_curve:
            with open(args.save_curve, "w", encoding="utf-8") as f:
                json.dump([[float(x), float(y)] for x, y in sim.curve["pts"]], f)
        return sim

    def prepare(self, sim, spf):
        a, ra = self.args, sim.rain
        if a.no_preroll:
            return self.preroll
        dt = sim.settings.dt                                            # unrecorded settle: the controller drives the pen (real ROM dynamics, no teleport) onto the curve point nearest its rest position
        sim.target_pt, sim.trace_on, ra.on = sim.curve_point(sim.trace_s), False, False
        ok, err = 0, None
        for _ in range(int(20.0 / (spf * dt))):
            sim.step(spf)
            if sim.error:
                break
            self.preroll["seconds"] += spf * dt
            err = float(np.linalg.norm(sim.pen_points()[sim.drag_pen][:2] - sim.target_pt))
            ok = ok + 1 if err < 0.005 else 0
            if ok >= 10:
                break
        self.preroll["final_error"] = err
        sim.trace_on = True
        return self.preroll

    def control(self, sim, t):
        a, ra = self.args, sim.rain
        ra.on = t >= a.rain_delay
        if self.t_done is None and sim.laps >= a.laps:                  # requested laps traced: freeze the reference point where it is, start the wait
            sim.trace_on, self.t_done = False, t
        return self.t_done is not None and t >= self.t_done + a.wait

    def duration(self):
        return self.t_trace + self.args.wait

    def extras(self, sim):
        ra = sim.rain
        if sim.target_pt is not None:
            self.track_err.append(float(np.linalg.norm(sim.pen_points()[sim.drag_pen][:2] - sim.target_pt)))
        out = {"target": np.full(2, np.nan) if sim.target_pt is None else np.asarray(sim.target_pt, float).copy(),
               "ball_counts": np.int64(len(ra.x)), "ball_xy": Ragged(np.asarray(ra.x, float).reshape(-1, 2).copy()),
               "contacts": np.int64(ra.n_contacts), "max_overlap": float(ra.max_pen)}
        out.update({f"pens_{n}": p.copy() for n, p in sim.pen_points().items()})
        return out

    def static(self, sim):
        return {"curve_ctrl": self.ctrl, "curve_pts": sim.curve["pts"], "ball_radius": self.args.ball_radius}

    def report(self, sim):
        te, ra, a = np.asarray(self.track_err), sim.rain, self.args
        return {"curve": os.path.abspath(a.curve), "curve_length": sim.curve["length"], "laps": sim.laps,
                "complete": bool(self.t_done is not None and not sim.error), "preroll": self.preroll,
                "balls_final": int(len(ra.x)), "ball_hits": int(ra.hits), "peak_overlap": float(ra.peak_pen), "overlap_over_radius": float(ra.peak_pen / a.ball_radius),
                "tracking_error": {"mean": float(te.mean()), "p95": float(np.percentile(te, 95)), "max": float(te.max())} if len(te) else None}


DRIVERS = {d.name: d for d in (FreeDriver, RestDriver, PoseDriver, JointDriver, PenDriver)}
