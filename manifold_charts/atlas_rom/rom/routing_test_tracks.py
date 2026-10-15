"""routing_test_tracks.py — isolated tracks for chart-routing experiments.

Shared pieces for chart-routing viewers (Track, Settings, pose metrics, references).  Every track
exposes the same small surface::

    reset()            back to the shared initial state
    step(controls)     advance one ``dt``; returns False when the step blocked
    pose()             decoded (n_bodies, 3, 4) pose
    diagnostics()      dict of live readouts (chart, handoffs, residuals, ...)

Routing CHOICE (which chart receives the state) and handoff INITIALIZATION
(which coordinate/velocity the receiver starts from) are separate strategy
objects, so a track is ``router x handoff``:

    CurrentRoutingTrack    plain ``PatchAtlasRuntime`` (hysteresis + E_j(D_i(u)))
    ProjectedHandoffTrack  same routing decisions, destination coordinate found
                           by decoder-space projection, velocity through the
                           decoder tangents
    ConstraintOracleTrack  every chart starts from the same last-good pose and
                           velocity, is advanced ``lookahead`` learned steps,
                           and the lowest joint-closure error wins (rollback
                           and halt when none is acceptable); no reference

No libuipc / FullSpaceSim import anywhere in this module.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp  # noqa: F401

from atlas_rom.rom.handoff import EncoderHandoff, ProjectedHandoff, decode_flat, decoder_jacobian, project_pose, velocity_from_pose_rate



# ─── pose metrics ─────────────────────────────────────────────────────────────

def pose_gap(a, b) -> tuple[float, float]:
    """(max body translation distance, max body rotation angle in degrees)."""
    a = np.asarray(a, float).reshape(-1, 3, 4)
    b = np.asarray(b, float).reshape(-1, 3, 4)
    trans = np.linalg.norm(a[:, :, 3] - b[:, :, 3], axis=1)
    rel = np.einsum("nij,nkj->nik", a[:, :, :3], b[:, :, :3])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(trans.max()), float(np.degrees(np.arccos(cos)).max())


# ─── decoder-space projection (shared by handoff + oracle) ───────────────────







def tangent_angle_deg(Ja, Jb) -> float:
    """Largest principal angle (degrees) between the column spaces of two decoder Jacobians."""
    Qa, _ = np.linalg.qr(np.asarray(Ja, float)); Qb, _ = np.linalg.qr(np.asarray(Jb, float))
    cmin = float(np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False).min(), -1.0, 1.0))
    return float(np.degrees(np.arccos(cmin)))




# ─── handoff strategies: how a receiver chart is initialized ─────────────────





# ─── routing strategies: which chart receives the state ──────────────────────





# ─── tracks ───────────────────────────────────────────────────────────────────

class Settings:
    """Identical across every track in one viewer session."""

    def __init__(self, *, dt=1e-3, gravity=None, sdf_enabled=True, project_boundary=True,
                 stall_guard=True, s_stay=0.10, s_receive=0.05, dwell=0,
                 oracle_margin=0.25, lookahead=3, oracle_tol=1e-2,
                 oracle_pose_tol=5e-3, oracle_stay_frac=0.1, damping=None, oracle_tangent_tol=3.0):
        self.dt, self.gravity = dt, gravity
        self.sdf_enabled, self.project_boundary = sdf_enabled, project_boundary
        self.stall_guard = stall_guard
        self.s_stay, self.s_receive = s_stay, s_receive
        self.dwell = int(dwell)              # min steps between handoffs
        self.oracle_margin = oracle_margin   # oracle: relative closure gain to switch
        self.lookahead = int(lookahead)      # oracle: candidate rollout steps
        self.oracle_tol = oracle_tol         # oracle: max acceptable closure residual
        # oracle: a receiver must reproduce the SOURCE pose.  Closure alone only says the continuation is a valid pose
        # on the manifold -- any chart's decode satisfies it -- so the decoder roundtrip delta
        # rms(D_j(u_j) - D_i(u_i)) (u_j = decoder projection of the source pose) must be <= oracle_pose_tol * std(source pose).
        # None disables the check (legacy behaviour).
        self.oracle_pose_tol = oracle_pose_tol
        # oracle: an acceptable active chart is only left once its closure exceeds oracle_stay_frac * oracle_tol; a relative
        # improvement between two closures far below the tolerance is noise (legacy behaviour: 0.0).
        self.oracle_stay_frac = oracle_stay_frac
        # oracle: velocity transport.  Even when the receiver reproduces the pose, a velocity only transfers faithfully if the receiver's
        # decoder tangent plane matches the source's: the largest principal angle (degrees) between range(J_Dsource) and range(J_Dreceiver)
        # at the projected pose must be <= oracle_tangent_tol.  None disables the check.
        self.oracle_tangent_tol = oracle_tangent_tol
        # Rayleigh velocity damping (1/s) of every track's runtime; None keeps the runtime's own default (4.0).
        self.damping = damping


class Track:
    """Common state, readouts and dwell handling; subclasses build ``rt``."""
    label = "track"

    def __init__(self, rom, settings: Settings, reference=None):
        self.rom = rom
        self.settings = settings
        self.reference = reference
        self.rt = self._make_runtime()
        self.apply_settings()
        self.reset()

    # -- construction / settings ---------------------------------------------
    def _make_runtime(self):
        raise NotImplementedError

    def apply_settings(self):
        s, rt = self.settings, self.rt
        rt.dt = float(s.dt)
        if s.gravity is not None:
            rt.gravity = float(s.gravity)
        rt.set_sdf_enabled(s.sdf_enabled)
        rt.set_boundary_projection(s.project_boundary)
        rt.stall_guard = bool(s.stall_guard)
        rt.s_stay, rt.s_receive = float(s.s_stay), float(s.s_receive)
        if s.damping is not None:
            rt.velocity_damping = max(0.0, float(s.damping))

    def set_reference(self, reference):
        self.reference = reference
        self.reset()

    # -- interface -------------------------------------------------------------
    def reset(self):
        self.state = self.rt.initial_state(chart=0)
        self.steps = 0
        self._since_handoff = 10 ** 9
        self._jump = None            # latest handoff measurements
        self._residual = None
        self._pose_cache = None
        self.blocked_last = False

    def pose(self):
        return self.rt.pose(self.state)

    def step(self, controls) -> bool:
        self.apply_settings()
        before = self.state["handoffs"]
        ok = self._advance(controls)
        self.steps += 1
        self._since_handoff = 0 if self.state["handoffs"] != before else self._since_handoff + 1
        self.blocked_last = not ok
        return ok

    def _advance(self, controls) -> bool:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        st, rt = self.state, self.rt
        bd = rt.boundary_diagnostics(st)
        e = rt.energy(st)
        q = decode_flat(self.rom, st["chart"], st["u"])
        d = {"track": self.label, "chart": st["chart"], "n_charts": self.rom.n_patches,
             "handoffs": st["handoffs"], "stalls": st["stalls"], "blocked": st["blocked"],
             "time": st["time"], "support": bd["support"], "sdf": bd["sdf"],
             "closure": rt._closure_residual(q), "energy": e["total"],
             "kinetic": e["kinetic"], "potential": e["potential"],
             "jump_trans": None, "jump_rot": None, "projection_residual": self._residual,
             "err_trans": None, "err_rot": None}
        if self._jump is not None:
            d["jump_trans"], d["jump_rot"] = self._jump
        if self.reference is not None:
            d["err_trans"], d["err_rot"] = pose_gap(self.pose(), self.reference.pose_at(st["time"]))
        return d

    # -- handoff measurement ---------------------------------------------------
    def _record_handoff(self, q_pre_pose, residual):
        self._jump = pose_gap(self.pose(), q_pre_pose)
        self._residual = residual










# ─── reference trajectories ───────────────────────────────────────────────────



class Reference:
    """Packed-pose trajectory ``(T, n_bodies, 3, 4)`` sampled every ``dt``."""

    POSE_KEYS = ("poses", "pose", "q", "T", "transforms")

    def __init__(self, name, poses, dt, controls=None):
        self.name = name
        self.poses = np.asarray(poses, float)
        self.poses = self.poses.reshape(self.poses.shape[0], -1, 3, 4)
        self.dt = float(dt)
        self.controls = None if controls is None else np.asarray(controls, float)

    @property
    def n_frames(self):
        return len(self.poses)

    def index(self, t):
        return int(min(max(round(t / self.dt), 0), self.n_frames - 1))

    def exhausted(self, t):
        return round(t / self.dt) >= self.n_frames - 1

    def pose_at(self, t):
        return self.poses[self.index(t)]

    def flat_at(self, t):
        return self.pose_at(t).reshape(-1)

    def rate_at(self, t):
        k = self.index(t)
        lo, hi = max(k - 1, 0), min(k + 1, self.n_frames - 1)
        if hi == lo:
            return np.zeros(self.poses[k].size)
        return (self.poses[hi] - self.poses[lo]).reshape(-1) / ((hi - lo) * self.dt)

    def controls_at(self, t):
        return None if self.controls is None else self.controls[self.index(t)]


def load_references(path, *, default_dt) -> list[Reference]:
    """``.npz`` with a pose array under ``poses``/``pose``/``q``/``T``/
    ``transforms`` shaped ``(T, n, 3, 4)`` or ``(N, T, n, 3, 4)`` (N
    trajectories); optional ``dt`` (scalar) and ``controls`` ((T, n_act) or
    (N, T, n_act))."""
    import os
    data = np.load(path, allow_pickle=True)
    key = next((k for k in Reference.POSE_KEYS if k in data), None)
    if key is None:
        raise ValueError(f"{path}: expected one of {Reference.POSE_KEYS}, found {list(data.keys())}")
    arr = np.asarray(data[key], float)
    dt = float(data["dt"]) if "dt" in data else float(default_dt)
    ctl = np.asarray(data["controls"], float) if "controls" in data else None
    base = os.path.splitext(os.path.basename(path))[0]
    if arr.ndim <= 4:
        return [Reference(base, arr, dt, ctl)]
    if arr.ndim == 5:
        return [Reference(f"{base}[{i}]", arr[i], dt, None if ctl is None else ctl[i])
                for i in range(len(arr))]
    raise ValueError(f"{path}: cannot interpret pose array of shape {arr.shape}")
