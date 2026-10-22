"""viewers/rom_sim.py -- the case-agnostic ROM stepper: the learned fields + net-only router, with no pen, curve, rain or any other task attached.

`RomBase` owns what every viewer / exporter needs from a ROM and nothing case-specific: the router track (`AdvRouterTrack`), the rest-pose reset, `step(n)`, `pose()`, the local meshes, actuator
angles and the loop-closure residual.  Controls come from one overridable hook, `_controls()`; the default applies none (the ROM's free dynamics).  Subclasses add behaviour through three hooks
(`_pre_step`, `_controls`, `_post_step`) -- `viewers.rom_target.RomSim` adds the pen, `viewers.rom_drivers` supplies the controllers used for export.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from atlas_rom.rom import routing_test_tracks as T, ops  # noqa: E402
from atlas_rom.rom.adv_router import AdvRouterTrack  # noqa: E402
from atlas_rom.charts.validity import validity_from_case  # noqa: E402
from atlas_rom.spec.geometry import build_bodies  # noqa: E402


class RomBase:
    def __init__(self, rom, dt=0.01, damping=0.0, gravity=None):
        self.rom = rom
        self.acts = list(rom.case.actuators)
        self.joints = [rom.case.mechanism.joint(a.target) for a in self.acts]
        self.validity = validity_from_case(rom.case, include_grounded=False)
        self.anchor_trees = [cKDTree(np.asarray(a, float) / np.maximum(np.asarray(a, float).std(0), 1e-4)) for a in rom.anchors]
        self.anchor_std = [np.maximum(np.asarray(a, float).std(0), 1e-4) for a in rom.anchors]
        self.settings = T.Settings(dt=dt, lookahead=1, damping=damping, gravity=gravity)   # gravity None = the case's own
        self.trk = AdvRouterTrack(rom, self.settings)
        self.control_on = True
        self.controller = None                                      # optional callable(sim) -> (controls, rows); None = no control
        bodies = build_bodies(rom.mechanism)
        self.mesh_local = list(zip(bodies["V"], bodies["F"]))       # local-frame (V, F) per body, in `rom.body_order`
        self.mesh_color = [b.color for b in rom.mechanism.driven_bodies]   # the case's declared rgb per body (None where undeclared), same order
        self.rows = []
        self.error = ""
        self.reset()

    # ---- readouts --------------------------------------------------------------------------------------------------------------------------------------
    def pose(self):
        return self.trk.pose()

    def time(self):
        return float(self.trk.state["time"])

    def angles_of(self, Tm):
        pose = torch.as_tensor(Tm, dtype=torch.float64)
        return [float(ops.joint_angle_torch(pose, j, self.rom.body_order)) for j in self.joints]

    def angles(self):
        return self.angles_of(self.pose())

    def closure(self, Tm):
        if self.validity is None:
            return 0.0
        return float(np.mean(self.validity.residual(None, np.asarray(Tm, float))))

    # ---- state ----------------------------------------------------------------------------------------------------------------------------------------
    def reset(self):
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
        self.error = ""
        self.rows = []

    # ---- dynamics: three hooks, so a subclass changes what happens around a step without re-implementing the step -----------------------------------------
    def _pre_step(self, dt):
        pass

    def _controls(self):
        """(controls per actuator, rows).  Default: `self.controller(self)` if one is set, else no torque -- the ROM's free dynamics (gravity, learned fields, router)."""
        if self.controller is None:
            return [0.0] * len(self.acts), []
        return self.controller(self)

    def _post_step(self, dt):
        pass

    def step(self, n=1):
        rows = self.rows
        for _ in range(n):
            self._pre_step(self.settings.dt)
            ctl, rows = self._controls()
            try:
                self.trk.step(ctl)
            except (RuntimeError, np.linalg.LinAlgError, ValueError) as ex:
                self.error = f"{type(ex).__name__}: {ex}"
                break
            self._post_step(self.settings.dt)
        self.rows = rows
        return rows


class ReferenceSim:
    """No ROM at all: the case's own meshes at the rest pose (every body at its identity transform).  A ground-truth *reference* for stills of cases that have no trained ROM -- nothing
    here is learned or simulated, so a render from it must be labelled as the declared geometry, not as a ROM result.  Provides the slice of `RomBase` the recorder reads."""

    def __init__(self, case_name):
        import importlib
        self.case = importlib.import_module(f"atlas_rom.cases.{case_name}").CASE
        bodies = build_bodies(self.case.mechanism)
        self.rom = type("NoRom", (), {"body_order": list(bodies["names"])})()
        self.mesh_local = list(zip(bodies["V"], bodies["F"]))
        self.mesh_color = [b.color for b in self.case.mechanism.driven_bodies]
        self.acts, self.error = [], ""
        self.settings = T.Settings(dt=0.01, lookahead=1, damping=0.0)

    def pose(self):
        return np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (len(self.mesh_local), 1, 1))

    def time(self):
        return 0.0
