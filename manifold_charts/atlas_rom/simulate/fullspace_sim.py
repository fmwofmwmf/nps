"""fullspace_sim.py — production full-space libuipc simulator, control-schema driven.

Wraps ``oracle.build_oracle`` behind a ``reset()/step(controls)/pose()``
interface keyed on the SAME ordered ``case.control_schema`` a staged ROM's
``PatchAtlasRuntime`` is driven by, so a control vector means the same thing
in either backend (see ``run_patch_atlas.py``'s unified Full/ROM runner and
``patch_rom.selfcheck``'s cross-backend parity checks).

v1 scope, deliberately, matching ``patch_rom.runtime.PatchAtlasRuntime``:
only ``joint_torque`` actuators are drivable (the oracle's per-joint
``AffineBodyRevoluteJointExternalForce`` channel).  A ``coordinate_force``
actuator has no full-space force channel wired here either -- refused at
construction, not silently ignored.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.simulate.oracle import build_oracle


class FullSpaceSim:
    def __init__(self, case, *, dt=2e-2, gravity=None, joint_strength=1e9,
                 d_hat=None, max_kappa=None, newton_max_iter=None, log=print):
        self.case = case
        self.dt = float(dt)
        self.gravity = float(case.mechanism.gravity if gravity is None else gravity)
        self.actuators = list(case.actuators)
        for a in self.actuators:
            if a.kind != "joint_torque":
                raise NotImplementedError(
                    f"actuator {a.name!r} is a {a.kind!r} actuator; FullSpaceSim v1 "
                    "only drives joint_torque actuators (no full-space force "
                    "channel is wired for coordinate_force here)")
        sampling = getattr(case.build, "sampling", None)
        self.oracle = build_oracle(
            case, dt=self.dt, gravity=self.gravity, joint_strength=joint_strength,
            d_hat=d_hat if d_hat is not None else getattr(sampling, "d_hat", None),
            max_kappa=max_kappa if max_kappa is not None else getattr(sampling, "max_kappa", None),
            newton_max_iter=(newton_max_iter if newton_max_iter is not None
                             else getattr(sampling, "newton_max_iter", None)),
            log=log)
        self.oracle.state["drive_on"] = False       # free dynamics by default
        self._initial_q = self.oracle.pack().copy()
        self.time = 0.0

    def reset(self) -> np.ndarray:
        self.oracle.state["drive_on"] = True
        self.oracle._set_target(self._initial_q)
        self.oracle._settle(4)
        self.oracle.state["drive_on"] = False
        for a in self.actuators:
            self.oracle.set_joint_torque(a.target, 0.0)
        self.time = 0.0
        return self.pose()

    def pose(self) -> np.ndarray:
        n = len(self.case.mechanism.driven_bodies)
        return self.oracle.pack().reshape(n, 3, 4)

    def step(self, controls=None) -> np.ndarray:
        controls = (np.zeros(len(self.actuators)) if controls is None
                   else np.asarray(controls, float).reshape(-1))
        if controls.shape[0] != len(self.actuators):
            raise ValueError(f"controls has {controls.shape[0]} entries, "
                             f"expected {len(self.actuators)} (one per actuator)")
        for a, tau in zip(self.actuators, controls):
            self.oracle.set_joint_torque(a.target, a.gain * float(tau))
        if getattr(self.oracle, "reduced_coordinates", False):
            self.oracle.advance_reduced(1)
        else:
            self.oracle.world.advance()
            self.oracle.world.retrieve()
        self.time += self.dt
        return self.pose()
