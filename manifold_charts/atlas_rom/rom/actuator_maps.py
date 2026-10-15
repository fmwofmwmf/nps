"""actuator_maps.py — per-actuator generalized-force maps for a staged ROM.

Every channel in the case's ``control_schema`` must resolve to a validated
map before a ROM package may enter ROM mode:

    joint_torque      dtheta/du, differentiated through the patch decoder
                      (``ops.actuator_jacobian``).  Always resolves wherever
                      the declared joint exists on the mechanism -- no
                      training needed, so "validated" means "structurally
                      resolvable" at stage time and "numerically sane" at
                      ``--selfcheck`` time (see ``selfcheck._check_control_parity``).

    coordinate_force  Needs a per-chart reduced actuator map calibrated
                      against the target coordinate's physical action.  The
                      patch-atlas pipeline has no live evaluator for a
                      declared coordinate outside the archived kinematics
                      stage, so this is NOT YET calibrated here -- marked
                      ``unsupported`` rather than silently applying a wrong
                      or absent force.  A case that declares one is refused
                      ROM mode, honestly, rather than running with a missing
                      control channel.
"""
from __future__ import annotations


def validate(case) -> dict:
    """Structural validation only (no trained model needed): can every
    declared actuator's target even be resolved on this mechanism?  Runtime
    numerical validation (does the map actually move the state, in the right
    direction) is ``selfcheck``'s job, not staging's."""
    per_actuator = []
    ok = True
    for a in case.actuators:
        if a.kind == "joint_torque":
            try:
                joint = case.mechanism.joint(a.target)
                if joint.kind != "revolute":
                    raise ValueError(f"joint {a.target!r} is {joint.kind!r}, "
                                     "not revolute -- no torque channel")
                entry = {"name": a.name, "kind": a.kind, "target": a.target,
                         "status": "analytic"}
            except (KeyError, ValueError) as exc:
                entry = {"name": a.name, "kind": a.kind, "target": a.target,
                         "status": "failed", "reason": str(exc)}
                ok = False
        elif a.kind == "coordinate_force":
            entry = {"name": a.name, "kind": a.kind, "target": a.target,
                     "status": "unsupported",
                     "reason": "coordinate_force actuator maps require per-chart "
                               "calibration against a live coordinate evaluator, "
                               "not yet implemented in patch_rom"}
            ok = False
        else:
            entry = {"name": a.name, "kind": a.kind, "target": a.target,
                     "status": "failed", "reason": f"unknown actuator kind {a.kind!r}"}
            ok = False
        per_actuator.append(entry)
    return {"pass": ok, "actuators": per_actuator}
