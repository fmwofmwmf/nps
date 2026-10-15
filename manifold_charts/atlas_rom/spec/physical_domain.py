"""Physical travel and workspace limits, independent of chart coordinates."""
from __future__ import annotations

import numpy as np


def _axis_angle(rotation, axis):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    cosine = np.clip((np.trace(rotation) - 1.) * .5, -1., 1.)
    angle = float(np.arccos(cosine))
    skew = np.array((rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]))
    return float((np.sign(axis @ skew) or 1.) * angle)


def joint_travel(transforms, joint, body_order):
    """Physical displacement of a revolute or sliding joint from its rest pose."""
    poses = np.asarray(transforms, float)
    ia = body_order.index(joint.body_a) if joint.body_a in body_order else None
    ib = body_order.index(joint.body_b) if joint.body_b in body_order else None
    if joint.kind == "revolute":
        left = np.eye(3) if ia is None else poses[ia, :, :3]
        right = np.eye(3) if ib is None else poses[ib, :, :3]
        return _axis_angle(left.T @ right, joint.axis)

    def anchor(index):
        return (np.asarray(joint.pivot, float) if index is None else
                poses[index, :, :3] @ np.asarray(joint.pivot, float) + poses[index, :, 3])

    rail = np.eye(3) if ia is None else poses[ia, :, :3]
    return float((anchor(ib) - anchor(ia)) @ (rail @ np.asarray(joint.axis, float)))


def violation(spec, transforms):
    """Return physical-bound diagnostics for a pose, else ``None``.

    Only mechanism joint and workspace declarations participate.  In particular,
    this module never reads ``CoordinateDef`` or the chart-coordinate map.
    """
    poses = np.asarray(transforms, float)
    order = spec.mechanism.body_order
    for joint in spec.mechanism.joints:
        if not joint.limited:
            continue
        travel = joint_travel(poses, joint, order)
        if travel < joint.limit_lo or travel > joint.limit_hi:
            return {"physical_boundary": "joint_limit", "joint": joint.name,
                    "travel": travel, "limit_lo": joint.limit_lo, "limit_hi": joint.limit_hi}
    for bound in spec.mechanism.workspace_bounds:
        body = spec.mechanism.body_index(bound.body)
        value = float(poses[body, "xyz".index(bound.axis), 3])
        if value < bound.lo or value > bound.hi:
            return {"physical_boundary": "workspace_bound", "workspace_bound": bound.name,
                    "value": value, "limit_lo": bound.lo, "limit_hi": bound.hi}
    return None


def signed_margin(spec, transforms):
    """Normalized signed distance to the nearest declared physical bound, ``> 0`` inside.

    For every limited joint and workspace bound, ``min(value - lo, hi - value) / (hi - lo)``; the pose's
    margin is the minimum over them (``inf`` when nothing is declared).  The same quantities and bounds as
    :func:`violation`, so ``signed_margin < 0`` exactly when ``violation`` is not ``None``.
    """
    poses = np.asarray(transforms, float)
    order = spec.mechanism.body_order
    margins = []
    for joint in spec.mechanism.joints:
        if not joint.limited:
            continue
        travel = joint_travel(poses, joint, order)
        span = max(float(joint.limit_hi) - float(joint.limit_lo), 1e-12)
        margins.append(min(travel - joint.limit_lo, joint.limit_hi - travel) / span)
    for bound in spec.mechanism.workspace_bounds:
        body = spec.mechanism.body_index(bound.body)
        value = float(poses[body, "xyz".index(bound.axis), 3])
        span = max(float(bound.hi) - float(bound.lo), 1e-12)
        margins.append(min(value - bound.lo, bound.hi - value) / span)
    return float(min(margins)) if margins else float("inf")
