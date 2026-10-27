"""Closed-form clearance for declared contact pairs between sphere bodies.

A contact pair whose two bodies are both plain ``sphere`` meshes has an exact
non-penetration condition, ``|c_a - c_b| >= r_a + r_b``, read straight off the
declared geometry -- no mesh proximity query and no IPC barrier.  This module
only recognises such pairs; it never invents geometry for any other mesh kind,
so a mechanism with any non-sphere contact pair is reported as unsupported and
stays on the contact (libuipc) oracle.
"""
from __future__ import annotations

import numpy as np


def sphere_of(body) -> tuple[np.ndarray, float] | None:
    """(world centre at rest, radius) of a plain sphere body, else ``None``."""
    mesh = body.mesh
    if mesh.kind != "sphere":
        return None
    return (np.asarray(mesh.params.get("center", (0.0, 0.0, 0.0)), float),
            float(mesh.params.get("radius", 1.0)))


def sphere_pairs(mechanism) -> list[dict] | None:
    """Every declared contact pair as a sphere-sphere clearance, or ``None``.

    ``None`` when contact pairs are left to the broad phase (undeclared), when
    none are declared, or when any pair involves a non-sphere body: the closed
    form is only exact when it covers every contact the mechanism declares.
    """
    pairs = mechanism.contact_pairs
    if not pairs:
        return None
    bodies = {body.name: body for body in mechanism.bodies}
    out = []
    for name_a, name_b in pairs:
        if name_a not in bodies or name_b not in bodies:
            return None
        sphere_a, sphere_b = sphere_of(bodies[name_a]), sphere_of(bodies[name_b])
        if sphere_a is None or sphere_b is None:
            return None
        out.append({"body_a": name_a, "body_b": name_b,
                    "center_a": sphere_a[0], "center_b": sphere_b[0],
                    "clearance": sphere_a[1] + sphere_b[1]})
    return out
