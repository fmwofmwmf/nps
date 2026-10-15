"""mesh_view.py — generic rigid-mesh rendering for any declared mechanism.

Extracted out of the archived ``atlas_rom_experiments/assembly.py`` (which also
carried the old ``atlas.json`` chart-decoder assembly, gone here) as the one
piece both the production full-space simulator and the ROM viewer still need:
build each body's mesh from the same declarative rule the simulator used, and
pose it by a decoded/simulated transform.  Depends only on ``geometry.py``, so
neither viewer needs a renderer written per case.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec import geometry


class MeshAdapter:
    """Draw any rigid-mesh mechanism: static bodies once, driven bodies per frame."""

    DRIVEN_COLOR = (0.90, 0.45, 0.20)
    STATIC_COLOR = (0.60, 0.62, 0.66)

    def __init__(self, mechanism):
        self.mechanism = mechanism
        bodies = geometry.build_bodies(mechanism)
        self.names = bodies["names"]
        self.local = list(zip(bodies["V"], bodies["F"]))
        self.colors = [b.color for b in mechanism.driven_bodies]
        self.static = geometry.build_static(mechanism)
        self.up_dir = f"{mechanism.render_up_axis}_up"
        self.meshes = []

    def posed(self, transforms):
        return [geometry.apply_transform(V, T)
                for (V, _), T in zip(self.local, np.asarray(transforms, float))]

    def register(self, ps, transforms, *, name_prefix="", include_static=True,
                 color_override=None):
        if include_static:
            for body in self.static:
                m = ps.register_surface_mesh(name_prefix + body["name"],
                                             body["V"].astype(np.float32), body["F"])
                m.set_color(tuple(body["color"]) if body["color"] is not None
                            else self.STATIC_COLOR)
                m.set_transparency(0.35)
        self.meshes = []
        for name, (_, F), color, verts in zip(self.names, self.local, self.colors,
                                              self.posed(transforms)):
            m = ps.register_surface_mesh(name_prefix + name, verts.astype(np.float32), F)
            m.set_color(tuple(color_override) if color_override is not None
                        else (tuple(color) if color is not None else self.DRIVEN_COLOR))
            self.meshes.append(m)

    def update(self, transforms):
        for m, verts in zip(self.meshes, self.posed(transforms)):
            m.update_vertex_positions(verts.astype(np.float32))
