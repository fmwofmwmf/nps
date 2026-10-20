"""viewers/rom_record.py -- records any `RomBase` run into the arrays `viewers/blender_import_rom.py` reads.  No driver, pen or Blender knowledge lives here.

`Recorder.record(extras)` stores the current link transforms and time plus whatever per-frame channels a driver hands over:
  plain array / scalar  -> stacked to (F, ...)
  `Ragged(array)`       -> concatenated along axis 0 (variable count per frame, e.g. balls); pair it with a per-frame count channel to split it again

    frames.npz   pose (F, nb, 3, 4) in `body_order`; time (F,); fps; dt; body_order; every extras channel; every driver `static` array
    meshes.npz   V_<i>, F_<i> local-frame mesh of body i; world vertex = pose[i,:, :3] @ V + pose[i,:, 3]
"""
from __future__ import annotations

import os

import numpy as np


class Ragged:
    def __init__(self, a):
        self.a = np.asarray(a, float)


class Recorder:
    def __init__(self, sim):
        self.sim = sim
        self.pose, self.time, self.chan = [], [], {}

    def record(self, extras=None):
        self.pose.append(np.asarray(self.sim.pose(), float).copy())
        self.time.append(self.sim.time())
        for k, v in (extras or {}).items():
            self.chan.setdefault(k, []).append(v)

    def __len__(self):
        return len(self.pose)

    def save(self, out, fps, dt, static=None):
        os.makedirs(out, exist_ok=True)
        arrays = {}
        for k, vs in self.chan.items():
            if isinstance(vs[0], Ragged):
                arrays[k] = np.concatenate([v.a for v in vs], 0)
            else:
                arrays[k] = np.asarray(vs)
        np.savez(os.path.join(out, "frames.npz"), pose=np.asarray(self.pose), time=np.asarray(self.time), fps=fps, dt=dt,
                 body_order=np.asarray(list(self.sim.rom.body_order)), **arrays, **(static or {}))
        np.savez(os.path.join(out, "meshes.npz"), **{f"{k}_{i}": np.asarray(a) for i, (V, Fc) in enumerate(self.sim.mesh_local) for k, a in (("V", V), ("F", Fc))})
