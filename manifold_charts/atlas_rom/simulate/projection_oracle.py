"""libuipc-backed projection oracle retained by the physical-AE pipeline."""

import numpy as np
from uipc import view

from atlas_rom.simulate.oracle_base import ManifoldOracle


class LibuipcConfigOracle(ManifoldOracle):
    """Project a packed configuration by drive, settle, then joint-only relax."""

    def __init__(self, world, scene, driven, target, state,
                 drive_iters=40, relax_iters=40, settle_tol=1e-8,
                 min_iters=2, probe_count=32, engine=None, actuated=None,
                 hold_drive_during_projection=False):
        super().__init__()
        self.engine = engine
        self.world = world
        self.scene = scene
        self.driven = driven
        self._target = target
        self.state = state
        self.actuated = set(actuated) if actuated is not None else {
            name for name, _slot in driven
        }
        self.hold_drive_during_projection = bool(hold_drive_during_projection)
        self.drive_iters = int(drive_iters)
        self.relax_iters = int(relax_iters)
        self.settle_tol = float(settle_tol)
        self.min_iters = int(min_iters)
        self.probe_count = int(probe_count)
        self.dim_ambient = 12 * len(driven)
        self.n_settles = 0
        self.n_advances = 0

    def snapshot(self, settled):
        """Capture simulator-owned transforms and drive state for probe replay."""
        return {"packed": np.asarray(settled, float).copy(),
                "targets": {name: np.asarray(value, float).copy()
                            for name, value in self._target.items()},
                "drive_on": bool(self.state.get("drive_on", False))}

    def restore(self, snapshot):
        """Put the UIPC scene back before every directional command.

        This deliberately restores transforms directly before a settle; merely
        driving from the previous probe result made probe order a hidden input.
        """
        packed = np.asarray(snapshot["packed"], float)
        blocks = packed.reshape(len(self.driven), 3, 4)
        for block, (_name, slot) in zip(blocks, self.driven):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :4] = block
            view(slot.geometry().transforms())[0] = transform
        self._target.update({name: np.asarray(value, float).copy()
                             for name, value in snapshot["targets"].items()})
        self.state["drive_on"] = bool(snapshot["drive_on"])
        # Do not re-project here: this is a state restore, not another physical
        # response.  The following commanded settle is the one ledgered query.
        return packed.copy()

    def pack(self):
        parts = []
        for _name, slot in self.driven:
            transform = np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4)
            parts.append(transform[:3, :4].reshape(12))
        return np.concatenate(parts)

    def seed(self):
        return self.pack()

    def _set_target(self, q):
        q = np.asarray(q, dtype=float).reshape(len(self.driven), 3, 4)
        for i, (name, _slot) in enumerate(self.driven):
            if name not in self.actuated:
                continue
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :4] = q[i]
            left, _singular, right_t = np.linalg.svd(transform[:3, :3])
            rotation = left @ right_t
            if np.linalg.det(rotation) < 0:
                left[:, -1] *= -1.0
                rotation = left @ right_t
            transform[:3, :3] = rotation
            self._target[name] = transform

    def _settle(self, n_iters):
        previous = self.pack()
        current = previous
        for iteration in range(n_iters):
            self.world.advance()
            self.world.retrieve()
            self.n_advances += 1
            current = self.pack()
            if iteration >= self.min_iters and np.linalg.norm(current - previous) < self.settle_tol:
                break
            previous = current
        return current

    def project(self, q):
        self._set_target(q)
        self.state["drive_on"] = True
        self._settle(self.drive_iters)
        self.state["drive_on"] = self.hold_drive_during_projection
        current = self._settle(self.relax_iters)
        self.n_settles += 1
        return current

    def tangent_frame(self, p, k=None, eps=1e-3, n_probe=None,
                      rel_tol=0.05, rng=None):
        if n_probe is None:
            n_probe = self.probe_count
        return super().tangent_frame(p, k=k, eps=eps, n_probe=n_probe,
                                     rel_tol=rel_tol, rng=rng)
