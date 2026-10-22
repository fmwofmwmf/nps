"""libuipc-backed projection oracle retained by the physical-AE pipeline."""

import numpy as np
from uipc import view

from atlas_rom.simulate.oracle_base import ManifoldOracle


PROJECTION_MODES = ("drive_relax", "proximal_step")


class LibuipcConfigOracle(ManifoldOracle):
    """Project a packed configuration onto the IPC-feasible set.

    ``drive_relax`` (default): soft-drive toward the target for up to
    ``drive_iters`` frames, then relax for up to ``relax_iters``.

    ``proximal_step``: ONE implicit-Euler step whose inertial target is the
    command.  The solver state is set to the baseline ``x_n`` with affine
    velocity ``(q - x_n) / dt``, so ``x_tilde = q`` and the step solves
    ``argmin_x 1/2 |x - q|_M^2 + dt^2 (contact + joints + ABD)(x)`` with CCD
    along the path from the baseline.  State is read and written through
    ``AffineBodyStateAccessorFeature`` because the backend owns it after
    ``world.init``.

    One step stops at libuipc's Newton tolerance, which is not a projection
    tolerance.  The step is therefore repeated from its own result toward the
    same target (same objective, since ``x_tilde = q`` again) until the largest
    packed change is ``<= proximal_tol``; failing that within
    ``proximal_max_steps`` raises, so the ledger records a non-converged
    response instead of a loose pose.
    """

    def __init__(self, world, scene, driven, target, state,
                 drive_iters=40, relax_iters=40, settle_tol=1e-8,
                 min_iters=2, probe_count=32, engine=None, actuated=None,
                 hold_drive_during_projection=False,
                 projection_mode="drive_relax", accessor=None, body_index=None,
                 dt=None, proximal_tol=1e-5, proximal_max_steps=8):
        super().__init__()
        self.proximal_tol = float(proximal_tol)
        self.proximal_max_steps = int(proximal_max_steps)
        self.last_proximal_steps = 0
        self.last_proximal_change = float("nan")
        if projection_mode not in PROJECTION_MODES:
            raise ValueError(f"unknown projection_mode {projection_mode!r}; "
                             f"expected one of {PROJECTION_MODES}")
        self.projection_mode = projection_mode
        if projection_mode == "proximal_step":
            if accessor is None or body_index is None or dt is None:
                raise ValueError("proximal_step needs the ABD state accessor, "
                                 "body_index and dt")
            self._accessor = accessor
            self._body_index = [int(body_index[name]) for name, _slot in driven]
            self._dt = float(dt)
            self._state_geo = accessor.create_geometry()
            from uipc import Matrix4x4, builtin
            self._state_geo.instances().create(builtin.transform, Matrix4x4.Zero())
            self._state_geo.instances().create(builtin.velocity, Matrix4x4.Zero())
            state["drive_on"] = False
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
        if self.projection_mode == "proximal_step":
            self._write_state(packed.reshape(len(self.driven), 3, 4), None)
            self.state["drive_on"] = False
            return packed.copy()
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

    def oracle_fingerprint(self):
        base = super().oracle_fingerprint()
        return base if self.projection_mode == "drive_relax" else f"{base}:{self.projection_mode}"

    def _read_state(self):
        """Backend transforms of the driven bodies, shape (n_driven, 4, 4)."""
        self._accessor.copy_to(self._state_geo)
        transforms = np.asarray(view(self._state_geo.transforms()), float).reshape(-1, 4, 4)
        return transforms[self._body_index].copy()

    def _write_state(self, blocks, velocities):
        """Push driven-body transforms (n, 3, 4) and affine velocities (n, 3, 4)
        to the backend; ``velocities=None`` means at rest.  Undriven bodies keep
        their backend state."""
        from uipc import builtin
        self._accessor.copy_to(self._state_geo)
        transforms = view(self._state_geo.transforms())
        velocity = view(self._state_geo.instances().find(builtin.velocity))
        for k, row in enumerate(self._body_index):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :4] = blocks[k]
            transforms[row] = transform
            rate = np.zeros((4, 4), dtype=np.float64)
            if velocities is not None:
                rate[:3, :4] = velocities[k]
            velocity[row] = rate
        self._accessor.copy_from(self._state_geo)
        self.world.retrieve()

    def _proximal_step(self, current):
        rates = np.zeros_like(current)
        for i, (name, _slot) in enumerate(self.driven):
            if name not in self.actuated:
                continue                       # followers: x_tilde = x_n
            rates[i] = (self._target[name][:3, :4] - current[i]) / self._dt
        self._write_state(current, rates)
        self.world.advance()
        self.world.retrieve()
        self.n_advances += 1
        return self._read_state()[:, :3, :4]

    def _proximal_project(self, q):
        self._set_target(q)                    # orthonormalized per-body targets
        current = self._read_state()[:, :3, :4]
        change = float("inf")
        steps = 0
        while steps < self.proximal_max_steps:
            result = self._proximal_step(current)
            steps += 1
            change = float(np.abs(result - current).max())
            current = result
            # The first step moves toward the target by design; convergence is
            # only evidenced by a later step that no longer moves.
            if steps >= 2 and change <= self.proximal_tol:
                break
        # Leave the backend at rest so nothing downstream inherits the
        # injected velocity.
        self._write_state(current, None)
        self.last_proximal_steps, self.last_proximal_change = steps, change
        if change > self.proximal_tol:
            raise RuntimeError(f"proximal projection not converged after {steps} steps: "
                               f"last packed change {change:.3e} > tol {self.proximal_tol:.1e}")
        return current.reshape(-1)

    def pack(self):
        if self.projection_mode == "proximal_step":
            return self._read_state()[:, :3, :4].reshape(-1)
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
        if self.projection_mode == "proximal_step":
            current = self._proximal_project(q)
            self.n_settles += 1
            return current
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
