"""SuperDex Physics contact-backed projection oracle.

Bodies interact through SDF contact.  The only non-contact elements are
revolutes to a fixed body, which have no ground geometry to touch: each is two
world-anchored pivot constraints on the joint axis.  Body-to-body joints are
rejected, so links can only be located by contact.

``project(q)`` is a proximal iteration.  Every step gives each body the velocity
that would carry it to ``q`` in one step, so backward Euler solves
``argmin_x 1/2 |x - q|_M^2 + dt^2 E(x)``.  Velocity is discarded after each step.
The push runs until the pose changes by at most ``push_tol``, then uncommanded
steps run until it changes by at most ``tol``, which releases contact load.
Failing that it raises, so the ledger records a non-converged response instead
of a loose pose.  There is no continuous collision detection, so the per-step
body motion is bounded by ``max_step_translation`` and ``max_step_rotation`` to
keep bodies from tunnelling through walls.

Density here is a numerical inertia for the projection, not the body's physical
mass, because the settled pose should be determined by contact.

With ``n_workers > 1``, batched responses run in separate processes.  Those are
spawned, so a script using them must guard its entry point with
``if __name__ == "__main__":``.
"""
from __future__ import annotations

import atexit
from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait as wait_for_connections
import os
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from atlas_rom.simulate.oracle_base import DirectionalResponse, ManifoldOracle
from atlas_rom.spec import geometry

MAX_SDF_CELLS = 128 ** 3            # per body; 8 bytes per grid vertex in fp64
_initialized = []


def _import_physics():
    """Import SuperDex in fp64; precision is fixed at first import."""
    os.environ.setdefault("SUPERDEX_PRECISION", "fp64")
    import superdex.physics as sdp
    if not sdp.uses_double_precision():
        raise RuntimeError("SuperDex oracle requires fp64 (SUPERDEX_PRECISION=fp64)")
    return sdp


def _sdf_params(sdp, vertices, voxel, max_cells):
    """SDF settings for one body, refusing a grid over budget before allocating."""
    cells = np.ceil(np.ptp(vertices, axis=0) / float(voxel)).astype(np.int64) + 1
    total = int(np.prod(cells))
    if total > int(max_cells):
        raise MemoryError(f"SDF grid {cells.tolist()} has {total} cells > budget {max_cells}")
    params = sdp.GridSdfParams()
    params.resolution_mode = sdp.GridSdfResolutionMode.EXPLICIT
    params.resolution_delta = [float(voxel)] * 3
    return params


def baked_sdf_grid(vertices, faces, voxel=0.0625, max_cells=MAX_SDF_CELLS):
    """The signed-distance grid SuperDex bakes for one body's contact collider.

    Same mesh and :func:`_sdf_params` as the oracle's colliders.  Returns
    ``(values, low, high)``: values indexed ``[x, y, z]`` (negative inside) at
    vertices spaced uniformly from ``low`` to ``high``.
    """
    sdp = _import_physics()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "body.obj")
        _write_obj(path, vertices, faces)
        model = sdp.model.load_from_file(path)
    sdp.model.bake_sdf(model, _sdf_params(sdp, vertices, voxel, max_cells))
    grid = model.sdf
    if grid.rotation is not None or grid.scale is not None or grid.translation is not None:
        raise ValueError("baked SDF grid carries a runtime transform; not supported")
    dims = [int(n) for n in grid.dims]
    values = np.asarray(grid.values, float).reshape(dims)        # z fastest
    low = np.asarray(list(grid.bounds.min), float)
    high = np.asarray(list(grid.bounds.max), float)
    return values, low, high


def _write_obj(path, vertices, faces):
    with open(path, "w", encoding="ascii") as fh:
        for x, y, z in vertices:
            fh.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in np.asarray(faces, int) + 1:
            fh.write(f"f {a} {b} {c}\n")


def _quaternion_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotation_vector(rotation):
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    sine = 0.5 * np.linalg.norm(axis)
    if sine < 1e-9:
        return 0.5 * axis
    angle = np.arctan2(sine, np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    return axis * (angle / (2.0 * sine))


def _worker_main(connection, spec, options):
    """Own one scene; answer (index, state bytes, target) until told to stop."""
    oracle = SuperDexContactOracle(spec, log=lambda *_: None, **options)
    connection.send("ready")
    while True:
        message = connection.recv()
        if message is None:
            return
        index, state, target = message
        try:
            if state is None:                # settle at the target itself
                oracle._write_pose(target)
            else:
                oracle.scene.restore_state_from_bytes(state)
            connection.send((index, oracle.project(target), None))
        except Exception as error:  # reported as a failed physical measurement
            connection.send((index, None, f"{type(error).__name__}: {error}"))


class _WorkerPool:
    """Fixed processes, one command each at a time; stuck or dead ones are replaced."""

    def __init__(self, spec, options, count, command_timeout, build_timeout=300.0):
        self._context = mp.get_context("spawn")
        self._spec, self._options = spec, options
        self.command_timeout = float(command_timeout)
        self.build_timeout = float(build_timeout)
        self.restarts = 0
        self._workers = [self._launch() for _ in range(int(count))]
        for worker in self._workers:
            self._await_ready(worker)

    def _launch(self):
        parent, child = self._context.Pipe()
        process = self._context.Process(target=_worker_main, daemon=True,
                                        args=(child, self._spec, self._options))
        process.start()
        child.close()
        return {"process": process, "connection": parent}

    def _await_ready(self, worker):
        if not worker["connection"].poll(self.build_timeout):
            raise RuntimeError("SuperDex worker did not finish building in time")
        worker["connection"].recv()

    def _replace(self, worker):
        worker["process"].kill()
        worker["process"].join()
        worker["connection"].close()
        replacement = self._launch()
        self._await_ready(replacement)
        self.restarts += 1
        return replacement

    def run(self, state, targets):
        """Return one (result, error) per target, in order."""
        results = [None] * len(targets)
        pending = deque(enumerate(targets))
        idle = list(self._workers)
        busy = {}                              # connection -> (worker, index, started)
        while pending or busy:
            while pending and idle:
                index, target = pending.popleft()
                worker = idle.pop()
                worker["connection"].send((index, state, target))
                busy[worker["connection"]] = (worker, index, time.monotonic())
            ready = wait_for_connections(list(busy), timeout=0.05)
            for connection in ready:
                worker, index, _started = busy.pop(connection)
                try:
                    _i, result, error = connection.recv()
                    results[index] = (result, error)
                    idle.append(worker)
                except (EOFError, OSError):
                    results[index] = (None, "RuntimeError: SuperDex worker exited")
                    idle.append(self._swap(worker))
            now = time.monotonic()
            for connection in [c for c, (_w, _i, s) in busy.items()
                               if now - s > self.command_timeout]:
                worker, index, _started = busy.pop(connection)
                results[index] = (None, f"TimeoutError: command exceeded {self.command_timeout} s")
                idle.append(self._swap(worker))
        return results

    def _swap(self, worker):
        replacement = self._replace(worker)
        self._workers[self._workers.index(worker)] = replacement
        return replacement

    def close(self):
        for worker in self._workers:
            try:
                worker["connection"].send(None)
            except (OSError, ValueError):
                pass
        for worker in self._workers:
            worker["process"].join(timeout=5.0)
            if worker["process"].is_alive():
                worker["process"].kill()
                worker["process"].join()
            worker["connection"].close()
        self._workers = []


class SuperDexContactOracle(ManifoldOracle):
    """Project a packed configuration onto SuperDex's contact-feasible set."""

    def __init__(self, spec, *, dt=0.02, density=1.0, hinge_stiffness=1e7, hinge_damping=0.0,
                 hinge_half_length=1.0, penalty_coefficient=None, sdf_voxel=0.0625,
                 max_sdf_cells=MAX_SDF_CELLS, tol=1e-4, push_tol=1e-3, max_steps=2000,
                 max_step_translation=0.05, max_step_rotation=0.02, max_point_step=None,
                 num_worker_threads=0, probe_directions=20, n_workers=1, command_timeout=60.0,
                 log=print):
        super().__init__()
        self.spec = spec
        self.dt = float(dt)
        self.tol = float(tol)
        self.push_tol = float(push_tol)
        # Closure tolerance callers apply to ``residual`` (re-settling a settled pose moves it ~3e-4 on
        # rotate_contact, against the exact-joint default 1e-6): the push phase's own acceptance tolerance.
        self.residual_tol = self.push_tol
        self.max_steps = int(max_steps)
        self.max_step_translation = float(max_step_translation)
        self.max_step_rotation = float(max_step_rotation)
        # When set, one cap on each body's step replaces the two above: its largest
        # material-point displacement |dt| + |w| r_max, with r_max the farthest
        # mesh vertex from its centre of mass.  A long body then cannot sweep a
        # point further than a compact one in a step, which the separate caps allow.
        self.max_point_step = None if max_point_step is None else float(max_point_step)
        self.sdf_voxel = float(sdf_voxel)
        self.max_sdf_cells = int(max_sdf_cells)
        self.probe_directions = int(probe_directions)
        self.n_workers = int(n_workers)
        self.command_timeout = float(command_timeout)
        self.n_settles = self.n_steps = 0
        self.last_steps = 0
        self.last_change = float("nan")
        self._log = log
        self._pool = None
        self._worker_options = dict(
            dt=dt, density=density, hinge_stiffness=hinge_stiffness, hinge_damping=hinge_damping,
            hinge_half_length=hinge_half_length, penalty_coefficient=penalty_coefficient,
            sdf_voxel=sdf_voxel, max_sdf_cells=max_sdf_cells, tol=tol, push_tol=push_tol,
            max_steps=max_steps, max_step_translation=max_step_translation,
            max_step_rotation=max_step_rotation, max_point_step=max_point_step,
            num_worker_threads=num_worker_threads)
        self._sdp = sdp = _import_physics()
        if not _initialized:                # one process-wide context
            sdp.initialize(num_worker_threads=int(num_worker_threads))
            _initialized.append(True)
        self.scene = sdp.create_scene(f"{spec.name} contact oracle")

        mechanism = spec.mechanism
        self.order = list(mechanism.body_order)
        self.dim_ambient = 12 * len(self.order)
        contact = sdp.ContactParams()
        contact.coulomb_friction_coefficient = 0.0
        if penalty_coefficient is not None:
            contact.penalty_coefficient = float(penalty_coefficient)

        driven = set(self.order)
        hinged = []
        for joint in mechanism.joints:
            moving = [name for name in (joint.body_a, joint.body_b) if name in driven]
            if joint.kind != "revolute" or len(moving) != 1:
                raise ValueError(f"{spec.name}: joint {joint.name!r} is not a revolute to a "
                                 "fixed body; links must be located by contact")
            hinged.append((joint, moving[0]))

        self.actors = {}
        vertices = {}
        with tempfile.TemporaryDirectory() as directory:
            for name in self.order:
                V, F = geometry.build_mesh(mechanism.body(name).mesh)
                vertices[name] = V
                path = os.path.join(directory, f"{name}.obj")
                _write_obj(path, V, F)
                self.actors[name] = self.scene.create_rigid_actor(
                    name=name, layer=name, shape=sdp.load_shape_from_file(file_path=path),
                    collider_type=sdp.ColliderType.SDF,
                    sdf=_sdf_params(sdp, V, sdf_voxel, max_sdf_cells),
                    contact=contact, has_gravity=False, density=float(density),
                    world_from_local=sdp.TransformRT())

        self.hinges = []
        for joint, moving in hinged:
            axis = np.asarray(joint.axis, float)
            axis = axis / np.linalg.norm(axis)
            for sign in (-1.0, 1.0):
                anchor = np.asarray(joint.pivot, float) + sign * float(hinge_half_length) * axis
                self.hinges.append(self.scene.create_rigid_pivot_position_constraint(
                    sdp.RigidPivotPositionConstraintParams(
                        actor=self.actors[moving].get_handle(),
                        local_position=anchor.tolist(), target_position=anchor.tolist(),
                        stiffness=float(hinge_stiffness), damping=float(hinge_damping),
                        saturation=-1.0)))

        # Local centre of mass: bodies are baked at rest, so root = identity here.
        self._com_local = {
            name: np.asarray(actor.get_center_of_mass_transform().translation.tolist(), float)
            for name, actor in self.actors.items()}
        self._reach = {name: float(np.linalg.norm(vertices[name] - self._com_local[name], axis=1).max())
                       for name in self.order}
        self._seed = self.pack()
        self._log(f"[superdex] {spec.name}: {len(self.order)} bodies, "
                  f"{len(self.hinges)} hinge anchors, precision {sdp.PRECISION_NAME}")

    def oracle_fingerprint(self):
        return (f"{ManifoldOracle.oracle_fingerprint(self)}:dt={self.dt}:tol={self.tol}:"
                f"push_tol={self.push_tol}:steps={self.max_steps}:"
                f"caps={self.max_step_translation},{self.max_step_rotation}:"
                f"sdf={self.sdf_voxel},{self.max_sdf_cells}:point_step={self.max_point_step}")

    def seed(self):
        return self._seed.copy()

    def pack(self):
        packed = np.empty(self.dim_ambient)
        block = np.empty((3, 4))
        for i, name in enumerate(self.order):
            transform = self.actors[name].get_root_transform()
            block[:, :3] = _quaternion_matrix(transform.rotation.tolist())
            block[:, 3] = transform.translation.tolist()
            packed[12 * i:12 * (i + 1)] = block.ravel()
        return packed

    def _blocks(self, q):
        """Packed pose -> (n, 3, 4) with each rotation orthonormalized."""
        blocks = np.array(q, float).reshape(len(self.order), 3, 4)
        for block in blocks:
            left, _singular, right_t = np.linalg.svd(block[:, :3])
            if np.linalg.det(left @ right_t) < 0:
                left[:, -1] *= -1.0
            block[:, :3] = left @ right_t
        return blocks

    def _write_pose(self, q):
        sdp = self._sdp
        for name, block in zip(self.order, self._blocks(q)):
            quat = Rotation.from_matrix(block[:, :3]).as_quat()
            self.actors[name].set_root_transform(
                sdp.TransformRT(sdp.Quaternion(*quat.tolist()),
                                sdp.Real3(*block[:, 3].tolist())))
        self._rest()

    def _rest(self):
        zero = self._sdp.Real3(0.0, 0.0, 0.0)
        for actor in self.actors.values():
            actor.set_velocity(zero, zero)

    def _command(self, current, target):
        """Velocity toward the target, with per-step motion bounded."""
        sdp = self._sdp
        for name, now, goal in zip(self.order, current, target):
            com = self._com_local[name]
            shift = (goal[:, :3] @ com + goal[:, 3]) - (now[:, :3] @ com + now[:, 3])
            turn = _rotation_vector(goal[:, :3] @ now[:, :3].T)
            shift_norm, turn_norm = np.linalg.norm(shift), np.linalg.norm(turn)
            if self.max_point_step is not None:
                reach = shift_norm + turn_norm * self._reach[name]
                if reach > self.max_point_step:
                    shift = shift * (self.max_point_step / reach)
                    turn = turn * (self.max_point_step / reach)
            else:
                if shift_norm > self.max_step_translation:
                    shift = shift * (self.max_step_translation / shift_norm)
                if turn_norm > self.max_step_rotation:
                    turn = turn * (self.max_step_rotation / turn_norm)
            self.actors[name].set_velocity(sdp.Real3(*(shift / self.dt).tolist()),
                                           sdp.Real3(*(turn / self.dt).tolist()))

    def _step(self):
        self.scene.step(self.dt)
        self.n_steps += 1
        status = self.scene.get_solver_stats().convergence_status
        if status == self._sdp.ConvergenceStatus.DIVERGED:
            raise RuntimeError("SuperDex solver diverged")
        self._rest()
        return self.pack()

    def project(self, q):
        target = self._blocks(q)
        current = self.pack()
        change = float("inf")
        steps = 0
        while steps < self.max_steps:
            self._command(current.reshape(target.shape), target)
            result = self._step()
            steps += 1
            change = float(np.abs(result - current).max())
            current = result
            if steps >= 2 and change <= self.push_tol:
                break
        # The push balances the pull toward q against contact, leaving contact
        # loaded.  Uncommanded steps release that load; they share the same step
        # budget and must meet the final tolerance.
        change = float("inf")
        relax = 0
        while steps < self.max_steps:
            result = self._step()
            steps += 1
            relax += 1
            change = float(np.abs(result - current).max())
            current = result
            if relax >= 2 and change <= self.tol:
                break
        self.last_steps, self.last_change = steps, change
        if change > self.tol:
            raise RuntimeError(f"contact projection not converged after {steps} steps: "
                               f"last packed change {change:.3e} > tol {self.tol:.1e}")
        self.n_settles += 1
        return current

    def baseline(self, q):
        """Settle at ``q`` itself, not at the end of a walk from the last probe."""
        self._write_pose(q)
        return super().baseline(q)

    def snapshot(self, settled):
        """Capture solver state for ``settled``; bytes are freed with the snapshot."""
        settled = np.asarray(settled, float).copy()
        if np.abs(self.pack() - settled).max() > 0.0:
            self._write_pose(settled)
        data = self._sdp.DynamicArrayUint8()
        self.scene.capture_state_to_bytes(data)
        return {"packed": settled, "state": data}

    def restore(self, snapshot):
        self.scene.restore_state_from_bytes(snapshot["state"])
        return snapshot["packed"].copy()

    def directional_responses(self, snapshot, directions, scale, *, scales=None):
        """Responses for many directions from one baseline, on workers if configured.

        ``scales`` gives each direction its own scale in place of ``scale``.
        """
        sizes = ([float(scale)] * len(directions) if scales is None
                 else [float(value) for value in scales])
        if len(sizes) != len(directions):
            raise ValueError("scales must give one scale per direction")
        if self.n_workers <= 1:
            return [self.directional_response(snapshot, direction, size)
                    for direction, size in zip(directions, sizes)]
        if self._pool is None:
            self._pool = _WorkerPool(self.spec, self._worker_options, self.n_workers,
                                     self.command_timeout)
            atexit.register(self.close)
        baseline = np.asarray(snapshot["packed"], float).copy()
        fingerprint = self.oracle_fingerprint()
        responses, misses = [], []
        for direction, scale in zip(directions, sizes):
            direction = np.asarray(direction, float).reshape(-1)
            direction = direction / np.linalg.norm(direction)
            self._response_sequence += 1
            sequence = self._response_sequence
            cached = self.ledger.lookup(fingerprint, baseline, direction, scale)
            if cached is None:
                misses.append((len(responses), direction, sequence, scale))
                responses.append(None)
            else:
                responses.append(DirectionalResponse(
                    baseline, direction, scale, np.asarray(cached["result"], float),
                    bool(cached["converged"]), bool(cached["valid"]),
                    dict(cached.get("diagnostics", {})), sequence, None, reused=True))
        state = bytes(bytearray(snapshot["state"]))
        answers = self._pool.run(state, [baseline + scale * d for _i, d, _s, scale in misses])
        for (slot, direction, sequence, scale), (result, error) in zip(misses, answers):
            if error is None:
                converged = valid = bool(np.isfinite(result).all())
                diagnostics = {"response_norm": float(np.linalg.norm(result - baseline))}
                if valid:
                    domain_valid, domain = self.response_validity(result)
                    valid = bool(domain_valid)
                    diagnostics.update(domain)
            else:
                result = baseline.copy()
                converged = valid = False
                diagnostics = {"error": error}
            self.ledger.store(fingerprint, baseline, direction, scale, result, converged, valid,
                              diagnostics, sequence=sequence)
            responses[slot] = DirectionalResponse(baseline, direction, scale, result, converged,
                                                  valid, diagnostics, sequence)
        return responses

    def residual(self, q):
        """Physical deviation of ``q``: how far settling from ``q`` itself moves it.

        Zero on the contact-feasible set; a pose that does not settle answers
        infinity, so a test on it fails rather than passes.
        """
        target = self._blocks(q).reshape(-1)
        self._write_pose(q)
        try:
            return self.project(q) - target
        except RuntimeError:
            return np.full_like(target, np.inf)

    def project_from(self, start, q, *, max_steps=None):
        """``project(q)`` walked from the pose ``start`` rather than from where the scene was left.

        ``start`` should be feasible (a settled sample), so the walk only covers the distance from it
        to ``q``.  ``max_steps`` bounds this one projection; it still raises when not converged.
        """
        self._write_pose(start)
        if max_steps is None:
            return self.project(q)
        saved = self.max_steps
        self.max_steps = int(max_steps)
        try:
            return self.project(q)
        finally:
            self.max_steps = saved

    def residuals(self, poses):
        """``residual`` for many poses, on workers if configured."""
        if self.n_workers <= 1:
            return [self.residual(q) for q in poses]
        if self._pool is None:
            self._pool = _WorkerPool(self.spec, self._worker_options, self.n_workers,
                                     self.command_timeout)
            atexit.register(self.close)
        targets = [self._blocks(q).reshape(-1) for q in poses]
        answers = self._pool.run(None, targets)
        return [np.full_like(target, np.inf) if error is not None or result is None
                else np.asarray(result, float) - target
                for target, (result, error) in zip(targets, answers)]

    def close(self):
        """Stop any worker processes."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def contact_force_norms(self):
        """Total contact force per body; needs ``register_contact_force_queries``."""
        return {name: float(np.linalg.norm(actor.get_contact_force_world().tolist()))
                for name, actor in self.actors.items()}

    def register_contact_force_queries(self):
        for actor in self.actors.values():
            actor.register_query(self._sdp.QueryType.TOTAL_CONTACT_FORCE)

    def hinge_error(self):
        """Largest hinge anchor deviation."""
        return max(float(np.linalg.norm(np.asarray(hinge.get_deviation(), float)))
                   for hinge in self.hinges)


def build_superdex_oracle(spec, *, log=print, **options):
    return SuperDexContactOracle(spec, log=log, **options)
