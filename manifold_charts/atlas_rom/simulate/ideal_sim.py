"""ideal_sim.py — full-space rigid-body dynamics on the analytic ideal-joint oracle.

The libuipc `FullSpaceSim` can only realise joints libuipc has a constitution
for, so a case with a `gear_ratio` / `phase_lock` / `bevel_gear` law has no
full-space backend there.  `IdealJointSim` integrates the SAME packed
(n_bodies, 3, 4) poses the explorer samples, with every joint law — gears
included — held exactly as a row of `IdealJointOracle`'s constraint system.

Same surface as `FullSpaceSim` (`reset()`, `pose()`, `step(controls)`, `time`,
`actuators`), so `run_patch_atlas`'s Full mode drives either one unchanged.

METHOD.  Position-level projection (the SHAKE / position-based-dynamics family):

    q~     = q + dt * qdot + dt^2 * W^-1 F          unconstrained predictor
    q_new  = argmin 1/2 |q - q~|_W^2  s.t.  C(q) = 0  mass-metric projection
    qdot   = (q_new - q) / dt

`W` is the exact kinetic metric of the affine coordinates: a body's pose block
A (3x4, row-major) moves a material point p by  dA [p; 1],  so its kinetic
energy is  sum_r dA_r Mbar dA_r^T  with  Mbar = int rho [p;1][p;1]^T  (4x4),
i.e.  W = blockdiag(kron(I_3, Mbar)).  The oracle's orthonormality rows keep
every block rigid, so this is rigid-body dynamics, gyroscopic terms included,
to first order in dt.  The scheme dissipates a little energy per step rather
than gaining it.

FORCES are generalized forces in the same coordinates: a world force f at a
body-local point p contributes  outer(f, [p; 1])  to that body's block.
Gravity acts at each CoM along -`up_axis`; a `joint_torque` actuator is a pure
force couple about the joint's current world axis, +tau on `body_b` and -tau on
`body_a` (ground takes nothing), so positive torque increases `joint_travel`.

JOINT LIMITS (`limit_lo`/`limit_hi`) are inequality constraints: after a
projection, any limited joint past its stop gets an equality row pinning it at
the stop and the projection is redone.  The velocity into the stop then
vanishes through `qdot = (q_new - q) / dt` (an inelastic stop); the active set
is rebuilt every step, so a joint leaves its stop as soon as it is pushed away.

This is a ground-truth reference simulator for viewing and parity, not a
runtime component: the ROM never calls it.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec import geometry
from atlas_rom.spec.coordinates import joint_travel
from atlas_rom.simulate.ideal_oracle import IdealJointOracle, supports


def _second_moment(V, F, density):
    """Mbar = int rho [p;1][p;1]^T over the body (4x4, SPD)."""
    m, I_com, c = geometry.body_inertia(V, F, density)
    S = 0.5 * np.trace(I_com) * np.eye(3) - I_com      # int rho (p-c)(p-c)^T
    Mbar = np.empty((4, 4))
    Mbar[:3, :3] = S + m * np.outer(c, c)
    Mbar[:3, 3] = Mbar[3, :3] = m * c
    Mbar[3, 3] = m
    return m, c, Mbar


def _perpendicular(n):
    trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = np.cross(n, trial)
    return e / np.linalg.norm(e)


class IdealJointSim:
    def __init__(self, case, *, dt=2e-2, gravity=None, damping=0.0,
                 tol=1e-10, max_iterations=30, log=print):
        if not supports(case):
            raise NotImplementedError(
                f"{case.name}: not an ideal-joint case, IdealJointSim cannot run it")
        self.case = case
        self.dt = float(dt)
        self.gravity = float(case.mechanism.gravity if gravity is None else gravity)
        self.damping = float(damping)
        self.tol, self.max_iterations = float(tol), int(max_iterations)
        self.actuators = list(case.actuators)
        for a in self.actuators:
            if a.kind != "joint_torque":
                raise NotImplementedError(
                    f"actuator {a.name!r} is a {a.kind!r} actuator; IdealJointSim "
                    "only drives joint_torque actuators")
        mech = case.mechanism
        self.oracle = IdealJointOracle(case, log=lambda *_a: None)
        self.order = self.oracle.order
        self.n = len(self.order)

        # per-body mass data, in the oracle's packing order
        self.mass, self.com = np.empty(self.n), np.empty((self.n, 3))
        W_half_inv = np.zeros((12 * self.n, 12 * self.n))
        for i, name in enumerate(self.order):
            body = mech.body(name)
            V, F = geometry.build_mesh(body.mesh)
            self.mass[i], self.com[i], Mbar = _second_moment(V, F, body.density)
            # W_i = kron(I3, Mbar) = L L^T  =>  W^(1/2) := L^T,  W^(-1/2) = L^-T
            L_inv_T = np.linalg.inv(np.linalg.cholesky(Mbar)).T
            W_half_inv[12 * i:12 * i + 12, 12 * i:12 * i + 12] = np.kron(np.eye(3), L_inv_T)
        self._W_half_inv = W_half_inv
        self._W_inv = W_half_inv @ W_half_inv.T

        self._torque_joints = {a.name: mech.joint(a.target) for a in self.actuators}
        for a, j in zip(self.actuators, self._torque_joints.values()):
            if j.kind != "revolute":
                raise NotImplementedError(
                    f"actuator {a.name!r}: joint_torque on a {j.kind!r} joint")
        self._limited = [j for j in mech.joints if j.limited]
        self._index = {name: i for i, name in enumerate(self.order)}
        self._initial_q = self.oracle.seed()
        self.residual_max = 0.0
        self.active_limits: list[str] = []
        self.reset()
        log(f"ideal sim: {self.n} bodies, {len(mech.joints)} joints, "
            f"{len(self._limited)} limited, dt={self.dt:g}")

    # -- FullSpaceSim surface ----------------------------------------------------
    def reset(self) -> np.ndarray:
        self.q = self._initial_q.copy()
        self.qdot = np.zeros_like(self.q)
        self.time = 0.0
        return self.pose()

    def pose(self) -> np.ndarray:
        return self.q.reshape(self.n, 3, 4).copy()

    def step(self, controls=None) -> np.ndarray:
        controls = (np.zeros(len(self.actuators)) if controls is None
                    else np.asarray(controls, float).reshape(-1))
        if controls.shape[0] != len(self.actuators):
            raise ValueError(f"controls has {controls.shape[0]} entries, "
                             f"expected {len(self.actuators)} (one per actuator)")
        force = self._generalized_force(controls)
        qdot = self.qdot * (1.0 - self.damping * self.dt)
        q_pred = self.q + self.dt * qdot + self.dt ** 2 * (self._W_inv @ force)

        pinned: dict[str, float] = {}
        for _round in range(len(self._limited) + 1):
            q_new = self._project(q_pred, pinned)
            T = q_new.reshape(self.n, 3, 4)
            past = {}
            for j in self._limited:
                if j.name in pinned:
                    continue
                travel = joint_travel(T, j, self.order)
                if travel < j.limit_lo or travel > j.limit_hi:
                    past[j.name] = j.limit_lo if travel < j.limit_lo else j.limit_hi
            if not past:
                break
            pinned.update(past)
        self.active_limits = sorted(pinned)
        self.qdot = (q_new - self.q) / self.dt
        self.q = q_new
        self.time += self.dt
        return self.pose()

    # -- internals ---------------------------------------------------------------
    def _point_force(self, out, i, world_point, f):
        """outer(f, [p; 1]) into body i's block, p = its local image of the point."""
        A = self.q[12 * i:12 * i + 12].reshape(3, 4)
        p = np.linalg.solve(A[:, :3], np.asarray(world_point) - A[:, 3])
        out[12 * i:12 * i + 12] += np.outer(f, np.append(p, 1.0)).ravel()

    def _generalized_force(self, controls):
        F = np.zeros(12 * self.n)
        up = self.case.mechanism.up_vector
        for i in range(self.n):
            F[12 * i:12 * i + 12] += np.outer(-self.gravity * up,
                                              self.mass[i] * np.append(self.com[i], 1.0)).ravel()
        T = self.q.reshape(self.n, 3, 4)
        for a, tau in zip(self.actuators, controls):
            tau = a.gain * float(tau)
            if tau == 0.0:
                continue
            j = self._torque_joints[a.name]
            ia, ib = self._index.get(j.body_a), self._index.get(j.body_b)
            # the axis belongs to body_a (joint_travel reads R_a^T R_b about it)
            n = np.asarray(j.axis, float) if ia is None else T[ia, :, :3] @ j.axis
            n = n / np.linalg.norm(n)
            carrier = ib if ib is not None else ia
            x = T[carrier, :, :3] @ np.asarray(j.pivot, float) + T[carrier, :, 3]
            e, d = _perpendicular(n), 0.05
            f = tau / (2.0 * d) * np.cross(n, e)          # couple of moment tau * n
            for body, sign in ((ib, 1.0), (ia, -1.0)):
                if body is not None:
                    self._point_force(F, body, x + d * e, sign * f)
                    self._point_force(F, body, x - d * e, -sign * f)
        return F

    def _limit_rows(self, q, pinned):
        """Residual rows (travel - stop) and their Jacobian, by central difference
        over the two involved bodies' columns."""
        if not pinned:
            return np.zeros(0), np.zeros((0, q.size))
        res, J, h = [], np.zeros((len(pinned), q.size)), 1e-7
        for r, (name, stop) in enumerate(pinned.items()):
            joint = self.case.mechanism.joint(name)
            travel = lambda x: joint_travel(x.reshape(self.n, 3, 4), joint, self.order)
            res.append(travel(q) - stop)
            for body in (joint.body_a, joint.body_b):
                i = self._index.get(body)
                if i is None:
                    continue
                for c in range(12 * i, 12 * i + 12):
                    dq = np.zeros_like(q)
                    dq[c] = h
                    J[r, c] = (travel(q + dq) - travel(q - dq)) / (2.0 * h)
        return np.asarray(res), J

    def _project(self, q_pred, pinned):
        """argmin |q - q_pred|_W  s.t.  C(q) = 0 (and pinned stops), by
        Gauss-Newton with the target pull kept: with z = W^(1/2) (q - q_pred),
        each iterate takes the minimum-norm z satisfying the linearised
        constraints, which tolerates the redundant rows of planar linkages."""
        q = q_pred.copy()
        for _ in range(self.max_iterations):
            lres, lJ = self._limit_rows(q, pinned)
            res = np.concatenate([self.oracle.residual(q), lres])
            if np.linalg.norm(res, np.inf) <= self.tol:
                break
            J = np.vstack([self.oracle.jacobian(q), lJ])
            A = J @ self._W_half_inv
            z, *_ = np.linalg.lstsq(A, -res + J @ (q - q_pred), rcond=None)
            q = q_pred + self._W_half_inv @ z
        self.residual_max = float(np.linalg.norm(self.oracle.residual(q), np.inf))
        return q
