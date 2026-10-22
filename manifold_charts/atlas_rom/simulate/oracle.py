"""oracle.py — build the full-space libuipc simulator for ANY declared mechanism.

This is the one scene assembly, driven by the spec: bodies come from the
declared meshes, joints from the declared joints, contact from the declared
contact model. The oracle class itself (`LibuipcConfigOracle`) is unchanged
and stays authoritative — this only replaces the per-case wiring around it.

This is the production entry point: nothing under `atlas_rom/` may import
`atlas_rom_experiments` (that boundary is checked by
`patch_rom.selfcheck._check_production_isolation`), so this module -- not the
archived copy it was lifted from -- is what `atlas_rom.rom.run_patch_atlas`'s Full
mode and any other production full-space viewer must import.

Importing this module imports `uipc`, so it needs CUDA. Every other stage of
the build is import-clean without it, which is why the import lives inside
`build_oracle` rather than at module scope: a ROM can be trained, packaged and
run on a machine that has no GPU, as long as the samples already exist.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # atlas_rom/
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "general"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas_rom.spec import geometry


class CylindricalReducedOracle:
    """Hard two-coordinate cylindrical kinematics around a contact oracle.

    The wrapped libuipc body is allowed to resolve IPC contact, but every target,
    returned sample, and viewer pose is reconstructed from only axial travel and
    axial spin. Forbidden translation/tilt never enters the atlas state.
    """

    reduced_coordinates = True

    def __init__(self, raw, mechanism, joint, *, dt, gravity):
        self.raw = raw
        self.joint = joint
        self.order = mechanism.body_order
        if joint.body_b not in self.order or not mechanism.body(joint.body_a).fixed:
            raise NotImplementedError(
                "reduced cylindrical joints currently require fixed body_a and driven body_b")
        self.body = self.order.index(joint.body_b)
        self.axis = np.asarray(joint.axis, float)
        self.axis /= np.linalg.norm(self.axis)
        self.pivot = np.asarray(joint.pivot, float)
        self.T0 = np.asarray(mechanism.body(joint.body_b).initial_transform, float)
        self.local_pivot = (np.linalg.inv(self.T0) @ np.r_[self.pivot, 1.0])[:3]
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(trial @ self.axis)) > 0.9:
            trial = np.array([0.0, 0.0, 1.0])
        self.e1 = trial - self.axis * float(trial @ self.axis)
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.axis, self.e1)
        self.dim_ambient = raw.dim_ambient
        self.probe_count = 2
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.travel_velocity = 0.0
        self.spin_velocity = 0.0

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _parameters(self, block):
        R, t = block[:, :3], block[:, 3]
        D = R @ self.T0[:3, :3].T
        theta = float(np.arctan2(self.e2 @ (D @ self.e1),
                                 self.e1 @ (D @ self.e1)))
        world_pivot = R @ self.local_pivot + t
        travel = float(self.axis @ (world_pivot - self.pivot))
        return travel, theta

    def _pose(self, travel, theta):
        a = self.axis
        K = np.array([[0.0, -a[2], a[1]],
                      [a[2], 0.0, -a[0]],
                      [-a[1], a[0], 0.0]])
        D = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        R = D @ self.T0[:3, :3]
        t = self.pivot + travel * a - R @ self.local_pivot
        return np.column_stack((R, t))

    def clamp(self, q):
        blocks = np.asarray(q, float).reshape(len(self.order), 3, 4).copy()
        travel, theta = self._parameters(blocks[self.body])
        blocks[self.body] = self._pose(travel, theta)
        return blocks.reshape(-1)

    def pack(self):
        return self.clamp(self.raw.pack())

    def seed(self):
        return self.clamp(self.raw.seed())

    def project(self, q):
        target = self.clamp(q)
        # Alternating projection: IPC resolves clearance, then the analytic map
        # removes every forbidden body mode. Keeping the drive active prevents
        # an unconstrained relax phase from reintroducing those modes.
        for _ in range(3):
            self.raw._set_target(target)
            self.raw.state["drive_on"] = True
            landed = self.raw._settle(self.raw.drive_iters)
            next_target = self.clamp(landed)
            if np.linalg.norm(next_target - target) < self.raw.settle_tol:
                target = next_target
                break
            target = next_target
        self.raw.n_settles += 1
        return target

    def tangent_frame(self, p, k=None, eps=1e-3, n_probe=None,
                      rel_tol=0.05, rng=None):
        """Measure mobility after the IPC/contact projection.

        The old analytic two-column frame described the *cylindrical joint*,
        rather than the feasible contact stratum.  Consequently an engaged
        thread could never be observed as a one-dimensional screw motion.
        Probing ``self.project`` preserves the cylindrical clamp while allowing
        contact to suppress its incompatible axial/spin direction.
        """
        from atlas_rom.simulate.oracle_base import ManifoldOracle

        if n_probe is None:
            n_probe = max(4, self.probe_count)
        return ManifoldOracle.tangent_frame(
            self, p, k=k, eps=eps, n_probe=n_probe,
            rel_tol=rel_tol, rng=rng)

    def tangent_measurement(self, p, k=None, eps=1e-3, n_probe=None,
                            rel_tol=0.05, rng=None, scales=(1.0,)):
        from atlas_rom.simulate.oracle_base import ManifoldOracle
        if n_probe is None:
            n_probe = max(4, self.probe_count)
        return ManifoldOracle.tangent_measurement(
            self, p, k=k, eps=eps, n_probe=n_probe, rel_tol=rel_tol,
            rng=rng, scales=scales)

    def advance_reduced(self, steps=1):
        """Integrate gravity in Y while clamping the other four body modes.

        IPC remains responsible for opposing infeasible axial targets.  When
        contact prevents the requested travel, the realized displacement resets
        the axial velocity instead of accumulating energy behind the barrier.
        """
        for _ in range(int(steps)):
            current = self.clamp(self.raw.pack())
            blocks = current.reshape(len(self.order), 3, 4)
            travel, theta = self._parameters(blocks[self.body])
            self.travel_velocity -= self.gravity * self.dt
            wanted_travel = travel + self.travel_velocity * self.dt
            wanted_theta = theta + self.spin_velocity * self.dt
            target_blocks = blocks.copy()
            target_blocks[self.body] = self._pose(wanted_travel, wanted_theta)
            target = target_blocks.reshape(-1)
            self.raw._set_target(target)
            self.raw.state["drive_on"] = True
            self.raw.world.advance()
            self.raw.world.retrieve()
            self.raw.n_advances += 1
            realized = self.clamp(self.raw.pack()).reshape(len(self.order), 3, 4)
            realized_travel, realized_theta = self._parameters(realized[self.body])
            self.travel_velocity = (realized_travel - travel) / self.dt
            self.spin_velocity = (realized_theta - theta) / self.dt
        return self.pack()


def tangent_frame_for_spec(spec, oracle, configuration, *, rel_tol=None,
                           return_diagnostics=False, witnessed_secants=None):
    """Return the tangent basis used by the explorer for a declared mechanism.

    UIPC's generic affine-body path does not expose a planar-rigid body
    constitution, so planar cases declare their allowed ambient subspace here:
    per body, translations in x/y and rotation about z.  The same projection
    is used by sampling and diagnostics, making the viewer an exact view of the
    explorer's tangent directions.
    """
    # A random unit probe in n-D ambient space carries only ~1/sqrt(n) of its
    # length in the tangent plane, so a fixed eps buries the tangential signal in
    # the settle's own noise as a mechanism gains bodies — at 228 ambient
    # dimensions the second singular value reaches 0.85 of the first and the
    # measured tangent is partly noise.  Scale the probe so the expected
    # tangential displacement is the same for any body count (12 = one body,
    # which reproduces the previous behaviour exactly).
    ambient = 12 * len(spec.mechanism.driven_bodies)
    configured_eps = spec.build.sampling.tangent_eps
    eps = (float(configured_eps) if configured_eps is not None
           else 1e-3 * float(np.sqrt(ambient / 12.0)))
    if eps <= 0.0:
        raise ValueError("sampling.tangent_eps must be positive")
    if hasattr(oracle, "tangent_measurement"):
        kwargs = {"eps": eps, "witnessed_secants": witnessed_secants}
        if rel_tol is not None:
            kwargs["rel_tol"] = float(rel_tol)
        basis, singular, dim, diagnostics = oracle.tangent_measurement(configuration, **kwargs)
    elif rel_tol is None:
        basis, singular, dim = oracle.tangent_frame(configuration, eps=eps)
        diagnostics = {"probe_confidence": 1.0}
    else:
        basis, singular, dim = oracle.tangent_frame(
            configuration, eps=eps, rel_tol=float(rel_tol))
        diagnostics = {"probe_confidence": 1.0}
    planar = all(b.configuration == "planar_rigid"
                 for b in spec.mechanism.driven_bodies)
    if not planar:
        return (basis, singular, dim, diagnostics) if return_diagnostics else (basis, singular, dim)
    B = np.asarray(basis, float)
    if B.ndim == 1:
        B = B[:, None]
    n = len(spec.mechanism.driven_bodies)
    if B.shape[0] != 12 * n:
        raise ValueError("planar tangent projection expects packed 3x4 poses")
    mask = np.zeros((n, 3, 4), bool)
    mask[:, 0, 0:2] = True
    mask[:, 1, 0:2] = True
    mask[:, 0, 3] = True
    mask[:, 1, 3] = True
    P = B * mask.reshape(-1, 1)
    Q, R = np.linalg.qr(P, mode="reduced")
    diag = np.abs(np.diag(R)) if R.size else np.empty(0)
    tol = max(P.shape) * np.finfo(float).eps * (diag.max() if len(diag) else 1.0)
    rank = int(np.sum(diag > tol))
    return ((Q[:, :rank], singular, rank, diagnostics) if return_diagnostics
            else (Q[:, :rank], singular, rank))


def build_oracle(spec, *, dt=0.01, abd_stiffness=None, joint_strength=1e9,
                 drive_strength_ratio=1e4, drive_iters=30, relax_iters=20,
                 settle_tol=1e-4, probe_count=8, d_hat=None, max_kappa=None,
                 newton_max_iter=None,
                 gravity=0.0, actuated_bodies=None, initial_configuration=None,
                 hold_drive_during_projection=False,
                 projection_mode="drive_relax", newton_velocity_tol=None,
                 proximal_tol=1e-5, proximal_max_steps=8,
                 log=print):
    """MechanismSpec -> a ready `LibuipcConfigOracle` (world already initialized).

    Contact is enabled only when the case declares a contact model; a pure ideal-
    joint linkage runs with contact off, which is both faster and the only way its
    manifold is a clean joint-constrained set.

    Gravity is ZERO regardless of the mechanism's rendering/potential gravity: the
    oracle is a quasistatic projection, and a gravity load would bias `project()`
    off the nearest-point map it is defined to be.  Gravity re-enters later, in
    V(u), where it belongs.

    `d_hat`/`max_kappa` relax the IPC contact when supplied — the fix for
    `advance()` hanging on a near-singular inner solve.  Passing them is the
    difference between a mobility probe that reports the truth and one where stiff
    defaults falsely jam every direction.
    """
    import uipc.builtin as builtin
    from uipc import view
    from uipc.core import Engine, World, Scene
    from uipc.constitution import (AffineBodyConstitution,
                                   AffineBodyFixedJoint,
                                   AffineBodyPrismaticJoint,
                                   AffineBodyRevoluteJoint,
                                   AffineBodyRevoluteJointExternalForce,
                                   AffineBodySphericalJoint,
                                   SoftTransformConstraint)
    from uipc.geometry import trimesh as uipc_trimesh, label_surface
    from uipc.unit import GPa

    from atlas_rom.simulate.projection_oracle import LibuipcConfigOracle

    mech = spec.mechanism
    contact_on = spec.contact_model != "none"
    stiffness = 1 * GPa if abd_stiffness is None else abd_stiffness

    engine = Engine("cuda")
    world = World(engine)
    cfg = Scene.default_config()
    cfg["dt"] = dt
    g = float(gravity)
    axis = "xyz".index(mech.up_axis)
    gravity_vec = np.zeros((3, 1), dtype=float)
    gravity_vec[axis, 0] = -g
    cfg["gravity"] = gravity_vec.tolist()
    cfg["contact"]["enable"] = bool(contact_on)
    cfg["contact"]["friction"]["enable"] = False        # else project() is path-
    cfg["sanity_check"]["enable"] = 0                   # dependent, so multivalued
    if d_hat is not None:
        cfg["contact"]["d_hat"] = float(d_hat)
    if max_kappa is not None:
        cfg["contact"]["adaptive"]["max_kappa"] = float(max_kappa)
        cfg["contact"]["adaptive"]["min_kappa"] = min(
            float(cfg["contact"]["adaptive"]["min_kappa"]), float(max_kappa))
    if newton_max_iter is not None:
        cfg["newton"]["max_iter"] = int(newton_max_iter)
    if newton_velocity_tol is not None:
        # Newton stops when max |dx| / dt < velocity_tol, i.e. an absolute
        # displacement tolerance of velocity_tol * dt.
        cfg["newton"]["velocity_tol"] = float(newton_velocity_tol)
    scene = Scene(cfg)
    scene.contact_tabular().default_model(0.5, 1.0 * GPa)
    scene.animator().substep(1)

    abd = AffineBodyConstitution()

    slots, objects = {}, {}
    for body in mech.bodies:
        V, F = geometry.build_mesh(body.mesh)
        sc = uipc_trimesh(np.ascontiguousarray(V, np.float64),
                          np.ascontiguousarray(F, np.int32))
        label_surface(sc)
        # Keep IPC mass and inertia consistent with the case declaration.
        # Omitting this argument silently uses libuipc's 1000 default density.
        abd.apply_to(sc, stiffness, float(body.density))
        view(sc.transforms())[:] = np.asarray(body.initial_transform, np.float64)
        if body.fixed:
            view(sc.instances().find(builtin.is_fixed))[:] = 1
        obj = scene.objects().create(body.name)
        slots[body.name] = obj.geometries().create(sc)[0]
        objects[body.name] = obj

    _WORLD_BODY = "__oracle_world__"
    _world_slot_cache = []

    def world_slot():
        """Lazily-created fixed dummy body standing in for `body_b is None`.

        `JointDef.body_b is None` means "pinned to ground" (see its docstring);
        several other consumers — decoder.py, mechanism_constraints.py,
        mechanism_contact_solver.py — already rely on that sentinel, so it must
        stay `None` on the spec.  This full-space builder is the one place that
        needs an actual second ABD body to attach the joint constitution to, so
        it manufactures one here instead, at identity (world) transform.
        """
        if not _world_slot_cache:
            h = 1e-3
            V = np.array([[0.0, 0.0, 0.0], [h, 0.0, 0.0],
                          [0.0, h, 0.0], [0.0, 0.0, h]], np.float64)
            F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
            sc = uipc_trimesh(V, F)
            label_surface(sc)
            abd.apply_to(sc, stiffness)
            contacts = scene.contact_tabular()
            ghost = contacts.create(f"__{_WORLD_BODY}_contact")
            default = contacts.default_element()
            contacts.insert(ghost, default, 0.0, 0.0, False)
            contacts.insert(ghost, ghost, 0.0, 0.0, False)
            ghost.apply_to(sc)
            view(sc.instances().find(builtin.is_fixed))[:] = 1
            obj = scene.objects().create(_WORLD_BODY)
            _world_slot_cache.append(obj.geometries().create(sc)[0])
        return _world_slot_cache[0]

    def body_transform(name):
        return (np.eye(4, dtype=np.float64) if name == _WORLD_BODY else
                np.asarray(mech.body(name).initial_transform, float))

    def cylindrical_carriage(name, pivot):
        """An internal collision-free carrier used to compose a cylindrical joint.

        libuipc provides slider and hinge constitutions but no cylindrical one.
        Their serial composition has the exact two desired freedoms.  The
        carrier is a tiny body at the joint frame and is removed from the IPC
        contact table.  A former implementation put this mesh at 1e6 to avoid
        collision; that gave the helper an enormous lever arm/inertia and made
        the serial joint visibly compliant under contact load.
        """
        h = 1e-3
        V = np.array([[0.0, 0.0, 0.0], [h, 0.0, 0.0],
                      [0.0, h, 0.0], [0.0, 0.0, h]], np.float64)
        F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
        sc = uipc_trimesh(V, F)
        label_surface(sc)
        abd.apply_to(sc, stiffness)
        contacts = scene.contact_tabular()
        ghost = contacts.create(f"__cylindrical_carriage_contact_{name}")
        default = contacts.default_element()
        contacts.insert(ghost, default, 0.0, 0.0, False)
        contacts.insert(ghost, ghost, 0.0, 0.0, False)
        ghost.apply_to(sc)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.asarray(pivot, float)
        view(sc.transforms())[:] = T
        obj = scene.objects().create(f"__cylindrical_carriage_{name}")
        return obj.geometries().create(sc)[0]

    if initial_configuration is not None:
        q = np.asarray(initial_configuration, float).reshape(
            len(mech.driven_bodies), 3, 4)
        for body, block in zip(mech.driven_bodies, q):
            T = np.eye(4, dtype=np.float64)
            U, _s, Vt = np.linalg.svd(block[:, :3])
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1.0
                R = U @ Vt
            T[:3, :3] = R
            T[:3, 3] = block[:, 3]
            view(slots[body.name].geometry().transforms())[0] = T
    log(f"oracle: {len(mech.bodies)} bodies "
        f"({len(mech.driven_bodies)} driven), contact={'on' if contact_on else 'off'}")

    # The contact table.  Left at the default model alone, EVERY body collides
    # with every other one — including the pairs the spec joins with a pin,
    # whose bosses overlap at the rest pose by construction, so IPC starts at
    # t=0 against a barrier that is already violated.  `contact_pairs` and
    # `JointDef.exclude_collision` say which pairs are real; they were declared
    # and serialized but nothing here read them.  Same precedence as
    # `general/mechanism_spec.collision_candidates`: a declared pair list is
    # exact, otherwise everything except joint-excluded and fixed-fixed pairs.
    if contact_on:
        from itertools import combinations
        contacts = scene.contact_tabular()
        names = [b.name for b in mech.bodies]
        if mech.contact_pairs is not None:
            allowed = {frozenset(p) for p in mech.contact_pairs}
            unknown = sorted({n for p in allowed for n in p} - set(names))
            if unknown:
                raise ValueError(f"contact_pairs names unknown bodies {unknown}")
        else:
            excluded = {frozenset((j.body_a, j.body_b)) for j in mech.joints
                        if j.exclude_collision and j.body_b is not None}
            fixed_names = {b.name for b in mech.bodies if b.fixed}
            allowed = {frozenset(p) for p in combinations(names, 2)
                       if p[0] not in fixed_names or p[1] not in fixed_names}
            allowed -= excluded
        element = {}
        for name in names:
            element[name] = contacts.create(f"body_{name}")
            element[name].apply_to(slots[name].geometry())
        for a, b in combinations(names, 2):
            on = frozenset((a, b)) in allowed
            contacts.insert(element[a], element[b],
                            0.5 if on else 0.0,
                            (1.0 * GPa) if on else 0.0, on)
        for name in names:              # an ABD body cannot collide with itself
            contacts.insert(element[name], element[name], 0.0, 0.0, False)
        log(f"oracle: contact enabled on {len(allowed)} of "
            f"{len(names) * (len(names) - 1) // 2} body pairs "
            f"({'declared' if mech.contact_pairs is not None else 'broad phase'})")

    joint_torque = {}
    for joint in mech.joints:
        body_b_slot = world_slot() if joint.body_b is None else slots[joint.body_b]
        if joint.kind not in ("fixed", "revolute", "prismatic", "cylindrical", "spherical"):
            raise NotImplementedError(
                f"joint {joint.name!r}: unsupported generic joint kind {joint.kind!r}")
        pivot = np.asarray([joint.pivot], np.float64)
        if joint.axis is not None:
            axis_pt = np.asarray([joint.pivot + joint.axis], np.float64)
        if joint.kind == "cylindrical":
            # Enforced by CylindricalReducedOracle below, not by a compliant
            # full-space helper body.
            continue
        if joint.kind == "spherical":
            # A ball-and-socket pins ONE point and leaves relative orientation
            # entirely free — one coincident anchor, where the revolute needs a
            # second point to also pin its axis.  The world-position overload
            # localises the anchor against each body's current transform, which
            # is the rest pose at scene-build time.
            jmesh = AffineBodySphericalJoint().create_geometry(
                pivot, [slots[joint.body_a]], np.array([0], np.int32),
                [body_b_slot], np.array([0], np.int32),
                np.array([joint_strength], np.float64))
        elif joint.kind == "fixed":
            # Three non-collinear coincident points lock both translation and
            # relative orientation, unlike a revolute joint which leaves one
            # angular DOF.  The points are localised from their common rest-pose
            # world locations so the weld is valid for arbitrary initial poses.
            p = np.asarray(joint.pivot, float)
            e1 = np.array([0.02, 0.0, 0.0])
            e2 = np.array([0.0, 0.02, 0.0])
            world_points = np.asarray([p, p + e1, p + e2], float)
            ta = np.asarray(mech.body(joint.body_a).initial_transform, float)
            tb = body_transform(joint.body_b)
            la = (np.linalg.inv(ta) @ np.column_stack(
                [world_points, np.ones(len(world_points))]).T).T[:, :3]
            lb = (np.linalg.inv(tb) @ np.column_stack(
                [world_points, np.ones(len(world_points))]).T).T[:, :3]
            jmesh = AffineBodyFixedJoint().create_geometry(
                la, lb, [slots[joint.body_a]] * 3, np.zeros(3, np.int32),
                [body_b_slot] * 3, np.zeros(3, np.int32),
                np.full(3, min(float(joint_strength), 1e6), np.float64))
        elif joint.kind == "revolute":
            jmesh = AffineBodyRevoluteJoint().create_geometry(
                pivot, axis_pt, [slots[joint.body_a]], [0],
                [body_b_slot], [0], [joint_strength])
            # This is a generalized torque on the *joint*, not a pose spring
            # on either body.  The animation callback below updates its value
            # every substep, so full-space IPC and the reduced actuator map
            # consume the same physical units (N m).
            AffineBodyRevoluteJointExternalForce().apply_to(jmesh, 0.0)
        else:
            # The prismatic constitution uses two points on each side to
            # define a shared world-space rail axis.  Convert the declared
            # world pivot/axis to each body's local frame at the rest pose.
            ta = np.asarray(mech.body(joint.body_a).initial_transform)
            tb = body_transform(joint.body_b)
            inv_a, inv_b = np.linalg.inv(ta), np.linalg.inv(tb)
            p0 = np.asarray(joint.pivot, float)
            p1 = p0 + np.asarray(joint.axis, float)
            l0 = (inv_a @ np.r_[p0, 1.0])[:3]
            l1 = (inv_a @ np.r_[p1, 1.0])[:3]
            r0 = (inv_b @ np.r_[p0, 1.0])[:3]
            r1 = (inv_b @ np.r_[p1, 1.0])[:3]
            jmesh = AffineBodyPrismaticJoint().create_geometry(
                np.asarray([l0]), np.asarray([l1]),
                np.asarray([r0]), np.asarray([r1]),
                [slots[joint.body_a]], np.array([0], dtype=np.int32),
                [body_b_slot], np.array([0], dtype=np.int32),
                np.array([joint_strength], dtype=np.float64))
        joint_object = scene.objects().create(f"joint_{joint.name}")
        joint_slot = joint_object.geometries().create(jmesh)[0]
        if joint.kind == "revolute":
            joint_torque[joint.name] = {"slot": joint_slot, "value": 0.0}

            def _apply_torque(info, entry=joint_torque[joint.name]):
                geo = info.geo_slots()[0].geometry()
                enabled = geo.edges().find("external_torque/is_constrained")
                value = geo.edges().find("external_torque")
                if enabled is not None:
                    view(enabled)[:] = 1 if entry["value"] != 0.0 else 0
                if value is not None:
                    view(value)[:] = entry["value"]

            scene.animator().insert(joint_object, _apply_torque)
    if mech.joints:
        log(f"oracle: {len(mech.joints)} joints at strength {joint_strength:g}")

    # Soft drive: the projection's pull toward q.  A contact-coupled follower
    # must not receive its own independent target: that over-constrains a tangent
    # predictor and makes the contact solve chase incompatible poses.  It remains
    # in the packed configuration, but is positioned by the mechanism.
    driven = [(b.name, slots[b.name]) for b in mech.driven_bodies]
    driven_names = {name for name, _slot in driven}
    if actuated_bodies is not None:
        actuated = set(actuated_bodies)
        unknown = sorted(actuated - driven_names)
        if unknown:
            raise ValueError(f"oracle: actuated bodies {unknown} are not driven bodies "
                             f"{sorted(driven_names)}")
    else:
        actuated = driven_names
    stc = SoftTransformConstraint()
    for name, slot in driven:
        if name not in actuated:
            continue
        stc.apply_to(slot.geometry(),
                      np.array([drive_strength_ratio, drive_strength_ratio]))

    target = {name: np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4).copy()
              for name, slot in driven}
    state = {"drive_on": True}
    animator = scene.animator()

    def _anim(name):
        def _fn(info):
            geo = info.geo_slots()[0].geometry()
            view(geo.instances().find(builtin.is_constrained))[0] = \
                1 if state["drive_on"] else 0
            view(geo.instances().find(builtin.aim_transform))[0] = target[name]
        return _fn

    for name, _slot in driven:
        if name not in actuated:
            continue
        animator.insert(objects[name], _anim(name))

    world.init(scene)
    world.retrieve()
    accessor = body_index = None
    if projection_mode == "proximal_step":
        if any(j.kind == "cylindrical" for j in mech.joints):
            raise NotImplementedError(
                "projection_mode='proximal_step' does not support the reduced "
                "cylindrical wrapper yet")
        from uipc.core import AffineBodyStateAccessorFeature
        accessor = world.features().find(AffineBodyStateAccessorFeature)
        if accessor is None:
            raise RuntimeError("this uipc build has no AffineBodyStateAccessorFeature")
        # Backend ABD rows follow geometry creation order; the declared bodies
        # are created first, before any helper (world/carriage) body.
        body_index = {b.name: i for i, b in enumerate(mech.bodies)}
        if accessor.body_count() < len(mech.bodies):
            raise RuntimeError(f"accessor reports {accessor.body_count()} ABD bodies, "
                               f"spec declares {len(mech.bodies)}")
        probe = accessor.create_geometry()
        from uipc import Matrix4x4
        probe.instances().create(builtin.transform, Matrix4x4.Zero())
        accessor.copy_to(probe)
        rows = np.asarray(view(probe.transforms()), float).reshape(-1, 4, 4)
        for body in mech.bodies:
            scene_T = np.asarray(view(slots[body.name].geometry().transforms())[0]).reshape(4, 4)
            if not np.allclose(rows[body_index[body.name]], scene_T, atol=1e-9):
                raise RuntimeError(f"accessor row {body_index[body.name]} does not match "
                                   f"body {body.name!r}; creation-order mapping is wrong")
    raw = LibuipcConfigOracle(world, scene, driven, target, state,
                              drive_iters=drive_iters, relax_iters=relax_iters,
                              settle_tol=settle_tol, probe_count=probe_count,
                              engine=engine, actuated=actuated,
                              hold_drive_during_projection=hold_drive_during_projection,
                              projection_mode=projection_mode, accessor=accessor,
                              body_index=body_index, dt=dt,
                              proximal_tol=proximal_tol,
                              proximal_max_steps=proximal_max_steps)
    raw.joint_torques = joint_torque

    def set_joint_torque(name, value):
        if name not in joint_torque:
            raise KeyError(f"no revolute IPC torque channel named {name!r}; "
                           f"have {sorted(joint_torque)}")
        joint_torque[name]["value"] = float(value)

    raw.set_joint_torque = set_joint_torque
    cylindrical = [j for j in mech.joints if j.kind == "cylindrical"]
    if cylindrical:
        if len(cylindrical) != 1:
            raise NotImplementedError("one reduced cylindrical joint per mechanism is supported")
        log("oracle: cylindrical body uses hard 2-DOF analytic coordinates")
        return CylindricalReducedOracle(raw, mech, cylindrical[0],
                                        dt=dt, gravity=gravity)
    return raw
