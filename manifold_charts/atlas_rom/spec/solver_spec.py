"""Small, solver-independent input and local-stratum records.

This is deliberately a data boundary, not a second mechanism framework.  The
existing libuipc and projection-oracle builders remain authoritative; adapters
only describe their coordinates, ideal joints, and collision geometry to the
atlas layer.
"""
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

import numpy as np


class ConfigurationType(str, Enum):
    FIXED = "fixed"
    HINGE = "hinge"
    SLIDER = "slider"
    PLANAR_TRANSLATION = "planar_translation"
    PLANAR_RIGID = "planar_rigid"
    FREE_RIGID = "free_rigid"

    @property
    def dimension(self):
        return {
            self.FIXED: 0,
            self.HINGE: 1,
            self.SLIDER: 1,
            self.PLANAR_TRANSLATION: 2,
            self.PLANAR_RIGID: 3,
            self.FREE_RIGID: 6,
        }[self]


class ContactModel(str, Enum):
    """Which contact-derived object the atlas is asked to traverse.

    This selects the physical interpretation, not an expected rank or
    dimension.  A maintained-contact atlas locks onto the independently
    certified seed stratum; exact-clearance exploration may release into its
    full unilateral feasible set.
    """

    EXACT_CLEARANCE = "exact_clearance"
    MAINTAINED_CONTACT = "maintained_contact"


@dataclass(frozen=True)
class BodySpec:
    name: str
    configuration: ConfigurationType
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    initial_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Bounds are expanded by this amount during automatic candidate generation.
    # It is a motion bound, not a declaration that a pair is in contact.
    sweep_radius: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "configuration",
                           ConfigurationType(self.configuration))
        transform = np.asarray(self.initial_transform, float)
        if transform.shape != (4, 4):
            raise ValueError(f"{self.name}: initial_transform must be 4x4")
        object.__setattr__(self, "initial_transform", transform.copy())
        if self.vertices is not None:
            vertices = np.asarray(self.vertices, float)
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError(f"{self.name}: vertices must have shape (n, 3)")
            object.__setattr__(self, "vertices", vertices)
        if self.faces is not None:
            object.__setattr__(self, "faces", np.asarray(self.faces, np.int32))

    @property
    def coordinate_dimension(self):
        return self.configuration.dimension

    def world_bounds(self):
        if self.vertices is None or not len(self.vertices):
            return None
        homogeneous = np.column_stack((self.vertices, np.ones(len(self.vertices))))
        world = (self.initial_transform @ homogeneous.T).T[:, :3]
        radius = float(self.sweep_radius)
        return world.min(axis=0) - radius, world.max(axis=0) + radius


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    body_a: str
    body_b: str | None = None
    body_c: str | None = None
    pivot: np.ndarray | None = None
    axis: np.ndarray | None = None
    axis2: np.ndarray | None = None
    ratio: float | None = None
    exclude_collision: bool = True

    def __post_init__(self):
        if self.pivot is not None:
            object.__setattr__(self, "pivot", np.asarray(self.pivot, float))
        for attr in ("axis", "axis2"):
            value = getattr(self, attr)
            if value is None:
                continue
            vec = np.asarray(value, float)
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                raise ValueError(f"{self.name}: joint {attr} is zero")
            object.__setattr__(self, attr, vec / norm)
        if self.kind == "gear_ratio":
            if self.body_b is None or self.body_c is None:
                raise ValueError(f"{self.name}: gear_ratio needs two gears and a carrier")
            if self.ratio is None or not np.isfinite(float(self.ratio)) or float(self.ratio) <= 0.0:
                raise ValueError(f"{self.name}: gear_ratio needs a finite positive ratio")
            object.__setattr__(self, "ratio", float(self.ratio))


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    bodies: tuple[BodySpec, ...]
    joints: tuple[JointSpec, ...] = ()
    seeds: tuple[np.ndarray, ...] = ()
    contact_model: ContactModel = ContactModel.EXACT_CLEARANCE

    def __post_init__(self):
        object.__setattr__(self, "bodies", tuple(self.bodies))
        object.__setattr__(self, "joints", tuple(self.joints))
        object.__setattr__(self, "contact_model",
                           ContactModel(self.contact_model))
        names = [body.name for body in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("body names must be unique")
        known = set(names)
        for joint in self.joints:
            if (joint.body_a not in known or (joint.body_b is not None and
                                              joint.body_b not in known) or
                    (joint.body_c is not None and joint.body_c not in known)):
                raise ValueError(f"{joint.name}: joint refers to an unknown body")

    @property
    def ambient_dimension(self):
        return sum(body.coordinate_dimension for body in self.bodies)

    @property
    def fixed_bodies(self):
        return tuple(body.name for body in self.bodies
                     if body.configuration is ConfigurationType.FIXED)

    @property
    def coordinate_slices(self):
        """Minimal-coordinate slice for every non-fixed body."""
        result = {}
        start = 0
        for body in self.bodies:
            stop = start + body.coordinate_dimension
            if stop != start:
                result[body.name] = slice(start, stop)
            start = stop
        return result

    def collision_candidates(self, margin=np.inf):
        """Generate possible pairs without declaring an expected contact set.

        Infinite ``margin`` is the conservative broad phase and is useful when a
        fixture lacks motion bounds.  A finite margin culls disjoint swept AABBs.
        Fixed-fixed and explicitly joint-excluded pairs are always omitted.
        """
        excluded = {
            frozenset((joint.body_a, joint.body_b))
            for joint in self.joints
            if joint.exclude_collision and joint.body_b is not None
        }
        result = []
        for a, b in combinations(self.bodies, 2):
            if (a.configuration is ConfigurationType.FIXED and
                    b.configuration is ConfigurationType.FIXED):
                continue
            if frozenset((a.name, b.name)) in excluded:
                continue
            if np.isfinite(margin):
                ba, bb = a.world_bounds(), b.world_bounds()
                if ba is not None and bb is not None:
                    separated = np.any(ba[1] + margin < bb[0]) or \
                        np.any(bb[1] + margin < ba[0])
                    if separated:
                        continue
            result.append((a.name, b.name))
        return tuple(result)


@dataclass(frozen=True)
class ConstraintObservation:
    """One equality or nonpenetration observation at a configuration."""
    name: str
    value: float
    jacobian: np.ndarray
    alternate_jacobian: np.ndarray | None = None
    kind: str = "contact"
    clearance: float = 0.0
    equality: bool = False
    material_support: object | None = None

    @property
    def slack(self):
        return float(self.value - self.clearance)


@dataclass(frozen=True)
class LocalStratum:
    q: np.ndarray
    active_constraints: tuple[ConstraintObservation, ...]
    inactive_clearances: tuple[ConstraintObservation, ...]
    stratum_rank: int | None
    stratum_dimension: int | None
    feasible_dimension: int | None
    lineality_dimension: int | None
    tangent_basis: np.ndarray
    feasible_cone_rows: np.ndarray
    rank_certificate: object
    status: str

    # Temporary compatibility names for the first foundation callers.  They
    # mean the selected maintained-contact stratum, not the entire feasible set.
    @property
    def rank(self):
        return self.stratum_rank

    @property
    def intrinsic_dimension(self):
        return self.stratum_dimension
