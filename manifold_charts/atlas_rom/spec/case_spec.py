"""case_spec.py — the ONE declarative description of an atlas-ROM case.

A case is data, not code.  `atlas_rom/cases/<name>.py` contains a single module
level binding::

    CASE = AtlasCaseSpec(
        name="graphics",
        mechanism=MechanismSpec(...),     # bodies, meshes, fixed bodies, joints
        seeds=[...],
        contact_model="exact_clearance",
        coordinates=[...],
        actuators=[...],
        build=BuildOptions(...),
        scenarios={...},
    )

and nothing else: no `load()`, no decoder subclass, no renderer class, no build
script.  Every stage of `atlas_rom.build` reads this object; the runtime reads the
`atlas.json` it produces.  The declared expectations (`build.expect`) are
*validation assertions* — they gate the build, they never wire it.

Relationship to `atlas_rom/solver_spec.py`: that module is the solver-facing data
boundary (bodies with baked vertices, ideal joints, collision candidates) and stays
authoritative.  This module is the *authoring* layer above it: meshes are declared
by construction rule (`box`/`sphere`/`link`/`file`) rather than by vertex array, and
`MechanismSpec.to_solver_spec()` lowers one into the other.

Extension points (step 8) are deliberately awkward: a case that genuinely needs
nonstandard behaviour names a `plugin` identifier.  The normal pipeline is
plugin-free, and adding a standard mechanism means adding one spec file.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # atlas_rom/
_ROOT = os.path.dirname(_HERE)                              # testing/manifold_charts
for _p in (os.path.join(_ROOT, "general"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPEC_FORMAT = "atlas_case_v1"

# The kinds each declarative field may take.  These are closed sets on purpose:
# a value outside them is a spelling mistake, not an extension point.
MESH_KINDS = ("box", "sphere", "cylinder", "link", "file", "compound", "bevel_gear",
              "torus", "spur_gear")
JOINT_KINDS = ("revolute", "prismatic", "cylindrical", "spherical", "universal",
               "fixed", "phase_lock", "gear_ratio", "bevel_gear", "planar")
# Kinds with exactly ONE freedom, so "the joint's travel" is a single number a
# `limit_lo`/`limit_hi` pair can bound.
LIMITED_JOINT_KINDS = ("revolute", "prismatic")
CONFIGURATIONS = ("fixed", "hinge", "slider", "planar_translation",
                  "planar_rigid", "free_rigid")
CONTACT_MODELS = ("exact_clearance", "maintained_contact", "none")
TOPOLOGIES = ("euclidean", "periodic", "bounded")
COORDINATE_SOURCES = ("body_translation", "body_angle", "joint_angle",
                      "sample_field", "isomap")
ACTUATOR_KINDS = ("coordinate_force", "joint_torque")
CHART_MODES = ("single", "mesh_atlas", "coordinate_split")
BOUNDARY_KINDS = ("none", "sdf", "cloud", "interval")


def _as_array(value, shape=None, dtype=float):
    if value is None:
        return None
    arr = np.asarray(value, dtype)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    return arr


def _jsonable(value):
    """Recursively make a spec value JSON-serializable (arrays -> lists)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ─── geometry / mechanism ─────────────────────────────────────────────────────

@dataclass
class MeshSpec:
    """How to construct one body's LOCAL mesh.

    Vertices are baked at the body's rest world pose (the convention the whole
    pipeline already uses: libuipc starts each body at transform = identity and
    the recorded config T places those verts, see viz_cases.py).  Parameters per
    kind:

        box       center=(3,)  half=(3,) | float
        sphere    radius=float center=(3,) subdiv=int
        cylinder  radius=float height=float center=(3,) axis=(3,) sections=int
        link      p0=(3,) p1=(3,) thick=float        (a thin bar spanning p0->p1)
        spur_gear pitch_radius=float teeth=int width=float tooth_phase=float
                  center=(3,) axis=(3,)   (extruded trapezoidal-tooth spur gear)
        file      path=str  node=str|None  weld=bool  scale=float|(3,)
                  (GLB/OBJ; node picks a
                  scene node, and its world matrix is baked in)
        compound  parts=[MeshSpec, ...]     (one body made of several primitives:
                  a welded triangle of bars, a link with a boss)
    """
    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {self.kind!r}; have {MESH_KINDS}")
        self.params = dict(self.params)
        if self.kind == "compound":
            self.params["parts"] = [p if isinstance(p, MeshSpec)
                                    else MeshSpec.from_dict(p)
                                    for p in self.params.get("parts", [])]

    def to_dict(self) -> dict:
        params = dict(self.params)
        if self.kind == "compound":
            params["parts"] = [p.to_dict() for p in params.get("parts", [])]
        return {"kind": self.kind, "params": _jsonable(params)}

    @classmethod
    def from_dict(cls, d: dict) -> "MeshSpec":
        return cls(kind=d["kind"], params=dict(d.get("params", {})))


@dataclass
class BodyDef:
    """One rigid body: its mesh, its ambient configuration type, its material.

    `configuration` is the body's ambient freedom *as posed by the solver*, and
    per the mesh-contact rule it defaults to `free_rigid` — only constraints the
    case actually states (a joint, `fixed=True`) may reduce it.

    `density` is the libuipc ABD default (1000); mass and the full 3x3 CoM-frame
    inertia are computed from the mesh at build time, never declared by hand.
    """
    name: str
    mesh: MeshSpec
    configuration: str = "free_rigid"
    fixed: bool = False
    density: float = 1000.0
    initial_transform: Any = None                  # (4,4), default identity
    sweep_radius: float = 0.0
    color: Any = None                              # (3,) rgb for rendering
    render_only: bool = False                      # static context, not a DOF

    def __post_init__(self):
        if isinstance(self.mesh, dict):
            self.mesh = MeshSpec.from_dict(self.mesh)
        if self.fixed:
            self.configuration = "fixed"
        if self.configuration not in CONFIGURATIONS:
            raise ValueError(f"{self.name}: unknown configuration "
                             f"{self.configuration!r}; have {CONFIGURATIONS}")
        self.initial_transform = (np.eye(4) if self.initial_transform is None
                                  else _as_array(self.initial_transform, (4, 4)))
        self.color = _as_array(self.color, (3,))

    @property
    def is_driven(self) -> bool:
        """Driven bodies are the ones the decoder poses (the config packing)."""
        return not (self.fixed or self.render_only)

    def to_dict(self) -> dict:
        return {"name": self.name, "mesh": self.mesh.to_dict(),
                "configuration": self.configuration, "fixed": self.fixed,
                "density": self.density,
                "initial_transform": _jsonable(self.initial_transform),
                "sweep_radius": self.sweep_radius,
                "color": _jsonable(self.color), "render_only": self.render_only}

    @classmethod
    def from_dict(cls, d: dict) -> "BodyDef":
        return cls(name=d["name"], mesh=MeshSpec.from_dict(d["mesh"]),
                   configuration=d.get("configuration", "free_rigid"),
                   fixed=bool(d.get("fixed", False)),
                   density=float(d.get("density", 1000.0)),
                   initial_transform=d.get("initial_transform"),
                   sweep_radius=float(d.get("sweep_radius", 0.0)),
                   color=d.get("color"),
                   render_only=bool(d.get("render_only", False)))


@dataclass
class JointDef:
    """An ideal joint between two bodies (or one body and ground when body_b is
    None).  `pivot` is the world-frame joint point at the rest pose; `axis` the
    world-frame rotation/slide axis.  A `cylindrical` joint permits translation
    and rotation about this one axis; a `spherical` joint constrains only the
    pivot's coincidence, so it has no axis at all.  A `universal` (Hooke) joint
    pins the pivot AND holds two cross-pins perpendicular: `axis` is the pin
    fixed in `body_a`, `axis2` the pin fixed in `body_b`, perpendicular at the
    rest pose.  It removes exactly the relative spin about `axis`, leaving two
    bending freedoms — a spherical joint with its idle roll taken away.
    Attachment points are stored in world frame
    and converted to body-local at build time, so a pivot off a body's origin
    stays correct under large rotations.

    `limit_lo`/`limit_hi` bound the joint's OWN travel — radians about the axis
    for a revolute, metres along it for a prismatic — measured from the rest
    pose, so 0 is the assembly the spec declares.  This is how a stop belonging
    to the joint gets stated: a slot with closed ends, a hinge that fouls, a
    rail with end blocks.  `WorkspaceBound` cannot stand in for one, because it
    bounds a body's ABSOLUTE translation, and wherever the carrier moves too the
    two are barely related — measured on the Stephenson slot, the tightest
    translation bound that admits every legal pose also admits 80% of the
    illegal ones."""
    name: str
    kind: str
    body_a: str
    body_b: str | None = None
    # Everything past the two bodies is keyword-only.  These fields are not a
    # stable positional order -- `body_c` was inserted here for `gear_ratio`
    # and silently shifted `pivot`/`axis` by one in every spec that passed them
    # positionally, which reads as "revolute joint needs an axis" in a case file
    # that plainly has one.  Naming them makes that failure impossible.
    _: KW_ONLY
    body_c: str | None = None            # carrier for a gear_ratio constraint
    pivot: Any = None
    axis: Any = None
    axis2: Any = None                    # universal: the pin fixed in body_b
    ratio: float | None = None           # driven/driver tooth-count ratio
    exclude_collision: bool = True
    limit_lo: float | None = None
    limit_hi: float | None = None

    def __post_init__(self):
        if self.kind not in JOINT_KINDS:
            raise ValueError(f"{self.name}: unknown joint kind {self.kind!r}; "
                             f"have {JOINT_KINDS}")
        if (self.limit_lo is None) != (self.limit_hi is None):
            raise ValueError(f"{self.name}: give both limit_lo and limit_hi, or neither")
        if self.limit_lo is not None:
            if self.kind not in LIMITED_JOINT_KINDS:
                raise ValueError(f"{self.name}: a {self.kind} joint has no single "
                                 f"travel to limit; have {LIMITED_JOINT_KINDS}")
            self.limit_lo, self.limit_hi = float(self.limit_lo), float(self.limit_hi)
            if not self.limit_lo < self.limit_hi:
                raise ValueError(f"{self.name}: joint limit needs lo < hi")
            if not self.limit_lo <= 0.0 <= self.limit_hi:
                raise ValueError(f"{self.name}: travel is measured FROM the rest pose, "
                                 f"so the limits must bracket 0 (got "
                                 f"[{self.limit_lo}, {self.limit_hi}])")
        self.pivot = _as_array(self.pivot, (3,))
        self.axis = _as_array(self.axis, (3,))
        if self.axis is not None:
            n = float(np.linalg.norm(self.axis))
            if n == 0.0:
                raise ValueError(f"{self.name}: joint axis is zero")
            self.axis = self.axis / n
        if self.kind in ("revolute", "prismatic", "cylindrical", "phase_lock", "gear_ratio",
                         "bevel_gear", "universal") and self.axis is None:
            raise ValueError(f"{self.name}: {self.kind} joint needs an axis")
        if self.kind == "phase_lock" and self.body_b is None:
            raise ValueError(f"{self.name}: phase_lock needs two driven bodies")
        if self.kind == "gear_ratio":
            if self.body_b is None or self.body_c is None:
                raise ValueError(f"{self.name}: gear_ratio needs two gears and a carrier")
            if self.ratio is None or not np.isfinite(float(self.ratio)) or float(self.ratio) <= 0.0:
                raise ValueError(f"{self.name}: gear_ratio needs a finite positive ratio")
            self.ratio = float(self.ratio)
        if self.kind == "bevel_gear":
            # body_a = pinion, body_b = mating gear, body_c = carrier; `axis` is
            # the pinion axle (fixed in the carrier), `axis2` the mating gear's
            # spin axis.  A signed `ratio` carries the mesh handedness, so the
            # two side meshes of a differential pass +k and -k.
            if self.body_b is None or self.body_c is None:
                raise ValueError(f"{self.name}: bevel_gear needs a pinion, a mating gear and a carrier")
            if self.axis2 is None:
                raise ValueError(f"{self.name}: bevel_gear needs axis2 (the mating gear spin axis)")
            if self.ratio is None or not np.isfinite(float(self.ratio)) or float(self.ratio) == 0.0:
                raise ValueError(f"{self.name}: bevel_gear needs a finite nonzero ratio")
            self.ratio = float(self.ratio)
        if self.kind in ("spherical", "universal") and self.pivot is None:
            raise ValueError(f"{self.name}: {self.kind} joint needs a pivot")
        self.axis2 = _as_array(self.axis2, (3,))
        if self.axis2 is not None:
            n2 = float(np.linalg.norm(self.axis2))
            if n2 == 0.0:
                raise ValueError(f"{self.name}: joint axis2 is zero")
            self.axis2 = self.axis2 / n2
        if self.kind == "universal":
            if self.axis2 is None:
                raise ValueError(f"{self.name}: universal joint needs axis2 "
                                 f"(the pin fixed in body_b)")
            if abs(float(self.axis @ self.axis2)) > 1e-6:
                raise ValueError(f"{self.name}: universal joint axis and axis2 "
                                 f"must be perpendicular at rest "
                                 f"(got dot {float(self.axis @ self.axis2):.3e})")

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "body_a": self.body_a,
                "body_b": self.body_b, "body_c": self.body_c, "pivot": _jsonable(self.pivot),
                "axis": _jsonable(self.axis), "axis2": _jsonable(self.axis2),
                "ratio": self.ratio,
                "exclude_collision": self.exclude_collision,
                "limit_lo": self.limit_lo, "limit_hi": self.limit_hi}

    @property
    def limited(self) -> bool:
        return self.limit_lo is not None

    @classmethod
    def from_dict(cls, d: dict) -> "JointDef":
        return cls(name=d["name"], kind=d["kind"], body_a=d["body_a"],
                   body_b=d.get("body_b"), body_c=d.get("body_c"), pivot=d.get("pivot"),
                   axis=d.get("axis"), axis2=d.get("axis2"),
                   ratio=d.get("ratio"),
                   exclude_collision=bool(d.get("exclude_collision", True)),
                   limit_lo=d.get("limit_lo"), limit_hi=d.get("limit_hi"))


@dataclass
class WorkspaceBound:
    """Case-declared translational workspace boundary for a driven body."""
    name: str
    body: str
    axis: str
    lo: float
    hi: float

    def __post_init__(self):
        if self.axis not in ("x", "y", "z"):
            raise ValueError(f"{self.name}: workspace axis must be x|y|z")
        self.lo, self.hi = float(self.lo), float(self.hi)
        if not self.lo < self.hi:
            raise ValueError(f"{self.name}: workspace bound needs lo < hi")

    def to_dict(self):
        return {"name": self.name, "body": self.body, "axis": self.axis,
                "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass
class MechanismSpec:
    """Bodies + joints + world conventions.  Nothing about charts or training."""
    bodies: list[BodyDef] = field(default_factory=list)
    joints: list[JointDef] = field(default_factory=list)
    up_axis: str = "z"                  # the axis GRAVITY acts against
    render_up_axis: str | None = None   # the viewer's up, when it differs
    gravity: float = 9.8
    units: str = "m"
    contact_pairs: list[tuple[str, str]] | None = None   # None => auto broad phase
    workspace_bounds: list[WorkspaceBound] = field(default_factory=list)

    def __post_init__(self):
        self.bodies = [b if isinstance(b, BodyDef) else BodyDef.from_dict(b)
                       for b in self.bodies]
        self.joints = [j if isinstance(j, JointDef) else JointDef.from_dict(j)
                       for j in self.joints]
        self.workspace_bounds = [
            b if isinstance(b, WorkspaceBound) else WorkspaceBound.from_dict(b)
            for b in self.workspace_bounds]
        # A planar mechanism lying in the xy-plane has gravity along +y WITHIN that
        # plane while the camera still wants +z up; conflating the two silently
        # applies gravity out of the plane of motion, where it does nothing.
        self.render_up_axis = self.render_up_axis or self.up_axis
        for label, axis in (("up_axis", self.up_axis),
                            ("render_up_axis", self.render_up_axis)):
            if axis not in ("x", "y", "z"):
                raise ValueError(f"{label} must be x|y|z, got {axis!r}")
        names = [b.name for b in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("body names must be unique")
        known = set(names)
        for j in self.joints:
            for ref in (j.body_a, j.body_b, j.body_c):
                if ref is not None and ref not in known:
                    raise ValueError(f"joint {j.name!r} refers to unknown body {ref!r}")
        for bound in self.workspace_bounds:
            if bound.body not in known:
                raise ValueError(
                    f"workspace bound {bound.name!r} refers to unknown body "
                    f"{bound.body!r}")
        if self.contact_pairs is not None:
            self.contact_pairs = [tuple(p) for p in self.contact_pairs]

    # -- derived views ---------------------------------------------------------
    @property
    def driven_bodies(self) -> list[BodyDef]:
        """Config-packing order: exactly the bodies the decoder poses."""
        return [b for b in self.bodies if b.is_driven]

    @property
    def body_order(self) -> list[str]:
        return [b.name for b in self.driven_bodies]

    @property
    def static_bodies(self) -> list[BodyDef]:
        return [b for b in self.bodies if not b.is_driven]

    def body(self, name: str) -> BodyDef:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(f"unknown body {name!r}; have {[b.name for b in self.bodies]}")

    def body_index(self, name: str) -> int:
        """Index into the driven/config packing."""
        order = self.body_order
        if name not in order:
            raise KeyError(f"{name!r} is not a driven body; driven={order}")
        return order.index(name)

    def joint(self, name: str) -> JointDef:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"unknown joint {name!r}; have {[j.name for j in self.joints]}")

    @property
    def up_vector(self) -> np.ndarray:
        return np.eye(3)["xyz".index(self.up_axis)]

    def to_solver_spec(self, name: str, contact_model: str = "exact_clearance"):
        """Lower to `atlas_rom.spec.solver_spec.MechanismSpec` (baked vertices).

        The solver-facing spec is what the atlas/oracle layer consumes; this keeps
        that module authoritative instead of growing a second mechanism framework.
        """
        from atlas_rom.spec.solver_spec import BodySpec, JointSpec, MechanismSpec as SolverMechanismSpec
        from atlas_rom.spec import geometry

        bodies = []
        for b in self.bodies:
            V, F = geometry.build_mesh(b.mesh)
            bodies.append(BodySpec(name=b.name, configuration=b.configuration,
                                   vertices=V, faces=F,
                                   initial_transform=b.initial_transform,
                                   sweep_radius=b.sweep_radius))
        joints = [JointSpec(name=j.name, kind=j.kind, body_a=j.body_a,
                            body_b=j.body_b, body_c=j.body_c, pivot=j.pivot, axis=j.axis,
                            axis2=j.axis2, ratio=j.ratio, exclude_collision=j.exclude_collision)
                  for j in self.joints]
        model = "exact_clearance" if contact_model == "none" else contact_model
        return SolverMechanismSpec(name=name, bodies=tuple(bodies),
                                   joints=tuple(joints), contact_model=model)

    def to_dict(self) -> dict:
        return {"bodies": [b.to_dict() for b in self.bodies],
                "joints": [j.to_dict() for j in self.joints],
                "up_axis": self.up_axis, "render_up_axis": self.render_up_axis,
                "gravity": self.gravity, "units": self.units,
                "workspace_bounds": [b.to_dict()
                                     for b in self.workspace_bounds],
                "contact_pairs": (None if self.contact_pairs is None
                                  else [list(p) for p in self.contact_pairs])}

    @classmethod
    def from_dict(cls, d: dict) -> "MechanismSpec":
        return cls(bodies=[BodyDef.from_dict(b) for b in d.get("bodies", [])],
                   joints=[JointDef.from_dict(j) for j in d.get("joints", [])],
                   up_axis=d.get("up_axis", "z"),
                   render_up_axis=d.get("render_up_axis"),
                   gravity=float(d.get("gravity", 9.8)),
                   units=d.get("units", "m"),
                   workspace_bounds=[
                       WorkspaceBound.from_dict(value)
                       for value in d.get("workspace_bounds", [])],
                   contact_pairs=d.get("contact_pairs"))


# ─── coordinates / actuators ──────────────────────────────────────────────────

@dataclass
class CoordinateDef:
    """One intrinsic coordinate of the reduced model.

    `source` says how to READ this coordinate off a recorded configuration —
    which is what makes the decoder addressable and the whole pipeline
    case-independent:

        body_translation  args: body=str, axis="x"|"y"|"z"
        body_angle        args: body=str, axis=(3,), reference="first"|"identity"
        joint_angle       args: joint=str            (signed angle about its axis)
        sample_field      args: key=str, scale=float (a column recorded by the
                          sampling stage, e.g. a loop parameter)
        isomap            args: n_neighbors=int, component=int (nonlinear
                          intrinsic coordinate of the whole config cloud)

    `topology` drives BOTH the decoder's input encoding (periodic -> Fourier
    features) and the runtime's seam wrapping — declared once, honoured everywhere.
    """
    name: str
    topology: str = "euclidean"
    period: float | None = None
    lo: float | None = None
    hi: float | None = None
    source: str = "sample_field"
    args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"{self.name}: bad topology {self.topology!r}")
        if self.source not in COORDINATE_SOURCES:
            raise ValueError(f"{self.name}: bad source {self.source!r}; "
                             f"have {COORDINATE_SOURCES}")
        if self.topology == "periodic":
            self.period = 2.0 * np.pi if self.period is None else float(self.period)
            if self.period <= 0:
                raise ValueError(f"{self.name}: periodic needs period > 0")
        if self.topology == "bounded" and (self.lo is None or self.hi is None
                                           or self.hi <= self.lo):
            raise ValueError(f"{self.name}: bounded needs lo < hi")
        self.args = dict(self.args)

    @property
    def periodic(self) -> bool:
        return self.topology == "periodic"

    def to_dict(self) -> dict:
        return {"name": self.name, "topology": self.topology,
                "period": self.period, "min": self.lo, "max": self.hi,
                "source": self.source, "args": _jsonable(self.args)}

    @classmethod
    def from_dict(cls, d: dict) -> "CoordinateDef":
        return cls(name=d["name"], topology=d.get("topology", "euclidean"),
                   period=d.get("period"), lo=d.get("min"), hi=d.get("max"),
                   source=d.get("source", "sample_field"),
                   args=dict(d.get("args", {})))


@dataclass
class ActuatorDef:
    """A named generalized-force channel.

        coordinate_force  target=<coordinate name>.  The coordinate IS the
                          actuated quantity (puck's xy push, a 1-DOF crank
                          torque): F = a * e_i, no Jacobian needed.
        joint_torque      target=<joint name>.  A torque about the joint axis,
                          mapped to generalized force by F = (dθ/du)ᵀ τ with
                          θ(u) the decoded joint angle — differentiated through
                          the decoder, so it needs no per-case kinematics.
    """
    name: str
    kind: str = "coordinate_force"
    target: str = ""
    lo: float | None = None
    hi: float = 1.0
    gain: float = 1.0
    unit: str | None = None       # slider unit label; defaults per `kind` below

    _DEFAULT_UNIT = {"joint_torque": "N·m", "coordinate_force": "N"}

    def __post_init__(self):
        if self.kind not in ACTUATOR_KINDS:
            raise ValueError(f"{self.name}: bad actuator kind {self.kind!r}; "
                             f"have {ACTUATOR_KINDS}")
        if not self.target:
            raise ValueError(f"{self.name}: actuator needs a target")
        self.hi = float(self.hi)
        self.lo = -self.hi if self.lo is None else float(self.lo)
        self.gain = float(self.gain)
        if self.gain == 0.0:
            raise ValueError(f"{self.name}: actuator gain must be nonzero")
        self.unit = self.unit or self._DEFAULT_UNIT.get(self.kind, "")

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "target": self.target,
                "lo": self.lo, "hi": self.hi, "gain": self.gain, "unit": self.unit}

    @classmethod
    def from_dict(cls, d: dict) -> "ActuatorDef":
        return cls(name=d["name"], kind=d.get("kind", "coordinate_force"),
                   target=d["target"], lo=d.get("lo"), hi=float(d.get("hi", 1.0)),
                   gain=float(d.get("gain", 1.0)), unit=d.get("unit"))


# ─── build options ────────────────────────────────────────────────────────────

@dataclass
class SamplingOptions:
    """Sampling configuration.

    There are exactly two physical samplers: ``analytic`` for an ideal
    joint/closure manifold, and ``contact`` for a contact-constrained
    manifold.  ``reuse``, ``target_grid`` and the continuation settings are
    input/proposal/execution details, not additional sampler kinds.
    """
    sampler: str | None = None  # inferred from case.contact_model when omitted
    # Strategy and backend are independent selections.  ``sampler`` remains as
    # a compatibility alias for older case files during the migration.
    exploration_strategy: str = "physical_advancing_front"
    oracle_backend: str | None = None
    # ``None`` means explore until the physical frontier is exhausted (or the
    # wall-clock budget expires), rather than imposing a sample-count cutoff.
    n_samples: int | None = 2000
    step: float = 0.08
    seed: int = 0
    max_seconds: float = 1800.0
    reuse: str | None = None      # existing npz/glob to adopt instead of exploring
    target_grid: dict | None = None  # explicit joint-target grid for ideal cases
    seed_reuse: str | None = None # existing cloud used only for an exploration seed
    seed_index: int = 0
    representation: str = "packed"  # packed transforms or planar_xytheta
    tangent_rel_tol: float = 0.05    # PCA singular-value ratio for tangent rank
    tangent_eps: float | None = None # local projected-mobility probe radius
    rank_gap_ratio: float = 1e3       # first singular-value gap accepted as rank evidence
    checkpoint_seconds: float | None = None
    # How many nearest neighbours the `graph` stage offers as chord candidates
    # when it completes the explorer's spanning tree.  Each one is certified
    # individually, so this only bounds how much company a sample may keep; it
    # buys nothing to raise it past ~2^d neighbours in every direction.
    graph_k: int = 16
    # Cap on how many directions the advancing front may probe from ONE node
    # before giving up on it (marking it retired rather than re-queuing
    # forever). `None` keeps the old behaviour (retry until the cone's max
    # gap closes, however long that takes) -- only needed for a case with a
    # localized high-curvature pocket where every probe direction fails the
    # same way regardless of which one is tried (see pantograph_2d, whose
    # theta_rel~=0/+-pi fold/extension configurations stalled at ~585 probes
    # on a handful of nodes instead of the ~16 a well-conditioned 2-D cone
    # needs).
    max_probes_per_node: int | None = None
    # Oracle settle cost, the explorer's dominant expense.  One frontier pop
    # spends `probe_count` projections measuring the tangent frame before it
    # takes a single step, and each projection runs `drive_iters` driving plus
    # `relax_iters` relaxing solver frames.  On a many-body mechanism that is
    # tens of seconds per pop, so these are case-tunable rather than fixed —
    # `None` keeps the pipeline defaults.
    probe_count: int | None = None
    drive_iters: int | None = None
    relax_iters: int | None = None
    predictor_body_displacement: float | None = None
    hold_drive_during_projection: bool = False
    # Isolate every contact-continuation candidate in a fresh worker process.
    # This is for CUDA/IPC failures that cannot be recovered in-process.
    fault_tolerant: bool = False
    continuation_timeout: float = 90.0
    continuation_min_step: float | None = None
    continuation_replay_window: int = 8
    d_hat: float | None = None
    max_kappa: float | None = None      # cap IPC contact Hessian conditioning
    newton_max_iter: int | None = None  # bound one pathological IPC advance
    # Bodies that receive the projection's soft target.  Leave followers out:
    # their poses must be solved from joints/contact, not independently driven
    # toward an inconsistent predicted transform.
    actuated_bodies: list[str] | None = None
    # libuipc contact oracle: "drive_relax" (soft drive, then relax frames) or
    # "proximal_step" (one implicit step whose inertial target is the command).
    projection_mode: str = "drive_relax"
    # Newton displacement tolerance is newton_velocity_tol * dt; proximal_step
    # needs it well below proximal_tol's scale (None keeps libuipc's 0.05).
    newton_velocity_tol: float | None = None
    proximal_tol: float = 1e-5        # fixed-point tolerance of proximal_step
    proximal_max_steps: int = 8

    def to_dict(self) -> dict:
        return {"sampler": self.sampler,
                "exploration_strategy": self.exploration_strategy,
                "oracle_backend": self.oracle_backend,
                "n_samples": self.n_samples, "step": self.step, "seed": self.seed,
                "max_seconds": self.max_seconds, "reuse": self.reuse,
                "target_grid": self.target_grid,
                "seed_reuse": self.seed_reuse, "seed_index": self.seed_index,
                "representation": self.representation,
                "tangent_rel_tol": self.tangent_rel_tol,
                "tangent_eps": self.tangent_eps,
                "rank_gap_ratio": self.rank_gap_ratio,
                "checkpoint_seconds": self.checkpoint_seconds,
                "graph_k": self.graph_k,
                "max_probes_per_node": self.max_probes_per_node,
                "probe_count": self.probe_count,
                "drive_iters": self.drive_iters,
                "relax_iters": self.relax_iters,
                "predictor_body_displacement": self.predictor_body_displacement,
                "hold_drive_during_projection": self.hold_drive_during_projection,
                "fault_tolerant": self.fault_tolerant,
                "continuation_timeout": self.continuation_timeout,
                "continuation_min_step": self.continuation_min_step,
                "continuation_replay_window": self.continuation_replay_window,
                "d_hat": self.d_hat, "max_kappa": self.max_kappa,
                "newton_max_iter": self.newton_max_iter,
                "actuated_bodies": self.actuated_bodies,
                "projection_mode": self.projection_mode,
                "newton_velocity_tol": self.newton_velocity_tol,
                "proximal_tol": self.proximal_tol,
                "proximal_max_steps": self.proximal_max_steps}

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class ChartOptions:
    """How the sample cloud is cut into charts.

    `single` is the whole manifold in one coordinate patch (a global chart, with
    the feasible region enforced by the boundary field rather than by chart
    membership).  `mesh_atlas` triangulates the coordinate cloud and grows
    overlapping patches, giving membership routing and chart transitions.
    """
    mode: str = "single"
    n_charts: int | None = None       # mesh_atlas: target patch count
    overlap: float = 0.25             # mesh_atlas: patch overlap fraction
    import_mesh: str | None = None    # optional certified cells/UV cover to retain
    split_centres: list[float] | None = None  # coordinate_split chart centres
    split_half_width: float = 0.0             # defaults to 0.60*pi

    def __post_init__(self):
        if self.mode not in CHART_MODES:
            raise ValueError(f"bad chart mode {self.mode!r}; have {CHART_MODES}")

    def to_dict(self) -> dict:
        return {"mode": self.mode, "n_charts": self.n_charts,
                "overlap": self.overlap, "import_mesh": self.import_mesh,
                "split_centres": self.split_centres,
                "split_half_width": self.split_half_width}

    @classmethod
    def from_dict(cls, d: dict) -> "ChartOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class DecoderOptions:
    epochs: int = 6000
    lr: float = 2e-3
    hidden_width: int = 128
    hidden_layers: int = 4
    n_harmonics: int = 1
    rot_weight: float = 1.0
    smooth_lambda: float = 0.0
    smooth_eps: float = 0.05
    # Graph-edge secants supervise the decoder derivative field.  This is off
    # by default because it is only meaningful when the input coordinate has a
    # certified neighbour graph in the same physical branch.
    edge_tangent_weight: float = 0.0
    holdout: float = 0.2
    # How many contiguous geodesic regions half the holdout is drawn from.  0
    # keeps the old purely random split, which on a continuation cloud measures
    # interpolation between training neighbours rather than generalization; see
    # `decoder.split_holdout`.
    holdout_regions: int = 8
    seed: int = 0
    closure_weight: float = 0.0       # loop-closure loss on declared joints
    seam_weight: float = 5.0          # shared physical sample pose consistency
    val_every: int = 25               # validation cadence for best-weight selection

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class DynamicsOptions:
    """M/V teacher + optional distillation.

    The teacher is always analytic: body inertias from the declared meshes,
    Jacobians by autodiff through the trained decoder.  `distill` amortises it
    into the shared MassNet/PotentialNet; with `distill=False` the runtime calls
    the teacher directly (exact, slower).
    """
    distill: bool = True
    epochs: int = 4000
    width: int = 64
    layers: int = 3
    lr: float = 1e-3
    n_teacher: int = 4000             # teacher points sampled over the charts

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicsOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class BoundaryOptions:
    """How the feasible region is enforced at runtime."""
    kind: str = "none"
    k: float = 1e8
    margin: float = 0.0
    damping: float = 0.0
    epochs: int = 3000
    width: int = 128
    layers: int = 4

    def __post_init__(self):
        if self.kind not in BOUNDARY_KINDS:
            raise ValueError(f"bad boundary kind {self.kind!r}; have {BOUNDARY_KINDS}")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundaryOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class Expectations:
    """Declared truths the build must VERIFY, never wiring it consumes.

    A mismatch fails the build loudly instead of silently producing a ROM of the
    wrong mechanism.  `None` means "don't assert this".
    """
    dimension: int | None = None
    n_charts: int | None = None
    n_bodies: int | None = None
    contact_pairs: list[tuple[str, str]] | None = None

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "n_charts": self.n_charts,
                "n_bodies": self.n_bodies,
                "contact_pairs": (None if self.contact_pairs is None
                                  else [list(p) for p in self.contact_pairs])}

    @classmethod
    def from_dict(cls, d: dict) -> "Expectations":
        pairs = d.get("contact_pairs")
        return cls(dimension=d.get("dimension"), n_charts=d.get("n_charts"),
                   n_bodies=d.get("n_bodies"),
                   contact_pairs=None if pairs is None else [tuple(p) for p in pairs])


@dataclass
class Tolerances:
    """The fixed gates every build must pass (step 10)."""
    decoder_translation_rmse: float = 0.15
    decoder_rotation_deg: float = 5.0
    mass_min_eigenvalue: float = 1e-9
    closure: float = 1e-3
    handoff_energy_drift: float = 6e-2
    free_energy_drift: float = 3e-1
    energy_work_residual: float = 5e-1
    validation_energy_err: float = 1e-6

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "Tolerances":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class CanonicalOptions:
    """Whether and how to build the canonical coordinate `z`.

    `dimension` defaults to 2*d (Whitney), which is enough.  Measured on the
    five-bar's genus-4 surface, with the canonical decoder's validation error as
    the thing that actually matters:

        d_z=4   purity 0.990  sigma_min 0.21    0.0262 / 0.76 deg
        d_z=5   purity 1.000  sigma_min 0.73    0.0113 / 0.36 deg
        d_z=6   purity 1.000  sigma_min 0.82    0.0088 / 0.31 deg

    All three pass the case's declared tolerances (0.05 / 3.0 deg); the
    chart-conditioned decoder they replace passes none of them (0.168 / 5.18).
    So 2*d is a working default and the extra column buys a factor of two, with
    the next one buying almost nothing.  Raise it when the purity warning fires
    and the margin matters; do not assume a low dimension is what is wrong with
    a bad decoder, which is the mistake made here once already — a mis-scaled
    tangent loss, not d_z, was responsible for a 22-degree fit at d_z=4.

    `target_separation` is the one setting that matters most, and it is measured
    in ambient edge lengths.  At 1.0 the refinement is satisfied as soon as far
    pairs are one sample spacing apart, which is INSIDE the local neighbourhood:
    36% of samples then keep a stranger among their eight nearest neighbours
    while every scalar gate reads healthy.  The ambient cloud achieves 2.5, so
    that is the reference the target is set against.
    """
    enabled: bool = False
    dimension: int | None = None
    n_candidates: int = 40
    far_hops: int = 4
    refine_steps: int = 1500
    target_separation: float = 2.5
    target_sigma: float = 0.75
    planarity: float = 0.12
    # Largest |dh|/|du| between neighbouring samples a chart may carry, i.e.
    # how far its surface may stand up over its own projection plane.  This is
    # what makes the height single-valued LOCALLY; `planarity` only bounds the
    # patch's worst off-plane distance against its radius, which every chart of
    # the first three-RRS cover satisfied while still reaching slope 72.
    max_slope: float = 0.5
    # Fit each chart height from HEIGHTS ALONE and take its tangent and
    # curvature by differentiating that one function, rather than constraining
    # the fit with separately measured slopes.  The slopes are not independent
    # truth — `local_jacobians` returns a curvature-biased secant estimate — so
    # imposing them makes position and tangent contradictory by construction and
    # the fit oscillates between the nodes to satisfy both.  Differentiating a
    # single fitted height cannot disagree with itself.
    height_only: bool = False
    # Discover chart supports directly from the canonical samples and their
    # certified neighbour graph.  The legacy mesh cover is then evaluation-only.
    residual_discovery: bool = False
    # Replace the linear-projection charts with learned residual autoencoders
    # (`chart_autoencoder`).  `R_i` and `E_i` both become nonlinear residuals on
    # the PCA chart they are initialised from, which is what lets a patch grow
    # past a fold: `max_slope` above exists only because a graph over a plane
    # cannot represent one, and it is what forces the chart count.  Position,
    # tangent and curvature then all come from differentiating one function, so
    # the height/slope contradiction has nowhere to live.  The cost is that
    # `R_i . E_i = id` stops being algebraic and has to be trained and certified.
    autoencoder_charts: bool = False
    autoencoder_width: int = 128
    autoencoder_depth: int = 3
    autoencoder_seed_epochs: int = 1500
    # No warm start between growth steps (the frame is re-derived each time),
    # so a step costs the same as the seed and should get the same budget.
    autoencoder_ring_epochs: int = 1500
    autoencoder_tangent_weight: float = 0.3
    # Gauge fixing on the decoder's in-plane residual, NOT regularisation: with
    # it free, every reparameterisation of `u` describes the same chart and the
    # optimiser wanders that null direction instead of fitting.  See
    # `chart_autoencoder.fit_patch`.
    autoencoder_plane_weight: float = 1.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class BuildOptions:
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    charts: ChartOptions = field(default_factory=ChartOptions)
    canonical: CanonicalOptions = field(default_factory=CanonicalOptions)
    decoder: DecoderOptions = field(default_factory=DecoderOptions)
    dynamics: DynamicsOptions = field(default_factory=DynamicsOptions)
    boundary: BoundaryOptions = field(default_factory=BoundaryOptions)
    expect: Expectations = field(default_factory=Expectations)
    tolerances: Tolerances = field(default_factory=Tolerances)

    def to_dict(self) -> dict:
        return {"sampling": self.sampling.to_dict(), "charts": self.charts.to_dict(),
                "canonical": self.canonical.to_dict(),
                "decoder": self.decoder.to_dict(), "dynamics": self.dynamics.to_dict(),
                "boundary": self.boundary.to_dict(), "expect": self.expect.to_dict(),
                "tolerances": self.tolerances.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "BuildOptions":
        return cls(sampling=SamplingOptions.from_dict(d.get("sampling", {})),
                   charts=ChartOptions.from_dict(d.get("charts", {})),
                   canonical=CanonicalOptions.from_dict(d.get("canonical", {})),
                   decoder=DecoderOptions.from_dict(d.get("decoder", {})),
                   dynamics=DynamicsOptions.from_dict(d.get("dynamics", {})),
                   boundary=BoundaryOptions.from_dict(d.get("boundary", {})),
                   expect=Expectations.from_dict(d.get("expect", {})),
                   tolerances=Tolerances.from_dict(d.get("tolerances", {})))


# ─── the case ─────────────────────────────────────────────────────────────────

@dataclass
class AtlasCaseSpec:
    """Everything the pipeline needs to know about one mechanism."""
    name: str
    mechanism: MechanismSpec
    coordinates: list[CoordinateDef] = field(default_factory=list)
    seeds: list[Any] = field(default_factory=list)
    contact_model: str = "exact_clearance"
    actuators: list[ActuatorDef] = field(default_factory=list)
    build: BuildOptions = field(default_factory=BuildOptions)
    scenarios: dict[str, dict] = field(default_factory=dict)
    plugin: str | None = None          # step 8: opt-in, exceptional

    def __post_init__(self):
        if isinstance(self.mechanism, dict):
            self.mechanism = MechanismSpec.from_dict(self.mechanism)
        self.coordinates = [c if isinstance(c, CoordinateDef)
                            else CoordinateDef.from_dict(c) for c in self.coordinates]
        self.actuators = [a if isinstance(a, ActuatorDef)
                          else ActuatorDef.from_dict(a) for a in self.actuators]
        if isinstance(self.build, dict):
            self.build = BuildOptions.from_dict(self.build)
        if self.contact_model not in CONTACT_MODELS:
            raise ValueError(f"bad contact model {self.contact_model!r}; "
                                 f"have {CONTACT_MODELS}")
        sampler = self.build.sampling.sampler
        if sampler is None:
            sampler = "analytic" if self.contact_model == "none" else "contact"
            self.build.sampling.sampler = sampler
        if self.build.sampling.oracle_backend is None:
            self.build.sampling.oracle_backend = sampler
        if self.build.sampling.exploration_strategy != "physical_advancing_front":
            raise ValueError(f"{self.name}: unsupported exploration strategy "
                             f"{self.build.sampling.exploration_strategy!r}")
        if self.build.sampling.oracle_backend not in ("analytic", "contact", "sdf"):
            raise ValueError(f"{self.name}: unsupported oracle backend "
                             f"{self.build.sampling.oracle_backend!r}")
        if sampler not in ("analytic", "contact"):
            raise ValueError(f"{self.name}: sampling.sampler must be 'analytic' or "
                             f"'contact', got {sampler!r}")
        if sampler == "analytic" and self.contact_model != "none":
            raise ValueError(f"{self.name}: analytic sampling requires contact_model='none'")
        # A prior-free atlas begins with physical poses and an oracle, not a
        # coordinate declaration.  Its intrinsic dimension is still recorded
        # as the discovered/certified expectation for downstream tensor shapes.
        prior_free = self.build.canonical.enabled and self.build.canonical.residual_discovery
        if not self.coordinates and not prior_free:
            raise ValueError(f"{self.name}: a case must declare its coordinates")
        names = [c.name for c in self.coordinates]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: coordinate names must be unique")
        self.seeds = [np.asarray(s, float) if not isinstance(s, dict) else s
                      for s in self.seeds]
        # actuator targets must resolve NOW, not at runtime
        for a in self.actuators:
            if a.kind == "coordinate_force" and a.target not in names:
                raise ValueError(f"{self.name}: actuator {a.name!r} targets unknown "
                                 f"coordinate {a.target!r}; have {names}")
            if a.kind == "joint_torque":
                self.mechanism.joint(a.target)      # raises if unknown
        # declared dimension, if any, must match the declared coordinates
        expect = self.build.expect
        if (self.coordinates and expect.dimension is not None
                and expect.dimension != len(self.coordinates)):
            raise ValueError(f"{self.name}: expect.dimension={expect.dimension} but "
                             f"{len(self.coordinates)} coordinates declared")

    # -- derived ---------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return len(self.coordinates) or int(self.build.expect.dimension or 0)

    @property
    def n_bodies(self) -> int:
        return len(self.mechanism.driven_bodies)

    @property
    def periodic_mask(self) -> list[bool]:
        return [c.periodic for c in self.coordinates]

    @property
    def periods(self) -> list[float | None]:
        return [c.period for c in self.coordinates]

    def coordinate(self, name: str) -> CoordinateDef:
        for c in self.coordinates:
            if c.name == name:
                return c
        raise KeyError(f"unknown coordinate {name!r}")

    def coordinate_index(self, name: str) -> int:
        return [c.name for c in self.coordinates].index(name)

    @property
    def control_schema(self) -> list[dict]:
        """The canonical, ORDERED control schema: one entry per declared
        actuator, in declaration order.  Both the full-space and ROM runners
        build their sliders from exactly this list (and nothing else), so a
        control vector's index always means the same actuator in either
        backend."""
        return [{"name": a.name, "kind": a.kind, "target": a.target,
                 "lo": a.lo, "hi": a.hi, "gain": a.gain, "unit": a.unit}
                for a in self.actuators]

    def scenario(self, name: str | None = None) -> dict:
        if not self.scenarios:
            return {}
        if name is None:
            name = "default" if "default" in self.scenarios else next(iter(self.scenarios))
        if name not in self.scenarios:
            raise KeyError(f"unknown scenario {name!r}; have {sorted(self.scenarios)}")
        return dict(self.scenarios[name])

    # -- serialization (work/<case>/mechanism.json) ----------------------------
    def to_dict(self) -> dict:
        return {"format": SPEC_FORMAT, "name": self.name,
                "mechanism": self.mechanism.to_dict(),
                "coordinates": [c.to_dict() for c in self.coordinates],
                "seeds": _jsonable(self.seeds),
                "contact_model": self.contact_model,
                "actuators": [a.to_dict() for a in self.actuators],
                "build": self.build.to_dict(),
                "scenarios": _jsonable(self.scenarios),
                "plugin": self.plugin}

    @classmethod
    def from_dict(cls, d: dict) -> "AtlasCaseSpec":
        if d.get("format") != SPEC_FORMAT:
            raise ValueError(f"not an {SPEC_FORMAT} spec (got {d.get('format')!r})")
        return cls(name=d["name"],
                   mechanism=MechanismSpec.from_dict(d["mechanism"]),
                   coordinates=[CoordinateDef.from_dict(c) for c in d["coordinates"]],
                   seeds=[np.asarray(s, float) for s in d.get("seeds", [])],
                   contact_model=d.get("contact_model", "exact_clearance"),
                   actuators=[ActuatorDef.from_dict(a) for a in d.get("actuators", [])],
                   build=BuildOptions.from_dict(d.get("build", {})),
                   scenarios=dict(d.get("scenarios", {})),
                   plugin=d.get("plugin"))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "AtlasCaseSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
