"""cases/puck.py — a puck free in a square arena around a central peg.

The reachable set of puck positions is an ANNULUS: a disk with a hole punched by
the peg.  Two things make it the right first migration to the generic pipeline:

  * the coordinate is Cartesian, so `body_translation` reads it off directly and
    there is nothing clever to hide a per-case shortcut in;
  * the manifold's boundary is genuine contact, not a joint limit, so the feasible
    region has to come from a learned domain field rather than chart membership —
    which exercises the `boundary` half of the manifest.

Everything below is data.  There is no `load()`, no decoder class, no renderer.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BoundaryOptions, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

_PUCK_RADIUS = 0.3
_PEG_RADIUS = 1.0
_ARENA_HALF = 3.0
_WALL_THICK = 0.3
_Z_HALF = 1.5
_A = _ARENA_HALF + _WALL_THICK
_LONG = _ARENA_HALF + 2.0 * _WALL_THICK


def _wall(name, center, half):
    return BodyDef(name=name, mesh=MeshSpec("box", {"center": center, "half": half}),
                   fixed=True, color=(0.60, 0.62, 0.66))


CASE = AtlasCaseSpec(
    name="puck",
    mechanism=MechanismSpec(
        bodies=[
            # the puck's position lives in its TRANSFORM (verts centred at the
            # origin), which is what puts it in the config vector at all
            BodyDef(name="puck",
                    mesh=MeshSpec("sphere", {"radius": _PUCK_RADIUS, "subdiv": 2}),
                    initial_transform=[[1, 0, 0, 2.0], [0, 1, 0, 0.0],
                                       [0, 0, 1, 0.0], [0, 0, 0, 1]],
                    color=(0.90, 0.35, 0.20)),
            # a SPHERE peg: a box peg's sharp edges send sphere-vs-edge contact
            # into degenerate CCD work and stall the settle
            BodyDef(name="__peg",
                    mesh=MeshSpec("sphere", {"radius": _PEG_RADIUS, "subdiv": 2}),
                    fixed=True, color=(0.55, 0.30, 0.30)),
            _wall("__wall_px", [_A, 0.0, 0.0], [_WALL_THICK, _LONG, _Z_HALF]),
            _wall("__wall_nx", [-_A, 0.0, 0.0], [_WALL_THICK, _LONG, _Z_HALF]),
            _wall("__wall_py", [0.0, _A, 0.0], [_LONG, _WALL_THICK, _Z_HALF]),
            _wall("__wall_ny", [0.0, -_A, 0.0], [_LONG, _WALL_THICK, _Z_HALF]),
        ],
        joints=[],                       # no joints: the boundary is pure contact
        up_axis="y",                     # gravity acts in the plane of motion...
        render_up_axis="z",              # ...while the camera looks down the plane
        gravity=9.8,
    ),
    seeds=[],                            # the rest pose is the seed
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef(name="x", topology="euclidean", source="body_translation",
                      args={"body": "puck", "axis": "x"}),
        CoordinateDef(name="y", topology="euclidean", source="body_translation",
                      args={"body": "puck", "axis": "y"}),
    ],
    actuators=[
        # the coordinate IS the puck's Cartesian position, so a generalized force
        # here is literally a push — the degenerate case of the same actuator rule
        ActuatorDef(name="push x", kind="coordinate_force", target="x", hi=200.0),
        ActuatorDef(name="push y", kind="coordinate_force", target="y", hi=200.0),
    ],
    build=BuildOptions(
        # the shard pool is the FULL ring; the single-process puck_chart.npz covers
        # only ~63% of it, and a decoder trained on the partial ring extrapolates
        # into the untrained half the moment the dynamics swings there
        sampling=SamplingOptions(reuse="output/shards/shard_*.npz", n_samples=4000),
        charts=ChartOptions(mode="single"),
        decoder=DecoderOptions(epochs=4000, n_harmonics=1, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=1500),
        boundary=BoundaryOptions(kind="sdf", k=1e8),
        expect=Expectations(dimension=2, n_charts=1, n_bodies=1),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=1.0,
                              free_energy_drift=3e-1),
    ),
    scenarios={"default": {"dt": 2e-3, "initial_u": [0.0, 2.0]}},
)
