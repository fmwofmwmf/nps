"""Atlas Explorer case for ``examples/8_screw_and_nut``.

The source example fixes the nut and leaves the screw in direct full-space ABD
contact dynamics.  The recorded gravity trajectories use that same setup: there
is no synthetic joint or helical constraint; screw travel and spin are read from
the resulting contact poses.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, MechanismSpec, MeshSpec, SamplingOptions, Tolerances, WorkspaceBound, BoundaryOptions


_SCREW_MESH = "assets/sim_data/trimesh/screw-and-nut/screw-big-2.obj"
_NUT_MESH = "assets/sim_data/trimesh/screw-and-nut/nut-big-2.obj"
_Y = [0.0, 1.0, 0.0]
_SCALE = 1.0 / 30.0
_DENSITY_SCALE = _SCALE ** -3
_TANGENT_EPS = 0.035 * 3.0 * 4.0 / 30.0
_DOWN_TRAVEL_LO = -0.221
_INVERSE_TRAVEL = (0.0118180001 + 0.75 * (3.99644995 - 0.0118180001)
                   - 0.661768973) * _SCALE
# Local phase limits measured from the direct full-space recording.  They are
# not a global angle: each branch is integrated from its common start pose.
_SPIN_PHASE_LO = -31.737728146454714
_SPIN_PHASE_HI = 10.692525411536849


CASE = AtlasCaseSpec(
    name="screw",
    mechanism=MechanismSpec(
        bodies=[
            # The imported OBJ volumes are 383.579 (screw) and 105.591 (nut),
            # so these densities give each body unit mass.  The nut is static;
            # retaining its density keeps the spec physically complete.
            BodyDef("screw", MeshSpec("file", {"path": _SCREW_MESH,
                                                  "scale": _SCALE}),
                    density=_DENSITY_SCALE / 383.5787229270007,
                    color=(0.80, 0.52, 0.18)),
            BodyDef("nut", MeshSpec("file", {"path": _NUT_MESH,
                                                "scale": _SCALE}), fixed=True,
                    density=_DENSITY_SCALE / 105.59088094952897,
                    color=(0.42, 0.46, 0.52)),
        ],
        joints=[],
        up_axis="y", gravity=9.8,
        contact_pairs=[("screw", "nut")],
        # A cylindrical joint remains valid after the threads disengage, but
        # that free-travel branch is not part of this first contact atlas.
        workspace_bounds=[WorkspaceBound("engaged_screw_travel", body="screw",
                                         axis="y", lo=_DOWN_TRAVEL_LO,
                                         hi=_INVERSE_TRAVEL)],
    ),
    seeds=[],
    contact_model="exact_clearance",
    coordinates=[
        CoordinateDef("local_spin_phase", topology="bounded", lo=_SPIN_PHASE_LO,
                      hi=_SPIN_PHASE_HI, source="sample_field",
                      args={"key": "local_spin"}),
    ],
    actuators=[],
    build=BuildOptions(
        sampling=SamplingOptions(reuse="work/screw/fullspace_gravity_poses.npz",
                                  n_samples=8, step=0.06 * _SCALE,
                                  max_seconds=60.0, tangent_rel_tol=0.05,
                                  tangent_eps=_TANGENT_EPS, probe_count=8,
                                  drive_iters=3, relax_iters=2,
                                  predictor_body_displacement=0.06 * _SCALE,
                                  d_hat=0.02 * _SCALE,
                                  max_kappa=1e9, newton_max_iter=64,
                                  actuated_bodies=["screw"]),
        charts=ChartOptions(mode="single"),
        # The free-body contact trace has appreciable frame-to-frame angular
        # settle jitter (about 10 degrees locally).  Keep the decoder smooth so
        # it models the screw trajectory rather than memorising that jitter.
        decoder=DecoderOptions(epochs=4000, smooth_lambda=1e-3),
        dynamics=DynamicsOptions(distill=True, epochs=2000, n_teacher=800),
        # The sampled coordinate is a genuine closed interval.  Use its exact
        # interval barrier and a backward-Euler contact solve so the two travel
        # limits absorb velocity instead of behaving like elastic bumpers.
        boundary=BoundaryOptions(kind="interval", k=1e8, margin=0.0),
        expect=Expectations(dimension=1, n_bodies=1),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0),
    ),
    scenarios={"default": {"dt": 1e-3, "initial_u": [0.0],
                           "boundary_solver": "damped"}},
)
