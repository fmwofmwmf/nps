"""Ball-joint arm: an L-shaped chiral body hung off a spherical joint.

Configuration space is SO(3) = RP^3, exactly as in ``generic_so3`` -- with no
contact, the shape of a body on a ball joint cannot change the topology.  What
changes is the dynamics: the body's centre of mass sits well below the pivot
and off its axis, and the L (arm along -z, foot along +x, toe along +y) is
chiral, so the inertia tensor is fully anisotropic and gravity has a genuine
minimum, saddles and maximum on SO(3).  That makes this the dynamics rung on
a non-torus 3-manifold.

Mesh vertices are baked at the rest world pose and the pivot is the world
origin, so the body transform's translation stays zero and the packed pose is
the same 12-D rotation block ``generic_so3`` samples; its sampling step and
tangent probe radius carry over unchanged.

Parts are disjoint boxes that only share faces, so the concatenated compound's
volume and inertia are the sum of the parts'.
"""
from __future__ import annotations

from atlas_rom.cases.generic.generic_so3 import VIEW_EMBED  # noqa: F401  (viewer hook)
from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions

# Evaluation ground truth.  Not part of CASE or any artifact: the fitter must
# derive both numbers from measured evidence.
EVALUATOR_REFERENCE = {"configuration_space": "RP^3 (SO(3))", "intrinsic_dimension": 3,
                       "min_charts_ls_category": 4,
                       "representation": "body orientation about a fixed spherical joint"}


_L_BODY = MeshSpec("compound", {"parts": [
    # arm: z in [-0.67, -0.07]
    MeshSpec("box", {"center": (0.0, 0.0, -0.37), "half": (0.03, 0.03, 0.30)}),
    # foot along +x: x in [0, 0.40], z in [-0.77, -0.67]
    MeshSpec("box", {"center": (0.20, 0.0, -0.72), "half": (0.20, 0.05, 0.05)}),
    # toe along +y at the foot's end: y in [0.05, 0.35]
    MeshSpec("box", {"center": (0.35, 0.20, -0.72), "half": (0.05, 0.15, 0.05)}),
]})

CASE = AtlasCaseSpec(
    name="ball_joint_arm",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef(name="__ground", mesh=MeshSpec("sphere", {"radius": .05, "subdiv": 1}),
                    fixed=True, color=(.45, .47, .50)),
            BodyDef(name="arm", mesh=_L_BODY,
                    initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 1, 0], [0, 0, 0, 1]],
                    color=(.90, .52, .20)),
        ],
        joints=[JointDef("ball", "spherical", "arm", "__ground", pivot=(0, 0, 0))],
        up_axis="z",
        gravity=9.8,
    ),
    contact_model="none",
    # Discovery receives no coordinate labels, UVs, or declared chart count.
    coordinates=[],
    # Same packed-pose metric as generic_so3 (translation pinned at zero), so its
    # step / probe radius are reused: tangent_eps << step keeps SO(3)'s curvature
    # out of the secant-PCA rank.  No sample cap and no time budget: the
    # frontier must close on its own for the dataset to be trainable.
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.47307171, checkpoint_seconds=60.,
                                                tangent_eps=.05, max_seconds=None,
                                                max_probes_per_node=4),
                       canonical=CanonicalOptions(enabled=True, residual_discovery=True,
                                                  autoencoder_charts=True),
                       expect=Expectations(n_bodies=1)),
)
