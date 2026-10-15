"""SO(3) = RP^3: one body on a spherical joint, the ideal-joint (analytic) path.

SO(3) is diffeomorphic to real projective 3-space, so this is the RP^3 rung of
the generic family.  It is deliberately *not* an SDF case:

* RP^3 has no smooth embedding in R^4 (it is the lens space L(2,1), whose
  linking form does not bound), so it cannot reuse the R^4 ambient the S^3 / T^3
  cases use.
* SO(3) is codimension 6 in R^9.  The generic SDF oracle retracts onto the
  *codimension-1* zero level of one scalar field; no single scalar has SO(3) as
  a regular zero set, so that machinery cannot represent it.

The configuration manifold of a single rigid body whose centre is pinned by a
spherical joint is exactly SO(3): ``RᵀR = I`` per body plus the pivot
coincidence, three constraints for three rotational freedoms.  That is an
algebraic manifold, so ``ideal_oracle`` projects onto it in closed form (one
SVD) and the sampler routes there automatically because ``contact_model`` is
``"none"``.

``VIEW_EMBED`` maps a sample's packed pose to the body's principal rotation
vector -- the classic solid ball of radius pi with antipodal boundary
identification -- for ``viewers.view_sample_set``.
"""
from __future__ import annotations

import numpy as np

from atlas_rom.spec.case_spec import AtlasCaseSpec, BodyDef, BuildOptions, CoordinateDef, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions

# Evaluation ground truth.  Deliberately not part of CASE or any artifact: the
# fitter must derive both numbers from measured evidence.
GROUND_TRUTH = {"topology": "RP^3 (SO(3))", "intrinsic_dimension": 3,
                "min_charts_ls_category": 4}


def _rotvec(R):
    """Principal rotation vector; the antipodal pair at |v| = pi is RP^3's seam."""
    R = np.asarray(R, float).reshape(3, 3)
    cos = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-6:
        return np.zeros(3)
    if theta < np.pi - 1e-4:
        axis = np.array((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
        return theta * axis / np.linalg.norm(axis)
    values, vectors = np.linalg.eigh((R + np.eye(3)) * 0.5)  # +1 eigenvector
    axis = vectors[:, int(np.argmax(values))]
    return theta * axis / np.linalg.norm(axis)


def VIEW_EMBED(q):
    """(N, 12) packed poses -> (N, 3) axis-angle vectors, for view_sample_set."""
    T = np.asarray(q, float).reshape(len(q), 3, 4)
    return np.asarray([_rotvec(frame[:, :3]) for frame in T], float)


CASE = AtlasCaseSpec(
    name="generic_so3",
    mechanism=MechanismSpec(
        bodies=[
            BodyDef(name="__ground", mesh=MeshSpec("sphere", {"radius": .05, "subdiv": 1}),
                    fixed=True, color=(.45, .47, .50)),
            # An asymmetric brick so the orientation is legible in a pose
            # scrubber; the manifold sees only the 3x3 rotation block.
            BodyDef(name="probe", mesh=MeshSpec("box", {"half": (.34, .20, .11)}),
                    initial_transform=[[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 1, 0], [0, 0, 0, 1]],
                    color=(.25, .55, .90)),
        ],
        joints=[JointDef("centre", "spherical", "probe", "__ground", pivot=(0, 0, 0))],
    ),
    # System-observation declarations only; the analytic sampler derives chart
    # rank from measured pose secants.  Translation is pinned, so the readable
    # coordinates are body rotation angles (not a global chart -- rotations do
    # not commute -- which is exactly why the atlas needs more than one).
    coordinates=[CoordinateDef("rot_x", source="body_angle",
                               args={"body": "probe", "axis": (1, 0, 0), "reference": "identity"}),
                 CoordinateDef("rot_y", source="body_angle",
                               args={"body": "probe", "axis": (0, 1, 0), "reference": "identity"}),
                 CoordinateDef("rot_z", source="body_angle",
                               args={"body": "probe", "axis": (0, 0, 1), "reference": "identity"})],
    contact_model="none",
    # ``tangent_eps`` is set well below ``step``: SO(3)'s curvature over a full
    # walking step contaminates the secant-PCA tangent rank (it reads 4-6
    # instead of 3 and the front aborts on nonuniform ranks); a small probe
    # radius suppresses the second-order term.
    build=BuildOptions(sampling=SamplingOptions(n_samples=None, step=.47307171, checkpoint_seconds=60.,
                                                tangent_eps=.05, max_seconds=None,
                                                max_probes_per_node=4),
                       expect=Expectations(dimension=3, n_bodies=1)),
)
