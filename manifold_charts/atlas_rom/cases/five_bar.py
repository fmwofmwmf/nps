"""cases/five_bar.py — a planar 5R five-bar linkage with ideal joints, no contact.

Two grounded pivots A and B drive the proximal links a1 and a2; the distal links
b1 and b2 close the loop at the end-effector E.  DOF = 3*4 - 2*5 = 2.

This is the case that exercises the parts puck cannot:

  * MULTIPLE CHARTS.  The two motor angles are a perfectly good global coordinate
    except that they do not distinguish elbow-out from elbow-in — the same
    (theta1, theta2) is reached by more than one assembly.  The atlas stage finds
    those branches as configuration-space components and the decoder's chart
    embedding represents each one, instead of averaging them into a pose that is
    neither.
  * JOINT-TORQUE ACTUATORS.  A driver torque is not a force on a coordinate; it
    maps through d(theta)/du, differentiated through the decoder.
  * LOOP CLOSURE.  Pivots C, D and E sit off their bodies' origins, so a per-body
    pose loss can be small while the linkage visibly tears apart at its joints.
    `closure_weight` trains the declared pivots to stay coincident.

Geometry matches `viz_cases.five_bar_geometry()` exactly: links are rest-world-baked
bars, so each body starts at transform = identity, and the recorded config places
it the same way the solver did.
"""
from __future__ import annotations

from atlas_rom.spec.case_spec import ActuatorDef, AtlasCaseSpec, BodyDef, BuildOptions, CanonicalOptions, ChartOptions, CoordinateDef, DecoderOptions, DynamicsOptions, Expectations, JointDef, MechanismSpec, MeshSpec, SamplingOptions, Tolerances

# pivots of the rest assembly (base=2.0, la=lb=1.5, ey=1.6), elbow-out both sides
A = [-1.0, 0.0, 0.0]
B = [1.0, 0.0, 0.0]
C = [-1.488927, 1.418080, 0.0]
D = [1.488927, 1.418080, 0.0]
E = [0.0, 1.6, 0.0]
THICK = 0.12
Z = [0.0, 0.0, 1.0]                       # every pin is a planar hinge about +z


def _bar(name, p0, p1, color, thick=THICK):
    return BodyDef(name=name, mesh=MeshSpec("link", {"p0": p0, "p1": p1,
                                                     "thick": thick}),
                   color=color)


CASE = AtlasCaseSpec(
    name="five_bar",
    mechanism=MechanismSpec(
        bodies=[
            _bar("a1", A, C, (0.93, 0.53, 0.18)),      # driver, left
            _bar("a2", B, D, (0.93, 0.53, 0.18)),      # driver, right
            _bar("b1", C, E, (0.36, 0.62, 0.86)),
            _bar("b2", D, E, (0.36, 0.62, 0.86)),
            BodyDef(name="__ground",
                    mesh=MeshSpec("link", {"p0": [A[0] - 0.25, 0.0, 0.0],
                                           "p1": [B[0] + 0.25, 0.0, 0.0],
                                           "thick": THICK * 1.3}),
                    fixed=True, color=(0.45, 0.47, 0.50)),
        ],
        joints=[
            JointDef("A", "revolute", "__ground", "a1", pivot=A, axis=Z),
            JointDef("B", "revolute", "__ground", "a2", pivot=B, axis=Z),
            JointDef("C", "revolute", "a1", "b1", pivot=C, axis=Z),
            JointDef("D", "revolute", "a2", "b2", pivot=D, axis=Z),
            JointDef("E", "revolute", "b1", "b2", pivot=E, axis=Z),  # the closure
        ],
        up_axis="y",                     # planar linkage in xy: gravity is in-plane
        render_up_axis="z",
        gravity=9.8,
    ),
    seeds=[],
    contact_model="none",                # ideal joints; the manifold is the loop
    # The two motor angles look like the obvious coordinate and are NOT one on this
    # manifold: measured over the explored patch they are 96% correlated, so the
    # 2-D configuration set projects onto a near-curve and configurations 0.9 apart
    # share a coordinate to within the sample spacing.  The intrinsic (geodesic MDS)
    # coordinate of the configuration cloud is the one that actually parameterises
    # it — the same conclusion the earlier five-bar work reached independently.
    # Discovery receives no coordinate labels, UVs, or declared chart count.
    coordinates=[],
    actuators=[
        ActuatorDef("driver A torque", kind="joint_torque", target="A", hi=40.0),
        ActuatorDef("driver B torque", kind="joint_torque", target="B", hi=40.0),
    ],
    build=BuildOptions(
        # Discover the physical manifold directly.  No legacy atlas fields,
        # labels, UVs, or requested chart count enter the sampler or learner.
        sampling=SamplingOptions(
            n_samples=None, step=1.0, max_seconds=1800,
            representation="planar_xytheta", graph_k=32),
        charts=ChartOptions(mode="mesh_atlas", overlap=0.25),
        # The legacy four-chart cover remains available only for post-discovery
        # coverage/distortion/topology comparison; it cannot seed this learner.
        canonical=CanonicalOptions(enabled=True, dimension=2, planarity=0.2,
                                   residual_discovery=True,
                                   autoencoder_charts=True),
        decoder=DecoderOptions(epochs=6000, smooth_lambda=1e-3,
                               closure_weight=1.0),
        dynamics=DynamicsOptions(distill=True, epochs=3000, n_teacher=1500),
        expect=Expectations(dimension=2, n_bodies=4),
        tolerances=Tolerances(decoder_translation_rmse=0.05,
                              decoder_rotation_deg=3.0, closure=5e-2),
    ),
    scenarios={"default": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "exact": {"dt": 1e-3, "dynamics_backend": "exact_canonical"},
               "distilled": {"dt": 1e-3, "dynamics_backend": "distilled_canonical"},
               "projected": {"dt": 1e-3, "dynamics_backend": "exact_canonical",
                             "projected_patch": True},
               "direct": {"dt": 1e-3, "dynamics_backend": "direct_constrained"},
               "graph": {"dt": 1e-3, "dynamics_backend": "graph_constrained",
                         "kinematic_decoder": True}},
)


def VIEW_EMBED(q):
    """Display-only 3-D unfolding of the five-bar C-space for ``view_sample_set``.

    The two motor angles collapse elbow-in / elbow-out, and any single body's
    translation is just a curve, so neither shows the surface.  The five-bar
    C-space is a closed orientable surface (genus 4 for these link lengths);
    this places every sample by eigenvectors 1..3 of the normalised graph
    Laplacian of a kNN graph on the full pose -- an intrinsic embedding that
    reveals the handle structure (it self-intersects in R^3, as it must).
    Never passed to the fitter; recomputed from ``q`` alone.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    Q = np.asarray(q, float)
    n = len(Q)
    T = Q.reshape(n, -1, 3, 4)
    feat = np.concatenate((T[..., :3, :2].reshape(n, -1), T[..., 3].reshape(n, -1)), axis=1)
    k = min(12, n - 1)
    _, nn = cKDTree(feat).query(feat, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = nn[:, 1:].ravel()
    a = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    a = a.maximum(a.T)
    values, vectors = eigsh(laplacian(a, normed=True).astype(float), k=4, sigma=1e-8, which="LM")
    return vectors[:, np.argsort(values)[1:4]] * np.sqrt(n)


# Display / task-space hook for ``viewers.rom_target``: pen name -> (body, rest-world local point).  The rest pose
# is a nearly flat toggle: holding it takes ~1400 N.m (35x the declared 40), hence the larger torque limit.
PEN_TARGET = {
    "pens": {"E": ("b1", [float(x) for x in E])},
    "circles": [],
    "torque_limit": 80.0,
}
