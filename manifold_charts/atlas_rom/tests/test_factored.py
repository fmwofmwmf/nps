"""factored: factor-by-factor sampling of a product, on closed-form systems.

Each toy system has bodies that move with one factor only and a body that
moves with both (a rotation composed of the two, like a steered, driven
wheel).  The known answers live only here."""
from __future__ import annotations

import numpy as np
import pytest

from atlas_rom.sample.factored import FactoredOptions, LeafSystem, factored_explore
from atlas_rom.sample.physical_advancing_front import AdvancingFrontOptions
from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse


def _rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _block(R, t=(0., 0., 0.)):
    return np.column_stack((R, np.asarray(t, float))).reshape(-1)


class _Toy:
    """Response-only system over a hidden state; a command moves the state by
    the observation Jacobian's pseudo-inverse and re-observes."""

    def __init__(self, observe, dimension, valid):
        self.observe, self.dimension, self.valid = observe, dimension, valid
        self._state = {}

    def _obs(self, state):
        pose = self.observe(state)
        self._state[pose.tobytes()] = np.asarray(state, float)
        return PhysicalObservation(pose)

    def seed(self):
        return self._obs(np.zeros(self.dimension))

    def restore(self, observation):
        return self._state[np.asarray(observation.pose, float).reshape(-1).tobytes()]

    def command(self, session, direction, scale, *, parent=None):
        eps = 1e-7
        J = np.column_stack([(self.observe(session + eps * e) - self.observe(session - eps * e)) / (2 * eps)
                             for e in np.eye(self.dimension)])
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        state = session + scale * (np.linalg.pinv(J) @ d)
        valid = bool(self.valid(state))
        return PhysicalResponse(self._obs(state), True, valid,
                                {} if valid else {"physical_boundary": "limit"}, parent)

    def command_many(self, session, directions, scale, *, parent=None):
        return [self.command(session, d, scale, parent=parent) for d in directions]


def _circle_x_interval(epsilon=0.0):
    def observe(x):
        theta, s = x
        wheel = theta + epsilon * s * s * np.sin(2 * theta)
        return np.concatenate((_block(_rx(theta)),                          # moves with A only
                               _block(_rz(.5 * s), (s, 0, 0)),               # moves with B only
                               _block(_rz(.5 * s) @ _rx(wheel), (s, 0, 0)))) # both
    return _Toy(observe, 2, lambda x: abs(x[1]) <= 1.0)


def _torus_x_interval():
    def observe(x):
        theta, phi, s = x
        return np.concatenate((_block(_rx(theta)), _block(_ry(phi), (0, 2, 0)),
                               _block(_rz(.5 * s), (s, 0, 0)),
                               _block(_rz(.5 * s) @ _rx(theta), (s, 0, 0))))
    return _Toy(observe, 3, lambda x: abs(x[2]) <= 1.0)


FRONT = AdvancingFrontOptions(step=.4, max_seconds=300)


def test_exact_product_is_sampled_by_factors():
    result = factored_explore(_circle_x_interval(), FRONT, FactoredOptions(detect_samples=200),
                              log=lambda *_: None)
    audit = result["audit"]
    assert audit["factored"] and audit["frontier_complete"], audit
    assert audit["bodies"]["labels"] == ["factor1", "factor0", "mixed"] or \
        audit["bodies"]["labels"] == ["factor0", "factor1", "mixed"]
    ranks = sorted(leaf["rank"] for leaf in audit["leaves"])
    assert ranks == [1, 1]
    assert audit["composition"]["refined_residual_max"] < 1e-7
    assert audit["composition"]["other_order_max"] > 1e-3
    assert audit["sample_count"] < audit["product_count"]


def test_flat_split_from_bodies_without_boundary():
    """A partial front too small to see the interval's ends: flat, boundary
    ambiguous, so the split comes from the bodies' still directions."""
    result = factored_explore(_circle_x_interval(), AdvancingFrontOptions(step=.2, max_seconds=600),
                              FactoredOptions(detect_samples=30), log=lambda *_: None)
    audit = result["audit"]
    assert audit["split"]["split"]["kind"] == "flat_bodies", audit
    assert audit["factored"] and audit["frontier_complete"], audit
    assert sorted(audit["bodies"]["labels"]) == ["factor0", "factor1", "mixed"]


def test_flat_body_split_still_rejects_a_near_product():
    audit = factored_explore(_circle_x_interval(epsilon=.2), AdvancingFrontOptions(step=.2, max_seconds=600),
                             FactoredOptions(detect_samples=30), log=lambda *_: None)["audit"]
    assert not audit["factored"] and not audit["frontier_complete"], audit


def test_leaf_front_stays_on_its_leaf():
    """Every sample of the circle leaf keeps the interval-only body at its seed pose."""
    result = factored_explore(_circle_x_interval(), FRONT, FactoredOptions(detect_samples=200, certify=False),
                              log=lambda *_: None)
    labels = result["audit"]["bodies"]["labels"]
    for leaf in result["leaves"]:
        q = leaf["q"].reshape(len(leaf["q"]), 3, 12)
        other = [i for i, l in enumerate(labels) if l.startswith("factor")
                 and np.ptp(q[:, i], axis=0).max() < 1e-6]
        assert other, "a leaf moved every single-factor body"


def test_three_dimensional_product_splits_two_plus_one():
    result = factored_explore(_torus_x_interval(), AdvancingFrontOptions(step=.6, max_seconds=600),
                              FactoredOptions(detect_samples=300), log=lambda *_: None)
    audit = result["audit"]
    assert audit["factored"] and audit["frontier_complete"], audit
    assert sorted(leaf["rank"] for leaf in audit["leaves"]) == [1, 2]


def test_near_product_is_not_factored():
    result = factored_explore(_circle_x_interval(epsilon=.2), FRONT, FactoredOptions(detect_samples=200),
                              log=lambda *_: None)
    audit = result["audit"]
    assert not audit["factored"] and not audit["frontier_complete"], audit
    assert "reason" in audit


def test_leaf_system_refuses_foreign_sessions():
    system = _circle_x_interval()
    seed = system.seed()
    leaf = LeafSystem(system, seed, np.eye(36)[:, :2], np.eye(36)[:, :1], np.arange(12, 24),
                      tol=1e-8, max_newton=5, probe_h=.01)
    with pytest.raises(ValueError):
        leaf.restore(PhysicalObservation(np.ones(36)))


# ── kinetic metric ───────────────────────────────────────────────────────────
def _moments(points, masses):
    """Mbar = sum m [p;1][p;1]^T of point masses."""
    h = np.column_stack((np.asarray(points, float), np.ones(len(points))))
    return np.einsum("k,ki,kj->ij", np.asarray(masses, float), h, h)


_RING = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]   # isotropic, centred


def test_kinetic_metric_is_kinetic_energy():
    from atlas_rom.sample.factored import KineticMetric

    rng = np.random.default_rng(1)
    points, masses = rng.standard_normal((5, 3)), rng.uniform(.5, 2., 5)
    metric = KineticMetric(np.stack([_moments(points, masses)] * 2))
    dq = rng.standard_normal(24)
    assert np.allclose(metric.unweigh(metric.weigh(dq)), dq)
    dA = dq[:12].reshape(3, 4)
    energy = .5 * sum(m * np.sum((dA @ np.append(p, 1.)) ** 2) for p, m in zip(points, masses))
    assert np.isclose(.5 * np.sum(metric.weigh(dq)[:12] ** 2), energy)


def test_axisymmetric_masses_keep_the_product():
    moments = np.stack([_moments(_RING, np.ones(6))] * 3)
    audit = factored_explore(_circle_x_interval(), FRONT, FactoredOptions(detect_samples=200, body_moments=moments),
                             log=lambda *_: None)["audit"]
    assert audit["metric"] == "kinetic"
    assert audit["factored"] and audit["frontier_complete"], audit


def test_light_rotations_keep_the_product():
    """A ring of radius .1: rotations weigh ~100x less than translations, so the
    pose-spaced front is strongly anisotropic in kinetic coordinates.  Still a
    product, and still found as one -- given a partial front that reaches the
    boundary faces (at 200 samples it reaches 8 nodes and the planarity fit is
    under-resolved)."""
    ring = [tuple(.1 * c for c in p) for p in _RING]
    moments = np.stack([_moments(ring, np.ones(6))] * 3)
    audit = factored_explore(_circle_x_interval(), AdvancingFrontOptions(step=.2, max_seconds=300),
                             FactoredOptions(detect_samples=800, body_moments=moments),
                             log=lambda *_: None)["audit"]
    assert audit["split"]["parallel"]["count"] == 2
    assert audit["factored"] and audit["frontier_complete"], audit


def test_identity_moments_reproduce_the_pose_metric():
    pose = factored_explore(_circle_x_interval(), FRONT, FactoredOptions(detect_samples=200),
                            log=lambda *_: None)
    kinetic = factored_explore(_circle_x_interval(), FRONT,
                               FactoredOptions(detect_samples=200, body_moments=np.stack([np.eye(4)] * 3)),
                               log=lambda *_: None)
    assert np.allclose(pose["audit"]["split"]["parallel"]["mu"], kinetic["audit"]["split"]["parallel"]["mu"])
    for a, b in zip(pose["leaves"], kinetic["leaves"]):
        assert np.array_equal(a["q"], b["q"])


def test_off_axis_mass_on_the_mixed_body_breaks_the_product():
    """Same motions, a pose-metric product; a lump off the spin axis of the
    body that both spins and steers couples the factors' kinetic energy."""
    lump = _moments(_RING + [(0, .8, 0)], [1.] * 6 + [3.])
    moments = np.stack([_moments(_RING, np.ones(6))] * 2 + [lump])
    audit = factored_explore(_circle_x_interval(), FRONT, FactoredOptions(detect_samples=200, body_moments=moments),
                             log=lambda *_: None)["audit"]
    assert not audit["factored"] and not audit["frontier_complete"], audit


# ── a real mechanism: one steered, driven wheel through the ideal oracle ─────
def _steered_wheel(cardan):
    import dataclasses
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate import ideal_oracle
    from atlas_rom.simulate.oracle_base import LegacyPhysicalOracleAdapter
    from atlas_rom.spec.case_spec import JointDef

    spec = load_spec("steered_wheel")
    if cardan:   # evaluator-only control: the same wheel through a Hooke joint
        mech = spec.mechanism
        joints = [JointDef(j.name, "universal", j.body_a, j.body_b, pivot=j.pivot,
                           axis=(0., 1., 0.), axis2=(0., 0., 1.))
                  if j.kind == "constant_velocity" else j for j in mech.joints]
        spec = dataclasses.replace(spec, mechanism=dataclasses.replace(mech, joints=joints))
    return LegacyPhysicalOracleAdapter(ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None))


WHEEL_FRONT = AdvancingFrontOptions(step=.4, max_seconds=600)


def test_steered_wheel_is_sampled_by_factors():
    result = factored_explore(_steered_wheel(False), WHEEL_FRONT, FactoredOptions(detect_samples=150),
                              log=lambda *_: None)
    audit = result["audit"]
    assert audit["factored"] and audit["frontier_complete"], audit
    # A complete spin leaf must actually go round: evaluator readout of the
    # shaft's spin about x over the spin leaf's samples.
    spin = next(leaf for leaf in result["leaves"] if leaf["q"].shape[0] > 10)["q"].reshape(-1, 3, 3, 4)
    angles = np.sort(np.arctan2(spin[:, 0, 2, 1], spin[:, 0, 1, 1]))
    assert np.max(np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))) < 0.5
    assert audit["bodies"]["labels"] == ["factor0", "factor1", "mixed"]      # shaft, knuckle, wheel
    assert audit["composition"]["refined_residual_max"] < 1e-6
    assert audit["sample_count"] < audit["product_count"]


def test_steered_wheel_is_a_product_in_the_kinetic_metric():
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate.ideal_sim import body_second_moments

    system = _steered_wheel(False)
    moments = body_second_moments(load_spec("steered_wheel"), system.oracle.order)
    audit = factored_explore(system, WHEEL_FRONT, FactoredOptions(detect_samples=150, body_moments=moments),
                             log=lambda *_: None)["audit"]
    assert audit["metric"] == "kinetic"
    assert audit["factored"] and audit["frontier_complete"], audit
    assert audit["bodies"]["labels"] == ["factor0", "factor1", "mixed"]


def test_cardan_wheel_is_not_factored():
    audit = factored_explore(_steered_wheel(True), WHEEL_FRONT, FactoredOptions(detect_samples=150),
                             log=lambda *_: None)["audit"]
    assert not audit["factored"] and not audit["frontier_complete"], audit


def test_full_front_closes_against_joint_stops():
    """Nodes next to the kingpin stops measure their frame one-sidedly and
    record the stop, instead of becoming uncertain and leaving the front open."""
    from atlas_rom.sample.physical_advancing_front import explore

    q, _, frames, audit = explore(_steered_wheel(False), options=WHEEL_FRONT)
    summary = audit[-1]
    assert summary["frontier_complete"], summary
    assert summary["frontier_state_counts"]["uncertain_poses"] == 0
    records = audit[:len(q)]
    assert any(r["diagnostics"].get("boundary_probes") for r in records)
    assert any(r["diagnostics"].get("one_sided_pairs") for r in records)
    assert frames.shape[2] == 2


# ── decoupled subsystems: two independent slider-cranks ──────────────────────
def test_decoupled_pistons_split_by_body_groups():
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate import ideal_oracle
    from atlas_rom.simulate.oracle_base import LegacyPhysicalOracleAdapter

    spec = load_spec("twin_independent_pistons")
    system = LegacyPhysicalOracleAdapter(ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None))
    step = spec.build.sampling.step
    audit = factored_explore(system, AdvancingFrontOptions(step=step, max_seconds=600),
                             FactoredOptions(detect_samples=150), log=lambda *_: None)["audit"]
    assert audit["split"]["split"]["kind"] == "body_groups", audit
    assert sorted(len(g) for g in audit["split"]["split"]["groups"]) == [3, 3]   # crank, rod, piston each
    assert audit["factored"] and audit["frontier_complete"], audit
    assert [leaf["rank"] for leaf in audit["leaves"]] == [1, 1]
    assert "mixed" not in audit["bodies"]["labels"]
    assert audit["composition"]["refined_residual_max"] < 1e-6
    assert audit["sample_count"] < audit["product_count"]


def test_pooled_system_gives_the_same_samples():
    """Batches answered in worker processes are the serial responses."""
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate.pooled_system import PooledIdealSystem

    serial = factored_explore(_steered_wheel(False), WHEEL_FRONT, FactoredOptions(detect_samples=150),
                              log=lambda *_: None)
    with PooledIdealSystem(load_spec("steered_wheel"), workers=2) as system:
        pooled = factored_explore(system, WHEEL_FRONT, FactoredOptions(detect_samples=150),
                                  log=lambda *_: None)
        assert system.batched_commands > 0
    assert pooled["audit"]["factored"] and pooled["audit"]["frontier_complete"]
    for a, b in zip(serial["leaves"], pooled["leaves"]):
        assert a["q"].shape == b["q"].shape
        assert np.abs(a["q"] - b["q"]).max() < 1e-12
