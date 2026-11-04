"""SuperDex contact oracle: packing, budget refusal, projection and restore."""
from __future__ import annotations

import os

import numpy as np
import psutil
import pytest

os.environ.setdefault("SUPERDEX_PRECISION", "fp64")  # fixed at first import
pytest.importorskip("superdex.physics")

from atlas_rom.cases import load_spec
from atlas_rom.sample.sampling import _select_oracle
from atlas_rom.simulate.superdex_oracle import SuperDexContactOracle


@pytest.fixture(scope="module")
def oracle():
    return SuperDexContactOracle(load_spec("rotate_contact"), log=lambda *_: None)


def test_case_selects_superdex_backend():
    spec = load_spec("rotate_contact")
    assert spec.build.sampling.oracle_backend == "superdex"
    assert not any(j.body_a != "__ground" for j in spec.mechanism.joints)


def test_seed_is_identity_pose(oracle):
    blocks = oracle.seed().reshape(4, 3, 4)
    assert np.allclose(blocks[:, :, :3], np.eye(3))
    assert np.allclose(blocks[:, :, 3], 0.0)


def test_sdf_budget_refused_before_allocation():
    with pytest.raises(MemoryError):
        SuperDexContactOracle(load_spec("rotate_contact"), max_sdf_cells=10, log=lambda *_: None)


def test_restore_reproduces_a_command(oracle):
    settled, snapshot = oracle.baseline(oracle.seed())
    direction = np.zeros_like(settled)
    direction[3] = 1.0
    first = oracle.directional_response(snapshot, direction, 1e-3, reuse=False)
    second = oracle.directional_response(snapshot, direction, 1e-3, reuse=False)
    assert first.converged == second.converged
    assert np.array_equal(first.result, second.result)


def test_baseline_does_not_depend_on_where_the_scene_last_was(oracle):
    seed = oracle.seed()
    direct, _snapshot = oracle.baseline(seed)
    far = seed.copy()
    far[3] += 0.5                               # move body A far from the seed
    oracle.project(far)
    again, _snapshot = oracle.baseline(seed)
    assert np.array_equal(direct, again)


def test_worker_pool_answers_and_replaces_a_stuck_worker():
    oracle = SuperDexContactOracle(load_spec("rotate_contact"), n_workers=2, log=lambda *_: None)
    try:
        _settled, snapshot = oracle.baseline(oracle.seed())
        directions = np.eye(48)[[3, 15, 27, 39]]
        responses = oracle.directional_responses(snapshot, directions, 1e-3)
        assert all(r.converged and np.isfinite(r.result).all() for r in responses)
        reused = oracle.directional_responses(snapshot, directions, 1e-3)
        assert all(r.reused for r in reused)

        frozen = [psutil.Process(w["process"].pid) for w in oracle._pool._workers]
        for process in frozen:
            process.suspend()
        oracle._pool.command_timeout = 0.5
        stuck = oracle.directional_responses(snapshot, directions[:1], 2e-3)
        assert not stuck[0].converged and "TimeoutError" in stuck[0].diagnostics["error"]
        assert oracle._pool.restarts == 1
        for process in frozen:
            if process.is_running():
                process.resume()

        oracle._pool.command_timeout = 60.0
        again = oracle.directional_responses(snapshot, directions[:1], 3e-3)
        assert again[0].converged
    finally:
        oracle.close()


def test_case_oracle_options_reach_the_oracle():
    from dataclasses import replace
    spec = load_spec("rotate_contact")
    options = replace(spec.build.sampling, oracle_workers=1,
                      oracle_options={"sdf_voxel": 0.125, "max_step_translation": 0.02})
    oracle, _holds = _select_oracle(spec, options, None, lambda *_: None)
    try:
        assert oracle.sdf_voxel == 0.125 and oracle.max_step_translation == 0.02
        assert "sdf=0.125," in oracle.oracle_fingerprint()
    finally:
        oracle.close()
    with pytest.raises(TypeError):
        _select_oracle(spec, replace(options, oracle_options={"no_such_option": 1}), None,
                       lambda *_: None)


def test_point_step_caps_every_material_point():
    cap = 0.01
    oracle = SuperDexContactOracle(load_spec("rotate_contact"), max_point_step=cap, log=lambda *_: None)
    current = oracle._blocks(oracle.seed())
    target = current.copy()
    target[:, :, 3] += 5.0                                  # far away: the cap must bind
    oracle._command(current, target)
    for name in oracle.order:
        actor = oracle.actors[name]
        v = np.asarray(actor.get_linear_velocity().tolist(), float) * oracle.dt
        w = np.asarray(actor.get_angular_velocity().tolist(), float) * oracle.dt
        assert np.linalg.norm(v) + np.linalg.norm(w) * oracle._reach[name] <= cap * (1 + 1e-9)
    assert "point_step=0.01" in oracle.oracle_fingerprint()


def test_per_direction_scales_match_single_responses():
    spec = load_spec("rotate_contact")
    oracle = SuperDexContactOracle(spec, n_workers=2, log=lambda *_: None)
    serial = SuperDexContactOracle(spec, log=lambda *_: None)
    try:
        directions = list(np.eye(48)[[3, 15]])
        scales = [1e-3, 2e-3]
        _s, snapshot = oracle.baseline(oracle.seed())
        batched = oracle.directional_responses(snapshot, directions, None, scales=scales)
        _s, snap = serial.baseline(serial.seed())
        single = [serial.directional_response(snap, d, s, reuse=False) for d, s in zip(directions, scales)]
        assert [r.scale for r in batched] == scales
        for a, b in zip(batched, single):
            assert a.converged and b.converged
            assert np.allclose(a.result, b.result, atol=1e-9)
        with pytest.raises(ValueError, match="one scale per direction"):
            oracle.directional_responses(snapshot, directions, None, scales=scales[:1])
    finally:
        oracle.close()


def test_residual_is_zero_on_the_set_and_batches_on_workers():
    spec = load_spec("rotate_contact")
    serial = SuperDexContactOracle(spec, log=lambda *_: None)
    pooled = SuperDexContactOracle(spec, n_workers=2, log=lambda *_: None)
    try:
        seed = serial.seed()
        assert np.abs(serial.residual(seed)).max() < 1e-2
        off = seed.copy()
        off[15] += 1.0                          # body B pulled off its pin
        single = [serial.residual(q) for q in (seed, off)]
        batched = pooled.residuals([seed, off])
        assert np.abs(single[1]).max() > 0.1
        for a, b in zip(single, batched):
            assert np.allclose(a, b, atol=1e-9)
    finally:
        pooled.close()


def test_project_from_a_feasible_start_and_its_step_budget():
    from atlas_rom.rom.adv_labels import _project_rows
    spec = load_spec("rotate_contact")
    oracle = SuperDexContactOracle(spec, log=lambda *_: None)
    seed = oracle.seed()
    target = seed.copy()
    target[15] += 0.05                                     # a short pull on body B
    oracle._write_pose(seed)
    plain = oracle.project(target)
    warm = oracle.project_from(seed, target)
    assert np.allclose(plain, warm, atol=1e-9)
    assert oracle.max_steps == 2000
    with pytest.raises(RuntimeError):
        oracle.project_from(seed, seed + 2.0, max_steps=3)  # cannot settle in three steps
    assert oracle.max_steps == 2000                         # the budget is per call
    rows = np.stack([target, seed + 2.0])
    out = _project_rows(rows, oracle, starts=np.stack([seed, seed]), max_steps=3)
    assert np.isnan(out[1]).all()
    cold = _project_rows(rows[:1], oracle)                  # no starts: the original path
    assert np.isfinite(cold).all()


def test_body_to_body_joint_is_rejected():
    spec = load_spec("rotate")
    with pytest.raises(ValueError, match="located by contact"):
        SuperDexContactOracle(spec, log=lambda *_: None)
