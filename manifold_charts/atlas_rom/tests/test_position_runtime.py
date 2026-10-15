import os

import numpy as np
import pytest

ROM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "work", "polished_pipeline_20261007", "roms", "five_bar_tanband")


@pytest.fixture(scope="module")
def rom():
    if not os.path.isdir(ROM_DIR):
        pytest.skip(f"staged ROM not present: {ROM_DIR}")
    from atlas_rom.rom.loader import PatchAtlasROM
    return PatchAtlasROM.load(ROM_DIR)


def _rt(rom, **kw):
    from atlas_rom.rom.position_runtime import PositionAtlasRuntime
    rt = PositionAtlasRuntime(rom, dt=1e-3, gravity=0.0, velocity_damping=0.0, **kw)
    rt.set_sdf_enabled(False)
    return rt


def test_set_velocity_round_trips_through_the_position_pair(rom):
    rt = _rt(rom)
    s = rt.initial_state(0)
    rt.set_velocity(s, [0.3, -0.2])
    assert "p" not in s
    assert np.allclose(rt.velocity(s), [0.3, -0.2])


def test_rest_without_forces_is_a_fixed_point(rom):
    rt = _rt(rom)
    s = rt.initial_state(0)
    u0 = s["u"].copy()
    for _ in range(5):
        rt._step_single_chart(s, None)
    assert s["stalls"] == 0
    assert np.allclose(s["u"], u0, atol=1e-12)


def test_free_motion_conserves_kinetic_energy(rom):
    rt = _rt(rom)
    s = rt.initial_state(0)
    rt.set_velocity(s, [0.2, 0.1])
    e0 = rt.energy(s)["kinetic"]
    ke = []
    for _ in range(100):
        rt._step_single_chart(s, None)
        ke.append(rt.energy(s)["kinetic"])
    assert s["stalls"] == 0
    assert rt.last_newton["rel_residual"] < 1e-8
    assert max(abs(k - e0) for k in ke) / e0 < 1e-2


@pytest.mark.parametrize("transport", ["none", "projected"])
def test_self_handoff_reproduces_the_pair(rom, transport):
    rt = _rt(rom, transport=transport)
    u = np.array([0.05, -0.03]); u_prev = u - 1e-3 * np.array([0.2, 0.1])
    up2, u2, res = rt._handoff_pair(0, 0, u_prev, u)
    assert res < 1e-10
    assert np.allclose(u2, u, atol=1e-4) and np.allclose(up2, u_prev, atol=1e-4)
    # the projector's stopping error is common to both positions, so the implied velocity survives
    assert np.allclose((u2 - up2) / 1e-3, [0.2, 0.1], rtol=1e-3)
