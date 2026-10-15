import numpy as np
import torch

from atlas_rom.rom import adv_labels as L
from atlas_rom.rom.adv_router import Net, featurize


def test_featurize_pads_latent_and_onehots_source_and_receiver():
    sc = [np.array([2.0, 4.0]), np.array([1.0])]
    x = featurize(0, [2.0, 4.0], 1, sc, dmax=2, K=2)
    assert x.shape == (2 + 2 * 2,)
    assert np.allclose(x[:2], [1.0, 1.0])                 # u / anchor std
    assert np.allclose(x[2:4], [1, 0]) and np.allclose(x[4:], [0, 1])
    y = featurize(1, [3.0], 0, sc, dmax=2, K=2)
    assert np.allclose(y[:2], [3.0, 0.0])                 # shorter latent zero-padded
    assert Net(2, 2).double()(torch.as_tensor(x).unsqueeze(0)).shape == (1,)


class _Linear(torch.nn.Module):
    def __init__(self, W, b):
        super().__init__(); self.W, self.b = W, b

    def decode(self, U):
        return U @ self.W.T + self.b


def test_batched_lm_projection_recovers_an_exactly_representable_pose():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(6, 2, generator=g, dtype=torch.float64); b = torch.randn(6, generator=g, dtype=torch.float64)
    m = _Linear(W, b)
    U_true = torch.randn(5, 2, generator=g, dtype=torch.float64)
    Q = m.decode(U_true)
    U, c = L.proj_source(m, Q, torch.zeros_like(U_true))
    assert float(c.max()) < 1e-18
    assert torch.allclose(U, U_true, atol=1e-8)


def test_territory_states_cover_every_chart_with_anchor_jitter_and_interpolation():
    rng = np.random.default_rng(0)
    anchors = [rng.standard_normal((10, 2)), rng.standard_normal((8, 3))]
    sc = [np.maximum(a.std(0), 1e-4) for a in anchors]
    states = L.territory_states(anchors, sc, 9, rng)
    assert len(states) == 18
    assert [i for i, _ in states] == [0] * 9 + [1] * 9
    assert all(len(u) == anchors[i].shape[1] for i, u in states)


# ---------------------------------------------------------------- toy atlas: train_router end to end
class _ToyAE(torch.nn.Module):
    """Linear chart autoencoder on R^6 with a 2-d latent (encode/decode are exact pseudo-inverses)."""

    def __init__(self, basis):
        super().__init__()
        self.W = basis                                  # (6, 2) orthonormal columns

    def decode(self, U):
        return U @ self.W.T

    def encode(self, Q):
        return Q @ self.W


class _PlaneOracle:
    """The whole space is on-manifold: projection is the identity, residual zero."""

    def project(self, q):
        return np.asarray(q, float)

    def residual(self, q):
        return np.zeros(3)


def test_train_router_returns_a_loadable_payload_for_a_toy_atlas():
    from atlas_rom.rom.adv_router import AdvRouter
    g = torch.Generator().manual_seed(0)
    models = []
    for _ in range(3):
        W, _ = torch.linalg.qr(torch.randn(6, 2, generator=g, dtype=torch.float64))
        models.append(_ToyAE(W))
    rng = np.random.default_rng(0)
    anchors = [rng.standard_normal((12, 2)) for _ in models]
    q_all = rng.standard_normal((60, 6))
    payload, report = L.train_router(models, anchors, q_all, _PlaneOracle(), n_per=6, epochs=30, device="cpu",
                                     log=lambda *_: None)
    assert payload["K"] == 3 and payload["dmax"] == 2
    assert report["n_states"]["labelled"] > 0 and "wall_seconds" in report["timing"]
    router = AdvRouter.from_payload(payload)
    j, a, scores = router.best(0, np.array([0.1, -0.2]))
    assert j in (1, 2) and set(scores) == {1, 2} and np.isfinite(a)


# ---------------------------------------------------------------- package / loader / runtime
import os
import shutil
import pytest

_ROMS = os.path.join(os.path.dirname(__file__), "..", "work", "polished_pipeline_20261007", "roms")
_PANTO = os.path.join(_ROMS, "pantograph_2d_polished")
_needs_panto = pytest.mark.skipif(not os.path.isdir(_PANTO), reason="polished pantograph_2d ROM not present")


@_needs_panto
def test_multichart_rom_carries_a_router_and_runtime_hands_off_with_it():
    from atlas_rom.rom.loader import PatchAtlasROM
    from atlas_rom.rom.runtime import PatchAtlasRuntime
    rom = PatchAtlasROM.load(_PANTO)
    assert rom.router is not None and rom.router.K == rom.n_patches
    rt = PatchAtlasRuntime(rom, dt=0.01)
    state = rt.initial_state(0)
    rng = np.random.default_rng(1)
    ctl = np.zeros(len(rom.case.actuators))
    for k in range(900):
        if k % 60 == 0:
            ctl = rng.standard_normal(len(ctl)) * 15.0
        rt.step(state, ctl)
    assert state["handoffs"] > 0 and state["stalls"] == 0
    assert np.all(np.isfinite(state["u"])) and rt.router_info["mode"] == "adv"


@_needs_panto
def test_loader_refuses_a_multichart_package_without_a_router(tmp_path):
    from atlas_rom.rom.loader import PatchAtlasROM
    dst = tmp_path / "rom"
    shutil.copytree(_PANTO, dst, ignore=shutil.ignore_patterns("*.pre_router"))
    state = torch.load(dst / "patch_atlas.pt", map_location="cpu", weights_only=False)
    del state["physics_fields"]["router"]
    torch.save(state, dst / "patch_atlas.pt")
    with pytest.raises(ValueError, match="router"):
        PatchAtlasROM.load(str(dst))


@_needs_panto
def test_router_only_package_loads_without_support_fields(tmp_path):
    from atlas_rom.rom.loader import PatchAtlasROM
    dst = tmp_path / "rom"
    shutil.copytree(_PANTO, dst, ignore=shutil.ignore_patterns("*.pre_router"))
    state = torch.load(dst / "patch_atlas.pt", map_location="cpu", weights_only=False)
    del state["physics_fields"]["support_fields"]
    torch.save(state, dst / "patch_atlas.pt")
    rom = PatchAtlasROM.load(str(dst))
    assert rom.support_fields is None and rom.router is not None
    with pytest.raises(RuntimeError, match="no support fields"):
        rom.support_margin(0, np.zeros(rom.latent_dim))


@_needs_panto
def test_embed_rejects_a_router_for_the_wrong_chart_count(tmp_path):
    from atlas_rom.rom import router_build
    dst = tmp_path / "rom"
    shutil.copytree(_PANTO, dst, ignore=shutil.ignore_patterns("*.pre_router"))
    bad = {"state": {}, "dmax": 2, "K": 3, "sc": [], "eps": 0.1}
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="K=3"):
        router_build.embed(str(dst), str(tmp_path / "bad.pt"))


def test_single_chart_package_needs_no_router():
    roms = os.path.join(os.path.dirname(__file__), "..", "roms", "scissor_lift")
    if not os.path.isdir(roms):
        pytest.skip("roms/scissor_lift not present")
    from atlas_rom.rom.loader import PatchAtlasROM
    from atlas_rom.rom.runtime import PatchAtlasRuntime
    rom = PatchAtlasROM.load(roms)
    assert rom.n_patches == 1 and rom.router is None
    rt = PatchAtlasRuntime(rom, dt=0.01)
    state = rt.initial_state(0)
    for _ in range(50):
        rt.step(state, None)
    assert state["handoffs"] == 0 and np.all(np.isfinite(state["u"]))


def test_router_build_model_dims_work_for_shallow_and_deep_encoders():
    from atlas_rom.rom.router_build import _model_dims
    shallow = {"encoder.0.weight": torch.zeros(96, 90), "encoder.2.weight": torch.zeros(2, 96)}
    deep = {"encoder.0.weight": torch.zeros(64, 90), "encoder.2.weight": torch.zeros(64, 64),
            "encoder.4.weight": torch.zeros(3, 64)}
    assert _model_dims(shallow) == (2, 96)
    assert _model_dims(deep) == (3, 64)


@_needs_panto
def test_router_build_rebuilds_the_staged_models_from_the_raw_artifact():
    """The models rebuilt from patch_atlas.pt must be the ones the loader builds (same decode)."""
    from atlas_rom.rom.loader import PatchAtlasROM
    from atlas_rom.rom.router_build import _models_from_artifact
    state = torch.load(os.path.join(_PANTO, "patch_atlas.pt"), map_location="cpu", weights_only=False)
    models, anchors = _models_from_artifact(state)
    rom = PatchAtlasROM.load(_PANTO)
    assert len(models) == rom.n_patches
    u = torch.as_tensor(np.asarray(rom.anchors[0][:3], float))
    assert torch.allclose(models[0].decode(u), rom.models[0].decode(u))
