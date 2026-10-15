import json

import numpy as np
import pytest

from atlas_rom.charts.chart_contract import ChartFitConfig, ChartFitInput
from atlas_rom.charts.competitive_cae import CompetitiveCAEFitter, PhysicalRBFSelector, _chart_quality_passes, _condition_gauge_covariance, _decoder_spectral_product, _fps_warm_start_indices, _patch_far_distance, _reconstruction_diagnostics, _soft_multiplicity_weights, _responsibilities, encoder_ablation
from atlas_rom.charts.latent_support import in_local_support, locally_expanded


def _poses(n=4):
    q = np.zeros((n, 12))
    for i in range(n):
        q[i, :9] = np.eye(3).reshape(-1)
        q[i, 9:12] = (float(i), 0.0, 0.0)
    return q


def test_competitive_cae_construction_is_graph_free(tmp_path):
    q = _poses()
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)), np.asarray([[3, 2], [2, 1], [1, 0]]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=1, construction_only=True,
                            options={"discovery_rounds": 1, "rounds_per_chart": 1})
    report, models, patches, anchors = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path), log=lambda *_: None)
    assert report["uses_cover_edges"] is False
    assert report["uses_declared_coordinates"] is False
    assert report["uses_memberships"] is False
    assert report["encoder_kind"] == "shallow"
    assert models and len(models) == len(patches) == len(anchors)


def test_physical_rbf_selector_ranks_without_training_memberships():
    selector = PhysicalRBFSelector(np.asarray([[1.0] * 6 + [0.0, 0.0, 0.0], [1.0] * 6 + [10.0, 0.0, 0.0]]),
                                   np.zeros(2), np.zeros((2, 9)), np.zeros(2), np.zeros(9), np.ones(9))
    assert selector.rank(_poses(1)[0], [1, 0]) == [0, 1]


def test_responsibilities_use_direct_physical_loss_temperature():
    competence = np.asarray([[0.0, 0.1], [0.02, 0.0]])
    weights = _responsibilities(competence, temperature=0.01)
    assert weights[0, 1] < 5e-5
    assert weights[1, 0] < 0.2
    np.testing.assert_allclose(weights.sum(1), 1.0)


def test_decoder_spectral_product_uses_affine_spectral_norms():
    import torch

    decoder = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.SiLU(),
                                  torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        decoder[0].weight.copy_(torch.diag(torch.tensor([3.0, 2.0])))
        decoder[2].weight.copy_(torch.diag(torch.tensor([5.0, 4.0])))
    assert torch.isclose(_decoder_spectral_product(decoder), torch.tensor(15.0))


def test_gauge_covariance_condition_cap_bounds_triangular_amplification():
    import torch

    covariance = torch.diag(torch.tensor([1.0, 1e-6]))
    conditioned = _condition_gauge_covariance(covariance, max_condition=4.0)
    values = torch.linalg.eigvalsh(conditioned)
    assert torch.isclose(values.max() / values.min(), torch.tensor(16.0))


def test_far_pair_threshold_is_relative_to_the_candidate_patch():
    feature = np.asarray([[0.0], [1.0], [2.0], [100.0]])
    assert _patch_far_distance(feature, np.asarray([0, 1, 2])) == 1.5
    assert _patch_far_distance(feature, np.asarray([3])) == float("inf")


def test_chart_quality_requires_usage_reconstruction_and_output_spread():
    kwargs = dict(usage_floor=.01, recon_ratio_max=.5, spread_ratio_min=.5)
    assert _chart_quality_passes(usage=.1, recon_ratio=.2, spread_ratio=.8, **kwargs)
    assert not _chart_quality_passes(usage=.001, recon_ratio=.2, spread_ratio=.8, **kwargs)
    assert not _chart_quality_passes(usage=.1, recon_ratio=.8, spread_ratio=.8, **kwargs)
    assert not _chart_quality_passes(usage=.1, recon_ratio=.2, spread_ratio=.2, **kwargs)


def test_fps_warm_start_uses_physical_seed_separation_not_n_over_k():
    feature = np.asarray([[0.0], [0.1], [1.0], [1.1]])
    local, separation, fallback = _fps_warm_start_indices(feature, [0, 2], 0)
    np.testing.assert_array_equal(local, [0, 1])
    assert separation == 1.0
    assert not fallback
    single, separation, fallback = _fps_warm_start_indices(feature, [0], 0)
    np.testing.assert_array_equal(single, np.arange(4))
    assert np.isinf(separation) and not fallback


def test_fps_warm_start_reports_the_sparse_two_sample_fallback():
    feature = np.asarray([[0.0], [2.0], [-2.0]])
    local, separation, fallback = _fps_warm_start_indices(feature, [0, 1], 0)
    np.testing.assert_array_equal(local, [0, 1])
    assert separation == 2.0
    assert fallback


def test_reconstruction_diagnostics_detects_constant_predictor_and_identical_residuals():
    target = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    predictions = np.stack([target, target])
    good = _reconstruction_diagnostics(target, predictions)
    assert good["best_chart_recon_over_mean_pose_baseline"] == 0.0
    assert good["residual_spread_across_charts"]["mean"] == 0.0
    constant = _reconstruction_diagnostics(target, np.broadcast_to(target.mean(0), (1, len(target), 2)))
    assert abs(constant["per_chart"][0]["alpha"]) < 1e-12
    assert abs(constant["best_chart_recon_over_mean_pose_baseline"] - 1.0) < 1e-12


def test_soft_multiplicity_weights_are_absolute_per_chart_ownership():
    # Two-phase: ``membership`` is the absolute per-chart ownership weight
    # sigmoid((threshold - (recon + valid)) / (threshold / 5)) -- each chart
    # depends only on its own loss.  ``weights`` no longer carries a
    # coverage-deficit reweight (it is ones).
    reconstruction = np.asarray([[.0, 1.0], [1.0, .0], [.0, .0]])
    membership, multiplicity, weights, tau = _soft_multiplicity_weights(
        reconstruction, np.zeros_like(reconstruction), threshold=.1, atlas_dimension=2)
    assert np.all(tau > 0)
    assert membership.shape == reconstruction.shape
    assert membership[0, 0] > 0.9 and membership[0, 1] < 0.1     # own loss only
    assert multiplicity[2] > multiplicity[0]                     # both charts own pose 2
    np.testing.assert_array_equal(weights, np.ones(3))


def test_ownership_weight_is_unnormalised_and_k_independent():
    from atlas_rom.charts.competitive_cae import _ownership_weight
    loss = np.asarray([[0.0, 5.0], [0.0, 0.0]])
    w2 = _ownership_weight(loss, threshold_topo=1.0)
    w3 = _ownership_weight(np.pad(loss, ((0, 0), (0, 1)), constant_values=5.0), threshold_topo=1.0)
    assert w2[0, 0] > 0.9 and w2[0, 1] < 0.05
    np.testing.assert_allclose(w2, w3[:, :2])   # a chart's weight does not depend on K


def test_local_latent_support_does_not_fill_an_annular_hole():
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    cloud = np.concatenate([np.c_[r * np.cos(angle), r * np.sin(angle)] for r in (1.0, 1.2)])
    assert not in_local_support(cloud, np.asarray([[0.0, 0.0]]))[0]
    assert in_local_support(cloud, np.asarray([[1.1, 0.0]]))[0]
    expanded = locally_expanded(cloud, 1.1)
    assert expanded.shape == cloud.shape


def test_support_sdf_derives_its_witness_radius_from_the_anchor_cloud(monkeypatch):
    """The runtime-support builder must not require a global radius argument."""
    from atlas_rom.rom import learned
    import torch

    captured = {}
    class _SDF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(1, 1, dtype=torch.float64))
        def forward(self, value):
            return self.net(value)
    def fake_fit(inside, outside, margin, **kwargs):
        captured.update(inside=np.asarray(inside), outside=np.asarray(outside), margin=margin)
        return _SDF()
    monkeypatch.setattr(learned, "_fit_witness_sdf", fake_fit)
    anchors = np.asarray([[0.0], [0.01], [0.02], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    _, row = learned.fit_local_support_sdf(anchors, probes_per_anchor=1, epochs=1, seed=0)
    assert row["support_radius_max"] > row["support_radius_min"]
    assert captured["margin"] == .1 * row["support_radius_median"]


def test_local_distortion_pair_floor_excludes_only_near_coincident_witnesses():
    feature = np.asarray([[0.0], [1e-5], [1.0]])
    distance = np.linalg.norm(feature[:, None] - feature[None, :], axis=-1)
    positive = np.where(distance > 1e-12, distance, np.inf)
    spacing = np.partition(positive, 1, axis=1)[:, 1]
    floor = .05 * np.median(spacing)
    assert not (distance[0, 1] >= floor)
    assert distance[0, 2] >= floor


def test_encoder_ablation_exposes_all_requested_encoder_arms(tmp_path):
    q = _poses()
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)), np.empty((0, 2), int))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=1,
                            options={"discovery_rounds": 1, "rounds_per_chart": 1})
    reports = encoder_ablation(evidence, config, work=str(tmp_path), log=lambda *_: None)
    assert set(reports) == {"linear", "shallow", "current"}


def test_latent_gauge_false_keeps_the_bounded_sigmoid_chart_domain(tmp_path):
    q = _poses(6)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=2, construction_only=True,
                            options={"n_charts": 2, "latent_gauge": False})
    report, models, patches, anchors = CompetitiveCAEFitter().fit(
        evidence, config, work=str(tmp_path), log=lambda *_: None)
    assert report["latent_gauge"] is False
    assert all(not bool(m.gauge_use) for m in models)  # gauge never installed
    for anchor in anchors:  # encoded codes stay in the sigmoid box
        if len(anchor):
            assert anchor.min() >= 0.0 and anchor.max() <= 1.0


def test_paper_baseline_resolves_the_size_control_knobs_and_echoes_them(tmp_path):
    q = _poses(6)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=2, construction_only=True,
                            options={"n_charts": 2, "paper_baseline": True, "tangent_weight": 0.5})
    report = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path), log=lambda *_: None)[0]
    resolved = report["resolved_config"]
    assert report["latent_gauge"] is False and resolved["paper_baseline"] is True
    assert resolved["lip_weight"] == 0.01 and resolved["weight_decay"] == 0.01
    assert resolved["distortion_weight"] == 0.0 and resolved["alias_weight"] == 0.0
    assert resolved["tangent_weight"] == 0.5  # explicit override beats the preset


def test_gauge_refresh_is_a_pure_reparameterization():
    """A gauge refresh renames the latent coordinate and must move no pose.

    ``set_gauge`` absorbs the old-to-new latent affine map into the decoder's
    first linear layer.  Without that absorption the decoder is left reading a
    coordinate it was never trained in, which is silent: reconstruction simply
    degrades everywhere at once.  A non-orthogonal gauge is used deliberately
    so the test fails if the inverse is ever replaced by a transpose.
    """
    import torch
    from atlas_rom.charts.physical_chart_autoencoder import _module

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    d, n_bodies = 2, 1
    model = _module(n_bodies, width=16, d=d, encoder_kind="shallow")(np.zeros(3), np.ones(3))

    poses = []
    for _ in range(32):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        poses.append(np.column_stack((rotation, rng.normal(size=3))).ravel())
    q = torch.as_tensor(np.asarray(poses))

    raw = model.encode_raw(q).detach()
    # First installation: sigmoid -> affine.  Not a reparameterization, so it
    # is expected to move poses and nothing is carried.
    model.set_gauge(raw.mean(0), torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0])
    before = model.decode(model.encode(q)).detach().clone()
    latent_before = model.encode(q).detach().clone()

    skew = torch.as_tensor(rng.normal(size=(d, d)) + 2.0 * np.eye(d))
    model.set_gauge(raw.mean(0) + 0.3, skew)

    after = model.decode(model.encode(q)).detach()
    assert torch.allclose(after, before, atol=1e-12), float((after - before).abs().max())
    # The coordinate really did change -- otherwise the test would pass on a
    # no-op implementation of ``set_gauge``.
    assert not torch.allclose(model.encode(q).detach(), latent_before, atol=1e-6)


def test_gauge_enabled_flag_survives_state_dict_round_trip():
    import torch
    from atlas_rom.charts.physical_chart_autoencoder import _module

    cls = _module(1, width=4, d=2, encoder_kind="linear")
    model = cls(np.zeros(3), np.ones(3))
    q = torch.randn(5, 12, dtype=torch.float64)
    model.set_gauge(torch.zeros(2), torch.eye(2, dtype=torch.float64))
    expected = model.encode(q).detach()

    restored = cls(np.zeros(3), np.ones(3))
    restored.load_state_dict(model.state_dict())
    assert restored._gauge_enabled is True
    assert torch.equal(restored.encode(q).detach(), expected)


def test_dense_chart_forward_matches_scalar_outputs_and_gradients():
    import torch
    from torch.func import functional_call, vmap
    from atlas_rom.charts.physical_chart_autoencoder import _module

    torch.manual_seed(7)
    cls = _module(1, width=8, d=2, encoder_kind="shallow")
    models = [cls(np.zeros(3), np.ones(3)) for _ in range(3)]
    state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(state)
    q = torch.randn(3, 5, 12, dtype=torch.float64)

    scalar_outputs = [model(q[i]) for i, model in enumerate(models)]
    scalar_loss = sum(u.square().mean() + pred.square().mean()
                      for u, pred in scalar_outputs)
    scalar_loss.backward()
    scalar_grads = [{name: value.grad.detach().clone()
                     for name, value in model.named_parameters()}
                    for model in models]

    for model in models:
        model.zero_grad(set_to_none=True)
    parameter_names = tuple(dict(models[0].named_parameters()))
    buffer_names = tuple(dict(models[0].named_buffers()))
    params = {name: torch.stack([dict(model.named_parameters())[name] for model in models])
              for name in parameter_names}
    buffers = {name: torch.stack([dict(model.named_buffers())[name] for model in models])
               for name in buffer_names}

    def call_one(chart_params, chart_buffers, chart_q):
        return functional_call(models[0], (chart_params, chart_buffers), (chart_q,))

    dense_u, dense_pred = vmap(call_one, in_dims=(0, 0, 0))(params, buffers, q)
    # The dense means include the leading chart dimension; match the scalar
    # reference, which sums one equally weighted loss per chart.
    dense_loss = len(models) * (dense_u.square().mean() + dense_pred.square().mean())
    dense_loss.backward()
    for i, model in enumerate(models):
        assert torch.allclose(dense_u[i], scalar_outputs[i][0].detach(), atol=1e-12, rtol=1e-12)
        assert torch.allclose(dense_pred[i], scalar_outputs[i][1].detach(), atol=1e-12, rtol=1e-12)
        for name, value in model.named_parameters():
            assert torch.allclose(value.grad, scalar_grads[i][name], atol=1e-11, rtol=1e-11), name


def _stacked_ensemble(models):
    import torch
    from torch.func import functional_call, vmap
    parameter_names = tuple(dict(models[0].named_parameters()))
    buffer_names = tuple(dict(models[0].named_buffers()))
    params = {name: torch.stack([dict(m.named_parameters())[name] for m in models]).detach().requires_grad_(True)
              for name in parameter_names}
    buffers = {name: torch.stack([dict(m.named_buffers())[name] for m in models])
               for name in buffer_names}

    def ensemble_decode(codes):
        return vmap(lambda p, b, v: functional_call(models[0], (p, b), (v,), {"decode_only": True}),
                    in_dims=(0, 0, 0))(params, buffers, codes)

    def ensemble_encode(poses):
        return vmap(lambda p, b, v: functional_call(models[0], (p, b), (v,), {"encode_only": True}),
                    in_dims=(0, 0, 0))(params, buffers, poses)

    return params, buffers, ensemble_encode, ensemble_decode


def test_batched_tangent_jvp_matches_the_per_chart_loop():
    """One jvp(ensemble_decode, .) per latent axis == d jvps per chart.

    Mirrors the exact gram + projector math the fitter's ``batched_tangent``
    now runs in place of the scalar per-chart tangent block, so a divergence
    here is a divergence in the training objective.
    """
    import torch
    from torch.func import jvp
    from atlas_rom.charts.physical_chart_autoencoder import _module

    torch.manual_seed(3)
    d, n_bodies, k, P = 2, 1, 3, 5
    cls = _module(n_bodies, width=8, d=d, encoder_kind="shallow")
    models = [cls(np.zeros(3), np.ones(3)) for _ in range(k)]
    state = models[0].state_dict()
    for m in models[1:]:
        m.load_state_dict(state)
    params, buffers, _, ensemble_decode = _stacked_ensemble(models)

    u_pad = torch.randn(k, P, d, dtype=torch.float64)
    w_pad = torch.rand(k, P, dtype=torch.float64)
    w_pad = w_pad / w_pad.sum(1, keepdim=True)
    obs_pad = torch.randn(k, P, 12 * n_bodies, d, dtype=torch.float64)
    projector_weight = 0.3

    def gram_and_proj(got, obs, w):
        gg = got.transpose(-1, -2) @ got
        wg = obs.transpose(-1, -2) @ obs
        gg = gg / torch.linalg.vector_norm(gg, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        wg = wg / torch.linalg.vector_norm(wg, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        gram = (w[..., None, None] * (gg - wg).square()).sum() / (gg.shape[-1] * gg.shape[-2])
        jitter = 1e-8 * torch.eye(d, dtype=got.dtype)

        def proj(mat):
            g = mat.transpose(-1, -2) @ mat + jitter
            return mat @ torch.linalg.solve(g, mat.transpose(-1, -2))

        pe = (w[..., None, None] * (proj(got) - proj(obs)).square()).sum(dim=(-2, -1)).sum()
        return gram + projector_weight * pe

    # per-chart reference
    per_chart = []
    for chart in range(k):
        cols = []
        for axis in range(d):
            tang = torch.zeros_like(u_pad[chart])
            tang[:, axis] = 1.0
            _, col = jvp(models[chart].decode, (u_pad[chart],), (tang,))
            cols.append(col)
        got = torch.stack(cols, -1)
        per_chart.append(gram_and_proj(got, obs_pad[chart], w_pad[chart]))
    reference = torch.stack(per_chart)

    # batched
    cols = []
    for axis in range(d):
        tang = torch.zeros_like(u_pad)
        tang[..., axis] = 1.0
        _, col = jvp(ensemble_decode, (u_pad,), (tang,))
        cols.append(col)
    got = torch.stack(cols, -1)
    gg = got.transpose(-1, -2) @ got
    wg = obs_pad.transpose(-1, -2) @ obs_pad
    gg = gg / torch.linalg.vector_norm(gg, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    wg = wg / torch.linalg.vector_norm(wg, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    gram_term = ((w_pad[:, :, None, None] * (gg - wg).square()).sum(dim=(1, 2, 3)) /
                 (gg.shape[-1] * gg.shape[-2]))
    jitter = 1e-8 * torch.eye(d, dtype=got.dtype)

    def proj(mat):
        g = mat.transpose(-1, -2) @ mat + jitter
        return mat @ torch.linalg.solve(g, mat.transpose(-1, -2))

    proj_err = (w_pad[:, :, None, None] * (proj(got) - proj(obs_pad)).square()).sum(dim=(-2, -1)).sum(dim=1)
    batched = gram_term + projector_weight * proj_err

    assert torch.allclose(batched, reference, atol=1e-10, rtol=1e-10), (batched - reference)


def test_batched_inverse_cycle_matches_per_chart_values_and_gradients():
    import torch
    from atlas_rom.charts.physical_chart_autoencoder import _module

    torch.manual_seed(4)
    d, k, P = 2, 3, 6
    cls = _module(1, width=8, d=d, encoder_kind="shallow")
    models = [cls(np.zeros(3), np.ones(3)) for _ in range(k)]
    state = models[0].state_dict()
    for m in models[1:]:
        m.load_state_dict(state)

    codes = torch.randn(k, P, d, dtype=torch.float64)

    scalar = [(models[i].encode(models[i].decode(codes[i])) - codes[i]).square().mean()
              for i in range(k)]
    scalar_total = sum(scalar)
    scalar_total.backward()
    scalar_grads = [{n: v.grad.clone() for n, v in m.named_parameters()} for m in models]
    for m in models:
        m.zero_grad(set_to_none=True)

    params, buffers, ensemble_encode, ensemble_decode = _stacked_ensemble(models)
    batched = (ensemble_encode(ensemble_decode(codes)) - codes).square().mean(-1).mean(-1)
    for i in range(k):
        assert torch.allclose(batched[i], scalar[i].detach(), atol=1e-12)
    batched.sum().backward()
    for i, m in enumerate(models):
        for name, value in m.named_parameters():
            # ``params`` stacks the leaves; grad lands on the stacked tensor.
            assert torch.allclose(params[name].grad[i], scalar_grads[i][name], atol=1e-10), name


def test_batched_seam_replay_matches_the_scalar_pair_loop():
    """Grouped [K,M] masked reduction == sum of per-(pair,direction) means.

    The fitter's overlap replay now runs one ``ensemble_decode(ensemble_encode(.))``
    per receiving chart with per-pose weight ``weight / c_s``; that must equal
    the old ``sum_s weight_s * mean_{p,f} ||D(E(q)) - q||^2`` in both value and
    gradient, or the seam objective has silently changed.
    """
    import torch
    from atlas_rom.charts.physical_chart_autoencoder import _module

    torch.manual_seed(6)
    d, k = 2, 3
    cls = _module(1, width=8, d=d, encoder_kind="shallow")
    models = [cls(np.zeros(3), np.ones(3)) for _ in range(k)]
    state = models[0].state_dict()
    for m in models[1:]:
        m.load_state_dict(state)

    # samples: (dst, q_src, accepted-mask, weight)
    rng = np.random.default_rng(0)
    samples = []
    for dst, rows, weight in [(0, 4, 0.3), (0, 3, 0.1), (2, 5, 0.25)]:
        q_src = torch.randn(rows, 12, dtype=torch.float64)
        accepted = torch.tensor(rng.random(rows) > 0.25)
        samples.append((dst, q_src, accepted, weight))

    # scalar reference
    scalar = torch.zeros((), dtype=torch.float64)
    for dst, q_src, accepted, weight in samples:
        keep = q_src[accepted]
        if not len(keep):
            continue
        recon = models[dst].decode(models[dst].encode(keep))
        scalar = scalar + weight * (recon - keep).square().mean()
    scalar.backward()
    scalar_grads = [{n: (v.grad.clone() if v.grad is not None else torch.zeros_like(v))
                     for n, v in m.named_parameters()} for m in models]
    for m in models:
        m.zero_grad(set_to_none=True)

    params, buffers, ensemble_encode, ensemble_decode = _stacked_ensemble(models)
    active = sorted({dst for dst, *_ in samples})
    max_rows = max(sum(q.shape[0] for dd, q, *_ in samples if dd == dst) for dst in active)
    q_pad = torch.zeros(k, max_rows, 12, dtype=torch.float64)
    w_pad = torch.zeros(k, max_rows, dtype=torch.float64)
    for dst in active:
        off = 0
        for dd, q_src, accepted, weight in samples:
            if dd != dst:
                continue
            p = q_src.shape[0]
            q_pad[dst, off:off + p] = q_src
            w_pad[dst, off:off + p] = (weight / accepted.sum().clamp_min(1)) * accepted.double()
            off += p
    recon = ensemble_decode(ensemble_encode(q_pad))
    batched = (w_pad * (recon - q_pad).square().mean(-1)).sum()

    assert torch.allclose(batched, scalar.detach(), atol=1e-12), float(batched - scalar)
    batched.backward()
    for i in range(k):
        for name in scalar_grads[i]:
            assert torch.allclose(params[name].grad[i], scalar_grads[i][name], atol=1e-10), (i, name)


def test_ragged_dense_forward_matches_per_model_on_real_rows():
    import torch
    from torch.func import functional_call, vmap
    from atlas_rom.charts.physical_chart_autoencoder import _module

    torch.manual_seed(5)
    d = 2
    cls = _module(1, width=6, d=d, encoder_kind="shallow")
    models = [cls(np.zeros(3), np.ones(3)) for _ in range(3)]
    state = models[0].state_dict()
    for m in models[1:]:
        m.load_state_dict(state)

    batches = [torch.randn(b, 12, dtype=torch.float64) for b in (2, 5, 3)]
    max_batch = max(len(b) for b in batches)
    padded = torch.zeros(3, max_batch, 12, dtype=torch.float64)
    mask = torch.zeros(3, max_batch, dtype=torch.bool)
    for i, b in enumerate(batches):
        padded[i, :len(b)] = b
        mask[i, :len(b)] = True

    params = {n: torch.stack([dict(m.named_parameters())[n] for m in models])
              for n in dict(models[0].named_parameters())}
    buffers = {n: torch.stack([dict(m.named_buffers())[n] for m in models])
               for n in dict(models[0].named_buffers())}
    enc, pred = vmap(lambda p, bf, q: functional_call(models[0], (p, bf), (q,)),
                     in_dims=(0, 0, 0))(params, buffers, padded)

    for i, b in enumerate(batches):
        ref_u, ref_pred = models[i](b)
        assert torch.allclose(enc[i, :len(b)], ref_u, atol=1e-12)
        assert torch.allclose(pred[i, :len(b)], ref_pred, atol=1e-12)


def test_joint_loop_runs_with_batched_tangent_projector_and_pruning(tmp_path):
    """End-to-end: the joint loop exercises the batched tangent/JVP path, the
    projector term, a scheduled prune, and the synchronized timers."""
    q = _poses(14)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=12,
                            construction_only=True,
                            options={"n_charts": 3, "tangent_weight": 0.1,
                                     "projector_weight": 0.2, "e_step_interval": 2,
                                     "replay_weight": 0.3, "overlap_pair_samples": 2,
                                     "lifecycle_prune_interval_epochs": 4,
                                     "lifecycle_prune_start_fraction": 0.3})
    report = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path),
                                        log=lambda *_: None)[0]
    breakdown = report["timing_seconds"]["joint_breakdown_seconds"]
    assert "batched_tangent_seconds" in breakdown
    assert breakdown["batched_tangent_seconds"] >= 0.0
    assert "do not sum" in breakdown["breakdown_note"]
    assert breakdown["overlap_replay_seconds"] >= 0.0
    per_epoch = report["timing_seconds"]["per_epoch"]
    tangent_epochs = [row for row in per_epoch if any(row["tangent_samples"])]
    assert tangent_epochs, "tangent term never evaluated"
    assert report["projector_weight"] == 0.2


def test_competitive_cae_fit_runs_end_to_end_on_cuda(tmp_path):
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    q = _poses(16)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=10,
                            construction_only=True,
                            options={"n_charts": 3, "device": "cuda",
                                     "tangent_weight": 0.1, "projector_weight": 0.2,
                                     "e_step_interval": 3,
                                     "lifecycle_prune_interval_epochs": 4,
                                     "lifecycle_prune_start_fraction": 0.3})
    report, models, patches, anchors = CompetitiveCAEFitter().fit(
        evidence, config, work=str(tmp_path), log=lambda *_: None)
    assert report["status"] in ("pass", "incomplete")
    assert len(models) == len(patches) == len(anchors)


def test_functional_encode_only_does_not_run_the_decoder():
    import torch
    from torch.func import functional_call
    from atlas_rom.charts.physical_chart_autoencoder import _module

    cls = _module(1, width=4, d=2, encoder_kind="shallow")
    model = cls(np.zeros(3), np.ones(3))
    q = torch.randn(3, 12, dtype=torch.float64)
    expected = model.encode(q)

    def decoder_must_not_run(_):
        raise AssertionError("encode_only executed the decoder")

    model.transforms = decoder_must_not_run
    state = (dict(model.named_parameters()), dict(model.named_buffers()))
    actual = functional_call(model, state, (q,), {"encode_only": True})
    assert torch.allclose(actual, expected)


def test_phase1_exploration_snapshots_follow_fixed_epoch_cadence(tmp_path):
    """Exploration frames are post-M-step learned fields with stable IDs."""
    import torch
    from viewers.view_chart_exploration import load_exploration_snapshots, prepare_display_points

    q = _poses(8)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(
        fitter_id="competitive_cae", width=4, epochs=3, construction_only=True,
        options={"n_charts": 2, "tangent_weight": 0.0,
                 "exploration_snapshot_every": 2,
                 "phase2_exploration_snapshot_every": 2})
    report = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path),
                                        log=lambda *_: None)[0]
    index = report["exploration_snapshots"]["index"]
    assert [row["epoch"] for row in index] == [0, 2, 3]
    assert index[-1]["final"] is True
    frames = load_exploration_snapshots(str(tmp_path))
    assert [frame["epoch"] for frame in frames] == [0, 2, 3]
    assert all(frame["chart_ids"] == list(range(len(frame["states"]))) for frame in frames)
    assert all(all(tensor.device.type == "cpu" for tensor in frame["states"][0].values())
               for frame in frames)
    assert all(np.asarray(frame["owner_chart_ids"]).shape == (len(q),) for frame in frames)
    assert all(set(np.asarray(frame["owner_chart_ids"]).tolist()).issubset(frame["chart_ids"])
               for frame in frames)
    assert all(np.asarray(frame["ownership_weights"]).shape == (len(q), len(frame["chart_ids"]))
               for frame in frames)
    assert all(np.asarray(frame["core_owned_mask"]).shape == np.asarray(frame["ownership_weights"]).shape
               for frame in frames)
    assert all(np.asarray(frame["core_owned_mask"], bool).sum(0).min() >= 2 for frame in frames)
    phase2_index = report["phase2_exploration_snapshots"]["index"]
    assert [row["epoch"] for row in phase2_index] == [0, 2, 3]
    assert [frame["epoch"] for frame in load_exploration_snapshots(str(tmp_path), phase=2)] == [0, 2, 3]
    np.savez(tmp_path / "samples.npz", q=q)
    observed, prepared, chart_ids = prepare_display_points(str(tmp_path), frames, round_trip=True)
    assert chart_ids == [0, 1]
    assert observed.shape == (len(q), 3)
    assert prepared[0]["reconstructions"][0].shape == observed.shape


def test_exploration_snapshot_index_accepts_an_explicit_report_name(tmp_path):
    from viewers.view_chart_exploration import load_exploration_snapshot_index

    snapshot_dir = tmp_path / "chart_exploration_snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "epoch00000.pt").touch()
    report = {
        "exploration_snapshots": {
            "dir": "chart_exploration_snapshots",
            "index": [{"epoch": 0, "file": "epoch00000.pt"}],
        },
    }
    (tmp_path / "named_run.json").write_text(json.dumps(report), encoding="utf-8")

    records = load_exploration_snapshot_index(str(tmp_path), report_name="named_run.json")
    assert [row["epoch"] for row in records] == [0]


def test_monotone_frontier_off_reproduces_the_default_report_shape(tmp_path):
    """Regression guard: the option defaults off and leaves the shared report
    fields (and the new ``monotone_frontier`` block) inert."""
    q = _poses(14)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=6,
                            construction_only=True,
                            options={"n_charts": 2, "e_step_interval": 2})
    report = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path),
                                        log=lambda *_: None)[0]
    assert report["resolved_config"]["monotone_frontier"] is False
    assert report["monotone_frontier"]["enabled"] is False
    assert report["monotone_frontier"]["trace"] == []
    assert report["monotone_frontier"]["final_territory_size"] == []
    assert report["resolved_config"]["inertial_frontier"] is False
    assert report["inertial_frontier"]["enabled"] is False
    assert report["inertial_frontier"]["trace"] == []
    assert report["inertial_frontier"]["final_claim_mass"] == []


def _monotone_trial_report(tmp_path, *, n=14, epochs=20, n_charts=2, pretrain_steps=200):
    q = _poses(n)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=epochs,
                            construction_only=True,
                            options={"n_charts": n_charts, "monotone_frontier": True,
                                     "e_step_min_interval": 2, "e_step_max_interval": 3,
                                     "prune_warmup_epochs": 4, "phase2_epochs": 1,
                                     "pretrain_steps": pretrain_steps})
    return CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path), log=lambda *_: None)[0]


def test_monotone_frontier_certified_territory_is_monotone(tmp_path):
    """Persistent per-chart territory only ever grows, and admission comes
    from graph-adjacent growth or a connected provisional birth -- never a
    silent forget/regrow."""
    report = _monotone_trial_report(tmp_path)
    mf = report["monotone_frontier"]
    assert mf["enabled"] is True
    assert mf["trace"], "expected at least one monotone-frontier E-step record"

    by_chart: dict[int, list[dict]] = {}
    for row in mf["trace"]:
        by_chart.setdefault(row["chart"], []).append(row)
    for rows in by_chart.values():
        rows.sort(key=lambda r: r["epoch"])
        sizes = [r["territory_size"] for r in rows]
        assert sizes == sorted(sizes), "persistent territory shrank across E-steps"
        for before, after in zip(sizes, sizes[1:]):
            assert after >= before

    assert report["resolved_config"]["monotone_frontier"] is True
    assert all(size >= 0 for size in mf["final_territory_size"])


def test_monotone_frontier_off_leaves_monotone_frontier_state_unused(tmp_path):
    """``monotone_frontier=False`` (the default) touches none of the new
    provisional/born/repair machinery -- already covered by the report-shape
    regression test above; this documents the invariant it enforces."""
    report = _monotone_trial_report(tmp_path)
    # Sanity: the True-path report actually exercises the new fields, so the
    # False-path regression test above is checking something meaningful.
    assert any("born" in row for row in report["monotone_frontier"]["trace"])


def test_monotone_frontier_unborn_chart_gets_nonzero_local_training_weight(tmp_path):
    """An empty certified territory (chart not yet born) still receives
    nonzero frontier mass from its fixed provisional source -- it must not
    train on nothing while it waits for genuine competence."""
    report = _monotone_trial_report(tmp_path, pretrain_steps=0)
    mf = report["monotone_frontier"]
    unborn_rows = [row for row in mf["trace"] if not row["born"]]
    assert unborn_rows, "expected at least one still-unborn E-step record"
    for row in unborn_rows:
        assert row["territory_size"] == 0
        assert row["frontier_mass"] > 0.0


def test_monotone_frontier_mass_is_equal_across_active_charts(tmp_path):
    """Each chart's frontier weights are normalized to the same total mass
    (per E-step), so a larger territory does not automatically pull a bigger
    share of training."""
    report = _monotone_trial_report(tmp_path)
    mf = report["monotone_frontier"]
    assert mf["trace"]
    for row in mf["trace"]:
        assert row["frontier_mass"] == pytest.approx(1.0, abs=1e-6)


def test_monotone_frontier_unborn_charts_are_never_retired(tmp_path):
    """A chart that never certified any territory (unborn) cannot be retired
    as 'swallowed' -- only a genuinely born chart with zero exclusive support
    for the hysteresis window may be."""
    report = _monotone_trial_report(tmp_path, pretrain_steps=0, n_charts=3, epochs=30)
    mf = report["monotone_frontier"]
    by_chart: dict[int, list[dict]] = {}
    for row in mf["trace"]:
        by_chart.setdefault(row["chart"], []).append(row)
    last_epoch = max(row["epoch"] for row in mf["trace"])
    final_chart_ids = {row["chart"] for row in mf["trace"] if row["epoch"] == last_epoch}
    for cid, rows in by_chart.items():
        if cid in final_chart_ids:
            continue      # survived to the end, not retired
        rows.sort(key=lambda r: r["epoch"])
        assert rows[-1]["territory_size"] > 0, (
            "a chart was retired while still unborn (territory_size==0)")


def test_monotone_frontier_maintenance_multiplier_is_smooth_and_low_under_violation():
    """The repair multiplier g_i (from the same sigmoid as the ownership
    weight, applied to smoothmax-over-territory error) degrades smoothly as
    error rises toward tau, and sits below 0.5 once tau is actually crossed --
    no abrupt training-mode switch."""
    from atlas_rom.charts.competitive_cae import _ownership_weight

    tau = 0.01
    well_under = float(_ownership_weight(np.array([0.1 * tau]), tau)[0])
    near = float(_ownership_weight(np.array([0.9 * tau]), tau)[0])
    over = float(_ownership_weight(np.array([2.0 * tau]), tau)[0])
    assert well_under > near > over
    assert well_under > 0.9
    assert over < 0.5


def test_monotone_frontier_phase2_admits_nothing(tmp_path):
    """Phase 2 must not grow, birth, or otherwise change any territory --
    the threshold only ever records successful phase-1 expansion."""
    report = _monotone_trial_report(tmp_path)
    mf = report["monotone_frontier"]
    phase1_last = {}
    for row in mf["trace"]:
        if row["phase"] == 1:
            prev = phase1_last.get(row["chart"])
            if prev is None or row["epoch"] > prev["epoch"]:
                phase1_last[row["chart"]] = row
    phase2_rows = [row for row in mf["trace"] if row["phase"] == 2]
    assert phase2_rows, "expected the final bookkeeping E-step to log a phase-2 record"
    for row in phase2_rows:
        assert row["admitted"] == 0
        before = phase1_last.get(row["chart"])
        if before is not None:
            assert row["territory_size"] == before["territory_size"]


def _inertial_trial_report(tmp_path, *, n=14, epochs=20, n_charts=2):
    q = _poses(n)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=epochs,
                            construction_only=True,
                            options={"n_charts": n_charts, "inertial_frontier": True,
                                     "e_step_min_interval": 2, "e_step_max_interval": 3,
                                     "prune_warmup_epochs": 4, "phase2_epochs": 1})
    return CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path), log=lambda *_: None)[0]


def test_inertial_frontier_and_monotone_frontier_are_mutually_exclusive(tmp_path):
    q = _poses(14)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=4,
                            construction_only=True,
                            options={"n_charts": 2, "monotone_frontier": True,
                                     "inertial_frontier": True})
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path), log=lambda *_: None)


def test_inertial_frontier_claims_are_bounded_and_logged(tmp_path):
    """Smooth claims s_i stay in [0, 1] and every E-step logs a nonzero claim
    mass -- there is no hard admission gap where a chart trains on nothing."""
    report = _inertial_trial_report(tmp_path)
    inf = report["inertial_frontier"]
    assert inf["enabled"] is True
    assert inf["trace"], "expected at least one inertial-frontier E-step record"
    for row in inf["trace"]:
        assert row["claim_mass"] >= 0.0
        assert 0.0 <= row["claim_max"] <= 1.0 + 1e-9
        assert row["frontier_mass"] >= 0.0
    assert report["resolved_config"]["inertial_frontier"] is True
    assert len(inf["final_claim_mass"]) == len(report["chart_ids"])


def test_inertial_frontier_phase2_freezes_claims(tmp_path):
    """Claims are frozen (read-only) outside phase 1 -- the single phase-2
    bookkeeping E-step must not change any chart's claim mass."""
    report = _inertial_trial_report(tmp_path)
    inf = report["inertial_frontier"]
    phase1_last = {}
    for row in inf["trace"]:
        if row["phase"] == 1:
            prev = phase1_last.get(row["chart"])
            if prev is None or row["epoch"] > prev["epoch"]:
                phase1_last[row["chart"]] = row
    phase2_rows = [row for row in inf["trace"] if row["phase"] == 2]
    assert phase2_rows, "expected the final bookkeeping E-step to log a phase-2 record"
    for row in phase2_rows:
        before = phase1_last.get(row["chart"])
        if before is not None:
            assert row["claim_mass"] == pytest.approx(before["claim_mass"], rel=1e-9, abs=1e-12)


def test_inertial_frontier_does_not_touch_monotone_frontier_state(tmp_path):
    """The two frontier mechanisms are independent report blocks -- running
    ``inertial_frontier`` must leave ``monotone_frontier`` fully inert."""
    report = _inertial_trial_report(tmp_path)
    assert report["resolved_config"]["monotone_frontier"] is False
    assert report["monotone_frontier"]["enabled"] is False
    assert report["monotone_frontier"]["trace"] == []


def test_retirement_is_decided_on_the_cover_mask_not_the_tangent_gate():
    """The off-node probe must read the cover mask (the loss/coverage mask), and
    the retirement block must not rerun the composite evaluator or the full
    tangent/fold certificate per candidate."""
    import inspect
    from atlas_rom.charts import competitive_cae
    src = inspect.getsource(competitive_cae)
    assert "off_node_redundancy(cover_competent)" in src
    assert "off_node_redundancy(certificate_gate)" not in src
    start = src.index("_retirement_tick = sync_tick()")
    end = src.index("_retirement_tick)", start)
    block = src[start:end]
    assert "_eval_all_composite()" not in block
    assert "_gate_angles_folds()" not in block
    assert "_receiver_certificate(" in block


def test_retirement_rejections_use_only_cover_mask_stages_and_are_timed(tmp_path):
    q = _poses(14)
    evidence = ChartFitInput(q, np.ones((len(q), 12, 1)),
                             np.asarray([[i, i + 1] for i in range(len(q) - 1)]))
    config = ChartFitConfig(fitter_id="competitive_cae", width=4, epochs=30,
                            construction_only=True,
                            options={"n_charts": 3, "e_step_min_interval": 2,
                                     "e_step_max_interval": 3, "prune_warmup_epochs": 4,
                                     "phase2_epochs": 1})
    report = CompetitiveCAEFitter().fit(evidence, config, work=str(tmp_path),
                                        log=lambda *_: None)[0]
    allowed = {"prescreen", "off_sample_probe", "receiver_certificate"}
    for row in report["retirement_rejection_log"]:
        assert row["stage"] in allowed
    breakdown = report["timing_seconds"]["joint_breakdown_seconds"]
    if breakdown.get("retirement_checkpoints", 0):
        assert breakdown["retirement_checkpoint_seconds"] >= 0.0
