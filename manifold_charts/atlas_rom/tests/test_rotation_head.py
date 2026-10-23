"""The chart decoder's rotation heads: ``exp`` is a valid rotation with a bounded Jacobian at every rotation vector;
``r6`` artifacts still load and decode unchanged."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.func import jvp

from atlas_rom.charts import physical_chart_autoencoder as pca


def _model(head, n_bodies=3, seed=0):
    torch.manual_seed(seed)
    return pca._module(n_bodies, width=16, d=2, encoder_kind="shallow", rotation_head=head)(
        np.zeros(3), np.ones(3)).double().eval()


_rotation_exp = type(_model("exp"))._rotation_exp


@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-3, 0.5, np.pi, 2 * np.pi, 50.0])
def test_exp_head_is_a_rotation_at_any_magnitude(magnitude):
    rng = np.random.default_rng(1)
    axes = rng.standard_normal((64, 3))
    w = torch.as_tensor(axes / np.linalg.norm(axes, axis=1, keepdims=True) * magnitude)
    R = _rotation_exp(w)
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(len(R), dtype=R.dtype), atol=1e-12)


def test_exp_head_matches_the_axis_angle_rotation():
    angle = 0.7
    R = _rotation_exp(torch.tensor([[0.0, 0.0, angle]], dtype=torch.float64))[0]
    c, s = np.cos(angle), np.sin(angle)
    np.testing.assert_allclose(R.numpy(), [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], atol=1e-14)


def test_exp_head_jacobian_is_bounded_everywhere():
    """|dR|_F <= sqrt(2) |dw| at every w, including the series branch near 0 and the 2*pi shell."""
    rng = np.random.default_rng(2)
    radii = np.concatenate([[0.0, 1e-5, 1e-2], rng.uniform(0, 20, 200), [np.pi, 2 * np.pi, 4 * np.pi]])
    for r in radii:
        axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
        w = torch.as_tensor(axis * r)
        for _ in range(4):
            v = torch.as_tensor(rng.standard_normal(3))
            _, dR = jvp(_rotation_exp, (w[None],), (v[None],))
            assert torch.isfinite(dR).all()
            assert float(torch.linalg.norm(dR)) <= np.sqrt(2.0) * float(torch.linalg.norm(v)) * (1 + 1e-9)


def test_exp_head_gradient_is_finite_at_zero():
    w = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    _rotation_exp(w).sum().backward()
    assert torch.isfinite(w.grad).all()


def test_decoder_width_identifies_the_head_and_roundtrips():
    for head, per_body in (("exp", 6), ("r6", 9)):
        model = _model(head)
        state = model.state_dict()
        assert pca.rotation_head_of(state, 3) == head
        last = [k for k in state if k.startswith("decoder.") and k.endswith(".weight")][-1]
        assert state[last].shape[0] == per_body * 3
        rebuilt = pca._module(3, width=16, d=2, encoder_kind="shallow",
                              rotation_head=pca.rotation_head_of(state, 3))(np.zeros(3), np.ones(3)).double()
        rebuilt.load_state_dict(state); rebuilt.eval()
        u = torch.as_tensor(np.random.default_rng(3).standard_normal((10, 2)))
        with torch.no_grad():
            assert torch.equal(model.decode(u), rebuilt.decode(u))


def test_r6_default_decodes_exactly_as_before():
    """The default head is still r6 with the original Gram-Schmidt, so existing artifacts are unchanged."""
    torch.manual_seed(0)
    model = pca._module(2, width=8, d=2)(np.zeros(3), np.ones(3)).double().eval()
    assert model.rotation_head == "r6"
    u = torch.as_tensor(np.random.default_rng(4).standard_normal((5, 2)))
    with torch.no_grad():
        y = model.decoder(u).reshape(-1, 2, 9)
        a, b = y[..., :3], y[..., 3:6]
        e1 = a / a.norm(dim=-1, keepdim=True)
        b = b - (e1 * b).sum(-1, keepdim=True) * e1
        e2 = b / b.norm(dim=-1, keepdim=True)
        R = torch.stack((e1, e2, torch.linalg.cross(e1, e2, dim=-1)), -1)
        assert torch.allclose(model.transforms(u)[..., :3], R, atol=1e-14)
