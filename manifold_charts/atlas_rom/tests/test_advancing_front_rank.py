"""Explorer cone search and scale-invariant frame (CPU, no simulator)."""
import numpy as np
import pytest

from atlas_rom.sample.physical_advancing_front import _maximin_direction, _scale_invariant_frame


@pytest.mark.parametrize("d", [2, 3, 5])
def test_antipodal_pair_leaves_an_orthogonal_cone(d):
    # A fresh node's first two pops are a direction and its opposite.  The
    # widest unprobed cone is then the whole orthogonal great sphere (90 deg).
    probe = np.ones(d) / np.sqrt(d)
    direction, separation = _maximin_direction([probe, -probe], d, cone_angle_deg=22.5)
    assert np.degrees(separation) > 89.0
    assert abs(float(direction @ probe)) < 1e-6


def _secants(ambient, n_tangent, play_dims, c, h, n_probes=6, seed=0):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((ambient, ambient)))
    tangent, play = basis[:, :n_tangent], basis[:, n_tangent:n_tangent + play_dims]

    def project(command):
        return tangent @ (tangent.T @ command) + play @ np.clip(play.T @ command, -c, c)

    probes, _ = np.linalg.qr(rng.standard_normal((ambient, n_probes)))
    odd = lambda s: np.column_stack([(project(s * p) - project(-s * p)) / (2 * s) for p in probes.T])
    span = basis[:, :n_tangent + play_dims]
    return span, odd(h), odd(h / 2), tangent


def _angle_deg(frame, tangent):
    return float(np.degrees(np.arccos(np.linalg.svd(tangent.T @ frame, compute_uv=False).clip(0, 1).min())))


def test_clearance_play_is_removed_from_the_span():
    # Play axes saturate at c << h: their secant halves when h doubles.
    span, a_h, a_half, tangent = _secants(36, 1, play_dims=2, c=0.005, h=0.2)
    frame, ratios = _scale_invariant_frame(span, a_h, a_half, .75)
    assert frame.shape[1] == 1, ratios
    # One of these probes is nearly orthogonal to a play axis, so its play
    # component stays unsaturated at h/2 (ratio 1) and tilts the fit ~2.4 deg.
    assert _angle_deg(frame, tangent) < 3.0


@pytest.mark.parametrize("n_tangent", [1, 2, 3])
def test_smooth_span_is_kept_whole(n_tangent):
    span, a_h, a_half, tangent = _secants(12, n_tangent, play_dims=0, c=0.0, h=0.2)
    frame, ratios = _scale_invariant_frame(span, a_h, a_half, .75)
    assert frame.shape[1] == n_tangent, ratios
    assert np.allclose(ratios, 1.0)
