"""Specification for a clearance-aware tangent rank rule (CPU, no simulator).

Each synthetic projection moves a smooth manifold coordinate by the command's
component along declared tangent axes (the test's ground truth only) and, for
clearance systems, lets extra "play" axes follow the command up to +-c.

The first ladder rule (slope classification, enlarge-on-ambiguity) failed four
of these cases and was reverted: unsaturated play reads as tangent, slope alone
ignores magnitude, and few probes mix play into the frame.  These tests stay as
the acceptance bar for its replacement, ``_ladder_rank(odd_response, n_probes,
ambient, h, h_cap) -> (rank, frame, valid, doublings, diagnostics)``.
"""
import numpy as np
import pytest

try:
    from atlas_rom.sample.physical_advancing_front import _ladder_rank
except ImportError:
    pytest.skip("clearance-aware rank rule not implemented yet (specification only)",
                allow_module_level=True)


def _system(ambient, n_tangent, play_dims=0, c=0.005, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((ambient, ambient)))
    tangent = basis[:, :n_tangent]
    normal = basis[:, n_tangent:n_tangent + 2]          # curvature lives here
    play = basis[:, n_tangent + 2:n_tangent + 2 + play_dims]

    def project(command):
        t = tangent.T @ command
        bend = 0.8 * np.array([np.sum(t ** 2), np.sum(t ** 3)])   # curved embedding
        slack = np.clip(play.T @ command, -c, c) if play_dims else np.zeros(0)
        pose = tangent @ t + normal @ bend + (play @ slack if play_dims else 0.0)
        return pose + noise * rng.standard_normal(ambient)

    return project, tangent


def _rank(project, ambient, h=0.02, h_cap=0.32, n_probes=6, seed=1):
    probes, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((ambient, n_probes)))
    probes = probes.T

    def odd(j, s):
        return (project(s * probes[j]) - project(-s * probes[j])) / (2. * s)

    return _ladder_rank(odd, n_probes, ambient, h, h_cap)


def _span_angle_deg(frame, tangent):
    cosines = np.linalg.svd(tangent.T @ frame, compute_uv=False).clip(0., 1.)
    return float(np.degrees(np.arccos(cosines.min())))


@pytest.mark.parametrize("n_tangent", [1, 2])
def test_smooth_manifold_rank(n_tangent):
    project, tangent = _system(12, n_tangent)
    rank, frame, valid, doublings, _ = _rank(project, 12)
    assert rank == n_tangent and doublings == 0 and valid == 6
    assert _span_angle_deg(frame, tangent) < 1.0


def test_clearance_play_is_not_tangent():
    project, tangent = _system(36, 1, play_dims=6, c=0.005)
    rank, frame, _valid, _doublings, diag = _rank(project, 36)
    assert rank == 1, diag
    assert _span_angle_deg(frame, tangent) < 2.0


def test_projection_noise_is_not_tangent():
    project, tangent = _system(12, 1, noise=1e-5)
    rank, frame, *_ = _rank(project, 12)
    assert rank == 1
    assert _span_angle_deg(frame, tangent) < 2.0


def test_wide_play_enlarges_ladder():
    # Play wider than the first ladder: it reads as free motion until the
    # probes grow past it.
    project, tangent = _system(36, 1, play_dims=6, c=0.03)
    rank, frame, _valid, doublings, diag = _rank(project, 36, h=0.02, h_cap=2.0)
    assert doublings >= 1, diag
    assert rank == 1, diag
    assert _span_angle_deg(frame, tangent) < 5.0
