"""Tangent rank: decided by measured error, not by response symmetry or ratio gaps."""
from __future__ import annotations

import numpy as np

from atlas_rom.sample.physical_advancing_front import _vote_frames
from atlas_rom.simulate.oracle_base import ManifoldOracle


class _Plane(ManifoldOracle):
    """Projects onto a 2-D coordinate plane, optionally with even and noisy terms."""
    dim_ambient = 12
    probe_directions = 10

    def __init__(self, even=0.0, wobble=0.0):
        super().__init__()
        self.even, self.wobble = float(even), float(wobble)

    def project(self, q):
        q = np.asarray(q, float)
        result = np.zeros_like(q)
        result[:2] = q[:2]
        result[2] = self.even * float(q @ q)
        if self.wobble:
            # Deterministic off-plane error that does not scale with the command.
            result[3:] = self.wobble * np.sin(1e3 * q[3:])
        return result


def _measure(oracle, h=0.1, seed=0):
    return oracle.tangent_measurement(np.zeros(12), eps=h, rng=np.random.default_rng(seed))


def test_strongly_asymmetric_response_keeps_the_tangent():
    frame, _singular, rank, diagnostics = _measure(_Plane(even=20.0))
    assert rank == 2 and diagnostics["probe_outcome"] == "certified_regular"
    assert diagnostics["probe_signed_asymmetry"] > 1.0
    assert np.linalg.svd(frame[:2], compute_uv=False).min() > 0.999


def test_error_below_measured_noise_adds_no_rank():
    _frame, singular, rank, diagnostics = _measure(_Plane(wobble=2e-3))
    assert rank == 2 and diagnostics["probe_outcome"] == "certified_regular"
    assert singular[2] > 1e-8                     # the extra directions are really there


def _record(rank, outcome="certified_regular", columns=5):
    directions = np.eye(12)[:, :columns]
    return {"rank": rank, "outcome": outcome, "frame": directions[:, :rank], "directions": directions}


def test_vote_keeps_disagreeing_poses_with_a_flagged_top_k_frame():
    records = {0: _record(2), 1: _record(2), 2: _record(3), 3: _record(1),
               4: _record(0, "uncertain"), 5: _record(0, "uncertain", columns=1)}
    assert _vote_frames(records) == (2, [0, 1])            # by default they are dropped
    modal, kept = _vote_frames(records, keep_disagreeing=True)
    assert modal == 2 and kept == [0, 1, 2, 3, 4]          # 5 has too few directions
    assert all(records[i]["frame"].shape[1] == 2 for i in kept)
    assert not records[0].get("frame_low_confidence")
    assert all(records[i]["frame_low_confidence"] for i in (2, 3, 4))
    assert records[2]["rank"] == 3                         # the measurement is still recorded
    assert records[4]["outcome"] == "uncertain"            # the frontier still counts it


def test_default_batch_size_is_unchanged():
    assert ManifoldOracle.probe_directions == 4
