"""Contact (mesh penetration) validity: settled poses score ~0, overlapping poses do not."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from atlas_rom.cases import load_spec
from atlas_rom.charts.validity import ClosureValidity, CombinedValidity, validity_from_case

DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "work", "rotate_contact", "new_dataset.npz")


@pytest.fixture(scope="module")
def validity(tmp_path_factory):
    return validity_from_case(load_spec("rotate_contact"), include_contact=True,
                              contact_cache_dir=str(tmp_path_factory.mktemp("sdf")),
                              voxel=0.125, points_per_body=64)


def test_contact_is_opt_in():
    spec = load_spec("rotate_contact")
    assert isinstance(validity_from_case(spec), ClosureValidity)
    assert isinstance(validity_from_case(spec, include_contact=True, voxel=0.25, points_per_body=8),
                      CombinedValidity)
    assert isinstance(validity_from_case(load_spec("rotate"), include_contact=True), ClosureValidity)


def test_rest_pose_scores_zero_and_overlap_does_not(validity):
    contact = validity.parts[-1]
    rest = np.tile(np.hstack((np.eye(3), np.zeros((3, 1)))), (4, 1, 1))
    assert float(contact.residual(None, rest[None]).max()) < 1e-4
    moved = rest.copy()
    moved[1, :, 3] += [0.0, 0.0, -0.6]                 # push body B down through its neighbours
    assert float(contact.residual(None, moved[None]).max()) > 1e-3


def test_residual_is_differentiable_and_shaped(validity):
    rest = torch.as_tensor(np.tile(np.hstack((np.eye(3), np.zeros((3, 1)))), (2, 4, 1, 1)))
    rest = rest.clone()
    rest[:, 1, 2, 3] -= 0.6
    rest.requires_grad_(True)
    r = validity.residual(None, rest)
    assert r.shape == (2, 2 + 12)                      # two grounded hinges + twelve body pairs
    r.sum().backward()
    assert torch.isfinite(rest.grad).all() and rest.grad.abs().sum() > 0


@pytest.mark.skipif(not os.path.isfile(DATASET), reason="rotate_contact dataset not built")
def test_sampled_poses_score_near_zero(validity):
    q = np.load(DATASET)["q"][:50].reshape(-1, 4, 3, 4)
    assert float(validity.parts[-1].residual(None, q).max()) < 1e-4


def test_superdex_case_uses_the_oracle_grid(validity):
    assert validity.parts[-1].sdf_backend == "superdex"
    trimesh_part = validity_from_case(load_spec("rotate_contact"), include_contact=True,
                                      sdf_backend="trimesh", voxel=0.25, points_per_body=8).parts[-1]
    assert trimesh_part.sdf_backend == "trimesh"
