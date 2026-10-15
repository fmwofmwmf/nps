"""Smoke tests for the additional topology fixtures' physical SDFs."""
from __future__ import annotations

import numpy as np

from atlas_rom.cases import load_sdf_system, load_spec
from atlas_rom.simulate.sdf_oracle import SDFDirectionalOracle


def test_new_topology_cases_load_and_accept_their_seeds():
    names = ("generic_disk", "generic_u_region", "generic_pants",
             "generic_punctured_torus", "generic_genus2", "generic_genus3",
             "generic_genus4")
    for name in names:
        system = load_sdf_system(name)
        assert load_spec(name).name == name
        seed = np.asarray(system.seeds[0], float)
        if system.surface:
            projected = SDFDirectionalOracle(system).project(seed)
            assert abs(system.sdf(projected)) < 1e-7
        else:
            assert system.sdf(seed) <= 0.0


def test_planar_topology_fixtures_have_the_intended_holes_and_cut():
    disk = load_sdf_system("generic_disk")
    u_region = load_sdf_system("generic_u_region")
    pants = load_sdf_system("generic_pants")
    assert disk.surface_accept(np.array([0.0, 0.0, 0.0]))
    assert not disk.surface_accept(np.array([2.2, 0.0, 0.0]))
    assert u_region.surface_accept(np.array([-1.5, 0.0, 0.0]))
    assert not u_region.surface_accept(np.array([1.5, 0.0, 0.0]))  # removed radial chunk
    assert pants.surface_accept(np.array([0.0, 1.4, 0.0]))
    assert not pants.surface_accept(np.array([-.82, 0.0, 0.0]))
    assert not pants.surface_accept(np.array([.82, 0.0, 0.0]))


def test_punctured_torus_masks_the_removed_surface_disk():
    system = load_sdf_system("generic_punctured_torus")
    assert system.surface_accept is not None
    assert not system.surface_accept(np.array([2.8, 0.0, 0.0]))
    assert system.surface_accept(np.array([-2.8, 0.0, 0.0]))


def test_handle_surfaces_have_surface_seeds_and_bounded_exteriors():
    for name in ("generic_genus2", "generic_genus3", "generic_genus4"):
        system = load_sdf_system(name)
        seed = np.asarray(system.seeds[0], float)
        assert abs(system.sdf(seed)) < 1e-8
        assert system.tangent_eps == .01
        assert system.max_samples is None and system.max_seconds is None
        assert system.sdf(np.array([20.0, 20.0, 20.0])) > 0.0
