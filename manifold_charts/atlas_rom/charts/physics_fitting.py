"""Boundary between learned chart maps and learned physical fields."""
from __future__ import annotations

from typing import Any, Mapping


class PhysicsFitter:
    """Method-independent entry point for the existing field teachers."""

    fitter_id = "learned_physics_fields"

    def __init__(self, fit_dynamics, fit_support_and_boundary):
        self._fit_dynamics = fit_dynamics
        self._fit_support_and_boundary = fit_support_and_boundary

    def fit_chart(self, model, anchor, *, bodies, validity, seed, transition_positive=(),
                  transition_negative=(), local_map_negative=(), support_reference_q=(),
                  support_interior_q=(), epochs=800, validity_tol=0.1):
        mass, potential, mv_metrics = self._fit_dynamics(
            model, anchor, bodies["masses"], bodies["inertia"], bodies["com"],
            bodies["up"], bodies["gravity"], epochs=epochs)
        support, boundary, domain_metrics = self._fit_support_and_boundary(
            model, anchor, validity, epochs=epochs, seed=seed,
            validity_tol=validity_tol, transition_positive=transition_positive,
            transition_negative=transition_negative, local_map_negative=local_map_negative,
            support_reference_q=support_reference_q, support_interior_q=support_interior_q)
        return {
            "mass": mass, "potential": potential, "support": support,
            "boundary": boundary, "metrics": {**mv_metrics, **domain_metrics},
        }


def make_physics_fitter(learned_module):
    return PhysicsFitter(learned_module.fit_dynamics,
                         learned_module.fit_support_and_boundary)
