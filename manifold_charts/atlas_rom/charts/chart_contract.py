"""Stable, evidence-only contracts for learned physical chart fitters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ChartFitInput:
    """Simulator-produced evidence visible to a chart fitter.

    Deliberately absent are declared coordinates, topology, memberships,
    latent dimensions, and intended chart counts.
    """

    poses: np.ndarray
    frames: np.ndarray
    cover_edges: np.ndarray
    frame_confidence: np.ndarray | None = None
    physical_validity: Any | None = None

    def __post_init__(self) -> None:
        for name in ("poses", "frames", "cover_edges"):
            value = np.asarray(getattr(self, name))
            if value.ndim == 0:
                raise ValueError(f"{name} must be an array of evidence")
        if len(self.poses) != len(self.frames):
            raise ValueError("poses and frames must have the same sample count")
        if self.frame_confidence is not None and len(self.frame_confidence) != len(self.poses):
            raise ValueError("frame_confidence must match poses")


@dataclass(frozen=True)
class ChartFitConfig:
    """Method-owned fitting options; no physical prior is encoded here."""

    fitter_id: str = "physical_ae_patch"
    growth_steps: int = 4
    assignment_algorithm: str = "adaptive_front"
    overlap_hops: int = 2
    min_core_size: int = 16
    max_core_size: int | None = None
    max_seed_attempts: int | None = None
    epochs: int = 3000
    tangent_every: int = 1
    n_patches: int | None = None
    pretrain_epochs: int = 200
    phase_epochs: tuple[int, ...] = (150, 150, 100, 100)
    width: int = 96
    seed: int = 20260731
    simplify: bool = True
    construction_only: bool = False
    local_checkpoint: str | None = None
    out_report: str | None = None
    out_artifact: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class FittedChartAtlas(Protocol):
    charts: Sequence[Any]
    supports: Sequence[Any]
    reports: Sequence[Mapping[str, Any]]

    def transition(self, i: int, j: int, u_i: Any) -> Any: ...
    def serialize_payload(self) -> Mapping[str, Any]: ...


class ChartFitter(Protocol):
    fitter_id: str

    def fit(self, evidence: ChartFitInput, config: ChartFitConfig) -> FittedChartAtlas: ...


def evidence_from_sample_set(sample_set: Mapping[str, Any], validity: Any | None = None) -> ChartFitInput:
    """Build the contract without forwarding training-only metadata."""
    return ChartFitInput(
        poses=np.asarray(sample_set["q"]),
        frames=np.asarray(sample_set["frames"]),
        cover_edges=np.asarray(sample_set["cover_edges"]),
        frame_confidence=(None if sample_set.get("frame_confidence") is None
                          else np.asarray(sample_set["frame_confidence"])),
        physical_validity=validity,
    )
