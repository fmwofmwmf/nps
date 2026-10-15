"""Central registry for chart fitter implementations."""
from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, Any] = {}


def register(fitter_cls):
    fitter_id = getattr(fitter_cls, "fitter_id", None)
    if not fitter_id:
        raise ValueError("chart fitter must define fitter_id")
    _REGISTRY[fitter_id] = fitter_cls
    return fitter_cls


def resolve(fitter_id: str):
    if not _REGISTRY:
        from atlas_rom.charts.competitive_cae import CompetitiveCAEFitter
        register(CompetitiveCAEFitter)
    try:
        return _REGISTRY[fitter_id]()
    except KeyError as exc:
        raise ValueError(f"unknown chart fitter {fitter_id!r}") from exc
