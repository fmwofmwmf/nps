"""dataset.py — the standardized intermediate dataset every case flows through.

This is the CONTRACT between "mechanism / atlas discovery" and "ROM training".
Each stage writes exactly one file under `work/<case>/`, and every later stage
reads only these files — never a case module, never a stage's in-memory objects:

    mechanism.json      the normalized AtlasCaseSpec (what was actually built)
    samples.npz         q, per-body transforms, contact / mode labels
    atlas_data.npz      chart coordinates, memberships, overlaps, transitions
    dynamics_data.npz   q/u, M, V, force & actuator Jacobians

Because the contract is files, a stage can be re-run alone, a hand-produced
`samples.npz` can be dropped in from an older pipeline, and the build gates in
`validate.py` can check artifact completeness without importing any training code.

Every reader here validates shapes on load: a silently transposed or truncated
array is the failure mode that costs hours downstream.
"""
from __future__ import annotations

import os

import numpy as np

MECHANISM = "mechanism.json"
SAMPLES = "samples.npz"
ATLAS_DATA = "atlas_data.npz"
DYNAMICS_DATA = "dynamics_data.npz"
COORDINATE_STATE = "coordinate_state.json"

STAGE_ARTIFACTS = {
    "mechanism": (MECHANISM,),
    "samples": (SAMPLES,),
    "quotient": (),           # only where walked edges exist to quotient
    "graph": (),             # only where an analytic residual certifies chords
    "atlas": (ATLAS_DATA,),
    "canonical": (),          # only when the case enables it; checked by the stage
    "decoder": ("decoder.pt",),
    "dynamics": (DYNAMICS_DATA,),
    "boundary": (),          # only when the case declares one; checked by the stage
}


def work_dir(root: str, case: str) -> str:
    path = os.path.join(root, "work", case)
    os.makedirs(path, exist_ok=True)
    return path


def path_in(work: str, name: str) -> str:
    return os.path.join(work, name)


def exists(work: str, name: str) -> bool:
    return os.path.exists(path_in(work, name))


def _check(name, arr, ndim=None, shape=None):
    arr = np.asarray(arr)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim}-D, got shape {arr.shape}")
    if shape is not None:
        for axis, expected in enumerate(shape):
            if expected is not None and arr.shape[axis] != expected:
                raise ValueError(f"{name}: axis {axis} is {arr.shape[axis]}, "
                                 f"expected {expected}")
    return arr


# ─── samples.npz ──────────────────────────────────────────────────────────────

def save_samples(work: str, q, u, *, labels=None, frames=None, edges=None,
                 fields=None, contact_pairs=None):
    """The explored configurations and their coordinates.

    q       (N, 12*n_bodies)  flattened T[:3,:4] per driven body
    u       (N, d)            declared intrinsic coordinates of each sample
    labels  (N,) int          contact / mode label (0 = free), optional
    frames  (N, 12*n, d)      oracle tangent frames, when the sampler produced
                              them; a clean Jacobian beats differentiating a
                              value-fitted decoder for the reduced mass
    fields  dict[str, (N,)]   extra recorded scalars a coordinate may reference
    contact_pairs  list of "a|b" strings actually observed in contact
    """
    q = _check("q", q, ndim=2)
    u = _check("u", u, ndim=2, shape=(len(q), None))
    payload = {"q": q.astype(float), "u": u.astype(float)}
    if labels is not None:
        payload["labels"] = _check("labels", labels, ndim=1,
                                   shape=(len(q),)).astype(np.int64)
    if frames is not None:
        payload["frames"] = _check("frames", frames, shape=(len(q), None, None)
                                   ).astype(float)
    if edges is not None:
        edges = _check("edges", edges, ndim=2, shape=(None, 2)).astype(np.int64)
        if edges.size and (edges.min() < 0 or edges.max() >= len(q)):
            raise ValueError("edges contain a sample index outside q")
        payload["edges"] = edges
    if contact_pairs is not None:
        payload["contact_pairs"] = np.asarray(list(contact_pairs), dtype=object)
    for key, value in (fields or {}).items():
        payload[f"field_{key}"] = _check(f"field {key}", value, ndim=1,
                                         shape=(len(q),)).astype(float)
    np.savez_compressed(path_in(work, SAMPLES), **payload)
    return path_in(work, SAMPLES)


def load_samples(work: str) -> dict:
    data = np.load(path_in(work, SAMPLES), allow_pickle=True)
    out = {"q": np.asarray(data["q"], float), "u": np.asarray(data["u"], float),
           "labels": (np.asarray(data["labels"], int) if "labels" in data.files
                      else None),
           "frames": (np.asarray(data["frames"], float) if "frames" in data.files
                      else None),
           "edges": (np.asarray(data["edges"], int) if "edges" in data.files
                     else None),
           "contact_pairs": (list(data["contact_pairs"])
                             if "contact_pairs" in data.files else []),
           "fields": {k[len("field_"):]: np.asarray(data[k], float)
                      for k in data.files if k.startswith("field_")}}
    _check("u", out["u"], ndim=2, shape=(len(out["q"]), None))
    return out


# ─── atlas_data.npz ───────────────────────────────────────────────────────────

def save_atlas_data(work: str, *, chart_ids, u, membership, overlaps,
                    cells=None, chart_cells=None, chart_vertices=None, uv_maps=None,
                    transitions=None, certificate=None):
    """The chart decomposition of the sample cloud.

    chart_ids   (n_charts,) int          chart labels, 0..n-1
    u           (N, d)                   sample coordinates (same order as samples)
    membership  (N, n_charts) bool       which charts contain each sample
    overlaps    (E, 2) int               chart-adjacency edges
    cells / chart_cells / uv_maps        triangle-mesh atlas structure, when the
                                         chart mode produced one
    transitions dict[str, array]         per-edge chart-to-chart maps
    """
    u = _check("u", u, ndim=2)
    membership = _check("membership", membership, ndim=2,
                        shape=(len(u), len(chart_ids)))
    payload = {"chart_ids": np.asarray(chart_ids, int),
               "u": u.astype(float),
               "membership": membership.astype(bool),
               "overlaps": np.asarray(overlaps, int).reshape(-1, 2)}
    if cells is not None:
        payload["cells"] = np.asarray(cells, int)
    if chart_cells is not None:
        payload["chart_cells"] = np.asarray(chart_cells, dtype=object)
    if chart_vertices is not None:
        payload["chart_vertices"] = np.asarray(chart_vertices, dtype=object)
    if uv_maps is not None:
        payload["uv_maps"] = np.asarray(uv_maps, dtype=object)
    if certificate is not None:
        payload["certificate"] = np.asarray(certificate, dtype=object)
    for key, value in (transitions or {}).items():
        payload[f"transition_{key}"] = np.asarray(value)
    np.savez_compressed(path_in(work, ATLAS_DATA), **payload)
    return path_in(work, ATLAS_DATA)


def load_atlas_data(work: str) -> dict:
    data = np.load(path_in(work, ATLAS_DATA), allow_pickle=True)
    return {"chart_ids": np.asarray(data["chart_ids"], int),
            "u": np.asarray(data["u"], float),
            "membership": np.asarray(data["membership"], bool),
            "overlaps": np.asarray(data["overlaps"], int).reshape(-1, 2),
            "cells": (np.asarray(data["cells"], int) if "cells" in data.files
                      else None),
            "chart_cells": (list(data["chart_cells"]) if "chart_cells" in data.files
                            else None),
            "chart_vertices": (list(data["chart_vertices"]) if "chart_vertices" in data.files
                                else None),
            "uv_maps": (list(data["uv_maps"]) if "uv_maps" in data.files else None),
            "certificate": (data["certificate"].item() if "certificate" in data.files else None),
            "transitions": {k[len("transition_"):]: data[k] for k in data.files
                            if k.startswith("transition_")}}


def membership_pairs(work: str):
    """(sample index, chart, chart-local coordinate) for every ADMITTING chart.

    Training on the primary label alone would give an overlap-heavy chart almost
    no data and leave two charts free to disagree exactly where a handoff
    happens.  A sample in the overlap is a valid configuration of both charts, so
    it trains both — and, once canonical coordinates exist, it is also the pair
    that pins `E_a(u_a) == E_b(u_b) == z`.
    """
    atlas = load_atlas_data(work)
    membership = atlas["membership"]
    rows, charts = np.nonzero(membership)
    if atlas["chart_vertices"] is None or atlas["uv_maps"] is None:
        if membership.shape[1] != 1:
            raise ValueError("multi-chart atlas has no chart-local coordinates")
        return rows, charts, np.asarray(atlas["u"][rows], float)
    first_uv = np.asarray(atlas["uv_maps"][int(charts[0])], float)
    local = np.empty((len(rows), first_uv.shape[1]), float)
    for c in np.unique(charts):
        vertices = np.asarray(atlas["chart_vertices"][int(c)], int)
        uv = np.asarray(atlas["uv_maps"][int(c)], float)
        by_vertex = {int(v): uv[i] for i, v in enumerate(vertices)}
        for k in np.flatnonzero(charts == c):
            local[k] = by_vertex[int(rows[k])]
    return rows, charts, local


# ─── dynamics_data.npz ────────────────────────────────────────────────────────

def save_dynamics_data(work: str, *, chart, u, M, V, actuator_jacobian=None,
                       grad_V=None, q=None):
    """The (M, V) teacher dataset, per chart.

    chart              (N,) int          chart each teacher point belongs to
    u                  (N, d)
    M                  (N, d, d)         reduced mass, SPD
    V                  (N,)              potential
    grad_V             (N, d)            teacher gradient (the conservative force)
    actuator_jacobian  (N, n_act, d)     dθ_a/du per actuator, the generalized-
                                         force map's transpose
    q                  (N, 12*n_bodies)  decoded configuration, for regression
    """
    u = _check("u", u, ndim=2)
    d = u.shape[1]
    payload = {"chart": np.asarray(chart, int).reshape(len(u)),
               "u": u.astype(float),
               "M": _check("M", M, ndim=3, shape=(len(u), d, d)).astype(float),
               "V": _check("V", V, ndim=1, shape=(len(u),)).astype(float)}
    if grad_V is not None:
        payload["grad_V"] = _check("grad_V", grad_V, ndim=2,
                                   shape=(len(u), d)).astype(float)
    if actuator_jacobian is not None:
        payload["actuator_jacobian"] = _check(
            "actuator_jacobian", actuator_jacobian, ndim=3,
            shape=(len(u), None, d)).astype(float)
    if q is not None:
        payload["q"] = _check("q", q, ndim=2, shape=(len(u), None)).astype(float)
    np.savez_compressed(path_in(work, DYNAMICS_DATA), **payload)
    return path_in(work, DYNAMICS_DATA)


def load_dynamics_data(work: str) -> dict:
    data = np.load(path_in(work, DYNAMICS_DATA), allow_pickle=True)
    return {"chart": np.asarray(data["chart"], int),
            "u": np.asarray(data["u"], float),
            "M": np.asarray(data["M"], float),
            "V": np.asarray(data["V"], float),
            "grad_V": (np.asarray(data["grad_V"], float)
                       if "grad_V" in data.files else None),
            "actuator_jacobian": (np.asarray(data["actuator_jacobian"], float)
                                  if "actuator_jacobian" in data.files else None),
            "q": np.asarray(data["q"], float) if "q" in data.files else None}
