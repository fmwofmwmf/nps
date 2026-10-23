"""Live matplotlib inspector for a sample set in its own ambient space.

This is the interactive sibling of :mod:`viewers.generic_manifold_diagnostics`.
It draws the same layers -- physical samples, witnessed edges, the SDF backdrop,
chart supports/cores, certified contact strips, learned support isolines -- but:

* it reads the case's generic ``VISUAL_UVS`` display mapping when present;
  otherwise samples are placed by, in order of preference, a case-supplied
  ``VIEW_EMBED(q)`` map, the SDF ambient state (when 2- or 3-D),
  or the first driven body's translation;
* it opens an interactive window (pan/zoom, 3-D trackball, layer check boxes and
  a chart-isolation slider) instead of writing a PNG;
* ``--snapshots`` adds a second slider that steps through a competitive_cae run's
  prune-step snapshots (``<work>/prune_snapshots/prune_epoch*_{pre,post}.pt``,
  written when the fitter runs with ``snapshot_prune_steps=True``); each frame
  rebuilds the atlas from that snapshot's ``hard_competent`` membership.

Examples::

    python -m viewers.view_sample_set --case generic_sphere --work work/generic_sphere
    python -m viewers.view_sample_set --work work/generic_annulus_seam_baseline
    python -m viewers.view_sample_set --case generic_so3 --work work/generic_so3 --latent
    python -m viewers.view_sample_set --case generic_genus3 --work work/genus3_prune_snap --snapshots

``--case`` is optional when it can be inferred from the ``--work`` directory
(its recorded ``mechanism.json`` name, or the longest case-name prefix of the
directory stem); pass it explicitly when the inference is ambiguous or fails.

Run from ``testing/manifold_charts`` (where the ``work/`` tree lives); ``--work``
is resolved relative to the current directory.

A case whose ambient space is above 3-D (the S^3 / T^3 family in R^4, SO(3) in
R^9) shows nothing meaningful until its module defines::

    def VIEW_EMBED(q):               # (N, 12 * n_bodies) poses -> (N, 2) or (N, 3)
        ...

``VIEW_EMBED`` is deliberately separate from ``VISUAL_UVS``: it is a plain
projection for the eye, not a chart parametrization.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from viewers import generic_manifold_diagnostics as diag  # noqa: E402
from atlas_rom.cases import (  # noqa: E402
    available_specs, load_sdf_system, load_spec, resolve_case_name,
)


def _infer_case(work: str, stem: str) -> str:
    """Best-effort case name for a work dir when ``--case`` is omitted.

    Tries, in order: a ``name`` recorded in ``mechanism.json``; a ``case``
    recorded in a fitter report JSON (``competitive_cae_report.json`` /
    ``patch_atlas_report.json``); then the longest case name that is a prefix
    of the work-dir stem (``<case>`` or ``<case>_<suffix>``).  Raises if
    nothing matches or the prefix match is ambiguous, so we never silently
    guess the wrong mechanism.
    """
    import json

    manifest = os.path.join(work, "mechanism.json")
    if os.path.exists(manifest):
        try:
            with open(manifest) as fh:
                name = json.load(fh).get("name")
            if name:
                return resolve_case_name(name)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    import glob
    reports = (["competitive_cae_report.json", "patch_atlas_report.json"]
               + sorted(os.path.basename(p) for p in
                        glob.glob(os.path.join(work, "competitive_*_report.json"))))
    for report_name in reports:
        report_path = os.path.join(work, report_name)
        if os.path.exists(report_path):
            try:
                with open(report_path) as fh:
                    name = json.load(fh).get("case")
                if name:
                    return resolve_case_name(name)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    key = str(stem).strip().lower().replace("-", "_").replace(" ", "_")
    hits = [c for c in available_specs()
            if key == c.lower() or key.startswith(c.lower() + "_")]
    if hits:
        longest = max(len(c) for c in hits)
        best = [c for c in hits if len(c) == longest]
        if len(best) == 1:
            return best[0]
        raise ValueError(f"work dir {stem!r} matches multiple cases ({', '.join(best)}); "
                         "pass --case explicitly")
    raise ValueError(f"cannot infer --case from work dir {stem!r}; pass --case explicitly")


# --------------------------------------------------------------------------- #
# Ambient points
# --------------------------------------------------------------------------- #
def _view_embed(case: str):
    """Optional case-supplied projection into 2/3 dimensions (not VISUAL_UVS)."""
    return getattr(importlib.import_module(f"atlas_rom.cases.{case}"), "VIEW_EMBED", None)


def _load_samples(work: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(work, "samples.npz")
    if not os.path.exists(path):
        path = os.path.join(work, "new_dataset.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no samples.npz or new_dataset.npz in {work}")
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["q"], float)
        edges = np.asarray(data["edges"] if "edges" in data.files
                           else np.empty((0, 2)), int).reshape(-1, 2)
    return q, edges


def _points(case: str, q: np.ndarray) -> tuple[np.ndarray, str, bool]:
    """Place every sample in a 2- or 3-D display space.

    Returns ``(points, label, identity_2d)``.  ``identity_2d`` is True only when
    the display space *is* a 2-D ambient state space with no embedding applied,
    which is the one case where an SDF backdrop and learned support isolines
    are meaningful.
    """
    case = resolve_case_name(case)
    visual_uvs = diag._visual_uvs(case)
    if visual_uvs:
        spec = load_spec(case)
        points = np.asarray(diag._visual_points(case, q, spec, visual_uvs), float)
        names = visual_uvs.get("coordinates", ())
        label = (f"driver coordinates ({', '.join(names)})"
                 if visual_uvs.get("kind") == "driver_coordinates"
                 else "visual UV coordinates")
        return points, label, False

    embed = _view_embed(case)
    if embed is not None:
        points = np.asarray(embed(q), float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"{case}.VIEW_EMBED(q) must return (N, 2) or (N, 3); "
                             f"got {points.shape}")
        return points, f"VIEW_EMBED -> {points.shape[1]}D", False

    system = load_sdf_system(case)
    if system is not None and system.state_from_observation is not None:
        states = np.asarray([system.state_from_observation(row.reshape(-1))
                             for row in q], float)
        states = states.reshape(len(q), -1)
        d = states.shape[1]
        if d in (2, 3):
            return states, f"ambient {d}D (identity)", d == 2
        print(f"[view_sample_set] SDF ambient dimension is {d}; showing the first "
              f"3 coordinates. Add VIEW_EMBED(q) to cases/{case}.py for a real "
              f"projection.", file=sys.stderr)
        return states[:, :3], f"ambient {d}D -> first 3 coords", False

    # No embedding, no SDF state map: fall back to the first driven body's
    # translation (matches generic_manifold_diagnostics._points' non-UV branch).
    spec = load_spec(case)
    translated = next((c for c in spec.coordinates
                       if c.source == "body_translation"), None)
    if translated is None:
        raise ValueError(
            f"{case}: no VIEW_EMBED, no SDF state_from_observation, and no "
            "body_translation coordinate. Add `def VIEW_EMBED(q): ...` "
            "(q is (N, 12*n_bodies)) to the case module.")
    body = translated.args["body"]
    offset = 12 * spec.mechanism.body_order.index(body)
    return q[:, offset + np.asarray((3, 7, 11))], "body translation (3D)", False


# --------------------------------------------------------------------------- #
# Chart / contact metadata (all optional)
# --------------------------------------------------------------------------- #
def _chart_pass(report: dict, chart: int) -> bool:
    patches = report.get("patches")
    if patches and chart < len(patches):
        return bool(patches[chart].get("pass", False))
    return report.get("status") == "pass" or bool(report.get("all_pass"))


def _remap_indices(lists, inv: np.ndarray):
    """Filter+renumber index lists onto a kept-sample subset (inv: old -> new / -1)."""
    out = []
    for lst in lists or []:
        new = inv[np.asarray(lst, int)]
        out.append(new[new >= 0].tolist())
    return out


def _subsample(keep: np.ndarray, n: int, edges: np.ndarray, state):
    """Restrict samples to ``keep`` and remap edges / chart membership indices."""
    inv = np.full(n, -1, int)
    inv[keep] = np.arange(len(keep))
    if len(edges):
        mask = (inv[edges[:, 0]] >= 0) & (inv[edges[:, 1]] >= 0)
        edges = inv[edges[mask]]
    if state is not None:
        state = dict(state)
        state["patches"] = _remap_indices(state.get("patches"), inv)
        state["cores"] = _remap_indices(state.get("cores"), inv)
        contacts = []
        for contact in state.get("contacts", []) or []:
            contact = dict(contact)
            strip = inv[np.asarray(contact.get("strip_vertices", []), int)]
            contact["strip_vertices"] = strip[strip >= 0].tolist()
            contacts.append(contact)
        state["contacts"] = contacts
    return edges, state


def _chart_of_sample(n: int, state) -> np.ndarray:
    """Per-sample chart index from support membership, -1 where uncovered."""
    owner = np.full(n, -1, int)
    if state is None:
        return owner
    for chart, support in enumerate(state.get("patches", [])):
        idx = np.asarray(support, int)
        idx = idx[(idx >= 0) & (idx < n)]
        # last writer wins; overlap is expected and unimportant for coloring
        owner[idx] = chart
    return owner


def _load_chart_model(model_state, encoder_kind: str):
    """Rebuild one saved physical chart AE from its state dict (mirrors
    generic_manifold_diagnostics._latent_embeddings)."""
    import torch
    from atlas_rom.charts import physical_chart_autoencoder as pca

    width = model_state["encoder.0.weight"].shape[0]
    d = [v for k, v in model_state.items()
         if k.startswith("encoder.") and k.endswith(".bias")][-1].shape[0]
    n_bodies = int(model_state["encoder.0.weight"].shape[1] // 9)
    model = pca._module(n_bodies, width=width, d=d, encoder_kind=encoder_kind,
                        rotation_head=pca.rotation_head_of(model_state, n_bodies))(
        np.zeros_like(model_state["centre"].numpy()),
        np.ones_like(model_state["scale"].numpy())).double()
    model.load_state_dict(model_state)
    return model.eval()


def _roundtrip_poses(q: np.ndarray, owner: np.ndarray, state, *,
                     through_chart: int | None = None) -> np.ndarray:
    """Round-trip the overlay poses; keep GT samples fixed.

    Each pose goes through the chart that owns it -- UNLESS ``through_chart`` is
    given, in which case every overlay pose is reconstructed by that one chart's
    decoder ``D_c(E_c(q))``.  Nothing in training makes the charts agree on an
    overlap pose, so isolating a chart genuinely changes the round-trip there.
    """
    import torch

    out = np.asarray(q, float).copy()
    encoder_kind = state.get("encoder_kind", "current")
    Q = torch.as_tensor(out).double()
    states = state.get("states", []) or []
    overlay = set()
    for indices in state.get("patches", []) or []:
        overlay.update(int(i) for i in indices)
    for indices in state.get("cores", []) or []:
        overlay.update(int(i) for i in indices)
    for contact in state.get("contacts", []) or []:
        overlay.update(int(i) for i in contact.get("strip_vertices", []))
    overlay = np.asarray(sorted(i for i in overlay if 0 <= i < len(q)), int)
    if not len(overlay):
        return out

    if through_chart is not None:
        if not 0 <= through_chart < len(states):
            return out
        model = _load_chart_model(states[through_chart], encoder_kind)
        with torch.no_grad():
            out[overlay] = model.decode(model.encode(Q[torch.as_tensor(overlay)])).numpy()
        return out

    for chart, model_state in enumerate(states):
        idx = overlay[owner[overlay] == chart]
        if not len(idx):
            continue
        model = _load_chart_model(model_state, encoder_kind)
        with torch.no_grad():
            out[idx] = model.decode(model.encode(Q[torch.as_tensor(idx)])).numpy()
    return out


# --------------------------------------------------------------------------- #
# Prune-step snapshots (competitive_cae `snapshot_prune_steps=True`)
# --------------------------------------------------------------------------- #
def _load_prune_snapshots(snap_dir: str):
    """Sorted (label, raw_dict) for every prune_epochNNNNN_{pre,post}.pt.

    Files are ordered by epoch, then 'pre' before 'post'.  Each raw dict holds
    ``states`` (per-chart AE state dicts), ``encoder_kind`` and a
    ``hard_competent`` [n_samples, n_charts] boolean membership mask.
    """
    import torch

    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"no prune-snapshot directory at {snap_dir}")
    entries = [e for e in os.scandir(snap_dir)
               if e.name.startswith("prune_epoch") and e.name.endswith(".pt")]

    def _key(entry):
        stem = entry.name[: -len(".pt")].split("_")
        return int(stem[1][len("epoch"):]), 0 if stem[-1] == "pre" else 1

    out = []
    for entry in sorted(entries, key=_key):
        raw = torch.load(entry.path, map_location="cpu", weights_only=False)
        phase = "pre" if entry.name.endswith("_pre.pt") else "post"
        label = f"epoch {int(raw.get('epoch', 0))} {phase}  K={int(raw.get('n_charts', 0))}"
        out.append((label, raw))
    if not out:
        raise FileNotFoundError(f"no prune_epoch*.pt files in {snap_dir}")
    return out


def _snapshot_state(raw: dict, keep: np.ndarray | None, n_total: int):
    """Turn a prune snapshot into an atlas-like state the draw layers accept.

    Per-chart supports/cores are the hard-competent members; there are no
    certified contacts or support fields in a snapshot.
    """
    hard = np.asarray(raw["hard_competent"], bool)
    members = [np.flatnonzero(hard[:, i]) for i in range(hard.shape[1])]
    ids = list(raw.get("chart_ids") or range(hard.shape[1]))
    state = {"patches": [m.tolist() for m in members],
             "cores": [m.tolist() for m in members],
             "contacts": [],
             "states": raw.get("states", []),
             "encoder_kind": raw.get("encoder_kind", "current"),
             "chart_ids": ids}
    if keep is not None:
        chart_ids = state["chart_ids"]
        _, state = _subsample(np.asarray(keep, int), n_total, np.empty((0, 2), int), state)
        state["chart_ids"] = chart_ids
    return state


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class SampleSetView:
    def __init__(self, case: str, work: str, *, isolate: int = -1,
                 max_samples: int | None = None, seed: int = 0,
                 snapshots: str | None = None):
        self.case = case
        self.work = work
        q, self.edges = _load_samples(work)
        self.points, self.embed_label, self.identity_2d = _points(case, q)
        self.k = self.points.shape[1]
        self.report, self.state, self.source = diag._atlas_view(work)

        self.n_total = len(self.points)
        self._keep = None
        if max_samples is not None and self.n_total > max_samples:
            keep = np.sort(np.random.default_rng(seed).choice(
                self.n_total, int(max_samples), replace=False))
            self.edges, self.state = _subsample(keep, self.n_total, self.edges, self.state)
            self.points = self.points[keep]
            q = q[keep]
            self._keep = keep
            print(f"[view_sample_set] randomly kept {len(keep)}/{self.n_total} samples "
                  f"(seed {seed})", file=sys.stderr)
        self.q = q

        self.has_charts = self.state is not None and bool(self.state.get("patches"))
        self.n_charts = len(self.state.get("cores", [])) if self.state else 0
        self.owner = _chart_of_sample(len(self.points), self.state)
        # Lazily-built positions for learned overlays; GT samples/edges stay fixed.
        self.can_roundtrip = bool(self.state and self.state.get("states"))
        self._rt_points = None
        self._rt_key = None

        self.system = load_sdf_system(case)
        self.support_grid = None
        if self.identity_2d:
            try:
                limits = diag._limits(self.points)
                self.support_grid = diag._support_values(case, self.state, None, limits)
            except Exception as exc:  # stale checkpoint, missing torch pieces, ...
                print(f"[view_sample_set] learned support fields unavailable: {exc}",
                      file=sys.stderr)

        self.opts = {"samples": True, "edges": True, "sdf": self.identity_2d,
                     "supports": self.has_charts, "cores": self.has_charts,
                     "contacts": self.has_charts, "support_isolines": self.identity_2d,
                     "color_by_chart": self.has_charts, "round_trip": False}
        self.isolate = int(isolate)
        # Stable chart identity: for a plain atlas this is just position order,
        # but across prune snapshots it survives retirements (so a chart keeps
        # its colour and its "isolate" index when a lower-numbered chart goes).
        self.chart_ids = list((self.state or {}).get("chart_ids") or range(self.n_charts))

        # Optional: step through the fitter's prune-step snapshots.
        self.snapshots = None
        self.snapshot_label = None
        if snapshots is not None:
            self.snapshots = _load_prune_snapshots(snapshots)
            print(f"[view_sample_set] loaded {len(self.snapshots)} prune snapshots "
                  f"from {snapshots}", file=sys.stderr)
            self._apply_snapshot(0)

    def _apply_snapshot(self, index: int):
        """Swap the live atlas for prune snapshot ``index`` and refresh views."""
        index = int(np.clip(index, 0, len(self.snapshots) - 1))
        self.snapshot_index = index
        self.snapshot_label, raw = self.snapshots[index]
        self.state = _snapshot_state(raw, self._keep, self.n_total)
        self.has_charts = bool(self.state.get("patches"))
        self.n_charts = len(self.state.get("cores", []))
        self.owner = _chart_of_sample(len(self.points), self.state)
        self.can_roundtrip = bool(self.state.get("states"))
        self._rt_points = None
        self._rt_key = None
        self.chart_ids = list(self.state.get("chart_ids") or range(self.n_charts))
        self.source = f"prune snapshot: {self.snapshot_label}"
        if self.isolate >= self.n_charts:   # isolate is a size rank now
            self.isolate = -1

    def _overlay_points(self) -> np.ndarray:
        """Positions for learned overlays; samples remain in data space.

        When a chart is isolated the round-trip goes through THAT chart's
        decoder, so the overlay shows how the isolated chart alone reconstructs
        every pose it covers -- recomputed on isolate change.
        """
        if not self.opts.get("round_trip") or not self.can_roundtrip:
            return self.points
        vis = self._visible_charts() if self.isolate >= 0 else ()
        through = vis[0] if vis else None
        key = ("round_trip", through)
        if getattr(self, "_rt_key", None) != key:
            try:
                q_rt = _roundtrip_poses(self.q, self.owner, self.state, through_chart=through)
                self._rt_points = _points(self.case, q_rt)[0]
            except Exception as exc:
                print(f"[view_sample_set] round-trip positions unavailable: {exc}",
                      file=sys.stderr)
                self._rt_points = self.points
            self._rt_key = key
        return self._rt_points

    # -- artist layers ---------------------------------------------------- #
    def _palette(self):
        import matplotlib
        return matplotlib.colormaps["tab20"]

    def _cid(self, pos):
        """Stable chart id for a column position."""
        return self.chart_ids[pos] if 0 <= pos < len(self.chart_ids) else pos

    def _size_order(self):
        """Column positions ordered by patch (support) size, largest first."""
        patches = (self.state or {}).get("patches") or []
        return sorted(range(len(patches)), key=lambda i: len(patches[i]), reverse=True)

    def _visible_charts(self):
        """Column POSITIONS to draw.  ``isolate`` is a SIZE RANK: 0 = biggest
        chart, 1 = next, ...  (-1 = all).  Colour still follows the stable id."""
        if self.isolate < 0:
            return range(self.n_charts)
        order = self._size_order()
        return (order[self.isolate],) if 0 <= self.isolate < len(order) else ()

    def _draw(self):
        ax = self.ax
        ax.clear()
        P, overlay_P, k = self.points, self._overlay_points(), self.k
        palette = self._palette()

        if self.opts["sdf"] and self.identity_2d and self.system is not None \
                and int(self.system.dimension) == 2:
            try:
                diag._plot_sdf(ax, self.system, diag._limits(P))
            except Exception as exc:  # display-only; never fatal
                print(f"[view_sample_set] SDF backdrop skipped: {exc}", file=sys.stderr)

        if self.opts["edges"] and len(self.edges):
            segs = P[self.edges]  # (E, 2, k)
            for a, b in segs:
                ax.plot(*[[a[d], b[d]] for d in range(k)],
                        color="#90a4ae", lw=.35, alpha=.45, zorder=1)

        if self.opts["samples"]:
            if self.opts["color_by_chart"] and self.has_charts:
                vis = set(self._visible_charts())
                colors = np.array([palette(self._cid(o) % 20) if (o >= 0 and o in vis)
                                   else (.27, .35, .39, .55) for o in self.owner])
                ax.scatter(*[P[:, d] for d in range(k)], s=10, c=colors, zorder=2)
            else:
                ax.scatter(*[P[:, d] for d in range(k)], s=9, c="#455a64",
                           alpha=.65, zorder=2)

        if self.opts["supports"] and self.state is not None:
            for chart in self._visible_charts():
                support = np.asarray(self.state["patches"][chart], int)
                support = support[(support >= 0) & (support < len(overlay_P))]
                ax.scatter(*[overlay_P[support, d] for d in range(k)], s=18,
                           color=palette(self._cid(chart) % 20), zorder=3)

        if self.opts["cores"] and self.state is not None:
            for chart in self._visible_charts():
                core = np.asarray(self.state.get("cores", [])[chart], int)
                core = core[(core >= 0) & (core < len(overlay_P))]
                marker = "o" if _chart_pass(self.report, chart) else "x"
                ax.scatter(*[overlay_P[core, d] for d in range(k)], s=26,
                           color=palette(self._cid(chart) % 20), marker=marker, zorder=4)

        if self.opts["contacts"] and self.state is not None:
            for contact in self.state.get("contacts", []):
                strip = np.asarray(contact.get("strip_vertices", []), int)
                if not len(strip):
                    continue
                strip = strip[(strip >= 0) & (strip < len(overlay_P))]
                if not len(strip):
                    continue
                charts = contact.get("charts", [0, 0])
                ax.scatter(*[overlay_P[strip, d] for d in range(k)], s=44, marker="D",
                           facecolors=palette(int(charts[-1]) % 20),
                           edgecolors="#111827", linewidths=.65, zorder=6)

        if self.opts["support_isolines"] and self.support_grid is not None:
            _vis = self._visible_charts()
            chart = None if self.isolate < 0 else (_vis[0] if _vis else -1)
            try:
                diag._support_isolines(ax, self.support_grid, chart=chart)
            except Exception as exc:
                print(f"[view_sample_set] support isolines skipped: {exc}", file=sys.stderr)

        self._finish_axes(ax)
        self.fig.canvas.draw_idle()

    def _finish_axes(self, ax):
        P, overlay_P, k = self.points, self._overlay_points(), self.k
        all_P = np.vstack((P, overlay_P)) if self.opts.get("round_trip") else P
        lo, hi = all_P.min(0), all_P.max(0)
        pad = np.maximum(.08 * (hi - lo), 1e-6)
        lo, hi = lo - pad, hi + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if k == 3:
            ax.set_zlim(lo[2], hi[2]); ax.set_zlabel("z")
            ax.set_box_aspect(hi - lo)
        else:
            ax.set_aspect("equal")
        if not self.has_charts:
            scope = "no atlas"
        elif self.isolate < 0:
            scope = "all charts"
        else:
            _vis = self._visible_charts()
            if _vis:
                _p = _vis[0]
                scope = (f"chart {self._cid(_p)}  (#{self.isolate + 1}/{self.n_charts} by size, "
                         f"{len(self.state['patches'][_p])} pts)")
            else:
                scope = f"rank {self.isolate} (n/a)"
        shown = (f"{len(P)} samples" if len(P) == self.n_total
                 else f"{len(P)}/{self.n_total} samples")
        placement = (self.embed_label + " · round-trip D(E(q))"
                     if self.opts.get("round_trip") and self.can_roundtrip
                     else self.embed_label)
        snap = f" · snap {self.snapshot_index + 1}/{len(self.snapshots)} {self.snapshot_label}" \
            if self.snapshots else ""
        ax.set_title(
            f"{self.case}: {placement}, {shown}, "
            f"{len(self.edges)} edges, {self.n_charts} charts, "
            f"{len(self.state.get('contacts', [])) if self.state else 0} contacts "
            f"[{scope}]{snap}\n{self.source}")

    # -- interaction ---------------------------------------------------- #
    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, Slider

        self.fig = plt.figure(figsize=(11, 9))
        self.fig.subplots_adjust(left=0.30)
        if self.k == 3:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)

        labels = ["samples", "edges", "sdf", "supports", "cores", "contacts",
                  "support_isolines", "color_by_chart", "round_trip"]
        enabled = {
            "sdf": self.identity_2d and self.system is not None,
            "supports": self.has_charts, "cores": self.has_charts,
            "contacts": bool(self.state and self.state.get("contacts")),
            "support_isolines": self.support_grid is not None,
            "color_by_chart": self.has_charts,
            "round_trip": self.can_roundtrip,
        }
        labels = [l for l in labels if enabled.get(l, True)]
        cax = self.fig.add_axes([0.03, 0.45, 0.22, 0.40])
        cax.set_facecolor("#f3f4f6")
        cax.set_title("layers", fontsize=9, loc="left", pad=5, color="#374151")
        for spine in cax.spines.values():
            spine.set_color("#d1d5db")
        check = CheckButtons(
            cax, labels, [self.opts[l] for l in labels],
            label_props={"fontsize": [10], "color": ["#1f2937"]},
            frame_props={"s": 144, "facecolor": "white", "edgecolor": "#6b7280"},
            check_props={"s": 100, "facecolor": "#111827"},
        )
        self._check = check

        def on_check(label):
            self.opts[label] = not self.opts[label]
            self._draw()
        self._check.on_clicked(on_check)

        # "isolate" is a size rank: 0 = biggest chart, -1 = all.
        max_rank = ((max((int(r.get("n_charts", 1)) for _, r in self.snapshots), default=1) - 1)
                    if self.snapshots else self.n_charts - 1)
        if (self.has_charts or self.snapshots) and max_rank >= 0:
            sax = self.fig.add_axes([0.06, 0.30, 0.18, 0.04])
            self._slider = Slider(sax, "isolate\n(by size)", -1, max(max_rank, 0),
                                  valinit=self.isolate, valstep=1)

            def on_slide(value):
                self.isolate = int(round(value))
                self._draw()
            self._slider.on_changed(on_slide)

        if self.snapshots:
            snax = self.fig.add_axes([0.06, 0.23, 0.18, 0.04])
            self._snap_slider = Slider(snax, "snapshot", 0, len(self.snapshots) - 1,
                                       valinit=self.snapshot_index, valstep=1)

            def on_snap(value):
                self._apply_snapshot(int(round(value)))
                if hasattr(self, "_slider"):
                    self._slider.set_val(self.isolate)
                self._draw()
            self._snap_slider.on_changed(on_snap)

        self._draw()
        plt.show()


def show_latent(case: str, work: str):
    """Companion window: per-chart encoded construction supports."""
    import matplotlib.pyplot as plt

    _, state, source = diag._atlas_view(work)
    embeddings = diag._latent_embeddings(work, state)
    if not embeddings:
        print("[view_sample_set] no learned charts with latents to show", file=sys.stderr)
        return
    fig, axes = plt.subplots(1, len(embeddings),
                             figsize=(5 * len(embeddings), 5), squeeze=False)
    for chart, latent in enumerate(embeddings):
        diag._plot_latent(axes[0][chart], latent, chart)
    fig.suptitle(f"{case}: encoded construction supports; atlas from {source}")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None,
                        help="case name (optional if it can be inferred from --work)")
    parser.add_argument("--work", default=None,
                        help="dataset directory (default: <manifold_charts>/work/<case>)")
    parser.add_argument("--isolate", type=int, default=-1,
                        help="start focused on one chart index (-1 = all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="if more samples than this, randomly drop down to it")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --max-samples subsampling")
    parser.add_argument("--latent", action="store_true",
                        help="also open the per-chart latent-embedding window")
    parser.add_argument("--snapshots", nargs="?", const=True, default=None,
                        metavar="DIR",
                        help="step through competitive_cae prune-step snapshots with a "
                             "slider (default dir: <work>/prune_snapshots)")
    args = parser.parse_args()

    if not args.case and not args.work:
        parser.error("give --case, --work, or both")

    repo_root = os.path.dirname(os.path.dirname(_ROOT))
    roots = (os.getcwd(), _ROOT, repo_root)
    if args.work and os.path.isdir(args.work):
        work = args.work
    else:
        stem = args.work or resolve_case_name(args.case)
        candidates = [os.path.join(r, "work", stem) for r in roots]
        candidates += [os.path.join(r, stem) for r in roots]
        work = next((c for c in candidates if os.path.isdir(c)),
                    os.path.join(_ROOT, "work", stem))

    if args.case:
        case = resolve_case_name(args.case)
    else:
        stem = os.path.basename(os.path.normpath(args.work))
        case = _infer_case(work, stem)
        print(f"[view_sample_set] inferred --case {case} from work dir {stem!r}",
              file=sys.stderr)
    snapshots = None
    if args.snapshots is not None:
        snapshots = (os.path.join(work, "prune_snapshots")
                     if args.snapshots is True else args.snapshots)
    view = SampleSetView(case, work, isolate=args.isolate,
                         max_samples=args.max_samples, seed=args.seed,
                         snapshots=snapshots)
    if args.latent:
        show_latent(case, work)
    view.show()


if __name__ == "__main__":
    main()
