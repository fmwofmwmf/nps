"""Interactive, time-based view of Phase-1 competitive chart exploration.

Run after a competitive CAE fit configured with::

    --fitter-option exploration_snapshot_every=10

For example::

    python -m viewers.view_chart_exploration --work work/generic_sphere --phase 1
    python -m viewers.view_chart_exploration --work work/generic_sphere --phase 2
    python -m viewers.view_chart_exploration --work work/generic_sphere \
        --report competitive_cae_1k_report.json --phase 1

Every panel is drawn in the case's full physical display embedding.  The saved
frames contain learned AE weights and the fitter's overlapping P1 core-support
claims; this viewer does not use declared coordinates or topology.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _sample_poses(work: str) -> np.ndarray:
    for name in ("samples.npz", "new_dataset.npz"):
        path = os.path.join(work, name)
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data["q"], float)
    raise FileNotFoundError(f"no samples.npz or new_dataset.npz in {work}")


def load_exploration_snapshots(work: str, phase: int = 1) -> list[dict]:
    """Load saved exploration frames in epoch order, with an explicit schema."""
    import torch

    folder_name = "chart_exploration_snapshots" if int(phase) == 1 else "phase2_chart_exploration_snapshots"
    folder = os.path.join(work, folder_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"no exploration snapshots at {folder}; rerun with "
            f"--fitter-option {'exploration_snapshot_every' if int(phase) == 1 else 'phase2_exploration_snapshot_every'}=N")
    paths = sorted((entry.path for entry in os.scandir(folder)
                    if entry.name.startswith("epoch") and entry.name.endswith(".pt")),
                   key=lambda path: int(os.path.basename(path)[5:10]))
    frames = []
    for path in paths:
        frame = torch.load(path, map_location="cpu", weights_only=False)
        if frame.get("format") != "competitive_cae_exploration_v1":
            raise ValueError(f"unsupported exploration snapshot {path}")
        required = {"epoch", "chart_ids", "states", "encoder_kind", "n_bodies", "d"}
        missing = required.difference(frame)
        if missing:
            raise ValueError(f"exploration snapshot {path} missing {sorted(missing)}")
        frame["file"] = path
        if int(frame.get("phase", 0)) != int(phase):
            raise ValueError(f"exploration snapshot {path} belongs to phase {frame.get('phase')}, not {phase}")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no epoch*.pt exploration snapshots in {folder}")
    return frames


def load_exploration_snapshot_index(work: str, phase: int = 1,
                                   report_name: str = "competitive_cae_report.json") -> list[dict]:
    """Read lightweight snapshot metadata without loading every model state."""
    report_path = os.path.join(work, report_name)
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"no {report_name} in {work}")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    key = "exploration_snapshots" if int(phase) == 1 else "phase2_exploration_snapshots"
    section = report.get(key) or {}
    directory = section.get("dir")
    records = section.get("index") or []
    if not directory or not records:
        raise FileNotFoundError(f"no recorded Phase-{phase} exploration snapshots in {work}")
    folder = os.path.join(work, directory)
    result = []
    for record in records:
        path = os.path.join(folder, record["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"snapshot listed in report is missing: {path}")
        result.append({**record, "path": path})
    return result


def load_exploration_snapshot(record: dict, phase: int) -> dict:
    """Load one learned-state frame on demand and validate its phase."""
    import torch

    frame = torch.load(record["path"], map_location="cpu", weights_only=False)
    if frame.get("format") != "competitive_cae_exploration_v1":
        raise ValueError(f"unsupported exploration snapshot {record['path']}")
    if int(frame.get("phase", 0)) != int(phase):
        raise ValueError(f"exploration snapshot {record['path']} belongs to phase {frame.get('phase')}, not {phase}")
    required = {"ownership_weights", "core_owned_mask", "ownership_core_level"}
    missing = required.difference(frame)
    if missing:
        raise ValueError(
            f"exploration snapshot {record['path']} is missing saved fitter ownership {sorted(missing)}; "
            "rerun the fit with the current snapshot format")
    return frame


def _model_from_state(frame: dict, state: dict):
    import torch
    from atlas_rom.charts import physical_chart_autoencoder as pca

    centre = state["centre"].detach().cpu().numpy()
    scale = state["scale"].detach().cpu().numpy()
    width = int(state["encoder.0.weight"].shape[0])
    model = pca._module(int(frame["n_bodies"]), width=width, d=int(frame["d"]),
                        encoder_kind=frame["encoder_kind"],
                        rotation_head=pca.rotation_head_of(state, int(frame["n_bodies"])))(centre, scale).double()
    model.load_state_dict(state)
    model.eval()
    return model


def _scaled_poses(poses: np.ndarray, translation_scale: float) -> np.ndarray:
    """Packed (N, 12 * n_bodies) row-major 3x4 poses with translations rescaled.

    Rotation entries are unitless; the translation column is divided by one
    shared length so neither block dominates the display projection.
    """
    blocks = np.asarray(poses, float).reshape(len(poses), -1, 3, 4).copy()
    blocks[..., 3] /= translation_scale
    return blocks.reshape(len(poses), -1)


class DisplayEmbedding:
    """Display-only 3-D projection of the evidence poses (``pca`` or ``isomap``).

    Fit once on every sample pose so the view is fixed across snapshots.  It is
    never fed back into the pipeline.
    """

    def __init__(self, kind: str, poses: np.ndarray, edges: np.ndarray | None = None,
                 k_neighbours: int = 10):
        self.kind = kind
        translations = np.asarray(poses, float).reshape(len(poses), -1, 3, 4)[..., 3]
        spread = translations - translations.mean(axis=0)
        self.translation_scale = max(float(np.sqrt((spread ** 2).sum(axis=-1).mean())), 1e-12)
        x = _scaled_poses(poses, self.translation_scale)
        self.mean = x.mean(axis=0)
        x = x - self.mean
        if kind == "pca":
            _u, s, vt = np.linalg.svd(x, full_matrices=False)
            self.components = vt[:3]
            share = s ** 2 / max(float((s ** 2).sum()), 1e-300)
            print(f"[embed pca] explained variance top-3 {np.round(share[:3], 4).tolist()} "
                  f"(sum {share[:3].sum():.4f}); next-3 {np.round(share[3:6], 4).tolist()}")
            self.points = x @ self.components.T
        elif kind == "isomap":
            self.points = self._isomap(x, edges, k_neighbours)
        else:
            raise ValueError(f"unknown embedding {kind!r}")

    @staticmethod
    def _isomap(x: np.ndarray, edges: np.ndarray | None, k: int) -> np.ndarray:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components, shortest_path

        n = len(x)
        if edges is not None and len(edges):
            edges = np.asarray(edges, int)
            source = "witnessed sample edges"
        else:
            distances = np.linalg.norm(x[:, None] - x[None], axis=-1)
            neighbours = np.argsort(distances, axis=1)[:, 1:k + 1]
            edges = np.stack([np.repeat(np.arange(n), k), neighbours.reshape(-1)], axis=1)
            source = f"{k}-nearest neighbours"
        weights = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
        graph = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        n_components, labels = connected_components(graph, directed=False)
        if n_components != 1:
            sizes = np.bincount(labels).tolist()
            raise ValueError(f"isomap graph ({source}) is disconnected: "
                             f"{n_components} components of sizes {sizes}")
        geodesic = shortest_path(graph, directed=False)
        squared = geodesic ** 2
        centring = np.eye(n) - 1.0 / n
        gram = -0.5 * centring @ squared @ centring
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        positive = np.clip(values, 0.0, None)
        share = positive / max(float(positive.sum()), 1e-300)
        print(f"[embed isomap] graph: {source} ({len(edges)} edges); MDS eigen-share "
              f"top-6 {np.round(share[:6], 4).tolist()}")
        return vectors[:, :3] * np.sqrt(positive[:3])

    def transform(self, poses: np.ndarray) -> np.ndarray:
        if self.kind != "pca":
            raise ValueError("isomap has no out-of-sample map; --round-trip needs --embed native or pca")
        return (_scaled_poses(poses, self.translation_scale) - self.mean) @ self.components.T


def prepare_display_points(work: str, frames: list[dict], *, round_trip: bool) -> tuple[np.ndarray, list[dict], list[int]]:
    """Prepare a common full-space embedding and optional per-chart D(E(q))."""
    import torch
    from viewers.view_sample_set import _points

    poses = _sample_poses(work)
    q = torch.as_tensor(poses).double()
    case = str(frames[0]["case"])
    observed, _label, _identity = _points(case, poses)
    chart_ids = sorted({int(chart_id) for frame in frames for chart_id in frame["chart_ids"]})
    for frame in frames:
        reconstructions = {}
        if round_trip:
            with torch.no_grad():
                for chart_id, state in zip(frame["chart_ids"], frame["states"]):
                    model = _model_from_state(frame, state)
                    decoded = model.decode(model.encode(q)).cpu().numpy()
                    reconstructions[int(chart_id)] = _points(case, decoded)[0]
        frame["reconstructions"] = reconstructions
    return observed, frames, chart_ids


def _limits(observed: np.ndarray, frames: list[dict], chart_id: int) -> tuple[tuple[float, float], ...]:
    chunks = [observed] + [frame["reconstructions"][chart_id] for frame in frames
                           if chart_id in frame["reconstructions"]]
    points = np.concatenate(chunks, axis=0)
    def span(values):
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi - lo) * .06, 1e-6)
        return lo - pad, hi + pad
    return tuple(span(points[:, axis]) for axis in range(points.shape[1]))


class ExplorationView:
    def __init__(self, work: str, *, phase: int = 1, chart_id: int | None = None,
                 round_trip: bool = False, interval_ms: int = 350,
                 max_samples: int | None = 400, cache_size: int = 8,
                 hide_retired: bool = False, global_only: bool = False,
                 report_name: str = "competitive_cae_report.json",
                 embed: str = "native"):
        self.work = work
        self.phase = int(phase)
        self.round_trip = bool(round_trip)
        if embed == "isomap" and self.round_trip:
            raise ValueError("isomap has no out-of-sample map; --round-trip needs --embed native or pca")
        self.hide_retired = bool(hide_retired)
        self.global_only = bool(global_only)
        self.records = load_exploration_snapshot_index(work, self.phase, report_name)
        first = load_exploration_snapshot(self.records[0], self.phase)
        all_poses = _sample_poses(work)
        self.sample_indices = np.arange(len(all_poses), dtype=int)
        if max_samples is not None and len(all_poses) > int(max_samples):
            self.sample_indices = np.linspace(
                0, len(all_poses) - 1, int(max_samples), dtype=int)
        poses = all_poses[self.sample_indices]
        from viewers.view_sample_set import _points
        self.poses = poses
        self.case = str(first["case"])
        self.embedding = None
        if embed == "native":
            self.observed = _points(self.case, poses)[0]
        else:
            edges = None
            if embed == "isomap":
                for name in ("samples.npz", "new_dataset.npz"):
                    path = os.path.join(work, name)
                    if os.path.isfile(path):
                        with np.load(path, allow_pickle=False) as data:
                            if "edges" in data.files:
                                edges = np.asarray(data["edges"], int)
                        break
            self.embedding = DisplayEmbedding(embed, all_poses, edges)
            self.observed = self.embedding.points[self.sample_indices]
        self.embed = embed
        ids = sorted(int(chart_id) for chart_id in first["chart_ids"])
        if chart_id is not None and chart_id not in ids:
            raise ValueError(f"chart id {chart_id} is absent from these snapshots; choices: {ids}")
        self.chart_ids = [chart_id] if chart_id is not None else ids
        self.limits = {cid: _limits(self.observed, [], cid) for cid in self.chart_ids}
        self._cache_size = max(1, int(cache_size))
        self._frame_cache = OrderedDict()
        self.index = 0
        self.playing = False
        self.interval_ms = int(interval_ms)

    def _frame(self, index: int) -> dict:
        """Load/compute one frame, retaining only a small recent-frame cache."""
        cached = self._frame_cache.pop(index, None)
        if cached is not None:
            self._frame_cache[index] = cached
            return cached
        frame = load_exploration_snapshot(self.records[index], self.phase)
        reconstructions = {}
        if self.round_trip:
            import torch
            from viewers.view_sample_set import _points
            q = torch.as_tensor(self.poses).double()
            with torch.no_grad():
                for chart_id, state in zip(frame["chart_ids"], frame["states"]):
                    model = _model_from_state(frame, state)
                    decoded = model.decode(model.encode(q)).cpu().numpy()
                    reconstructions[int(chart_id)] = (
                        self.embedding.transform(decoded) if self.embedding is not None
                        else _points(self.case, decoded)[0])
        frame["reconstructions"] = reconstructions
        self._frame_cache[index] = frame
        while len(self._frame_cache) > self._cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def _draw(self):
        frame = self._frame(self.index)
        global_ax = self.axes[0]
        global_ax.set_visible(True)
        global_ax.clear()
        self._set_limits(global_ax, None)
        live_ids = [int(chart_id) for chart_id in frame["chart_ids"]]
        core_owned = np.asarray(frame["core_owned_mask"], dtype=bool)
        ownership = np.asarray(frame["ownership_weights"], dtype=float)
        if core_owned.ndim != 2 or ownership.shape != core_owned.shape:
            raise ValueError("snapshot ownership weights and core mask must have matching (poses, charts) shapes")
        if len(core_owned) <= int(self.sample_indices[-1]):
            raise ValueError(
                f"snapshot ownership has {len(core_owned)} poses, but needs index "
                f"{int(self.sample_indices[-1])}")
        core_owned = core_owned[self.sample_indices]
        if len(core_owned) != len(self.poses) or core_owned.shape[1] != len(live_ids):
            raise ValueError("snapshot core ownership does not align with displayed poses or live charts")
        if live_ids:
            for column, chart_id in enumerate(live_ids):
                mask = core_owned[:, column]
                points = (frame["reconstructions"][chart_id][mask] if self.round_trip
                          else self.observed[mask])
                self._scatter(global_ax, points,
                              color=__import__("matplotlib").colormaps["tab20"](chart_id % 20),
                              alpha=.72, size=10)
            global_ax.set_title("global: mapped core claims" if self.round_trip
                                else "global: learned core claims")
        else:
            self._missing_label(global_ax)
            global_ax.set_title("global")
        if not self.global_only:
            if self.hide_retired:
                # Keep Global in slot zero, but pack the live chart panels into the
                # remaining pre-created slots.  The figure/grid never changes size.
                for ax in self.axes[1:]:
                    ax.set_visible(False)
                panel_ids = [chart_id for chart_id in self.chart_ids if chart_id in live_ids]
                panel_axes = self.axes[1:1 + len(panel_ids)]
            else:
                panel_ids = list(self.chart_ids)
                panel_axes = self.axes[1:]
            for ax, chart_id in zip(panel_axes, panel_ids):
                ax.set_visible(True)
                ax.clear()
                points = frame["reconstructions"].get(chart_id)
                self._set_limits(ax, chart_id)
                if self.round_trip and points is None:
                    self._missing_label(ax)
                    ax.set_title(f"chart {chart_id}")
                    continue
                column = live_ids.index(chart_id) if chart_id in live_ids else None
                if column is None:
                    self._missing_label(ax)
                    ax.set_title(f"chart {chart_id}")
                    continue
                owned = core_owned[:, column]
                chart_points = points[owned] if self.round_trip else self.observed[owned]
                self._scatter(ax, chart_points, color=__import__("matplotlib").colormaps["tab20"](chart_id % 20),
                              alpha=.72, size=10)
                ax.set_title(f"chart {chart_id}: {'mapped core claim' if self.round_trip else 'core claim'}")
        suffix = " final" if frame.get("final") else ""
        mode = "mapped core claims D(E(q))" if self.round_trip else "learned core claims q"
        view = "" if self.embed == "native" else f" — {self.embed} view"
        self.fig.suptitle(f"Phase {self.phase} chart exploration — {mode}{view} — epoch {frame['epoch']}{suffix}")
        self.fig.canvas.draw_idle()

    @staticmethod
    def _missing_label(ax):
        if hasattr(ax, "text2D"):
            ax.text2D(.5, .5, "retired / unavailable", ha="center", va="center",
                      transform=ax.transAxes, color="#7f1d1d")
        else:
            ax.text(.5, .5, "retired / unavailable", ha="center", va="center",
                    transform=ax.transAxes, color="#7f1d1d")

    def _scatter(self, ax, points, *, color, alpha, size):
        if points.shape[1] == 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, color=color, alpha=alpha)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=size, color=color, alpha=alpha)

    def _set_limits(self, ax, chart_id):
        limits = self.limits[self.chart_ids[0] if chart_id is None else chart_id]
        ax.set(xlim=limits[0], ylim=limits[1])
        if self.observed.shape[1] == 3:
            ax.set(zlim=limits[2], box_aspect=tuple(hi - lo for lo, hi in limits))
        else:
            ax.set_aspect("equal")

    def _set_index(self, value):
        self.index = int(np.clip(round(value), 0, len(self.records) - 1))
        self.slider.eventson = False
        self.slider.set_val(self.index)
        self.slider.eventson = True
        self._draw()

    def _toggle(self, _event=None):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")

    def _tick(self):
        if self.playing:
            self._set_index((self.index + 1) % len(self.records))

    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        count = 1 if self.global_only else len(self.chart_ids) + 1
        cols = min(4, max(1, int(math.ceil(math.sqrt(count)))))
        rows = int(math.ceil(count / cols))
        self.fig = plt.figure(figsize=(4.5 * cols, 4.1 * rows))
        self.axes = [self.fig.add_subplot(rows, cols, index + 1,
                                          projection="3d" if self.observed.shape[1] == 3 else None)
                     for index in range(count)]
        self.fig.subplots_adjust(bottom=.17, top=.90)
        self.slider = Slider(self.fig.add_axes([.22, .07, .56, .03]), "snapshot", 0,
                             len(self.records) - 1, valinit=0, valstep=1)
        self.slider.on_changed(self._set_index)
        self.play_button = Button(self.fig.add_axes([.80, .055, .12, .06]), "Play")
        self.play_button.on_clicked(self._toggle)
        Button(self.fig.add_axes([.08, .055, .10, .06]), "‹").on_clicked(
            lambda _event: self._set_index(self.index - 1))
        Button(self.fig.add_axes([.93, .055, .05, .06]), "›").on_clicked(
            lambda _event: self._set_index(self.index + 1))
        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.show()

    def _on_key(self, event):
        if event.key == " ":
            self._toggle()
        elif event.key == "left":
            self._set_index(self.index - 1)
        elif event.key == "right":
            self._set_index(self.index + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="competitive CAE work directory")
    parser.add_argument("--report", default="competitive_cae_report.json",
                        help="report filename inside --work (default: competitive_cae_report.json)")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1,
                        help="separate snapshot sequence to view (default: 1)")
    parser.add_argument("--chart-id", type=int, default=None,
                        help="show one stable chart ID instead of the all-chart grid")
    parser.add_argument("--interval-ms", type=int, default=350, help="autoplay frame interval")
    parser.add_argument("--max-samples", type=int, default=400,
                        help="maximum evidence poses rendered per frame (default: 400; use 0 for all)")
    parser.add_argument("--round-trip", action="store_true",
                        help="display each learned claim at D(E(q)) instead of its original physical pose")
    parser.add_argument("--hide-retired", action="store_true",
                        help="compact live chart panels into the fixed grid while keeping Global top-left")
    parser.add_argument("--global-only", action="store_true",
                        help="show only the global, overlapping core-claim panel")
    parser.add_argument("--embed", choices=("native", "pca", "isomap"), default="native",
                        help="display placement: the case's own display hook (native), or a "
                             "display-only 3-D PCA / Isomap of the sample poses")
    args = parser.parse_args()
    ExplorationView(args.work, phase=args.phase, chart_id=args.chart_id,
                    round_trip=args.round_trip, interval_ms=args.interval_ms,
                    max_samples=(None if args.max_samples == 0 else args.max_samples),
                    hide_retired=args.hide_retired, global_only=args.global_only,
                    report_name=args.report, embed=args.embed).show()


if __name__ == "__main__":
    main()
