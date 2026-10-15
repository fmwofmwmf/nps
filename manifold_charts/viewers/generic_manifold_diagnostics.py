"""Offline visualization for a two-dimensional SDF sampling run.

This is evaluation-only: it reads saved physical pose observations and optional
completed chart artifacts.  It never supplies labels, topology, memberships, or
geometry to sampling, training, certification, or runtime.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _artifact(work: str):
    import torch

    path = os.path.join(work, "patch_atlas.pt")
    if not os.path.exists(path):
        candidates = sorted((entry.path for entry in os.scandir(work)
                             if entry.name.startswith("patch_atlas_dev_") and entry.name.endswith(".pt")),
                            key=os.path.getmtime)
        candidates += [entry.path for entry in os.scandir(work)
                       if entry.name.endswith(".pt")]
        path = max(candidates, key=os.path.getmtime) if candidates else path
    state = torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None
    return state.get("chart_atlas", state) if state is not None else None


def _load_json(path: str) -> dict:
    return json.loads(open(path, encoding="utf-8").read()) if os.path.exists(path) else {}


def _atlas_view(work: str):
    """Prefer the latest construction atlas over any older runtime artifact."""
    competitive = sorted((entry.path for entry in os.scandir(work)
                          if entry.name.startswith("competitive_") and entry.name.endswith(".pt")),
                         key=os.path.getmtime)
    if competitive:
        import torch
        report_path = os.path.splitext(competitive[-1])[0] + "_report.json"
        state = torch.load(competitive[-1], map_location="cpu", weights_only=False)
        return _load_json(report_path), state.get("chart_atlas", state), os.path.basename(competitive[-1])
    for name in ("trained_construction_report.json", "local_map_growth_diagnostic.json"):
        report = _load_json(os.path.join(work, name))
        membership = report.get("atlas_membership")
        if membership and membership.get("cores"):
            return report, {
                "cores": membership["cores"],
                "patches": membership.get("supports", membership["cores"]),
                "contacts": membership.get("contacts", []),
            }, name
    report = _load_json(os.path.join(work, "patch_atlas_report.json"))
    return report, _artifact(work), "patch_atlas.pt"


def _visual_uvs(case: str):
    import importlib
    return getattr(importlib.import_module(f"atlas_rom.cases.{case}"), "VISUAL_UVS", None)


def _unit_visual_uvs(visual_uvs: dict | None) -> bool:
    return bool(visual_uvs and visual_uvs.get("kind") in
                ("sphere_equirectangular", "torus_ring_tube"))


def _visual_points(case: str, q: np.ndarray, spec, visual_uvs: dict | None) -> np.ndarray:
    """Evaluate a case-declared display parametrization on observed poses."""
    if visual_uvs and visual_uvs.get("kind") == "driver_coordinates":
        from atlas_rom.spec.coordinates import CoordinateMap

        names = tuple(visual_uvs.get("coordinates", ()))
        by_name = {coordinate.name: index for index, coordinate in enumerate(spec.coordinates)}
        missing = [name for name in names if name not in by_name]
        if len(names) != 2 or missing:
            raise ValueError(f"{case}.VISUAL_UVS driver coordinates must name two "
                             f"declared coordinates; missing={missing}")
        values = np.asarray(CoordinateMap(spec)(q), float)
        return values[:, [by_name[name] for name in names]]

    translated = next((coordinate for coordinate in spec.coordinates
                       if coordinate.source == "body_translation"), None)
    if translated is None:
        raise ValueError(f"{case}: no body-translation observation is available for a 2D plot")
    body = visual_uvs.get("body") if visual_uvs else translated.args["body"]
    offset = 12 * spec.mechanism.body_order.index(body)
    xyz = q[:, offset + np.asarray((3, 7, 11))]
    if visual_uvs and visual_uvs.get("kind") == "sphere_equirectangular":
        unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True).clip(1e-12)
        return np.column_stack(((np.arctan2(unit[:, 1], unit[:, 0]) + np.pi) / (2 * np.pi),
                                np.arccos(np.clip(unit[:, 2], -1, 1)) / np.pi))
    if visual_uvs and visual_uvs.get("kind") == "torus_ring_tube":
        ring = float(visual_uvs["ring_radius"])
        ring_angle = np.arctan2(xyz[:, 1], xyz[:, 0])
        tube_angle = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]) - ring)
        return np.column_stack(((ring_angle + np.pi) / (2 * np.pi),
                                (tube_angle + np.pi) / (2 * np.pi)))
    return xyz[:, :2]


def _points(work: str, case: str) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Extract the observed XY translation of the case's first translated body."""
    from atlas_rom.cases import load_spec

    sample_path = os.path.join(work, "samples.npz")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(work, "new_dataset.npz")
    with np.load(sample_path, allow_pickle=False) as data:
        q = np.asarray(data["q"], float)
        edges = np.asarray(data["edges"] if "edges" in data else np.empty((0, 2)), int).reshape(-1, 2)
    spec = load_spec(case)
    visual_uvs = _visual_uvs(case)
    points = _visual_points(case, q, spec, visual_uvs)
    return points, edges, visual_uvs


def _limits(points: np.ndarray) -> tuple[float, float, float, float]:
    lo, hi = points.min(0), points.max(0)
    margin = np.maximum(.15 * (hi - lo), .25)
    return lo[0] - margin[0], hi[0] + margin[0], lo[1] - margin[1], hi[1] + margin[1]


def _plot_sdf(ax, sdf_system, limits: tuple[float, float, float, float]) -> None:
    if sdf_system is None or int(sdf_system.dimension) != 2:
        raise ValueError("diagnostic requires a case with a two-dimensional SDF system")
    xmin, xmax, ymin, ymax = limits
    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)
    xx, yy = np.meshgrid(x, y)
    phi = np.asarray([sdf_system.sdf(point)
                      for point in np.column_stack((xx.ravel(), yy.ravel()))], float).reshape(xx.shape)
    ax.contourf(xx, yy, phi <= 0.0, levels=[-1, .5, 1], colors=["#eceff1", "#e8f5e9"], alpha=.75)
    ax.contour(xx, yy, phi, levels=[0.0], colors="#263238", linewidths=1.3)


def _plot_visual_uv_domain(ax):
    ax.add_patch(__import__("matplotlib").patches.Rectangle((0, 0), 1, 1, facecolor="#e8f5e9", alpha=.75))
    ax.axvline(0, color="#263238", lw=1.3); ax.axvline(1, color="#263238", lw=1.3)
    ax.set(xlim=(0, 1), ylim=(1, 0), xlabel="visual longitude", ylabel="visual latitude")


def _plot_charts(ax, points: np.ndarray, state, report: dict) -> int:
    if state is None:
        return 0
    cores = state.get("cores", [])
    supports = state.get("patches", [])
    palette = __import__("matplotlib").colormaps["tab20"]
    for chart, support in enumerate(supports):
        color = palette(chart % 20)
        ax.scatter(points[np.asarray(support, int), 0], points[np.asarray(support, int), 1],
                   s=18, color=color, alpha=1.0, zorder=3)
    for chart, core in enumerate(cores):
        color = palette(chart % 20)
        passed = (chart < len(report.get("patches", []))
                  and report["patches"][chart].get("pass", False))
        if not report.get("patches"):
            passed = report.get("status") == "pass"
        ax.scatter(points[np.asarray(core, int), 0], points[np.asarray(core, int), 1],
                   s=24, color=color, marker="o" if passed else "x", alpha=.9, zorder=4)
    return len(cores)


def _plot_contacts(ax, points: np.ndarray, state) -> int:
    """Highlight the intentional, admission-certified physical strips."""
    if state is None:
        return 0
    contacts = state.get("contacts", [])
    palette = __import__("matplotlib").colormaps["tab20"]
    for contact in contacts:
        strip = np.asarray(contact.get("strip_vertices", []), int)
        if not len(strip):
            continue
        charts = contact.get("charts", [0, 0])
        color = palette(int(charts[-1]) % 20)
        ax.scatter(points[strip, 0], points[strip, 1], s=42, marker="D",
                   facecolors=color, edgecolors="#111827", linewidths=.65,
                   alpha=.9, zorder=6)
    return len(contacts)


def _support_values(case: str, state, visual_uvs, limits):
    """Evaluate saved learned support fields in display space only."""
    if state is None or not state.get("states") or not state.get("support_fields"):
        return None
    import torch
    from atlas_rom.charts import physical_chart_autoencoder as pca
    from atlas_rom.cases import load_sdf_system
    from atlas_rom.rom import learned

    system = load_sdf_system(case)
    if system is None or system.observe is None:
        return None
    xmin, xmax, ymin, ymax = limits
    x, y = np.linspace(xmin, xmax, 240), np.linspace(ymin, ymax, 240)
    xx, yy = np.meshgrid(x, y)
    if visual_uvs and visual_uvs.get("kind") == "sphere_equirectangular":
        longitude, latitude = 2 * np.pi * xx - np.pi, np.pi * yy
        states = 2.0 * np.stack((np.cos(longitude) * np.sin(latitude),
                                 np.sin(longitude) * np.sin(latitude),
                                 np.cos(latitude)), axis=-1)
    elif visual_uvs and visual_uvs.get("kind") == "torus_ring_tube":
        ring, tube = float(visual_uvs["ring_radius"]), float(visual_uvs["tube_radius"])
        ring_angle, tube_angle = 2 * np.pi * xx - np.pi, 2 * np.pi * yy - np.pi
        radius = ring + tube * np.cos(tube_angle)
        states = np.stack((radius * np.cos(ring_angle), radius * np.sin(ring_angle),
                           tube * np.sin(tube_angle)), axis=-1)
    elif int(system.dimension) == 2:
        states = np.stack((xx, yy), axis=-1)
    else:
        return None
    q = np.asarray([system.observe(point) for point in states.reshape(-1, states.shape[-1])], float)
    values = []
    for chart, (model_state, support_state) in enumerate(zip(state["states"], state["support_fields"])):
        width = model_state["encoder.0.weight"].shape[0]
        encoder_biases = [value for key, value in model_state.items()
                          if key.startswith("encoder.") and key.endswith(".bias")]
        d = encoder_biases[-1].shape[0]
        n_bodies = int(model_state["encoder.0.weight"].shape[1] // 9)
        model = pca._module(n_bodies, width=width, d=d,
                            encoder_kind=state.get("encoder_kind", "current"))(
            np.zeros_like(model_state["centre"].numpy()), np.ones_like(model_state["scale"].numpy())).double()
        model.load_state_dict(model_state); model.eval()
        support = learned.load_scalar(support_state)
        with torch.no_grad():
            values.append(support(model.encode(torch.as_tensor(q).double())).numpy().reshape(xx.shape))
    return xx, yy, values


def _support_isolines(ax, support_grid, *, chart: int | None = None, shade: bool = False) -> int:
    if support_grid is None:
        return 0
    xx, yy, values = support_grid
    palette = __import__("matplotlib").colormaps["tab20"]
    charts = range(len(values)) if chart is None else (chart,)
    for index in charts:
        color = palette(index % 20)
        if shade:
            ax.contourf(xx, yy, values[index], levels=[-np.inf, 0.0, np.inf],
                        colors=["none", color], alpha=.22, zorder=1.5)
        ax.contour(xx, yy, values[index], levels=[0.0], colors=[color],
                   linewidths=1.8, linestyles="--", zorder=5)
    return len(tuple(charts))


def _latent_embeddings(work: str, state):
    """Encode the exact displayed construction supports for each saved chart."""
    if state is None or not state.get("states"):
        return []
    import torch
    from atlas_rom.charts import physical_chart_autoencoder as pca

    sample_path = os.path.join(work, "samples.npz")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(work, "new_dataset.npz")
    with np.load(sample_path, allow_pickle=False) as data:
        q = np.asarray(data["q"], float)
    result = []
    for chart, model_state in enumerate(state["states"]):
        width = model_state["encoder.0.weight"].shape[0]
        encoder_biases = [value for key, value in model_state.items()
                          if key.startswith("encoder.") and key.endswith(".bias")]
        d = encoder_biases[-1].shape[0]
        n_bodies = int(model_state["encoder.0.weight"].shape[1] // 9)
        model = pca._module(n_bodies, width=width, d=d,
                            encoder_kind=state.get("encoder_kind", "current"))(
            np.zeros_like(model_state["centre"].numpy()), np.ones_like(model_state["scale"].numpy())).double()
        model.load_state_dict(model_state); model.eval()
        support = np.asarray(state.get("patches", [])[chart], int)
        support = support[(support >= 0) & (support < len(q))]
        with torch.no_grad():
            result.append(model.encode(torch.as_tensor(q[support]).double()).numpy() if len(support)
                          else np.empty((0, d)))
    return result


def _plot_latent(ax, embedding: np.ndarray, chart: int):
    color = __import__("matplotlib").colormaps["tab20"](chart % 20)
    if embedding.shape[1] == 1:
        ax.scatter(embedding[:, 0], np.zeros(len(embedding)), s=18, color=color, alpha=.85)
        ax.set(xlabel="latent 0", yticks=[])
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=18, color=color, alpha=.85)
        ax.set(xlabel="latent 0", ylabel="latent 1", aspect="equal")
    ax.set_title(f"chart {chart}: encoded construction support")


def _draw_samples(ax, points, edges, visual_uvs):
    for i, j in edges:
        if not (visual_uvs and abs(points[i, 0] - points[j, 0]) > .5):
            ax.plot(points[[i, j], 0], points[[i, j], 1], color="#90a4ae", lw=.35, alpha=.45, zorder=1)
    ax.scatter(points[:, 0], points[:, 1], s=9, c="#455a64", alpha=.65, zorder=2)


def render(case: str, work: str, *, output: str | None = None) -> str:
    """Render SDF isolines, physical samples, witnessed edges, and optional charts."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from atlas_rom.cases import load_sdf_system

    points, edges, visual_uvs = _points(work, case)
    report, state, source = _atlas_view(work)
    fig, ax = plt.subplots(figsize=(9, 8))
    limits = _limits(points)
    if _unit_visual_uvs(visual_uvs):
        _plot_visual_uv_domain(ax)
        limits = (0, 1, 1, 0)
    else:
        _plot_sdf(ax, load_sdf_system(case), limits)
    _draw_samples(ax, points, edges, visual_uvs)
    chart_count = _plot_charts(ax, points, state, report)
    contact_count = _plot_contacts(ax, points, state)
    support_grid = _support_values(case, state, visual_uvs, limits)
    support_count = _support_isolines(ax, support_grid)
    ax.set(aspect="equal", xlim=limits[:2], ylim=limits[2:])
    domain = "visual UV domain" if _unit_visual_uvs(visual_uvs) else \
             "driver-coordinate domain" if visual_uvs else "SDF isoline"
    ax.set_title(f"{case}: {domain}, {len(points)} samples, {len(edges)} witnessed edges, "
                 f"{chart_count} charts, {contact_count} certified contacts\n{source}")
    fig.tight_layout()
    output = output or os.path.join(work, f"{case}_sdf_samples.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def render_per_chart(case: str, work: str, *, output: str | None = None) -> str:
    """Render one shaded learned-support panel per chart plus an unshaded reference."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from atlas_rom.cases import load_sdf_system

    points, edges, visual_uvs = _points(work, case)
    report, state, source = _atlas_view(work)
    limits = (0, 1, 1, 0) if _unit_visual_uvs(visual_uvs) else _limits(points)
    support_grid = _support_values(case, state, visual_uvs, limits)
    chart_count = (len(support_grid[2]) if support_grid is not None
                   else len(state.get("cores", [])) if state is not None else 0)
    fig, axes = plt.subplots(1, chart_count + 1, figsize=(6 * (chart_count + 1), 6), squeeze=False)
    domain = "visual UV domain" if _unit_visual_uvs(visual_uvs) else \
             "driver-coordinate domain" if visual_uvs else "SDF isoline"
    for index, ax in enumerate(axes[0]):
        if _unit_visual_uvs(visual_uvs):
            _plot_visual_uv_domain(ax)
        else:
            _plot_sdf(ax, load_sdf_system(case), limits)
        _draw_samples(ax, points, edges, visual_uvs)
        if index == 0:
            _plot_charts(ax, points, state, report)
            _plot_contacts(ax, points, state)
            _support_isolines(ax, support_grid)
            ax.set_title("all charts and certified contact strips")
        else:
            chart = index - 1
            _support_isolines(ax, support_grid, chart=chart, shade=True)
            support = np.asarray(state.get("patches", [])[chart], int)
            core = np.asarray(state.get("cores", [])[chart], int)
            color = __import__("matplotlib").colormaps["tab20"](chart % 20)
            ax.scatter(points[support, 0], points[support, 1], s=22, color=color,
                       alpha=1.0, zorder=3)
            ax.scatter(points[core, 0], points[core, 1], s=24, color=color, alpha=.9, zorder=4)
            ax.set_title(f"chart {chart}: core and construction support")
        ax.set(aspect="equal", xlim=limits[:2], ylim=limits[2:],
               xlabel="visual longitude" if _unit_visual_uvs(visual_uvs)
               else "driver coordinate 0" if visual_uvs else "x",
               ylabel="visual latitude" if _unit_visual_uvs(visual_uvs)
               else "driver coordinate 1" if visual_uvs else "y")
    fig.suptitle(f"{case}: {domain}; atlas from {source}")
    fig.tight_layout()
    output = output or os.path.join(work, f"{case}_support_by_chart.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def render_with_latent(case: str, work: str, *, output: str | None = None) -> str:
    """Render the physical overview above color-aligned learned embeddings."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from atlas_rom.cases import load_sdf_system

    points, edges, visual_uvs = _points(work, case)
    report, state, source = _atlas_view(work)
    embeddings = _latent_embeddings(work, state)
    count = max(1, len(embeddings))
    fig = plt.figure(figsize=(5 * count, 10))
    grid = fig.add_gridspec(2, count, height_ratios=(1.4, 1.0))
    ax = fig.add_subplot(grid[0, :])
    limits = (0, 1, 1, 0) if _unit_visual_uvs(visual_uvs) else _limits(points)
    if _unit_visual_uvs(visual_uvs):
        _plot_visual_uv_domain(ax)
    else:
        _plot_sdf(ax, load_sdf_system(case), limits)
    _draw_samples(ax, points, edges, visual_uvs)
    chart_count = _plot_charts(ax, points, state, report)
    _plot_contacts(ax, points, state)
    _support_isolines(ax, _support_values(case, state, visual_uvs, limits))
    ax.set(aspect="equal", xlim=limits[:2], ylim=limits[2:],
           xlabel=("visual longitude" if _unit_visual_uvs(visual_uvs)
                   else "driver coordinate 0" if visual_uvs else "x"),
           ylabel=("visual latitude" if _unit_visual_uvs(visual_uvs)
                   else "driver coordinate 1" if visual_uvs else "y"),
           title=f"{case}: physical view; {chart_count} charts; atlas from {source}")
    for chart, latent in enumerate(embeddings):
        _plot_latent(fig.add_subplot(grid[1, chart]), latent, chart)
    fig.tight_layout()
    output = output or os.path.join(work, f"{case}_overview_latent.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render a two-dimensional SDF sampling diagnostic")
    parser.add_argument("case", help="case whose SDF and physical observations to render")
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-chart", action="store_true")
    parser.add_argument("--latent", action="store_true")
    args = parser.parse_args()
    renderer = render_with_latent if args.latent else render_per_chart if args.per_chart else render
    print(renderer(args.case, args.work, output=args.output))


if __name__ == "__main__":
    main()
