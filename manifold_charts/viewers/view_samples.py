"""Interactive inspector for the physical poses used to train chart AEs.

Examples::

    python -m viewers.view_samples --case five_bar
    python -m viewers.view_samples --case five_bar --dataset new_dataset.npz
    python -m viewers.view_samples --case jansen --work-dir work/jansen_live

With only ``--case``, the helper lists compatible pose datasets in
``work/<case>/``.  Pass one of those names through ``--dataset`` to open it.
Every accepted ``.npz`` artifact has a ``q`` array shaped ``(N, 12 *
n_driven_bodies)``; this includes ``new_dataset.npz``, the direct physical-AE
input.  The viewer never reconstructs poses through a decoder: every displayed
configuration is an exact saved sample.

When the archive also stores per-sample tangent ``frames`` shaped ``(N, 12 *
n_driven_bodies, d)`` (``(N, 12 * n)`` for d = 1), one offset slider per
tangent column appears, plus a reset button.  The displayed pose is then the
sample plus the weighted sum of its tangents, each body's rotation
re-orthonormalized for display only: an unprojected linear offset, never a
simulated pose.  The exact sample stays visible as a translucent ghost.  An
optional ``frame_values`` ``(N, d)`` array (e.g. singular values) is shown next
to each slider.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from atlas_rom.cases import load_spec  # noqa: E402
from viewers.mesh_view import MeshAdapter  # noqa: E402


def _load_dataset(path: str, n_bodies: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load poses plus scalar per-sample metadata suitable for the UI."""
    q, metadata, _frames, _values = _load_dataset_with_frames(path, n_bodies)
    return q, metadata


def _load_dataset_with_frames(path: str, n_bodies: int):
    """Poses, scalar metadata, and optional tangent frames ``(N, 12n, d)`` and
    their per-column values ``(N, d)``; frames of any other shape are ignored."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "q" not in data.files:
            raise ValueError(f"{path} has no 'q' pose array")
        q = np.asarray(data["q"], float)
        expected_width = 12 * n_bodies
        if q.ndim != 2 or q.shape[1] != expected_width:
            raise ValueError(
                f"{path}: q has shape {q.shape}; expected (N, {expected_width}) "
                f"for {n_bodies} driven bodies")
        metadata = {}
        for name in data.files:
            value = np.asarray(data[name])
            if name != "q" and value.ndim == 1 and len(value) == len(q):
                metadata[name] = value
        frames = values = None
        if "frames" in data.files:
            frames = np.asarray(data["frames"], float)
            if frames.ndim == 2:
                frames = frames[:, :, None]
            if frames.ndim != 3 or frames.shape[:2] != q.shape or frames.shape[2] == 0:
                frames = None
        if frames is not None and "frame_values" in data.files:
            values = np.asarray(data["frame_values"], float)
            if values.shape != (len(q), frames.shape[2]):
                values = None
    if not len(q):
        raise ValueError(f"{path}: q contains no samples")
    return q, metadata, frames, values


def _orthonormalized(transforms: np.ndarray) -> np.ndarray:
    """Nearest rotation per body block (display only)."""
    out = np.array(transforms, float, copy=True)
    for block in out:
        left, _s, right = np.linalg.svd(block[:, :3])
        if np.linalg.det(left @ right) < 0:
            left[:, -1] *= -1.0
        block[:, :3] = left @ right
    return out


def _body_motion(offset_pose: np.ndarray, sample: np.ndarray) -> list[tuple[float, float]]:
    """Per-body (translation distance, rotation degrees) of the offset pose."""
    rows = []
    for a, b in zip(offset_pose, sample):
        cos = (np.trace(b[:, :3].T @ a[:, :3]) - 1.0) / 2.0
        rows.append((float(np.linalg.norm(a[:, 3] - b[:, 3])),
                     float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))))
    return rows


def compatible_datasets(work: str, n_bodies: int) -> list[tuple[str, tuple[int, int]]]:
    """Return top-level pose archives in a work directory, ordered by name."""
    if not os.path.isdir(work):
        return []
    found = []
    for entry in sorted(os.scandir(work), key=lambda item: item.name.lower()):
        if not entry.is_file() or not entry.name.endswith(".npz"):
            continue
        try:
            q, _ = _load_dataset(entry.path, n_bodies)
        except (OSError, ValueError, KeyError):
            continue
        found.append((entry.name, tuple(q.shape)))
    return found


def view_samples(case_name: str, dataset_path: str, *, start: int = 0,
                 autoplay: bool = False, samples_per_frame: int = 1,
                 tangent_scale: float = 0.05) -> None:
    """Open a Polyscope scrubber for saved physical sample poses."""
    case = load_spec(case_name)
    n_bodies = len(case.mechanism.driven_bodies)
    q, metadata, frames, frame_values = _load_dataset_with_frames(dataset_path, n_bodies)
    transforms = q.reshape(len(q), n_bodies, 3, 4)
    index = int(np.clip(start, 0, len(q) - 1))
    n_tangents = 0 if frames is None else frames.shape[2]
    body_names = [body.name for body in case.mechanism.driven_bodies]

    import polyscope as ps
    from polyscope import imgui

    adapter = MeshAdapter(case.mechanism)
    ghost = MeshAdapter(case.mechanism) if n_tangents else None
    ps.init()
    ps.set_up_dir(adapter.up_dir)
    ps.set_ground_plane_mode("shadow_only")
    adapter.register(ps, transforms[index])
    if ghost is not None:
        ghost.register(ps, transforms[index], name_prefix="sample/", include_static=False,
                       color_override=(0.75, 0.75, 0.78))
        for mesh in ghost.meshes:
            mesh.set_transparency(0.3)

    state = {"index": index, "playing": bool(autoplay),
             "samples_per_frame": int(max(1, samples_per_frame)),
             "offsets": [0.0] * n_tangents, "ghost": True}

    def displayed_pose() -> np.ndarray:
        sample = transforms[state["index"]]
        if not n_tangents or not any(state["offsets"]):
            return sample
        delta = frames[state["index"]] @ np.asarray(state["offsets"], float)
        return _orthonormalized(sample + delta.reshape(n_bodies, 3, 4))

    def refresh() -> None:
        adapter.update(displayed_pose())
        if ghost is not None:
            ghost.update(transforms[state["index"]])
            for mesh in ghost.meshes:
                mesh.set_enabled(bool(state["ghost"]) and any(state["offsets"]))

    def set_index(value: int) -> None:
        state["index"] = int(np.clip(value, 0, len(q) - 1))
        refresh()

    def callback() -> None:
        changed, value = imgui.SliderInt("sample", state["index"], 0, len(q) - 1)
        if changed:
            set_index(value)
        if imgui.Button("Pause" if state["playing"] else "Play"):
            state["playing"] = not state["playing"]
        imgui.SameLine()
        if imgui.Button("Previous"):
            set_index(state["index"] - 1)
        imgui.SameLine()
        if imgui.Button("Next"):
            set_index(state["index"] + 1)
        changed, value = imgui.SliderInt(
            "samples/frame", state["samples_per_frame"], 1, min(100, len(q)))
        if changed:
            state["samples_per_frame"] = int(value)

        if state["playing"]:
            set_index((state["index"] + state["samples_per_frame"]) % len(q))

        imgui.Separator()
        imgui.Text(f"case={case.name}  sample={state['index'] + 1}/{len(q)}")
        imgui.Text(f"dataset={os.path.basename(dataset_path)}")
        for name, values in metadata.items():
            value = values[state["index"]]
            if np.issubdtype(values.dtype, np.integer):
                text = str(int(value))
            elif np.issubdtype(values.dtype, np.floating):
                text = f"{float(value):.6g}"
            else:
                text = str(value)
            imgui.Text(f"{name}={text}")

        if n_tangents:
            imgui.Separator()
            imgui.Text(f"tangent offsets (linear, unprojected; +/-{tangent_scale:g})")
            for k in range(n_tangents):
                label = f"t{k}"
                if frame_values is not None:
                    label += f" ({frame_values[state['index'], k]:.3g})"
                changed, value = imgui.SliderFloat(label, state["offsets"][k],
                                                   -tangent_scale, tangent_scale)
                if changed:
                    state["offsets"][k] = float(value)
                    refresh()
            if imgui.Button("Reset tangents"):
                state["offsets"] = [0.0] * n_tangents
                refresh()
            imgui.SameLine()
            changed, value = imgui.Checkbox("show sample ghost", state["ghost"])
            if changed:
                state["ghost"] = bool(value)
                refresh()
            if any(state["offsets"]):
                motion = _body_motion(displayed_pose(), transforms[state["index"]])
                for name, (dist, deg) in zip(body_names, motion):
                    imgui.Text(f"  {name}: move {dist:.4g}  turn {deg:.3g} deg")

    ps.set_user_callback(callback)
    print(f"sample viewer: case={case.name}, samples={len(q)}, dataset={dataset_path}")
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="cases/<case>.py spec name")
    parser.add_argument("--work-dir", default=None,
                        help="dataset directory (default: work/<case>)")
    parser.add_argument("--dataset", default=None,
                        help="compatible .npz filename or absolute path to view")
    parser.add_argument("--start", type=int, default=0, help="initial zero-based sample index")
    parser.add_argument("--play", action="store_true", help="start playback immediately")
    parser.add_argument("--samples-per-frame", type=int, default=1)
    parser.add_argument("--tangent-scale", type=float, default=0.05,
                        help="range of each tangent offset slider (packed pose units)")
    args = parser.parse_args()

    case = load_spec(args.case)
    work = args.work_dir or os.path.join(_ROOT, "work", case.name)
    if args.dataset is None:
        datasets = compatible_datasets(work, len(case.mechanism.driven_bodies))
        print(f"compatible pose datasets for case={case.name} in {work}:")
        if datasets:
            for name, shape in datasets:
                print(f"  {name:<52} q={shape}")
            print("\nOpen one with: python -m viewers.view_samples "
                  f"--case {case.name} --dataset <name>")
        else:
            print("  (none found)")
        return
    dataset_path = (args.dataset if os.path.isabs(args.dataset)
                    else os.path.join(work, args.dataset))
    view_samples(case.name, dataset_path, start=args.start, autoplay=args.play,
                 samples_per_frame=args.samples_per_frame,
                 tangent_scale=args.tangent_scale)


if __name__ == "__main__":
    main()
