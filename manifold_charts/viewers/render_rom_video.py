"""viewers/render_rom_video.py -- render a `viewers.export_rom_blender` output folder's scene.blend to an mp4.

    python -m viewers.render_rom_video work/blender/<export_dir> [--res 540] [--fast] [--crf 18]

Blender renders every frame headless (the scene's own Cycles setup from viewers/blender_import_rom.py, scaled from its
1080 px to --res) into <dir>/frames_<res>/f_####.png; ffmpeg then encodes them at the export's frame rate (frames.npz
`fps`) as H.264 yuv420p into <dir>/video_<res>.mp4.  Logs: <dir>/render_<res>.log, ffmpeg_<res>.log; report with a
timing block: <dir>/render_video_report.json.  Stale f_####.png from an earlier render of the same folder and
resolution are removed first so they cannot end up in the video.
"""
from __future__ import annotations

import argparse
import datetime as dt_mod
import glob
import json
import os
import shutil
import subprocess
import time

import numpy as np

SCENE_PX = 1080          # blender_import_rom.py renders 1080 x 1080


def find_blender(path=None):
    exe = path or next(iter(sorted(glob.glob("C:/Program Files/Blender Foundation/Blender */blender.exe"), reverse=True)), None)
    if exe is None:
        raise SystemExit("blender.exe not found; pass --blender")
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", help="a viewers.export_rom_blender output folder (scene.blend + frames.npz)")
    ap.add_argument("--res", type=int, default=540, help="output width/height in pixels (the scene is 1080 x 1080)")
    ap.add_argument("--fast", action="store_true", help="10x fewer Cycles samples, for iterating")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, larger)")
    ap.add_argument("--blender", default=None, help="path to blender.exe (default: newest under C:/Program Files/Blender Foundation)")
    ap.add_argument("--ffmpeg", default=None, help="path to ffmpeg (default: on PATH)")
    a = ap.parse_args()

    d = os.path.abspath(a.export_dir)
    blend, frames_npz = os.path.join(d, "scene.blend"), os.path.join(d, "frames.npz")
    for p in (blend, frames_npz):
        if not os.path.isfile(p):
            raise SystemExit(f"{p} not found; run viewers.export_rom_blender first")
    fps = float(np.load(frames_npz)["fps"])
    ffmpeg = a.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH; pass --ffmpeg")
    exe = find_blender(a.blender)

    t_start, started, timing = time.time(), dt_mod.datetime.now().isoformat(timespec="seconds"), {}
    out_frames = os.path.join(d, f"frames_{a.res}")
    os.makedirs(out_frames, exist_ok=True)
    for f in glob.glob(os.path.join(out_frames, "f_*.png")):
        os.remove(f)

    pct = max(1, min(100, int(round(100.0 * a.res / SCENE_PX))))
    expr = ("import bpy; s = bpy.context.scene; s.render.resolution_percentage = %d" % pct
            + ("; s.cycles.samples = max(1, s.cycles.samples // 10)" if a.fast else ""))
    cmd = [exe, "-b", blend, "--python-expr", expr, "-o", os.path.join(out_frames, "f_####"), "-F", "PNG", "-a"]
    t = time.time()
    with open(os.path.join(d, f"render_{a.res}.log"), "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    timing["render_s"] = time.time() - t
    pngs = sorted(glob.glob(os.path.join(out_frames, "f_*.png")))
    if r.returncode or not pngs:
        raise SystemExit(f"blender failed (exit {r.returncode}, {len(pngs)} frames); see render_{a.res}.log")

    video = os.path.join(d, f"video_{a.res}.mp4")
    cmd = [ffmpeg, "-y", "-framerate", f"{fps:.6f}", "-start_number", "1", "-i", os.path.join(out_frames, "f_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", video]
    t = time.time()
    with open(os.path.join(d, f"ffmpeg_{a.res}.log"), "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    timing["encode_s"] = time.time() - t
    if r.returncode or not os.path.isfile(video):
        raise SystemExit(f"ffmpeg failed (exit {r.returncode}); see ffmpeg_{a.res}.log")

    timing.update(started_at=started, finished_at=dt_mod.datetime.now().isoformat(timespec="seconds"),
                  wall_seconds=time.time() - t_start)
    report = {"export_dir": d, "video": video, "frames": len(pngs), "fps": fps, "seconds": len(pngs) / fps,
              "resolution": [int(SCENE_PX * pct / 100)] * 2, "fast": bool(a.fast), "crf": a.crf, "timing": timing}
    with open(os.path.join(d, "render_video_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {video}: {len(pngs)} frames @ {fps:.2f} fps ({len(pngs) / fps:.2f} s) [{timing['wall_seconds']:.0f}s]")


if __name__ == "__main__":
    main()
