"""viewers/export_rom_blender.py -- headless ROM export for Blender: pick a case + ROM, a driver (what moves it), and optionally render.  This file only wires the pieces together:

    viewers/rom_sim.py        RomBase: the case-agnostic router/learned-field stepper (no pen, no rain)
    viewers/rom_drivers.py    --drive free | joints | pen: how the mechanism is controlled and when the clip ends
    viewers/rom_record.py     Recorder: poses + driver channels -> frames.npz / meshes.npz
    viewers/blender_import_rom.py   scene.blend (and an optional still / animation render) from that folder

Output directory ``--out`` (default work/blender/<case>_<drive>[_<curve stem>]): frames.npz, meshes.npz, scene.blend, report.json (with a timing block), and the render if asked.

A still is a one-frame clip: ``--seconds 0`` (the default for `free` and `joints`) records the settled start pose only.  ``--render still.png`` renders it (``--render-frame N`` picks another frame of a clip).

    # one still of the Stephenson link, crank at 1.2 rad, reverser at 0.3 rad, perspective camera
    python -m viewers.export_rom_blender --case stephenson_link --rom stephenson_link_polished --drive pose --joint crank=1.2 --joint reverser=0.3 --camera persp --render steph.png
    # one revolution of the crank
    python -m viewers.export_rom_blender --case stephenson_link --rom stephenson_link_polished --drive pose --joint crank=0:6.283 --joint reverser=0.3 --seconds 6
    # a ROM-free reference still of any case (declared geometry at the rest pose)
    python -m viewers.export_rom_blender --case open_bevel_differential --drive rest --render differential.png
    # pantograph pen tracing a curve with raining balls (previous behaviour of this script)
    python -m viewers.export_rom_blender --case pantograph_2d --rom pantograph_2d_polished --drive pen --curve my_curve.json
"""
from __future__ import annotations

import argparse
import datetime as dt_mod
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from atlas_rom.rom.loader import PatchAtlasROM  # noqa: E402
from viewers.rom_drivers import DRIVERS  # noqa: E402
from viewers.rom_record import Recorder  # noqa: E402


def progress(frac, t0, width=30):
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(width * frac))
    el = time.time() - t0
    eta = el * (1.0 - frac) / frac if frac > 1e-6 else float("nan")
    sys.stdout.write(f"\r[{'#' * n}{'-' * (width - n)}] {100 * frac:5.1f}%  elapsed {el:6.1f}s  eta {eta:6.1f}s")
    sys.stdout.flush()


def parse():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--drive", choices=sorted(DRIVERS), default="free")
    drive = pre.parse_known_args()[0].drive
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre])
    ap.add_argument("--case", required=True)
    ap.add_argument("--rom", default=None, help="roms/<name> package; not needed for --drive rest")
    ap.add_argument("--roms-dir", default=os.path.join(ROOT, "roms"))
    ap.add_argument("--seconds", type=float, default=0.0, help="recorded clip length in simulated seconds; 0 = a single frame (a still).  The pen driver ends by its own laps/--wait instead")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--gravity", type=float, default=None, help="gravity magnitude for the ROM runtime (default: the case's own); not wired for --drive pen")
    ap.add_argument("--fps", type=float, default=30.0, help="exported frame rate (steps per frame = round(1 / (fps * dt)))")
    ap.add_argument("--out", default=None)
    b = ap.add_argument_group("blender")
    b.add_argument("--no-blend", action="store_true", help="skip building scene.blend (arrays only)")
    b.add_argument("--blender", default=None, help="path to blender.exe (default: newest under C:/Program Files/Blender Foundation)")
    b.add_argument("--camera", choices=["auto", "ortho", "persp"], default="auto", help="auto: orthographic along -z when the mechanism is planar, perspective otherwise")
    b.add_argument("--render", default=None, metavar="PNG", help="render one frame to this PNG (relative paths land in the output directory)")
    b.add_argument("--render-frame", type=int, default=0, help="0-based exported frame to render (default 0)")
    b.add_argument("--fast", action="store_true", help="10x fewer Cycles samples, for iterating")
    b.add_argument("--case-colors", action="store_true", help="colour each body with the case's declared BodyDef.color instead of the cycling paper palette")
    b.add_argument("--group-colors", action="store_true", help="one colour per body group (the body-name suffix after '@', e.g. one leg of an array case)")
    DRIVERS[drive].add_args(ap)
    return ap.parse_args()


def main():
    args = parse()
    t_start, started, timing = time.time(), dt_mod.datetime.now().isoformat(timespec="seconds"), {}
    driver = DRIVERS[args.drive](args)
    stem = os.path.splitext(os.path.basename(args.curve))[0] if args.drive == "pen" else None
    out = args.out or os.path.join(ROOT, "work", "blender", "_".join(filter(None, [args.case, args.drive, stem])))
    os.makedirs(out, exist_ok=True)

    t = time.time()
    if driver.needs_rom and not args.rom:
        raise SystemExit(f"--drive {args.drive} needs --rom")
    rom = PatchAtlasROM.load(os.path.join(args.roms_dir, args.rom)) if driver.needs_rom else None
    sim = driver.make_sim(rom, args)
    timing["load_s"] = time.time() - t

    spf = max(1, int(round(1.0 / (args.fps * args.dt))))
    fps = 1.0 / (spf * args.dt)
    t = time.time()
    prepare = driver.prepare(sim, spf)
    timing["prepare_s"] = time.time() - t

    rec = Recorder(sim)
    t0, sim_t = time.time(), 0.0
    total = driver.duration() if hasattr(driver, "duration") else args.seconds
    rec.record(driver.extras(sim))
    while not driver.control(sim, sim_t):
        driver.advance(sim, spf, sim_t)
        if sim.error:
            print(f"\nsim stopped: {sim.error}")
            break
        sim_t += spf * args.dt
        rec.record(driver.extras(sim))
        progress(sim_t / max(total, 1e-9), t0)
    if len(rec) > 1:
        progress(1.0 if not sim.error else sim_t / max(total, 1e-9), t0)
        print()
    timing["sim_s"] = time.time() - t0

    rec.save(out, fps, args.dt, driver.static(sim))
    if args.render and not (0 <= args.render_frame < len(rec)):
        raise SystemExit(f"--render-frame {args.render_frame} is outside the {len(rec)} exported frames")

    png = None
    if not args.no_blend:
        exe = args.blender or next(iter(sorted(glob.glob("C:/Program Files/Blender Foundation/Blender */blender.exe"), reverse=True)), None)
        if exe is None:
            raise SystemExit("blender.exe not found; pass --blender or use --no-blend")
        cmd = [exe, "-b", "-P", os.path.join(HERE, "blender_import_rom.py"), "--", out, "--camera", args.camera]
        if args.fast:
            cmd.append("--fast")
        if args.case_colors:
            cmd.append("--case-colors")
        if args.group_colors:
            cmd.append("--group-colors")
        if args.render:
            png = args.render if os.path.isabs(args.render) else os.path.join(out, args.render)
            cmd += ["--render", png, "--frame", str(args.render_frame)]
        t = time.time()
        subprocess.run(cmd, check=True, timeout=1800)
        timing["blend_s"] = time.time() - t

    timing.update(started_at=started, finished_at=dt_mod.datetime.now().isoformat(timespec="seconds"), wall_seconds=time.time() - t_start)
    report = {"case": args.case, "rom": args.rom, "drive": args.drive, "args": vars(args), "frames": len(rec), "fps": fps, "steps_per_frame": spf, "sim_seconds": sim_t,
              "error": sim.error, "prepare": prepare, "render": png, "timing": timing, **driver.report(sim)}
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {out}: {len(rec)} frames @ {fps:.2f} fps" + (f", render {png}" if png else ""))


if __name__ == "__main__":
    main()
