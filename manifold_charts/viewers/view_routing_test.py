"""view_routing_test.py — side-by-side chart-routing experiments (ROM only).

    python -m viewers.view_routing_test --rom twin_independent_pistons --tracks adv

Each requested track is an independent runtime over the same loaded staged ROM
(``patch_rom.routing_test_tracks``) and receives identical controls, timestep,
initial state, gravity and boundary settings.  ``run_patch_atlas.py`` and the
production runtime are untouched, and FullSpaceSim/libuipc is never imported.
The oracle is a constraint oracle (closure error of candidate continuations
from the same last-good state) and needs no external trajectory; ``--reference``
is optional and adds error readouts, a ghost column and control replay.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas_rom.rom.loader import PatchAtlasROM  # noqa: E402
from atlas_rom.rom.routing_test_tracks import Settings, load_references  # noqa: E402
from atlas_rom.rom.adv_router import AdvRouterTrack  # noqa: E402

# "adv" is the shipped learned advantage router (the support-field / closure-oracle tracks were retired).
TRACK_TYPES = {"adv": AdvRouterTrack}

TRACK_COLORS = {"adv": (0.90, 0.25, 0.25), "current": (0.25, 0.55, 0.95), "projected": (0.25, 0.80, 0.40),
                "oracle": (0.95, 0.60, 0.15), "oracle_enc": (0.85, 0.30, 0.75)}
REFERENCE_COLOR = (0.80, 0.80, 0.80)


def _verify_package(rom_path):
    """Warn if the staged artifact no longer matches its manifest hash."""
    with open(os.path.join(rom_path, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    want = manifest.get("source", {}).get("artifact_sha256")
    h = hashlib.sha256()
    with open(os.path.join(rom_path, manifest["artifact"]), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if want and h.hexdigest() != want:
        print(f"WARNING: {rom_path}: artifact sha256 differs from manifest")
    return manifest


def _layout(adapter, rom, n_columns):
    """Horizontal column offsets wide enough for the mechanism's whole range,
    estimated from decoded poses at anchors spread over every chart."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for chart, anchors in enumerate(rom.anchors):
        for u in anchors[:: max(len(anchors) // 8, 1)]:
            for verts in adapter.posed(rom.decode(chart, u)):
                lo, hi = np.minimum(lo, verts.min(0)), np.maximum(hi, verts.max(0))
    for body in adapter.static:
        lo, hi = np.minimum(lo, body["V"].min(0)), np.maximum(hi, body["V"].max(0))
    lateral = 1 if adapter.up_dir.startswith("x") else 0
    spacing = 1.25 * float(hi[lateral] - lo[lateral])
    offsets = []
    for i in range(n_columns):
        t = np.zeros(3)
        t[lateral] = i * spacing
        offsets.append(t)
    return offsets


def _translation(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def _fmt(x, spec="+.4f"):
    return "—" if x is None else format(x, spec)


def joint_pd_controls(rom, track, targets, w0, zeta, lim_mult, gcomp, manual):
    """Computed-torque PD in JOINT space for a track, using only the ROM's own learned quantities (decoder, mass field, potential field) plus the actuator joints for the angle readout.
    theta_ddot = Lam^-1 tau - J M^-1 dV/du   (J = d theta/du, Lam = (J M^-1 J^T)^-1 the joint-space inertia)  ->  tau = Lam (a_des + J M^-1 dV/du) with a_des = w0^2 e - 2 zeta w0 theta_dot.
    Returns (controls, rows[(name, angle, error, command, saturated)]); Coriolis terms are ignored (fine at the speeds this is used at)."""
    import torch
    from atlas_rom.rom import ops as _ops
    acts = list(rom.case.actuators)
    idx = [k for k, a in enumerate(acts) if a.kind == "joint_torque"]
    ch, u, p = int(track.state["chart"]), track.state["u"], track.state["p"]
    joints = [rom.case.mechanism.joint(acts[k].target) for k in idx]
    pose = torch.as_tensor(track.pose(), dtype=torch.float64)
    theta = np.array([float(_ops.joint_angle_torch(pose, j, rom.body_order)) for j in joints])
    J = np.stack([np.asarray(_ops.actuator_jacobian(rom.models[ch], u, j, rom.body_order), float) for j in joints])          # (n, d)
    M = track.rt._mass(ch, u)
    Minv_p = np.linalg.solve(M, p)
    theta_dot = J @ Minv_p
    err = np.array([((targets[k] - th + np.pi) % (2 * np.pi)) - np.pi for k, th in zip(idx, theta)])
    a_des = w0 ** 2 * err - 2.0 * zeta * w0 * theta_dot
    JMJ = J @ np.linalg.solve(M, J.T)
    ff = J @ np.linalg.solve(M, track.rt._potential(ch, u)[1]) if gcomp else 0.0
    tau = np.linalg.lstsq(JMJ + 1e-12 * np.eye(len(idx)), a_des + ff, rcond=1e-9)[0]
    ctl, rows = list(manual), []
    for r, k in enumerate(idx):
        a = acts[k]
        lim = lim_mult * float(a.hi)
        raw = float(tau[r]) / max(float(a.gain), 1e-12)
        cmd = float(np.clip(raw, -lim, lim))
        ctl[k] += cmd
        rows.append((a.name, float(theta[r]), float(err[r]), cmd, abs(raw) > lim))
    return ctl, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom", required=True, help="roms/<name>/ package name or path")
    ap.add_argument("--roms-dir", default=os.path.join(_ROOT, "roms"))
    ap.add_argument("--tracks", default="adv",
                    help="comma list of: " + ", ".join(TRACK_TYPES))
    ap.add_argument("--reference", nargs="+", default=None,
                    help="trajectory .npz file(s): poses (T,n,3,4) or (N,T,n,3,4), "
                         "optional dt and controls")
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--s-stay", type=float, default=0.10)
    ap.add_argument("--s-receive", type=float, default=0.05)
    ap.add_argument("--dwell", type=int, default=0, help="min steps between handoffs")
    ap.add_argument("--oracle-lookahead", type=int, default=3)
    ap.add_argument("--oracle-margin", type=float, default=0.25)
    ap.add_argument("--oracle-tol", type=float, default=1e-2)
    ap.add_argument("--oracle-pose-tol", type=float, default=5e-3,
                    help="oracle: max decoder roundtrip delta of the source pose into a receiver, relative to the pose's std "
                         "(closure alone cannot tell the same pose from any valid pose); <=0 disables")
    ap.add_argument("--damping", type=float, default=0.0,
                    help="Rayleigh velocity damping (1/s) of every track; the runtime's own default is 4.0 (live slider in the viewer)")
    ap.add_argument("--oracle-tangent-tol", type=float, default=3.0,
                    help="oracle: max principal angle (degrees) between the source and receiver decoder tangent planes, so a transported "
                         "velocity stays faithful; <=0 disables")
    ap.add_argument("--oracle-stay-frac", type=float, default=0.1,
                    help="oracle: an acceptable chart is only left once its closure exceeds this fraction of --oracle-tol")
    ap.add_argument("--disable-stalls", action="store_true")
    ap.add_argument("--replay-frames", type=int, default=3000,
                    help="rendered frames kept for the live replayer")
    args = ap.parse_args()

    names = [n.strip() for n in args.tracks.split(",") if n.strip()]
    bad = [n for n in names if n not in TRACK_TYPES]
    if bad or not names or len(set(names)) != len(names):
        ap.error(f"--tracks must be distinct names from {sorted(TRACK_TYPES)}, got {args.tracks!r}")

    rom_path = args.rom if os.path.isdir(args.rom) else os.path.join(args.roms_dir, args.rom)
    _verify_package(rom_path)
    rom = PatchAtlasROM.load(rom_path)

    references = []
    for path in args.reference or []:
        references += load_references(path, default_dt=args.dt)
    ref_index = 0

    settings = Settings(dt=args.dt, s_stay=args.s_stay, s_receive=args.s_receive,
                        dwell=args.dwell, stall_guard=not args.disable_stalls,
                        lookahead=args.oracle_lookahead, oracle_margin=args.oracle_margin,
                        oracle_tol=args.oracle_tol,
                        oracle_pose_tol=(args.oracle_pose_tol if args.oracle_pose_tol > 0 else None),
                        oracle_stay_frac=args.oracle_stay_frac, damping=args.damping,
                        oracle_tangent_tol=(args.oracle_tangent_tol if args.oracle_tangent_tol > 0 else None))
    tracks = {n: TRACK_TYPES[n](rom, settings, references[0] if references else None)
              for n in names}
    enabled = {n: True for n in names}

    from viewers.mesh_view import MeshAdapter
    import polyscope as ps
    from polyscope import imgui

    schema = rom.control_schema
    controls = [0.0] * len(schema)

    # ---- target driver (viewer-side controller, NOT part of the ROM): PD in joint space, gains scaled by each track's own effective joint inertia ------------------
    import torch
    from atlas_rom.rom import ops as _ops
    acts = list(rom.case.actuators)
    drive = {"on": False, "w0": 4.0, "zeta": 1.0, "lim": 20.0, "gcomp": True, "targets": [0.0] * len(acts), "inited": False, "last": {}}

    def _wrap(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def joint_readout(t, k):
        """(theta,) of actuator k on track t (viewer-side readout of the joint, used only to initialise the targets)."""
        a = acts[k]
        if a.kind != "joint_torque":
            return None
        return (float(_ops.joint_angle_torch(torch.as_tensor(t.pose(), dtype=torch.float64), rom.case.mechanism.joint(a.target), rom.body_order)),)

    def drive_controls(name, t):
        ctl, rows = joint_pd_controls(rom, t, drive["targets"], drive["w0"], drive["zeta"], drive["lim"], drive["gcomp"], controls)
        drive["last"][name] = rows
        return ctl

    def drive_targets_from(t):
        for k in range(len(acts)):
            ro = joint_readout(t, k)
            if ro is not None:
                drive["targets"][k] = float(ro[0])
        drive["inited"] = True

    ps.init()
    probe = MeshAdapter(rom.mechanism)
    ps.set_up_dir(probe.up_dir)
    ps.set_ground_plane_mode("shadow_only")

    columns = names + (["reference"] if references else [])
    offsets = _layout(probe, rom, len(columns))
    adapters = {}
    for col, offset in zip(columns, offsets):
        adapter = MeshAdapter(rom.mechanism)
        pose0 = tracks[col].pose() if col in tracks else references[0].pose_at(0.0)
        adapter.register(ps, pose0, name_prefix=f"{col}/",
                         color_override=TRACK_COLORS.get(col, REFERENCE_COLOR))
        for body in adapter.static:
            ps.get_surface_mesh(f"{col}/{body['name']}").set_transform(_translation(offset))
        for mesh in adapter.meshes:
            mesh.set_transform(_translation(offset))
        adapters[col] = adapter

    # Live replayer: one snapshot per rendered frame (poses + readouts of every
    # track, the reference pose, and the controls in force).  Replay only reads
    # this buffer; the tracks' own states are never touched, so "Resume live"
    # continues exactly where the simulation stopped.
    frames = collections.deque(maxlen=max(int(args.replay_frames), 2))
    replay = {"idx": None, "playing": False,    # idx None = live
              "speed": 1.0, "acc": 0.0}         # speed: frames per redraw, <0 = backwards

    ui = {"energy_q": "total", "energy_window": 400, "running": False, "spf": 4, "replay": bool(references and references[0].controls is not None),
          "error": ""}

    def show(frame):
        for n in tracks:
            adapters[n].update(frame["poses"][n])
            for m in adapters[n].meshes:
                m.set_enabled(enabled[n])
        if references:
            adapters["reference"].update(frame["reference"])

    def refresh():
        """Snapshot the live tracks into the buffer and draw them."""
        first = next(iter(tracks.values()))
        frame = {"time": first.state["time"], "controls": list(controls),
                 "poses": {n: t.pose().copy() for n, t in tracks.items()},
                 "diag": {n: t.diagnostics() for n, t in tracks.items()},
                 "reference": (references[ref_index].pose_at(first.state["time"]).copy()
                               if references else None)}
        frames.append(frame)
        show(frame)

    def do_reset():
        for t in tracks.values():
            t.reset()
        frames.clear()
        replay["idx"], replay["playing"] = None, False
        ui["error"] = ""
        refresh()

    def advance(n):
        first = next(iter(tracks.values()))
        for _ in range(n):
            ctl = controls
            if ui["replay"] and references:
                rc = references[ref_index].controls_at(first.state["time"])
                ctl = controls if rc is None else rc
            try:
                for name, t in tracks.items():
                    if enabled[name]:
                        t.step(drive_controls(name, t) if (drive["on"] and ctl is controls) else ctl)
            except (NotImplementedError, ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                ui["error"], ui["running"] = str(e), False
                break
        refresh()

    def replay_panel():
        n = len(frames)
        if replay["idx"] is None:
            if imgui.Button("Replay") and n > 1:
                ui["running"] = False
                replay["idx"], replay["playing"] = n - 1, False
            imgui.SameLine()
            imgui.Text(f"live  (buffer {n}/{frames.maxlen})")
            return
        i = replay["idx"] = min(replay["idx"], n - 1)
        if imgui.Button("Resume live"):
            replay["idx"], replay["playing"] = None, False
            show(frames[-1])
            return
        imgui.SameLine()
        if imgui.Button("Pause replay" if replay["playing"] else "Play replay"):
            replay["playing"] = not replay["playing"]
        imgui.SameLine()
        if imgui.Button("<"):
            i, replay["playing"] = max(i - 1, 0), False
        imgui.SameLine()
        if imgui.Button(">"):
            i, replay["playing"] = min(i + 1, n - 1), False
        if replay["playing"]:
            # Fractional speeds accumulate, so 0.1x advances one frame per 10 redraws.
            replay["acc"] += replay["speed"]
            whole = int(replay["acc"])
            replay["acc"] -= whole
            i = min(max(i + whole, 0), n - 1)
            if i in (0, n - 1) and whole != 0:
                replay["playing"] = False
        changed, v = imgui.SliderFloat("replay speed (frames/redraw)", replay["speed"], -8.0, 8.0)
        if changed:
            replay["speed"] = float(v)
        changed, v = imgui.SliderInt("replay frame", i, 0, n - 1)
        if changed:
            i, replay["playing"] = int(v), False
        replay["idx"] = i
        show(frames[i])
        imgui.TextColored((0.4, 0.8, 1.0, 1.0),
                          f"REPLAY frame {i}/{n - 1}  t={frames[i]['time']:.4f}  "
                          f"(tracks are paused; Resume live continues the simulation)")

    ENERGY_Q = (("total", "energy", "total energy (change over the window)"),
                ("kinetic", "kinetic", "kinetic energy"),
                ("potential", "potential", "potential energy (change over the window)"))

    def energy_panel():
        """Energy of every enabled track over the recorded frames (follows the replay frame).  Total / potential are shown as a change from
        the first frame of the window: the absolute levels are huge (V ~ m g h) and a handoff teleport shows up as a jump."""
        if not frames:
            return
        imgui.Separator()
        imgui.Text("energy")
        for key, _attr, label in ENERGY_Q:
            imgui.SameLine()
            if imgui.RadioButton(key, ui["energy_q"] == key):
                ui["energy_q"] = key
        attr, label = next((a, l) for k, a, l in ENERGY_Q if k == ui["energy_q"])
        end = replay["idx"] if replay["idx"] is not None else len(frames) - 1
        end = min(end, len(frames) - 1)
        start = max(0, end - ui["energy_window"] + 1)
        for n in names:
            if not enabled[n]:
                continue
            vals = np.array([frames[k]["diag"][n][attr] for k in range(start, end + 1)], dtype=float)
            if attr != "kinetic":
                vals = vals - vals[0]
            handoffs = frames[end]["diag"][n]["handoffs"]
            cur = vals[-1]
            try:
                imgui.PlotLines(f"{n}  {label}##energy_{n}", vals.astype(np.float32), graph_size=(0, 60),
                                overlay_text=f"{cur:+.3g}   (handoffs {handoffs})")
            except Exception:
                imgui.Text(f"{n}: {cur:+.3g}")

    def callback():
        nonlocal ref_index
        if replay["idx"] is None:
            if imgui.Button("Pause" if ui["running"] else "Play"):
                ui["running"] = not ui["running"]
            imgui.SameLine()
            if imgui.Button("Step"):
                advance(1)
            imgui.SameLine()
        if imgui.Button("Reset"):
            do_reset()
        replay_panel()
        changed, spf = imgui.SliderInt("steps/frame", ui["spf"], 1, 40)
        if changed:
            ui["spf"] = int(spf)

        imgui.Separator()
        for n in names:
            changed, value = imgui.Checkbox(f"track: {n}", enabled[n])
            if changed:
                enabled[n] = value
                show(frames[replay["idx"]] if replay["idx"] is not None else frames[-1])

        imgui.Separator()
        if references:
            if imgui.BeginCombo("reference", references[ref_index].name):
                for i, r in enumerate(references):
                    if imgui.Selectable(r.name, i == ref_index)[0] and i != ref_index:
                        ref_index = i
                        for t in tracks.values():
                            t.set_reference(r)
                        do_reset()
                imgui.EndCombo()
            r = references[ref_index]
            imgui.Text(f"frames={r.n_frames}  dt={r.dt:g}")
            if r.controls is not None:
                changed, value = imgui.Checkbox("replay reference controls", ui["replay"])
                if changed:
                    ui["replay"] = value
            first = next(iter(tracks.values()))
            if r.exhausted(first.state["time"]):
                imgui.TextColored((1.0, 0.6, 0.2, 1.0), "reference exhausted (holding last frame)")

        imgui.Separator()
        changed, g = imgui.SliderFloat("Gravity (m/s^2)",
                                       float(settings.gravity if settings.gravity is not None
                                             else rom.mechanism.gravity), 0.0, 20.0)
        if changed:
            settings.gravity = float(g)
        for label, attr, lo, hi in (("s_stay (hysteresis)", "s_stay", 0.0, 1.0),
                                    ("s_receive", "s_receive", 0.0, 1.0)):
            changed, v = imgui.SliderFloat(label, getattr(settings, attr), lo, hi)
            if changed:
                setattr(settings, attr, float(v))
        changed, v = imgui.SliderFloat("velocity damping (1/s)", float(settings.damping), 0.0, 20.0)
        if changed:
            settings.damping = float(v)
        changed, v = imgui.SliderInt("dwell (steps)", settings.dwell, 0, 500)
        if changed:
            settings.dwell = int(v)
        if any(n.startswith("oracle") for n in names):
            changed, v = imgui.SliderInt("oracle lookahead (steps)", settings.lookahead, 1, 10)
            if changed:
                settings.lookahead = int(v)
            changed, v = imgui.SliderFloat("oracle switch margin", settings.oracle_margin, 0.0, 0.9)
            if changed:
                settings.oracle_margin = float(v)
            changed, v = imgui.SliderFloat("oracle closure tolerance", settings.oracle_tol,
                                           0.0, 0.1, format="%.4f")
            if changed:
                settings.oracle_tol = float(v)
        energy_panel()
        for label, attr in (("Enable SDF boundary", "sdf_enabled"),
                            ("Project/retract at boundary", "project_boundary"),
                            ("Stall guard", "stall_guard")):
            changed, v = imgui.Checkbox(label, getattr(settings, attr))
            if changed:
                setattr(settings, attr, v)

        imgui.Separator()
        replaying = ui["replay"] and references and references[ref_index].controls is not None
        for i, ch in enumerate(schema):
            label = f"{ch['name']} ({ch['unit']})" if ch["unit"] else ch["name"]
            if replaying:
                imgui.TextDisabled(f"{label}: replayed")
                continue
            changed, v = imgui.SliderFloat(label, controls[i], ch["lo"], ch["hi"])
            if changed:
                controls[i] = float(v)

        imgui.Separator()
        if acts and any(a.kind == "joint_torque" for a in acts):
            changed, v = imgui.Checkbox("drive to target joint angles (PD, torque-limited)", drive["on"])
            if changed:
                drive["on"] = bool(v)
                if drive["on"] and not drive["inited"]:
                    drive_targets_from(next(iter(tracks.values())))
            if drive["on"]:
                for k, a in enumerate(acts):
                    if a.kind != "joint_torque":
                        continue
                    ch_, v_ = imgui.SliderFloat(f"target: {a.target} (rad)", float(drive["targets"][k]), -np.pi, np.pi)
                    if ch_:
                        drive["targets"][k] = float(v_)
                changed_g, vg = imgui.Checkbox("gravity compensation (feedforward from the learned potential)", drive["gcomp"])
                if changed_g:
                    drive["gcomp"] = bool(vg)
                if imgui.Button("target <- current pose of the first track"):
                    drive_targets_from(next(iter(tracks.values())))
                for key, lab, lo, hi in (("w0", "natural frequency w0 (rad/s)", 0.5, 20.0), ("zeta", "damping ratio zeta", 0.2, 3.0), ("lim", "torque limit (x declared hi)", 0.1, 50.0)):
                    ch_, v_ = imgui.SliderFloat(lab, float(drive[key]), lo, hi)
                    if ch_:
                        drive[key] = float(v_)
                for name in tracks:
                    for (an, th, err, cmd, sat) in drive["last"].get(name, []):
                        imgui.Text(f"  {name:8s} {an}: angle {th:+.3f}  error {err:+.3f}  torque {cmd:+.2f}{'  SATURATED' if sat else ''}")

        if ui["running"] and replay["idx"] is None:
            advance(ui["spf"])

        imgui.Separator()
        if ui["error"]:
            imgui.TextColored((1.0, 0.3, 0.3, 1.0), ui["error"])
        shown = frames[replay["idx"]] if replay["idx"] is not None else frames[-1]
        for n in tracks:
            d = shown["diag"][n]
            imgui.Separator()
            imgui.Text(f"[{n}] chart={d['chart']}/{d['n_charts']}  handoffs={d['handoffs']}  "
                       f"stalls={d['stalls']}" + ("  BLOCKED" if d["blocked"] else "")
                       + ("  HALTED (no acceptable candidate)" if d.get("halted") else "")
                       + ("" if enabled[n] else "  (paused)"))
            imgui.Text(f"  support={d['support']:+.4f}  sdf={d['sdf']:+.5f}  "
                       f"closure={d['closure']:.3e}")
            imgui.Text(f"  t={d['time']:.4f}  KE={d['kinetic']:.4f}  V={d['potential']:.4f}  "
                       f"E={d['energy']:.4f}")
            imgui.Text(f"  handoff jump: {_fmt(d['jump_trans'], '.4g')} m / "
                       f"{_fmt(d['jump_rot'], '.3g')} deg   "
                       f"proj residual: {_fmt(d['projection_residual'], '.3e')}")
            if references:
                imgui.Text(f"  vs reference: {_fmt(d['err_trans'], '.4g')} m / "
                           f"{_fmt(d['err_rot'], '.3g')} deg")

    refresh()
    ps.set_user_callback(callback)
    print(f"routing-test viewer: rom={os.path.basename(rom_path)} tracks={names} "
          f"references={len(references)}")
    ps.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
