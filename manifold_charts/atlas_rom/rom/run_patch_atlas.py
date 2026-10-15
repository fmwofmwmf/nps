"""run_patch_atlas.py — one runner, Full-space and ROM backends behind a toggle.

    python -m atlas_rom.rom.run_patch_atlas --case scissor_lift --selfcheck
    python -m atlas_rom.rom.run_patch_atlas --case scissor_lift --viewer

``--case`` always names a real ``cases/<case>.py`` spec -- Full mode needs
nothing else and is always available.  ROM mode additionally needs a staged
``roms/<rom-name>/`` package (``--rom`` names it; defaults to ``--case``) with
``manifest["rom_mode_ready"] is True`` -- a build/rollout/actuator-map gate
already enforced at staging time (see ``patch_rom.package.stage``).  A ROM
missing that gate is loaded for ``--selfcheck`` (which is exactly what
reports why) but refused for the ROM side of ``--viewer``.

The viewer shows ONE set of sliders, built from ``case.control_schema`` --
the same ordered, labeled, unit-carrying channel list both backends consume
identically (``atlas_rom.simulate.fullspace_sim.FullSpaceSim`` and
``atlas_rom.rom.PatchAtlasRuntime``) -- with a Full/ROM radio toggle,
Play/Step/Reset, and the live control vector routed to whichever backend is
selected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # atlas_rom/
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas_rom.cases import load_spec  # noqa: E402
from atlas_rom.rom.loader import PatchAtlasROM  # noqa: E402
from atlas_rom.rom.runtime import PatchAtlasRuntime  # noqa: E402
from atlas_rom.rom.selfcheck import run as run_selfcheck  # noqa: E402


def _default_roms_dir():
    return os.path.join(_ROOT, "roms")


def _rom_available(roms_dir, name):
    manifest_path = os.path.join(roms_dir, name, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, f"no staged ROM at roms/{name}/"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("rom_mode_ready", False):
        return False, "staged ROM exists but rom_mode_ready=False (see its manifest gates)"
    return True, None


# ─── backends: a common reset()/step(controls)/pose() surface ────────────────

class RomBackend:
    """Wraps ``PatchAtlasRuntime`` behind the shared viewer surface."""

    def __init__(self, rom_path, *, dt=1e-3, stall_guard=True):
        self.rom = PatchAtlasROM.load(rom_path)
        self.rt = PatchAtlasRuntime(self.rom, dt=dt, stall_guard=stall_guard)   # learned advantage router
        self.actuators = self.rt.actuators
        self.state = self.rt.initial_state(chart=0)

    def reset(self):
        self.state = self.rt.initial_state(chart=0)
        return self.pose()

    def pose(self):
        return self.rt.pose(self.state)

    def step(self, controls):
        self.rt.step(self.state, controls=controls)
        return self.pose()

    def set_gravity(self, gravity):
        # V was distilled at the case gravity; PatchAtlasRuntime rescales its
        # learned potential and force without reintroducing an analytic teacher.
        self.rt.gravity = float(gravity)

    def set_sdf_enabled(self, enabled):
        self.rt.set_sdf_enabled(enabled)

    def set_boundary_projection(self, enabled):
        self.rt.set_boundary_projection(enabled)

    def set_euclidean_retraction(self, enabled):
        self.rt.set_euclidean_retraction(enabled)

    def status_lines(self):
        e = self.rt.energy(self.state)
        chart = self.state["chart"]
        bd = self.rt.boundary_diagnostics(self.state)
        info = self.rt.router_info or {}
        route = f"A_best={info.get('adv', {}).get(info.get('best'), float('nan')):+.2f}"
        hyst = f"switch_ratio={self.rt.switch_ratio:.2f}"
        return [f"chart={chart}/{self.rom.n_patches}  "
               f"handoffs={self.state['handoffs']}  stalls={self.state['stalls']}",
               f"{route}  sdf={bd['sdf']:+.6f}  "
               f"|grad sdf|={bd['sdf_grad_norm']:.4f}",
               f"sdf force={bd['sdf_force_norm']:.4f}  "
               f"sdf={'on' if bd['sdf_enabled'] else 'off'}  {hyst}",
               f"t={self.state['time']:.4f}s  KE={e['kinetic']:.4f}  "
               f"V={e['potential']:.4f}  E={e['total']:.4f}",
               ("blocked: left every chart's trusted domain"
                if self.state["blocked"] else "")]


class CspaceInset:
    """Side inset showing the current ROM's encoded sample cloud and SDF."""

    def __init__(self, rom, dataset, pose):
        self.rom = rom
        data = np.load(dataset, allow_pickle=True)
        if "q" not in data:
            raise ValueError(f"{dataset}: expected q array")
        q = np.asarray(data["q"], float)
        self.u = np.asarray([rom.encode(0, row) for row in q], float)
        self.sdf = np.asarray([rom.boundary_value(0, u) for u in self.u], float)
        self.support = (np.asarray([rom.support_margin(0, u) for u in self.u], float)
                        if rom.support_fields is not None else np.zeros(len(self.u)))
        witness_path = os.path.join(os.path.dirname(os.path.abspath(dataset)),
                                    "sdf_witnesses.npz")
        self.failed_u = np.empty((0, self.u.shape[1]), float)
        self.failed_kind = np.empty(0, int)
        if os.path.isfile(witness_path):
            w = np.load(witness_path)
            kind = np.asarray(w["kind"], int)
            self.failed_u = np.asarray(w["u"], float)[kind == 1]
            self.failed_kind = kind[kind == 1]
        self.dataset = dataset
        self.cloud = None
        self.failed_cloud = None
        self.isoline = None
        self.marker = None
        self._place(pose)

    def _place(self, pose):
        centers = np.asarray(pose)[:, :, 3]
        extent = np.maximum(centers.max(0) - centers.min(0), 1e-3)
        span = np.maximum(self.u.max(0) - self.u.min(0), 1e-6)
        scale = 0.8 * max(float(extent.max()), 1.0) / float(span.max())
        side = 0
        yz_center = centers.mean(0)
        self.offset = np.array([centers.max(0)[side] + 0.8 * max(float(extent.max()), 1.0),
                                yz_center[1], yz_center[2]])
        self.scale = scale

    def point(self, u):
        p = np.asarray(u, float).reshape(-1)
        y = self.offset[1] + self.scale * p[0]
        z = self.offset[2] + self.scale * (p[1] if len(p) > 1 else 0.0)
        return np.array([self.offset[0], y, z], dtype=np.float32)

    def register(self, ps, u):
        points = np.asarray([self.point(p) for p in self.u], dtype=np.float32)
        self.cloud = ps.register_point_cloud("C-space / AE samples", points)
        self.cloud.set_radius(0.012, relative=False)
        self.cloud.add_scalar_quantity("SDF", self.sdf, cmap="coolwarm", enabled=True)
        self.cloud.add_scalar_quantity("support", self.support, cmap="viridis", enabled=False)
        if len(self.failed_u) and False:
            failed_points = np.asarray([self.point(p) for p in self.failed_u], dtype=np.float32)
            self.failed_cloud = ps.register_point_cloud("C-space / failed witnesses",
                                                        failed_points)
            self.failed_cloud.set_color((0.95, 0.12, 0.08))
            self.failed_cloud.set_radius(0.018, relative=False)
        if self.u.shape[1] == 2:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            gx = np.linspace(self.u[:, 0].min(), self.u[:, 0].max(), 96)
            gy = np.linspace(self.u[:, 1].min(), self.u[:, 1].max(), 96)
            X, Y = np.meshgrid(gx, gy, indexing="xy")
            Z = np.asarray([[self.rom.boundary_value(0, (x, y)) for x in gx] for y in gy])
            fig, ax = plt.subplots()
            contours = ax.contour(X, Y, Z, levels=[0.0])
            segments = [seg for seg in (contours.allsegs[0] if contours.allsegs else [])
                        if len(seg) >= 2]
            plt.close(fig)
            if segments:
                nodes = np.vstack([np.asarray([self.point(p) for p in seg], np.float32)
                                   for seg in segments])
                edges = []
                start = 0
                for seg in segments:
                    edges.extend((start + i, start + i + 1) for i in range(len(seg) - 1))
                    start += len(seg)
                if edges:
                    self.isoline = ps.register_curve_network(
                        "C-space / SDF zero isoline", nodes, np.asarray(edges, np.int32))
                    self.isoline.set_color((1.0, 0.85, 0.1))
                    self.isoline.set_radius(0.018, relative=False)
        self.marker = ps.register_point_cloud("C-space / current state",
                                              self.point(u)[None])
        self.marker.set_color((1.0, 1.0, 1.0))
        self.marker.set_radius(0.045, relative=False)

    def update(self, u):
        if self.marker is not None:
            self.marker.update_point_positions(self.point(u)[None])


class FullBackend:
    """Wraps a full-space simulator behind the same surface.

    ``backend="libuipc"`` is ``FullSpaceSim`` (production libuipc simulator);
    ``"ideal"`` is ``IdealJointSim`` (rigid-body dynamics projected onto the
    analytic ideal-joint oracle, which holds gear / phase-lock laws exactly);
    ``"auto"`` tries libuipc and falls back to ideal when libuipc has no
    constitution for one of the case's joints.
    """

    def __init__(self, case, *, dt=2e-2, backend="auto", log=print):
        self.case = case
        self.dt = float(dt)
        self.backend = backend
        self.log = log
        self.gravity = float(case.mechanism.gravity)
        self.sim = self._build()
        self.actuators = self.sim.actuators

    def _build(self):
        # once `auto` has fallen back, rebuilds (gravity changes) go straight
        # to the ideal sim instead of re-failing a libuipc world each time
        if self.backend in ("auto", "libuipc") and getattr(self, "active", None) != "ideal":
            from atlas_rom.simulate.fullspace_sim import FullSpaceSim
            try:
                sim = FullSpaceSim(self.case, dt=self.dt, gravity=self.gravity, log=self.log)
                self.active = "libuipc"
                return sim
            except NotImplementedError as e:
                if self.backend == "libuipc":
                    raise
                self.log(f"Full (libuipc) cannot build {self.case.name!r} ({e}); "
                         f"using the ideal-joint simulator")
        from atlas_rom.simulate.ideal_sim import IdealJointSim
        sim = IdealJointSim(self.case, dt=self.dt, gravity=self.gravity, log=self.log)
        self.active = "ideal"
        return sim

    def reset(self):
        return self.sim.reset()

    def pose(self):
        return self.sim.pose()

    def step(self, controls):
        return self.sim.step(controls)

    def set_gravity(self, gravity):
        # libuipc receives gravity at world construction, so changing it is a
        # reset/rebuild rather than a misleading mutation of a Python scalar.
        self.gravity = float(gravity)
        self.sim = self._build()
        self.actuators = self.sim.actuators
        return self.reset()

    def status_lines(self):
        if self.active == "ideal":
            stops = ", ".join(self.sim.active_limits) or "none"
            return [f"t={self.sim.time:.4f}s  full backend=ideal",
                    f"constraint residual max={self.sim.residual_max:.2e}",
                    f"joints at a stop: {stops}"]
        return [f"t={self.sim.time:.4f}s  full backend=libuipc", "", ""]


# ─── commands ─────────────────────────────────────────────────────────────────

def cmd_selfcheck(args):
    rom_path = os.path.join(args.roms_dir, args.rom or args.case)
    rom = PatchAtlasROM.load(rom_path)
    report = run_selfcheck(rom, n_steps=args.selfcheck_steps, dt=args.dt,
                           energy_drift_tol=args.energy_drift_tol)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


def cmd_viewer(args):
    from viewers.mesh_view import MeshAdapter

    case = load_spec(args.case)
    rom_name = args.rom or args.case
    rom_ok, rom_reason = _rom_available(args.roms_dir, rom_name)

    try:
        full = FullBackend(case, dt=args.dt_full, backend=args.full_backend)
    except NotImplementedError as e:
        # A case can be analytic-only (e.g. the jansen-array cases' virtual
        # `phase_lock` coupling has no libuipc/ABD joint equivalent), in which
        # case the real-physics backend simply cannot be built for it -- that
        # is not a reason to refuse viewing the ROM.
        full = None
        print(f"Full backend unavailable for {case.name!r}: {e}")
    rom = None
    if rom_ok:
        rom = RomBackend(os.path.join(args.roms_dir, rom_name), dt=args.dt_rom,
                         stall_guard=not args.disable_stalls)
    else:
        print(f"ROM mode unavailable: {rom_reason}")
    if full is None and rom is None:
        raise RuntimeError(f"{case.name!r}: neither the Full nor the ROM backend "
                           f"is available ({rom_reason})")

    schema = case.control_schema
    controls = [0.0] * len(schema)

    adapter = MeshAdapter(case.mechanism)

    import polyscope as ps
    from polyscope import imgui

    ps.init()
    ps.set_up_dir(adapter.up_dir)
    ps.set_ground_plane_mode("shadow_only")
    adapter.register(ps, (full or rom).pose())

    cspace = None
    cspace_dataset = args.cspace_dataset
    if rom is not None and cspace_dataset is None:
        # The staged package records the exact sampler work directory it was
        # built from.  Prefer it over a name-based guess: a case can have
        # several datasets (for example ``sarrus_stack`` and
        # ``sarrus_stack_outward``) whose encodings are different clouds.
        source_work = rom.rom.manifest.get("source", {}).get("work_dir")
        candidates = [
            (os.path.join(source_work, "new_dataset.npz")
             if source_work else None),
            os.path.join(os.path.dirname(args.roms_dir), "work", rom_name, "new_dataset.npz"),
            os.path.join(os.path.dirname(args.roms_dir), "work", rom_name + "_outward", "new_dataset.npz"),
        ]
        cspace_dataset = next((p for p in candidates if p and os.path.isfile(p)), None)
    if rom is not None and cspace_dataset:
        cspace = CspaceInset(rom.rom, cspace_dataset, rom.pose())
        cspace.register(ps, rom.state["u"])

    ui = {"running": False, "steps_per_frame": 4,
          "backend": "full" if full is not None else "rom",
          "gravity": float(case.mechanism.gravity)}

    def active():
        return full if (ui["backend"] == "full" or rom is None) and full is not None else rom

    def advance(n):
        b = active()
        pose = b.pose()
        for _ in range(n):
            pose = b.step(controls)
        adapter.update(pose)
        if cspace is not None and ui["backend"] == "rom":
            cspace.update(rom.state["u"])

    def callback():
        if full is not None:
            if imgui.RadioButton("Full", ui["backend"] == "full"):
                ui["backend"] = "full"
        else:
            imgui.TextDisabled("Full (unavailable)")
        imgui.SameLine()
        rom_label = "ROM" if rom is not None else "ROM (unavailable)"
        if rom is not None:
            if imgui.RadioButton(rom_label, ui["backend"] == "rom"):
                ui["backend"] = "rom"
        else:
            imgui.TextDisabled(rom_label)
        imgui.Separator()

        if imgui.Button("Pause" if ui["running"] else "Play"):
            ui["running"] = not ui["running"]
        imgui.SameLine()
        if imgui.Button("Step"):
            advance(1)
        imgui.SameLine()
        if imgui.Button("Reset"):
            if full is not None:
                full.reset()
            if rom is not None:
                rom.reset()
            adapter.update(active().pose())
            if cspace is not None and ui["backend"] == "rom":
                cspace.update(rom.state["u"])
        changed, spf = imgui.SliderInt("steps/frame", ui["steps_per_frame"], 1, 40)
        if changed:
            ui["steps_per_frame"] = int(spf)
        if rom is not None:
            changed, enabled = imgui.Checkbox("Enable SDF boundary", rom.rt.sdf_enabled)
            if changed:
                rom.set_sdf_enabled(enabled)
            changed, enabled = imgui.Checkbox(
                "Project/retract at boundary", rom.rt.project_boundary)
            if changed:
                rom.set_boundary_projection(enabled)
            changed, enabled = imgui.Checkbox(
                "Euclidean retraction", rom.rt.euclidean_retraction)
            if changed:
                rom.set_euclidean_retraction(enabled)

        imgui.Separator()
        for i, ch in enumerate(schema):
            label = f"{ch['name']} ({ch['unit']})" if ch["unit"] else ch["name"]
            changed, value = imgui.SliderFloat(label, controls[i], ch["lo"], ch["hi"])
            if changed:
                controls[i] = float(value)

        changed, gravity = imgui.SliderFloat("Gravity (m/s^2)", ui["gravity"], 0.0, 20.0)
        if changed:
            ui["gravity"] = float(gravity)
            if full is not None:
                full.set_gravity(gravity)
            if rom is not None:
                rom.set_gravity(gravity)
            adapter.update(active().pose())

        if ui["running"]:
            advance(ui["steps_per_frame"])

        imgui.Separator()
        imgui.Text(f"case={case.name}  backend={ui['backend']}  "
                   f"g={ui['gravity']:.3f}  controls={[round(c, 3) for c in controls]}")
        for line in active().status_lines():
            if line:
                imgui.Text(line)

    ps.set_user_callback(callback)
    print(f"patch_rom viewer: case={case.name}, ROM {'ready' if rom else 'unavailable'}")
    ps.show()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="cases/<case>.py spec name (Full mode)")
    ap.add_argument("--rom", default=None,
                    help="roms/<name>/ package to use for ROM mode (default: --case)")
    ap.add_argument("--roms-dir", default=_default_roms_dir())
    ap.add_argument("--dt", type=float, default=1e-3, help="--selfcheck's ROM step size")
    ap.add_argument("--dt-full", type=float, default=2e-2)
    ap.add_argument("--full-backend", choices=("auto", "libuipc", "ideal"), default="auto",
                    help="Full mode simulator: libuipc, the ideal-joint projection sim, "
                         "or libuipc with ideal fallback for joints libuipc cannot build")
    ap.add_argument("--dt-rom", type=float, default=1e-3)
    ap.add_argument("--cspace-dataset", default=None,
                    help="new_dataset.npz to display as the current AE C-space inset")
    ap.add_argument("--disable-stalls", action="store_true",
                    help="disable legacy seam-stall hold logic; learned SDF remains authoritative")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selfcheck", action="store_true")
    mode.add_argument("--viewer", action="store_true")
    ap.add_argument("--selfcheck-steps", type=int, default=200)
    ap.add_argument("--energy-drift-tol", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="write --selfcheck's report as JSON")
    args = ap.parse_args()

    if args.selfcheck:
        sys.exit(cmd_selfcheck(args))
    else:
        sys.exit(cmd_viewer(args))


if __name__ == "__main__":
    main()
