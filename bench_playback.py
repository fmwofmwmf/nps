"""
bench_playback.py

Generates sample configurations by running the full-space implicit-Euler
integrator for a list of (torque_strength, n_steps) entries.

After simulation, frames are saved to precompute/ and optionally played
back in Polyscope.

Edit SAMPLES and flags below.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from tqdm import tqdm

from Args import Args
from config_utils import load_config, system_to_name
from integrators import choose_integrator

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
DT      = 0.01
PROFILE = False   # print per-step timing breakdown
PLAYBACK = True   # open Polyscope after saving; set False to just save and exit

# Each entry: (torque_strength, n_steps)
SAMPLES = [
    (2**9,  150),
    (-2**9, 150)
]

# ─────────────────────────────────────────────────────────────────────────────


def _identity_state_to_system(system_def, state, space):
    return state


def run_sample(system, system_def, integrator, init_q, space, torque_strength, n_steps):
    """Simulate n_steps of full-space dynamics at the given torque_strength.

    Returns a list of CPU tensors (n_steps+1 frames, including the initial config).
    """
    system_def['external_forces']['torque_strength'] = float(torque_strength)

    q     = init_q.clone()
    q_dot = torch.zeros_like(q)

    step_times = []
    frames = [q.cpu()]

    bar = tqdm(range(n_steps), desc=f"  torque={torque_strength:+.3f}", unit="step", leave=True)
    for step in bar:
        q = q.detach()
        q[2::3] = q[2::3].remainder(2 * torch.pi)

        t0 = time.perf_counter()
        q, q_dot, residual = integrator(
            system, system_def, _identity_state_to_system,
            q, q_dot, space, dt=DT,
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

        frames.append(q.cpu())

        avg_ms = 1e3 * sum(step_times) / len(step_times)
        bar.set_postfix(res=f"{residual:.2e}", ms=f"{avg_ms:.1f}")

    if PROFILE and step_times:
        total = sum(step_times)
        avg   = total / len(step_times)
        print(f"  Profile: {n_steps} steps  total={total:.2f}s  avg={avg*1e3:.2f}ms/step  "
              f"min={min(step_times)*1e3:.2f}ms  max={max(step_times)*1e3:.2f}ms")

    return frames


def save_runs(all_runs, config, confirm=True):
    os.makedirs("precompute", exist_ok=True)
    path = os.path.join("precompute", f"{config.experiment_name}_dataset.pt")

    all_frames = []
    run_meta   = []
    for run in all_runs:
        stacked = torch.stack(run["frames"])
        run_meta.append({
            "torque":   run["torque"],
            "n_steps":  run["n_steps"],
            "start":    len(all_frames),
            "length":   len(stacked),
        })
        all_frames.append(stacked)

    q_all = torch.cat(all_frames, dim=0)
    if confirm:
        input(f"Press enter to save {len(q_all)} frames → {path}  (Ctrl-C to abort)")
    else:
        print(f"Saving {len(q_all)} frames → {path}")
    torch.save({"q": q_all, "runs": run_meta}, path)
    print("Saved.")


def main(args=None, samples=None, playback=None, dt=None, pipeline_mode=False):
    """
    args         – result of get_args() or Args(); if None, resolved here
    samples      – list of (torque_strength, n_steps); falls back to module SAMPLES
    playback     – open Polyscope after collection; falls back to module PLAYBACK
                   (always False when pipeline_mode=True)
    dt           – timestep; falls back to module DT
    pipeline_mode – skip all interactive prompts (input(), Polyscope)
    """
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(DEVICE)

    if args is None:
        from get_args import get_args
        args = get_args()

    config = load_config(args.config_file)

    # Resolve samples + dt: caller > config > module-level constants
    coll = config.get("collection", {})
    if samples is None:
        raw = coll.get("samples")
        samples = [(s["torque"], s["n_steps"]) for s in raw] if raw else SAMPLES
    dt_used = dt if dt is not None else coll.get("dt", DT)
    do_play = (PLAYBACK if playback is None else playback) and not pipeline_mode

    system, system_def = system_to_name(config)
    system.training = False

    space  = torch.ones(config.shape_space_dim)
    init_q = system_def['init_pos'].clone()

    integrator = choose_integrator("imp")   # full-space implicit Euler

    # ── Simulate ──────────────────────────────────────────────────────────────
    print(f"Device: {DEVICE}  |  {len(samples)} sample(s)  dt={dt_used}")

    all_runs = []
    for i, (torque, n_steps) in enumerate(samples):
        tqdm.write(f"[{i+1}/{len(samples)}] torque={torque:+.3f}  n_steps={n_steps}")
        frames = run_sample(system, system_def, integrator, init_q, space, torque, n_steps)
        all_runs.append({"torque": torque, "n_steps": n_steps, "frames": frames})

    if not do_play:
        save_runs(all_runs, config, confirm=not pipeline_mode)
        return

    # ── Polyscope playback ────────────────────────────────────────────────────
    print("Opening Polyscope...")
    ps.init()
    ps.set_ground_plane_mode('none')
    ps.set_automatically_compute_scene_extents(False)

    state = {
        "run_idx":   0,
        "frame_idx": 0,
        "playing":   False,
        "speed":     1,
    }

    def _show_current():
        run = all_runs[state["run_idx"]]
        fi  = min(state["frame_idx"], len(run["frames"]) - 1)
        system.visualize(system_def, run["frames"][fi].to(DEVICE), space)

    _show_current()
    system.visualize_set_nice_view(system_def, space)

    def callback():
        run      = all_runs[state["run_idx"]]
        n_frames = len(run["frames"])

        psim.TextUnformatted(
            f"Sample {state['run_idx']+1}/{len(all_runs)}  "
            f"torque = {run['torque']:+.3f}  "
            f"steps = {run['n_steps']}"
        )
        changed, new_idx = psim.SliderInt("Sample##run", state["run_idx"], 0, len(all_runs) - 1)
        if changed:
            state["run_idx"]   = new_idx
            state["frame_idx"] = 0
            _show_current()

        psim.Separator()

        psim.TextUnformatted(f"Frame {state['frame_idx']} / {n_frames - 1}")
        changed, new_f = psim.SliderInt("Frame##f", state["frame_idx"], 0, n_frames - 1)
        if changed:
            state["frame_idx"] = new_f
            _show_current()

        _, state["playing"] = psim.Checkbox("Play", state["playing"])
        psim.SameLine()
        _, state["speed"] = psim.SliderInt("Speed##spd", state["speed"], 1, 10)

        if psim.Button("|<"):
            state["frame_idx"] = 0
            _show_current()
        psim.SameLine()
        if psim.Button(">|"):
            state["frame_idx"] = n_frames - 1
            _show_current()

        if state["playing"]:
            state["frame_idx"] = (state["frame_idx"] + state["speed"]) % n_frames
            _show_current()

    ps.set_user_callback(callback)
    ps.show()

    save_runs(all_runs, config, confirm=True)


if __name__ == "__main__":
    main()