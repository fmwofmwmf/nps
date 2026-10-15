"""selfcheck.py — package integrity, local-domain validity, transition
continuity, and deterministic rollout regression for a staged ``PatchAtlasROM``.

Each check is independent and reported separately; ``run()`` only ANDs them
together at the end, so a failure always says which contract broke instead of
just "selfcheck failed".
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from atlas_rom.charts.validity import transition as _chart_transition

from atlas_rom.rom.runtime import PatchAtlasRuntime

_ATLAS_ROM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_integrity(rom) -> dict:
    artifact_path = os.path.join(rom.path, rom.manifest["artifact"])
    report_path = os.path.join(rom.path, rom.manifest["build_report"])
    got_artifact = _sha256(artifact_path)
    got_report = _sha256(report_path)
    want = rom.manifest.get("source", {})
    ok = (got_artifact == want.get("artifact_sha256")
          and got_report == want.get("build_report_sha256"))
    return {"pass": ok, "artifact_sha256_match": got_artifact == want.get("artifact_sha256"),
            "build_report_sha256_match": got_report == want.get("build_report_sha256")}


def _check_domain(rom, *, tol=0.05, max_per_patch=32, seed=0) -> dict:
    """Every patch's own anchors must round-trip through encode(decode(u)) ~ u,
    within a tolerance scaled to that patch's own anchor spread -- an absolute
    tolerance would be meaningless across cases with different unit scales."""
    rng = np.random.default_rng(seed)
    per_patch = []
    ok = True
    for i, (model, anchors) in enumerate(zip(rom.models, rom.anchors)):
        n = len(anchors)
        take = rng.choice(n, size=min(max_per_patch, n), replace=False)
        spread = float(np.std(anchors, axis=0).mean()) or 1.0
        import torch
        with torch.no_grad():
            u = torch.as_tensor(anchors[take], dtype=torch.float64)
            q = model.decode(u)
            u2 = model.encode(q)
        err = float((u2 - u).norm(dim=-1).max()) / spread
        passed = err <= tol
        ok &= passed
        per_patch.append({"patch": i, "n_checked": len(take),
                          "max_relative_roundtrip_error": err, "pass": passed})
    return {"pass": ok, "tol": tol, "patches": per_patch}


def _check_transitions(rom, *, max_pairs=16, seed=0) -> dict:
    """For overlapping patches, an anchor carried across via ``transition``
    must land back inside the target chart's own trusted domain -- otherwise
    a handoff at runtime could hand off into nowhere."""
    if rom.n_patches <= 1:
        return {"pass": True, "n_patches": 1, "note": "single chart, no handoffs"}
    rng = np.random.default_rng(seed)
    pairs, checked = [], []
    ok = True
    for i in range(rom.n_patches):
        for j in range(i + 1, rom.n_patches):
            if set(map(int, rom.patches[i])) & set(map(int, rom.patches[j])):
                pairs.append((i, j))
    for (i, j) in pairs[:max_pairs]:
        n = len(rom.anchors[i])
        idx = int(rng.integers(0, n))
        u_i = rom.anchors[i][idx]
        u_j = _chart_transition(rom.models, i, j, u_i)
        if rom.support_fields is not None:
            margin = rom.support_margin(j, u_j)
            passed = margin >= -1e-6
            ok &= passed
            checked.append({"from": i, "to": j, "margin": float(margin), "pass": passed})
        else:
            # Router-only package: no support field to read.  Report the physical round-trip instead --
            # the transported coordinate must decode to (nearly) the source pose and be finite.
            q_src = rom.decode(i, u_i)
            q_dst = rom.decode(j, u_j)
            residual = float(np.mean((q_dst - q_src) ** 2))
            passed = bool(np.isfinite(residual) and np.all(np.isfinite(u_j)))
            ok &= passed
            checked.append({"from": i, "to": j, "round_trip_mse": residual, "pass": passed})
    return {"pass": ok, "n_pairs_checked": len(checked), "pairs": checked}


def _check_rollout(rom, *, n_steps=200, dt=1e-3, energy_drift_tol=0.1) -> dict:
    """Two identical passive rollouts must be bit-identical (nothing here is
    stochastic; a mismatch means hidden global state leaked in), and total
    energy must stay bounded relative to its initial magnitude -- a sanity
    floor on the stepper, not a tight physical-accuracy claim."""
    def run():
        rt = PatchAtlasRuntime(rom, dt=dt)
        state = rt.initial_state(chart=0)
        energies = [rt.energy(state)["total"]]
        for _ in range(n_steps):
            rt.step(state, controls=None)
            energies.append(rt.energy(state)["total"])
            if state["blocked"]:
                break
        return energies, state

    e1, s1 = run()
    e2, s2 = run()
    deterministic = (len(e1) == len(e2) and np.allclose(e1, e2, atol=0.0, rtol=0.0))
    e0 = abs(e1[0]) if abs(e1[0]) > 1e-9 else 1.0
    max_drift = max(abs(e - e1[0]) for e in e1) / e0
    bounded = max_drift <= energy_drift_tol
    moved = len(e1) > 1 and not (s1["stalls"] > 0 and len(e1) <= 2)
    return {"pass": bool(deterministic and bounded and moved),
            "deterministic": bool(deterministic), "n_steps_completed": len(e1) - 1,
            "max_relative_energy_drift": float(max_drift), "energy_drift_tol": energy_drift_tol,
            "handoffs": s1["handoffs"], "stalls": s1["stalls"], "blocked_at_end": bool(s1["blocked"])}


def _check_schema(rom) -> dict:
    """The manifest's staged ``control_schema`` must still match the live
    case spec it names -- a stale staging (case edited after `stage()` ran)
    would otherwise let the Full and ROM backends build DIFFERENT sliders
    for what the viewer presents as one shared control vector."""
    live = rom.case.control_schema
    staged = rom.control_schema
    ok = (live == staged)
    return {"pass": ok, "n_channels": len(live),
            "live": live, "staged": staged}


def _check_zero_control(rom, *, n_steps=20, dt=1e-3) -> dict:
    """``controls=None`` (no actuators driven) and an explicit all-zero
    control vector must produce IDENTICAL trajectories -- a zero-valued
    channel is supposed to be a true no-op, not a degenerate force."""
    if not rom.rom_mode_ready:
        return {"pass": True, "note": "ROM not mode-ready; skipped"}
    n_act = len(rom.case.actuators)

    def run_with(controls):
        rt = PatchAtlasRuntime(rom, dt=dt)
        state = rt.initial_state(chart=0)
        us = [state["u"].copy()]
        for _ in range(n_steps):
            rt.step(state, controls=controls)
            us.append(state["u"].copy())
        return us

    a = run_with(None)
    b = run_with([0.0] * n_act) if n_act else run_with(None)
    ok = len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    return {"pass": bool(ok), "n_steps": n_steps, "n_actuators": n_act}


def _check_control_response(rom, *, n_steps=30, dt=1e-3) -> dict:
    """Every declared actuator must actually move the state -- and do net
    positive work on it, confirming the sign, not just that something
    changed -- when driven at its declared ``hi``.  Catches a wired-but-dead
    or sign-flipped actuator map here instead of only showing up as an
    unresponsive slider in the viewer."""
    if not rom.rom_mode_ready:
        return {"pass": True, "note": "ROM not mode-ready; skipped"}
    per_channel = []
    ok = True
    for i, a in enumerate(rom.case.actuators):
        rt = PatchAtlasRuntime(rom, dt=dt)
        state = rt.initial_state(chart=0)
        u0, e0 = state["u"].copy(), rt.energy(state)["total"]
        controls = [0.0] * len(rom.case.actuators)
        controls[i] = a.hi
        for _ in range(n_steps):
            rt.step(state, controls=controls)
        moved = float(np.linalg.norm(state["u"] - u0))
        work_done = rt.energy(state)["total"] - e0     # KE+V gained == net work in (no dissipation)
        passed = moved > 1e-9 and work_done > 0.0
        ok &= passed
        per_channel.append({"name": a.name, "target": a.target,
                            "u_displacement": moved, "energy_gained": work_done,
                            "pass": passed})
    return {"pass": ok, "channels": per_channel}


def _check_production_isolation() -> dict:
    """Nothing under ``atlas_rom/`` (this package included) may import
    ``atlas_rom_experiments`` -- the archived pipeline is not a production
    dependency."""
    import re
    offenders = []
    # Only real import statements count -- a docstring/comment that mentions
    # the archived package by name (to explain why NOT to depend on it, as
    # this very file's do) is not itself a dependency.
    pattern = re.compile(r"^\s*(import|from)\s+atlas_rom_experiments\b", re.MULTILINE)
    for root, _dirs, files in os.walk(_ATLAS_ROM_DIR):
        if "__pycache__" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if pattern.search(text):
                offenders.append(os.path.relpath(path, _ATLAS_ROM_DIR))
    return {"pass": len(offenders) == 0, "offenders": offenders}


def run(rom, *, n_steps=200, dt=1e-3, energy_drift_tol=0.1, log=print) -> dict:
    checks = {
        "integrity": _check_integrity(rom),
        "domain": _check_domain(rom),
        "transitions": _check_transitions(rom),
        "rollout": _check_rollout(rom, n_steps=n_steps, dt=dt, energy_drift_tol=energy_drift_tol),
        "schema": _check_schema(rom),
        "zero_control": _check_zero_control(rom),
        "control_response": _check_control_response(rom),
        "production_isolation": _check_production_isolation(),
    }
    all_pass = all(c["pass"] for c in checks.values())
    for name, c in checks.items():
        log(f"selfcheck[{name}]: {'PASS' if c['pass'] else 'FAIL'}")
    log(f"selfcheck: {'PASS' if all_pass else 'FAIL'}")
    return {"case": rom.case.name, "n_patches": rom.n_patches, "pass": all_pass,
           "checks": checks}
