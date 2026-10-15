"""package.py — stage a certified ``patch_atlas`` build into ``roms/<case>/``.

Only successful builds are packaged: the build report must read
``status: "pass"``.  Rollout evidence is recorded when available but is an
optional release gate; callers that require it opt in with
``require_rollout=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import torch

FORMAT = "patch_rom_v2"
ARTIFACT_FORMAT = "patch_atlas_v3"


def _payload_state(state: dict) -> dict:
    """Normalize legacy and v3 artifacts for package validation."""
    if state.get("format") == ARTIFACT_FORMAT:
        return {**state["chart_atlas"], **state["physics_fields"]}
    return state


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_dims(state_dict) -> tuple[int, int]:
    """(latent_dim, width) read off a ``PhysicalChartAE`` state dict's shapes --
    never guessed, so a package always matches the weights it ships.  Read from
    the decoder, whose shape (``mlp(d, 9*n_bodies)``) is invariant across every
    ``encoder_kind`` (linear / shallow / current)."""
    return int(state_dict["decoder.0.weight"].shape[1]), int(state_dict["decoder.0.weight"].shape[0])


def stage(work_dir: str, case: str, out_root: str | None = None, *,
          package_name: str | None = None,
          artifact: str = "patch_atlas.pt",
          report: str | None = None,
          rollout_report: str = "patch_atlas_rollout_report.json",
          require_pass: bool = True, require_rollout: bool = False,
          label: str | None = None, log=print) -> dict:
    """Copy a certified build from ``work_dir`` into ``roms/<package_name>/``
    and write its manifest.  ``package_name`` (default: ``case``) names the
    output directory only -- ``case`` must always be a real ``cases/<case>.py``
    spec name, so a re-sampled variant of the same mechanism (a denser work
    directory, say) can be staged under its own directory while still loading
    the one real spec at run time.  Raises ``ValueError`` on a build that did
    not pass (unless ``require_pass=False``)."""
    from atlas_rom.cases import load_spec
    package_name = package_name or case

    work_dir = os.path.abspath(work_dir)
    if report is None:
        report = ("competitive_cae_report.json" if os.path.isfile(os.path.join(work_dir, "competitive_cae_report.json"))
                  else "patch_atlas_report.json")
    artifact_path = os.path.join(work_dir, artifact)
    report_path = os.path.join(work_dir, report)
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"no {artifact!r} in {work_dir}")
    if not os.path.isfile(report_path):
        raise FileNotFoundError(f"no {report!r} in {work_dir}")

    with open(report_path, encoding="utf-8") as f:
        build_report = json.load(f)
    if require_pass and build_report.get("status") != "pass":
        raise ValueError(f"{report_path}: status={build_report.get('status')!r}, "
                         "refusing to package an unpassed build (require_pass=False overrides)")

    rollout_path = os.path.join(work_dir, rollout_report)
    rollout_summary = None
    if os.path.isfile(rollout_path):
        with open(rollout_path, encoding="utf-8") as f:
            rollout_data = json.load(f)
        trajectories = rollout_data.get("trajectories", [])
        n_pass = sum(1 for t in trajectories if t.get("pass"))
        rollout_summary = {"n_trajectories": len(trajectories), "n_pass": n_pass,
                           "all_pass": n_pass == len(trajectories) and len(trajectories) > 0}
        if require_rollout and not rollout_summary["all_pass"]:
            raise ValueError(f"{rollout_path}: {n_pass}/{len(trajectories)} rollouts passed, "
                             "refusing to package (require_rollout=False overrides)")
    elif require_rollout:
        raise ValueError(f"no {rollout_report!r} in {work_dir}: no rollout evidence to package "
                         "(require_rollout=False overrides)")

    from atlas_rom.rom import actuator_maps as _actuator_maps

    CASE = load_spec(case)
    control_check = _actuator_maps.validate(CASE)
    state = torch.load(artifact_path, map_location="cpu", weights_only=False)
    payload = _payload_state(state)
    n_bodies = int(payload["n_bodies"])
    required_fields = {"mass_fields", "potential_fields", "boundary_fields"}
    if len(payload["states"]) > 1:          # a single-chart atlas has nothing to route between
        required_fields.add("router")
    missing_fields = required_fields - set(payload)
    if missing_fields:
        raise ValueError(f"{artifact_path}: missing learned fields {sorted(missing_fields)} "
                         "(a missing router: `python -m atlas_rom.rom.router_build --work <dir> --case <case>`)")
    n_patches = len(payload["states"])
    dims = {_model_dims(sd) for sd in payload["states"]}
    if len(dims) != 1:
        raise ValueError(f"patches disagree on (latent_dim, width): {dims}")
    latent_dim, width = next(iter(dims))
    if n_bodies != CASE.n_bodies:
        raise ValueError(f"artifact n_bodies={n_bodies} != case {case!r} n_bodies={CASE.n_bodies}")

    out_root = out_root or os.path.join(os.path.dirname(os.path.dirname(work_dir)), "roms")
    out_dir = os.path.join(out_root, package_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(artifact_path, os.path.join(out_dir, "patch_atlas.pt"))
    shutil.copy2(report_path, os.path.join(out_dir, "patch_atlas_report.json"))
    if os.path.isfile(rollout_path):
        shutil.copy2(rollout_path, os.path.join(out_dir, "patch_atlas_rollout_report.json"))

    manifest = {
        "format": FORMAT,
        "case": case,
        "label": label,
        "n_bodies": n_bodies,
        "body_order": CASE.mechanism.body_order,
        "latent_dim": latent_dim,
        "width": width,
        "n_patches": n_patches,
        "control_schema": CASE.control_schema,
        "actuator_maps": control_check["actuators"],
        "artifact": "patch_atlas.pt",
        "build_report": "patch_atlas_report.json",
        "rollout_report": ("patch_atlas_rollout_report.json"
                           if os.path.isfile(rollout_path) else None),
        "source": {"work_dir": work_dir, "artifact_sha256": _sha256(artifact_path),
                   "build_report_sha256": _sha256(report_path)},
        "gates": {"build_status": build_report.get("status"),
                  "build_all_pass": build_report.get("all_pass"),
                  "n_uncovered": build_report.get("n_uncovered"),
                  "latent_gauge": build_report.get("latent_gauge", True),
                  "paper_baseline": build_report.get("paper_baseline", False),
                  "rollout": rollout_summary,
                  "rollout_required": bool(require_rollout),
                  "control_maps_ok": control_check["pass"]},
        # ROM mode's actual gate at run time (see run_patch_atlas.py): every
        # declared actuator must have a validated map, on top of the build gate
        # and an optional rollout gate.  A case with a coordinate_force
        # actuator is staged (so it can still be inspected / run in Full mode)
        # but never becomes ROM-mode-ready until that map is calibrated.
        "rom_mode_ready": bool(build_report.get("status") == "pass"
                               and (not require_rollout or (rollout_summary is not None
                                                            and rollout_summary["all_pass"]))
                               and control_check["pass"]),
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"patch_rom: staged {case!r} as {package_name!r} -> {out_dir} "
        f"(n_patches={n_patches}, latent_dim={latent_dim}, "
        f"rollout={rollout_summary})")
    return manifest
