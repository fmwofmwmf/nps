"""router_build.py -- train the chart router and store it in a build's ``patch_atlas.pt``.

The router replaces the per-chart support fields as the package's routing authority: it is stored as
``physics_fields["router"]`` and read by ``loader.PatchAtlasROM`` / ``runtime.PatchAtlasRuntime``.  It needs only the
final chart autoencoders, their anchors, the dataset poses and the case oracle (build-time label source); it does not
need the staged ROM or any of the mass / potential / boundary fields.

    python -m atlas_rom.rom.router_build --work <dir with patch_atlas.pt> --case <case> [--dataset <dir>]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from atlas_rom.charts import physical_chart_autoencoder as pca
from atlas_rom.sample import new_dataset as nd
from atlas_rom.rom.adv_labels import train_router


def _model_dims(sd):
    """(latent dim, width) from an autoencoder state dict, for any encoder depth: the width is the first encoder
    layer's output, the latent dim the last encoder layer's output."""
    layers = sorted((int(k.split(".")[1]), k) for k in sd if k.startswith("encoder.") and k.endswith(".weight"))
    return int(sd[layers[-1][1]].shape[0]), int(sd[layers[0][1]].shape[0])


def _models_from_artifact(state):
    chart = state["chart_atlas"] if state.get("format") == "patch_atlas_v3" else state
    n_bodies = int(chart["n_bodies"])
    kind = chart.get("encoder_kind", "current")
    models = []
    for sd in chart["states"]:
        d, width = _model_dims(sd)
        Model = pca._module(n_bodies, width, d=d, encoder_kind=kind)
        m = Model(np.zeros_like(sd["centre"]), np.ones_like(sd["scale"]))
        m.load_state_dict(sd, strict=False)
        models.append(m.eval())
    return models, [np.asarray(a, float) for a in chart["anchors"]]


def _refresh_manifest_hash(rom_dir: str, artifact: str) -> None:
    """If ``rom_dir`` is a staged ROM (has a manifest), record the rewritten artifact's new sha256."""
    manifest_path = os.path.join(rom_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return
    from atlas_rom.rom.package import _sha256
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("source", {})["artifact_sha256"] = _sha256(os.path.join(rom_dir, artifact))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def attach(work_dir: str, case: str, *, dataset_dir: str | None = None, artifact: str = "patch_atlas.pt",
           n_per: int = 300, epochs: int = 4000, device: str | None = None, log=print) -> dict:
    """Train the router for ``<work_dir>/<artifact>`` and write it into ``physics_fields["router"]``."""
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate import oracle_eval
    path = os.path.join(work_dir, artifact)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "patch_atlas_v3":
        raise ValueError(f"{path}: router_build supports patch_atlas_v3 artifacts only (got {state.get('format')!r})")
    models, anchors = _models_from_artifact(state)
    if len(models) < 2:
        log("router: single-chart atlas, nothing to route between -- no router stored")
        return {}
    q_all = np.asarray(nd.load(dataset_dir or work_dir)["q"], float)
    oracle = oracle_eval._build_oracle(load_spec(case))
    payload, report = train_router(models, anchors, q_all, oracle, n_per=n_per, epochs=epochs,
                                   device=device or ("cuda" if torch.cuda.is_available() else "cpu"), log=log)
    state["physics_fields"]["router"] = payload
    torch.save(state, path)
    _refresh_manifest_hash(work_dir, artifact)
    report.update(case=case, artifact=path, dataset=str(dataset_dir or work_dir))
    with open(os.path.join(work_dir, "router_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"router: wrote physics_fields['router'] into {path}")
    return report


def embed(rom_dir: str, router_file: str, *, artifact: str = "patch_atlas.pt", backup: bool = True) -> str:
    """Store an already-trained router (a ``{"state", "dmax", "K", "sc", "eps"}`` file) in a staged ROM's artifact.

    Migration path for packages built before the router was part of the package.  The previous artifact is kept as
    ``<artifact>.pre_router``."""
    import shutil
    path = os.path.join(rom_dir, artifact)
    state = torch.load(path, map_location="cpu", weights_only=False)
    payload = torch.load(router_file, map_location="cpu", weights_only=False)
    need = {"state", "dmax", "K", "sc", "eps"}
    if not need <= set(payload):
        raise ValueError(f"{router_file}: not a router file (missing {sorted(need - set(payload))})")
    n_charts = len(state["chart_atlas"]["states"])
    if int(payload["K"]) != n_charts:
        raise ValueError(f"{router_file}: router K={payload['K']} but {path} has {n_charts} charts")
    if backup and not os.path.exists(path + ".pre_router"):
        shutil.copy2(path, path + ".pre_router")
    state["physics_fields"]["router"] = payload
    torch.save(state, path)
    _refresh_manifest_hash(rom_dir, artifact)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="directory containing the build's patch_atlas.pt")
    ap.add_argument("--case", required=True)
    ap.add_argument("--dataset", default=None, help="directory containing new_dataset.npz (default: --work)")
    ap.add_argument("--n-per", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--embed-file", default=None,
                    help="instead of training, embed this existing router file into --work's artifact")
    a = ap.parse_args()
    if a.embed_file:
        print("embedded into", embed(a.work, a.embed_file))
        return
    attach(a.work, a.case, dataset_dir=a.dataset, n_per=a.n_per, epochs=a.epochs, device=a.device)


if __name__ == "__main__":
    main()
