"""loader.py — rebuild a runnable ``PatchAtlasROM`` from a staged directory.

Everything needed to reconstruct the trained models lives in the weight
shapes themselves (``package.py`` never invents metadata the state dict
doesn't already carry): ``n_bodies`` and the state dicts come straight out of
``patch_atlas.pt``, and the buffered ``centre``/``scale`` normalization is
restored by ``load_state_dict`` rather than recomputed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from atlas_rom.charts import physical_chart_autoencoder as pca
from atlas_rom.charts.validity import build_topology


class PatchAtlasROM:
    """A loaded, runnable patch atlas: models + patches + topology + case."""

    def __init__(self, *, manifest, case, models, patches, cores, anchors, topology,
                 mass_fields, potential_fields, support_fields, boundary_fields,
                 n_bodies, latent_dim, path, router=None):
        self.manifest = manifest
        self.case = case
        self.mechanism = case.mechanism
        self.models = models                # list[PhysicalChartAE], one per chart
        self.patches = patches              # list[np.ndarray] support vertex ids
        self.cores = cores                  # list[np.ndarray] core vertex ids
        self.anchors = anchors              # list[np.ndarray] (n_i, d) encoded anchors
        self.topology = topology            # CloudChartTopology over the anchors
        self.n_bodies = n_bodies
        self.latent_dim = latent_dim
        self.n_patches = len(models)
        self.path = path
        self.mass_fields = mass_fields
        self.potential_fields = potential_fields
        self.support_fields = support_fields    # optional, legacy routing only (None for router-only ROMs)
        self.router = router                    # AdvRouter: the shipped chart router
        self.boundary_fields = boundary_fields
        self.control_schema = manifest.get("control_schema", case.control_schema)
        self.actuator_maps = manifest.get("actuator_maps", [])
        self.rom_mode_ready = bool(manifest.get("rom_mode_ready", False))

        from atlas_rom.spec import geometry
        bodies = geometry.build_bodies(case.mechanism)
        self.body_order = bodies["names"]
        self.masses = bodies["masses"]
        self.inertia = bodies["inertia"]
        self.com = bodies["com"]

    @classmethod
    def load(cls, path: str) -> "PatchAtlasROM":
        from atlas_rom.cases import load_spec

        path = os.path.abspath(path)
        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "patch_rom_v2":
            raise ValueError(f"{manifest_path}: unsupported format {manifest.get('format')!r}")

        case = load_spec(manifest["case"])
        artifact_path = os.path.join(path, manifest["artifact"])
        state = torch.load(artifact_path, map_location="cpu", weights_only=False)
        artifact_format = state.get("format")
        if artifact_format == "patch_atlas_v3":
            fitter = state.get("chart_fitter", {})
            from atlas_rom.charts.registry import resolve as resolve_chart_fitter
            fitter_impl = resolve_chart_fitter(fitter.get("id"))
            chart_state = state["chart_atlas"]
            physics_state = state["physics_fields"]
            state = fitter_impl.load_payload(chart_state, physics_state,
                                             config=fitter.get("config"))
        elif artifact_format is not None:
            raise ValueError(f"{artifact_path}: unsupported artifact format {artifact_format!r}")
        n_bodies = int(state["n_bodies"])
        if n_bodies != manifest["n_bodies"]:
            raise ValueError(f"{artifact_path}: n_bodies={n_bodies} != "
                             f"manifest n_bodies={manifest['n_bodies']}")

        width = manifest["width"]
        d = manifest["latent_dim"]
        encoder_kind = state.get("encoder_kind", "current")
        Model = pca._module(n_bodies, width, d=d, encoder_kind=encoder_kind)
        models = []
        for sd in state["states"]:
            centre = np.zeros(3) if "centre" not in sd else np.zeros_like(sd["centre"])
            scale = np.ones(3) if "scale" not in sd else np.ones_like(sd["scale"])
            model = Model(centre, scale)
            # strict=False: pre-gauge artifacts lack the optional whitening-gauge
            # buffers; missing means identity gauge (the sigmoid encoder), which
            # is the behaviour those artifacts were trained with.
            model.load_state_dict(sd, strict=False)
            model.eval()
            models.append(model)

        patches = [np.asarray(p, int) for p in state["patches"]]
        cores = [np.asarray(c, int) for c in state["cores"]]
        anchors = [np.asarray(a, float) for a in state["anchors"]]
        from atlas_rom.rom import learned
        # A single-chart atlas has nothing to route between; a router is required only for 2+ charts.
        required = ("mass_fields", "potential_fields", "boundary_fields") + (("router",) if len(models) > 1 else ())
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"{artifact_path}: missing learned physical fields {missing}; rebuild the patch atlas "
                             "(a missing router is added with `python -m atlas_rom.rom.router_build`)")
        mass_fields = [learned.load_mass(x) for x in state["mass_fields"]]
        potential_fields = [learned.load_scalar(x) for x in state["potential_fields"]]
        support_fields = ([learned.load_scalar(x) for x in state["support_fields"]]
                          if "support_fields" in state else None)
        boundary_fields = [learned.load_sdf(x) for x in state["boundary_fields"]]
        from atlas_rom.rom.adv_router import AdvRouter
        router = AdvRouter.from_payload(state["router"]) if "router" in state else None
        if router is not None and router.K != len(models):
            raise ValueError(f"{artifact_path}: router was trained for {router.K} charts but the atlas has {len(models)}")
        topology = build_topology(models, support_fields)

        rom = cls(manifest=manifest, case=case, models=models, patches=patches,
                   cores=cores, anchors=anchors, topology=topology,
                   mass_fields=mass_fields, potential_fields=potential_fields,
                   support_fields=support_fields, boundary_fields=boundary_fields,
                   n_bodies=n_bodies, latent_dim=d, path=path, router=router)
        if "selector" in state:
            from atlas_rom.charts.competitive_cae import PhysicalRBFSelector
            rom.selector = PhysicalRBFSelector.from_payload(state["selector"])
        return rom

    def rank_receivers(self, q_flat, candidates):
        """Rank already competent receivers with the persisted physical RBF.

        The selector is only a tie-breaker after learned support has declared a
        receiver competent; it never queries anchors, memberships, or graphs.
        """
        if not hasattr(self, "selector"):
            return list(candidates)
        return self.selector.rank(np.asarray(q_flat, float), candidates)

    # -- convenience ------------------------------------------------------------
    def decode(self, chart: int, u) -> np.ndarray:
        """Decoded packed pose (n_bodies, 3, 4) at ``(chart, u)``."""
        with torch.no_grad():
            u_t = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
            T = self.models[int(chart)].transforms(u_t).squeeze(0)
        return T.numpy()

    def encode(self, chart: int, q_flat) -> np.ndarray:
        with torch.no_grad():
            q_t = torch.as_tensor(np.asarray(q_flat, float)).unsqueeze(0)
            u = self.models[int(chart)].encode(q_t).squeeze(0)
        return u.numpy()

    def support_margin(self, chart: int, u) -> float:
        """Signed learned membership score; zero is the classifier boundary."""
        if self.support_fields is None:
            raise RuntimeError("this ROM has no support fields (router-only package); use rom.router, "
                               "(support-field routing was retired)")
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
            return float(self.support_fields[int(chart)](x).item())

    def boundary_value(self, chart: int, u) -> float:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(u, float)).unsqueeze(0)
            return float(self.boundary_fields[int(chart)](x).item())
