"""patch_rom — package format, loader, and runtime for a built ``patch_atlas``.

``patch_atlas.py`` builds and certifies patches; nothing in that module reloads
them later or turns them into a runnable ROM.  This package is the other half:

    fit_fields.py    distill the learned per-chart mass/potential/support/
                     boundary fields (``learned.py``) into a raw
                     ``patch_atlas.pt`` build, BEFORE it can be staged
    package.py       stage a certified, field-augmented build
                     (``work/<case>/``) into a versioned, self-contained
                     ``roms/<case>/`` directory, gated on
                     ``actuator_maps.validate``
    loader.py        ``PatchAtlasROM.load(path)`` reconstructs the trained
                     models, patches, chart topology, and learned fields
                     from a staged directory
    ops.py           JVP/VJP reduced mass/potential/actuator operators --
                     the teacher ``fit_fields.py`` distills the learned
                     fields from, and the analytic fallback for actuator maps
    actuator_maps.py which declared control channels resolve to a validated
                     generalized-force map (``joint_torque``, always;
                     ``coordinate_force``, not yet -- see its docstring)
    runtime.py       ``PatchAtlasRuntime`` — passive + scripted-actuator
                     stepping over the LEARNED fields, chart hysteresis, and
                     safe exit outside the trusted domain
    selfcheck.py     package integrity, domain/transition/rollout regression,
                     and Full/ROM control-parity checks (``run_patch_atlas
                     --selfcheck``)
"""
from __future__ import annotations

__all__ = ["PatchAtlasROM", "stage", "PatchAtlasRuntime", "selfcheck"]


def __getattr__(name):
    """Keep field-training imports independent of optional runtime helpers."""
    if name == "PatchAtlasROM":
        from atlas_rom.rom.loader import PatchAtlasROM
        return PatchAtlasROM
    if name == "stage":
        from atlas_rom.rom.package import stage
        return stage
    if name == "PatchAtlasRuntime":
        from atlas_rom.rom.runtime import PatchAtlasRuntime
        return PatchAtlasRuntime
    if name == "selfcheck":
        from atlas_rom.rom.selfcheck import run
        return run
    raise AttributeError(name)
