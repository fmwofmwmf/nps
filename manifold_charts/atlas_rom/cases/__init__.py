"""cases/ — one declarative spec file per mechanism, and nothing else.

`cases/<name>.py` binds exactly one name::

    CASE = AtlasCaseSpec(...)

Adding a standard mechanism means adding one such file.  There is no registry to
edit (specs are discovered by module name), no `load()` to write, no decoder or
renderer subclass, and no build script — `python -m atlas_rom.build --case <name>`
reads the spec and produces the whole ROM.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

_HERE = os.path.dirname(os.path.abspath(__file__))


def available_specs() -> list[str]:
    """Every case that has a spec file, whether or not it has been built."""
    return sorted(m.name for m in pkgutil.iter_modules([_HERE])
                  if not m.name.startswith("_"))


def resolve_case_name(name: str) -> str:
    """Resolve an exact case name or an unambiguous prefix.

    CLI users commonly know the beginning of a case name (for example,
    ``--case steph``).  Keep the canonical on-disk name everywhere internally,
    while rejecting ambiguous prefixes instead of guessing.
    """
    names = available_specs()
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    exact = [candidate for candidate in names if candidate.lower() == normalized]
    if exact:
        return exact[0]
    matches = [candidate for candidate in names
               if candidate.lower().startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"case prefix {name!r} is ambiguous; choose one of: "
                         + ", ".join(matches))
    raise ValueError(f"unknown case {name!r}; available cases: "
                     + ", ".join(names))


def load_spec(name: str):
    """Import `cases/<name>.py` and return its `CASE`."""
    name = resolve_case_name(name)
    module = importlib.import_module(f"{__name__}.{name}")
    if not hasattr(module, "CASE"):
        raise AttributeError(f"cases/{name}.py must define CASE = AtlasCaseSpec(...)")
    spec = module.CASE
    if spec.name != name:
        raise ValueError(f"cases/{name}.py declares name={spec.name!r}; "
                         f"the spec name must match its file name")
    return spec


def load_sdf_system(name: str):
    """Return an optional first-class SDF system without reading ``CASE``."""
    name = resolve_case_name(name)
    module = importlib.import_module(f"{__name__}.{name}")
    return getattr(module, "SDF_SYSTEM", None)
