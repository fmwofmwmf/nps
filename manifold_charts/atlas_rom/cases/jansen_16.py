"""cases/jansen_16.py — sixteen Jansen legs on one crankshaft.

See `_jansen_array_common.py` for the layout formula; this file only fixes
the leg count so the leg-count speedup comparison (1, 4, 8, 16) has a real
built ROM at n=16.
"""
from __future__ import annotations

from atlas_rom.cases._jansen_array_common import make_case

CASE = make_case("jansen_16", n_legs=16)
