"""cases/jansen_04.py — four Jansen legs on one crankshaft.

See `_jansen_array_common.py` for the layout formula; this file only fixes
the leg count so the leg-count speedup comparison (1, 4, 8, 16) has a real
built ROM at n=4.
"""
from __future__ import annotations

from atlas_rom.cases._jansen_array_common import make_case

CASE = make_case("jansen_04", n_legs=4)
