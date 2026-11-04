"""Write circle_arm's moving bodies as STL, in mm, at the rest pose (phi = theta = 0).

These are the collision meshes for `cases/circle_arm_contact.py`.  The arm's
collider is its bar alone: the cage never touches another moving part over a
full turn, and its envelope would exceed the SDF budget.  Meshes are tessellated
far finer than the 1.5 mm ring clearance, which the viewer's fixed mesher is not.

    cad run export_sim_meshes.py
"""
from pathlib import Path

import build123d as bd

import circle_arm as ca

OUT = Path(__file__).resolve().parent / "sim"
TOLERANCE = 0.02        # mm chord deviation
ANGULAR = 0.1           # rad

BODIES = {
    "arm": ca.arm_bar,
    "pink": ca.pink_body,
    "blue": ca.blue_body,
    "yellow": ca.yellow_body,
    "coupling_A": lambda: ca.coupling_rod(0.0),
    "coupling_B": lambda: ca.coupling_rod(ca.PHASE_B),
    "output_A": lambda: ca.output_rod(0.0),
    "output_B": lambda: ca.output_rod(ca.PHASE_B),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, build in BODIES.items():
        shape = build()
        solids = shape.solids()
        assert len(solids) == 1, f"{name}: {len(solids)} solids, expected one"
        path = OUT / f"{name}.stl"
        bd.export_stl(shape, str(path), tolerance=TOLERANCE, angular_tolerance=ANGULAR)
        size = shape.bounding_box().size
        print(f"{name:11s} {size.X:7.1f} x {size.Y:6.1f} x {size.Z:5.1f} mm  "
              f"{path.stat().st_size / 1e6:.2f} MB")
