"""Maltese-cross (external Geneva drive, n=4 slots) mechanism.

https://www.firgelliauto.com/blogs/mechanisms/maltese-cross-mechanism

Two rigid bodies, no joints -- everything is contact:
  driver: a disc with one crank pin, driven at constant angular velocity by
          a RotatingMotor about its own fixed axle.
  wheel:  a 4-slot cross, free-floating except for the contact bearing on
          its own fixed axle. The driver's crank pin enters/exits a slot
          once per driver revolution and, in the idealized rigid-kinematic
          version of this geometry, would carry the wheel through exactly
          90 deg with zero relative sliding velocity at both the entry and
          exit point (geneva_geometry.py has the derivation: the pin's path
          is tangent to the wheel radius there). Verified the CSG'd meshes
          themselves trace that exact collision-free path (see the git
          history for the kinematic-only check). In this contact-penalty
          simulation, though, the actual index overshoots -- expect
          somewhat more than 90 deg and a couple of frames of "Newton exits
          with max iteration" noise right at the tangent point, where the
          contact Jacobian is closest to singular. That's a fidelity limit
          of driving an idealized zero-shock mechanism through IPC contact
          with a hard RotatingMotor forcing, not a geometry bug -- soften
          MOTOR_RAD_PER_S / the RotatingMotor strength / contact stiffness
          further if you want it closer to the ideal. No locking disc: an
          attempt at one is documented (and abandoned) in
          geneva_geometry.py. The driver starts mid-idle (not
          mid-engagement) so the first index plays out from a standing
          start instead of injecting spurious momentum at frame 0.

Both meshes are built procedurally with trimesh boolean CSG (no data files)
and handed to libuipc directly via uipc.geometry.trimesh(V, F).
"""

import argparse

import numpy as np
import polyscope as ps
import uipc.builtin as builtin
from asset_dir import AssetDir
from geneva_geometry import A, N_SLOTS, build_driver, build_driver_axle, build_wheel, build_wheel_axle
from polyscope import imgui
from uipc import AngleAxis, Animation, Engine, Logger, Timer, Transform, Vector3, World, view
from uipc.constitution import AffineBodyConstitution, RotatingMotor
from uipc.geometry import SimplicialComplexSlot, label_surface
from uipc.geometry import trimesh as uipc_trimesh
from uipc.gui import SceneGUI
from uipc.core import Scene

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.005
config["gravity"] = [[0.0], [0.0], [0.0]]
config["contact"]["enable"] = True
config["contact"]["d_hat"] = 0.012
config["contact"]["friction"]["enable"] = False
config["newton"]["velocity_tol"] = 0.1
scene = Scene(config)

abd = AffineBodyConstitution()
rm = RotatingMotor()
scene.contact_tabular().default_model(0.0, 1e7)


def make_geo(build_fn):
    V, F = build_fn()
    sc = uipc_trimesh(V, F)
    label_surface(sc)
    return sc


MOTOR_RAD_PER_S = np.pi / 3.0  # driver period 6 s

# driver: free ABD body, driven by RotatingMotor about its local +z.
# Start mid-idle (crank pointed away from the wheel) so the first engagement
# is a clean entry-to-exit index instead of starting half-engaged.
DRIVER_START_DEG = -90.0

driver_obj = scene.objects().create("driver")
driver_mesh = make_geo(build_driver)
abd.apply_to(driver_mesh, 1e8)
t = Transform.Identity()
t.rotate(AngleAxis(np.radians(DRIVER_START_DEG), Vector3.UnitZ()))
view(driver_mesh.transforms())[0] = t.matrix()
rm.apply_to(driver_mesh, 2000.0, motor_axis=Vector3.UnitZ(), motor_rot_vel=MOTOR_RAD_PER_S)
driver_slot, _ = driver_obj.geometries().create(driver_mesh)


def driver_animation(info: Animation.UpdateInfo):
    geo_slots = info.geo_slots()
    geo_slot: SimplicialComplexSlot = geo_slots[0]
    geo = geo_slot.geometry()
    is_constrained = geo.instances().find(builtin.is_constrained)
    view(is_constrained)[0] = 1
    RotatingMotor.animate(geo, info.dt())


scene.animator().insert(driver_obj, driver_animation)

# driver_axle: fixed bearing the driver disc's bore rides on
driver_axle_obj = scene.objects().create("driver_axle")
driver_axle_mesh = make_geo(build_driver_axle)
abd.apply_to(driver_axle_mesh, 1e8)
view(driver_axle_mesh.instances().find(builtin.is_fixed))[:] = 1
driver_axle_obj.geometries().create(driver_axle_mesh)

# wheel: free ABD body, indexed purely by contact with the driver's crank pin
wheel_obj = scene.objects().create("wheel")
wheel_mesh = make_geo(build_wheel)
abd.apply_to(wheel_mesh, 1e8)
wheel_slot, _ = wheel_obj.geometries().create(wheel_mesh)

# wheel_axle: fixed bearing the wheel's bore rides on
wheel_axle_obj = scene.objects().create("wheel_axle")
wheel_axle_mesh = make_geo(build_wheel_axle)
abd.apply_to(wheel_axle_mesh, 1e8)
view(wheel_axle_mesh.instances().find(builtin.is_fixed))[:] = 1
wheel_axle_obj.geometries().create(wheel_axle_mesh)

world.init(scene)


def wheel_angle_deg() -> float:
    T = np.asarray(view(wheel_slot.geometry().transforms())[0]).reshape(4, 4)
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def driver_angle_deg() -> float:
    T = np.asarray(view(driver_slot.geometry().transforms())[0]).reshape(4, 4)
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


cli = argparse.ArgumentParser()
cli.add_argument("--steps", type=int, default=0, help="advance N frames headlessly, skip the GUI, then exit")
args = cli.parse_args()

if args.steps > 0:
    for _ in range(args.steps):
        world.advance()
        world.retrieve()
        print(f"frame={world.frame()} driver={driver_angle_deg():+8.2f} deg  wheel={wheel_angle_deg():+8.2f} deg")
    raise SystemExit(0)

ps.init()
ps.set_window_size(1600, 1280)

sgui = SceneGUI(scene, "split")
sgui.register()
sgui.set_edge_width(1)

run = False


def on_update():
    global run
    if imgui.Button("run & stop"):
        run = not run
    imgui.Text(f"n_slots={N_SLOTS}  center_dist={A:.4f}")
    imgui.Text(f"driver angle: {driver_angle_deg():+8.2f} deg")
    imgui.Text(f"wheel angle:  {wheel_angle_deg():+8.2f} deg")
    if run:
        world.advance()
        world.retrieve()
        Timer.report()
    sgui.update()


ps.set_user_callback(on_update)
ps.show()
