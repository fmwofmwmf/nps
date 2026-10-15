"""
Driving prismatic joint + external force.

Two phases controlled by the Animator on the joint linemesh edges:
- Frames [0, 100]: driving only (`driving/is_constrained` = 1, `aim_distance` motor).
- Frames (100, 200): external force only (`external_force/is_constrained` = 1).
"""

import numpy as np
import polyscope as ps
from asset_dir import AssetDir
from polyscope import imgui
from uipc import Animation, Engine, Logger, Matrix4x4, Scene, World, view
import uipc.builtin as builtin
import uipc.unit as unit
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyDrivingPrismaticJoint,
    AffineBodyPrismaticJoint,
    AffineBodyPrismaticJointExternalForce,
)
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.gui import SceneGUI

Logger.set_level(Logger.Level.Info)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = True
config["line_search"]["report_energy"] = True
scene = Scene(config)
scene.contact_tabular().default_model(0.5, 1.0 * unit.GPa)

pre = Matrix4x4.Identity()
pre[0, 0] = pre[1, 1] = pre[2, 2] = 0.4

io = SimplicialComplexIO(pre)


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


left_link = scene.objects().create("left")
right_link = scene.objects().create("right")

abd = AffineBodyConstitution()

left_mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
abd.apply_to(left_mesh, 100.0 * unit.MPa)
left_mesh.instances().resize(1)
tl = Matrix4x4.Identity()
tl[0:3, 3] = np.array([-0.6, 0.0, 0.0], dtype=np.float64)
view(left_mesh.transforms())[0] = tl
view(left_mesh.instances().find(builtin.is_fixed))[0] = 1

right_mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
abd.apply_to(right_mesh, 100.0 * unit.MPa)
right_mesh.instances().resize(1)
tr = Matrix4x4.Identity()
tr[0:3, 3] = np.array([0.6, 0.0, 0.0], dtype=np.float64)
view(right_mesh.transforms())[0] = tr
view(right_mesh.instances().find(builtin.is_fixed))[0] = 0

left_slot = left_link.geometries().create(left_mesh)[0]
right_slot = right_link.geometries().create(right_mesh)[0]

# Prismatic joint along Z axis
prismatic = AffineBodyPrismaticJoint()
pos0s = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
pos1s = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
joint_mesh = prismatic.create_geometry(
    pos0s, pos1s, [left_slot], [0], [right_slot], [0], [100.0]
)

# 2) Driving constitution (active during frames 1..100)
driving = AffineBodyDrivingPrismaticJoint()
driving.apply_to(joint_mesh, 100.0)

# 3) External force constitution (active during frames 101..200)
ext = AffineBodyPrismaticJointExternalForce()
ext.apply_to(joint_mesh, 0.0)

joint_object = scene.objects().create("prismatic_joint")
joint_object.geometries().create(joint_mesh)


def animate_joint(info: Animation.UpdateInfo) -> None:
    for geo_slot in info.geo_slots():
        geo: SimplicialComplex = geo_slot.geometry()

        distances = geo.edges().find("distance")
        distances_view = view(distances)

        driving_phase = info.frame() <= 100

        # Toggle driving constraint
        drv_ic = geo.edges().find("driving/is_constrained")
        view(drv_ic)[:] = 1 if driving_phase else 0

        # Toggle external force constraint
        ext_ic = geo.edges().find("external_force/is_constrained")
        view(ext_ic)[:] = 0 if driving_phase else 1

        # Driving: set aim_distance based on velocity
        aim_dist = geo.edges().find("aim_distance")
        aim0 = float(view(aim_dist)[0])
        if driving_phase:
            velocity = -10.0 if info.frame() <= 50 else 10.0
            view(aim_dist)[:] = distances_view + info.dt() * velocity
            aim0 = float(view(aim_dist)[0])

        # External force
        ext_attr = geo.edges().find("external_force")
        ext0 = 0.0
        if driving_phase:
            view(ext_attr)[:] = 0.0
        else:
            ext0 = 1000.0 if info.frame() <= 150 else -1000.0
            view(ext_attr)[:] = ext0

        phase = "driving" if driving_phase else "external"
        print(
            f"Frame {info.frame()} phase={phase} "
            f"aim_distance[0]={aim0:.4f} distance[0]={float(distances_view[0]):.4f} "
            f"ext_force[0]={ext0:.2f}"
        )


scene.animator().insert(joint_object, animate_joint)

sgui = SceneGUI(scene)
world.init(scene)

ps.init()
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update():
    global run
    if imgui.Button("run & stop"):
        run = not run

    if run and world.frame() < 200:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()
