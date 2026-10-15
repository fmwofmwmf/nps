"""
Four-box revolute joint chain along the X axis.

Four cubes are placed at x = 0.0, 0.5, 1.0, 1.5 (0.5 m spacing).
Adjacent boxes are connected by revolute joints (rotation axis = Z).
A fixed anchor body is connected to the leftmost box via a revolute joint;
the rest swing freely under gravity.

GUI features:
- Display current angle (degrees) for each joint.
- Driving sliders to set target angle for each joint.
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
    AffineBodyDrivingRevoluteJoint,
    AffineBodyRevoluteJoint,
)
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.gui import SceneGUI

Logger.set_level(Logger.Level.Critical)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = False
scene = Scene(config)
print(config)
scene.contact_tabular().default_model(0.5, 1.0 * unit.GPa)


pre = Matrix4x4.Identity()
pre[1, 1] = pre[2, 2] = 0.1
pre[0, 0] = 0.6

io = SimplicialComplexIO(pre)


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


# ---------------------------------------------------------------------------
# Create 4 boxes along X axis, spaced 0.5 m apart
# ---------------------------------------------------------------------------
NUM_BOXES = 4
SPACING = 0.8
NUM_JOINTS = NUM_BOXES  # anchor->box0, box0->box1, box1->box2, box2->box3

box_meshes = []
box_objects = []
box_slots = []

abd = AffineBodyConstitution()

# Create a fixed anchor body at the leftmost position
anchor_obj = scene.objects().create("anchor")
anchor_mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
abd.apply_to(anchor_mesh, 100.0 * unit.MPa)
anchor_mesh.instances().resize(1)
t_anchor = Matrix4x4.Identity()
t_anchor[0:3, 3] = np.array([-SPACING, 0.0, 0.0], dtype=np.float64)
view(anchor_mesh.transforms())[0] = t_anchor
view(anchor_mesh.instances().find(builtin.is_fixed))[0] = 1
anchor_slot = anchor_obj.geometries().create(anchor_mesh)[0]

# Create 4 free boxes
for i in range(NUM_BOXES):
    obj = scene.objects().create(f"box_{i}")
    mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    abd.apply_to(mesh, 100.0 * unit.MPa)
    mesh.instances().resize(1)

    # Position each box along X axis
    t = Matrix4x4.Identity()
    t[0:3, 3] = np.array([i * SPACING, 0.0, 0.0], dtype=np.float64)
    view(mesh.transforms())[0] = t

    slot = obj.geometries().create(mesh)[0]
    box_meshes.append(mesh)
    box_objects.append(obj)
    box_slots.append(slot)

# ---------------------------------------------------------------------------
# Create revolute joints between adjacent boxes (with driving support)
# ---------------------------------------------------------------------------
revolute_joint = AffineBodyRevoluteJoint()
driving = AffineBodyDrivingRevoluteJoint()

AXIS_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # rotation axis = Z
Y_OFFSET = 0.6

# Store joint objects and meshes for animator access
joint_objects = []
joint_meshes = []

# Joint 0: anchor -> box_0
anchor_pos = np.array([-SPACING, 0.0, 0.0], dtype=np.float32)
anchor_joint_mesh = revolute_joint.create_geometry(
    np.array([anchor_pos], dtype=np.float32),
    np.array([anchor_pos + AXIS_DIR], dtype=np.float32),
    [anchor_slot], [0], [box_slots[0]], [0], [100.0],
)
driving.apply_to(anchor_joint_mesh, 100.0)
anchor_joint_obj = scene.objects().create("joint_anchor_0")
anchor_joint_obj.geometries().create(anchor_joint_mesh)
joint_objects.append(anchor_joint_obj)
joint_meshes.append(anchor_joint_mesh)

# Joints between adjacent boxes
for i in range(NUM_BOXES - 1):
    body_i_pos = np.array([i * SPACING, 0.0, 0.0], dtype=np.float32)
    joint_mesh = revolute_joint.create_geometry(
        np.array([body_i_pos], dtype=np.float32),
        np.array([body_i_pos + AXIS_DIR], dtype=np.float32),
        [box_slots[i]], [0], [box_slots[i + 1]], [0], [100.0],
    )
    driving.apply_to(joint_mesh, 100.0)

    joint_obj = scene.objects().create(f"joint_{i}_{i + 1}")
    joint_obj.geometries().create(joint_mesh)
    joint_objects.append(joint_obj)
    joint_meshes.append(joint_mesh)

# ---------------------------------------------------------------------------
# GUI state: driving control
# ---------------------------------------------------------------------------
# aim_angles[i] = target angle for joint i (radians)
aim_angles = [0.0] * NUM_JOINTS
driving_enabled = [False] * NUM_JOINTS
# current_angles[i] = last read angle for joint i (radians), for display
current_angles = [0.0] * NUM_JOINTS


# ---------------------------------------------------------------------------
# Animator: read current angles, apply driving aim_angle
# ---------------------------------------------------------------------------
def make_joint_animator(joint_idx: int):
    """Create an animator callback for a specific joint."""

    def animate(info: Animation.UpdateInfo) -> None:
        for geo_slot in info.geo_slots():
            if geo_slot is None:
                continue
            geo = geo_slot.geometry()
            if geo is None:
                continue

            # Read current angle
            angles_attr = geo.edges().find("angle")
            if angles_attr is not None:
                current_angles[joint_idx] = float(view(angles_attr)[0])

            # Set driving constraint
            drv_ic = geo.edges().find("driving/is_constrained")
            if drv_ic is not None:
                view(drv_ic)[:] = 1 if driving_enabled[joint_idx] else 0

            # Set aim angle
            aim_attr = geo.edges().find("aim_angle")
            if aim_attr is not None and driving_enabled[joint_idx]:
                view(aim_attr)[:] = aim_angles[joint_idx]

    return animate


for idx, jobj in enumerate(joint_objects):
    scene.animator().insert(jobj, make_joint_animator(idx))

# ---------------------------------------------------------------------------
# Init & GUI
# ---------------------------------------------------------------------------
JOINT_LABELS = ["Anchor->Box0"] + [f"Box{i}->Box{i + 1}" for i in range(NUM_BOXES - 1)]

sgui = SceneGUI(scene)
world.init(scene)

ps.init()
tri_surf, _, _ = sgui.register()
tri_surf.set_edge_width(1)

run = False


def on_update():
    global run

    if imgui.Button("Run & Stop"):
        run = not run

    imgui.Separator()
    imgui.Text(f"Frame: {world.frame()}")

    # --- Joint angle display ---
    imgui.Separator()
    imgui.Text("Joint Angles (current)")
    for i in range(NUM_JOINTS):
        deg = np.degrees(current_angles[i])
        imgui.Text(f"  {JOINT_LABELS[i]}: {deg:+.2f} deg")

    # --- Driving controls ---
    imgui.Separator()
    imgui.Text("Driving Control")
    for i in range(NUM_JOINTS):
        changed_en, driving_enabled[i] = imgui.Checkbox(f"Enable##{i}", driving_enabled[i])
        imgui.SameLine()
        aim_deg = np.degrees(aim_angles[i])
        changed_a, aim_deg = imgui.SliderFloat(
            f"{JOINT_LABELS[i]}##{i}",
            aim_deg,
            -180.0,
            180.0,
        )
        if changed_a:
            aim_angles[i] = np.radians(aim_deg)

    if run:
        world.advance()
        world.retrieve()
        sgui.update()


ps.set_user_callback(on_update)
ps.show()
