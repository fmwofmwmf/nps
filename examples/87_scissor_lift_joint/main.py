"""
Single-pair scissor lift, built from body-local revolute/prismatic joints.

Stage 1 ("build it first") for the scissor-lift atlas-ROM case.

A first pass wired leg_b's ends directly to the fixed ground and to the
platform with plain prismatic joints. That's wrong: a prismatic joint
here permits translation along its axis but forbids ANY relative
rotation between the two bodies it connects. With leg_b prismatic-tied
at both ends to non-rotating bodies (ground, platform), its own
orientation was locked to identity forever -- the whole assembly froze.

A real scissor foot/roller needs *both* slide and free rotation. That
takes two joints in series, not one: a tiny roller-block body slides
via prismatic (translation only -- fine, the block itself needn't
rotate), then a revolute joint from that block to the leg gives the
leg free rotation as it opens/closes. Both roller ends use this
pattern below.

Bodies (7 total: 2 fixed anchors + 5 moving):
  ground_pin     fixed anchor, base pivot for leg_a
  ground_rail    fixed anchor, base rail for the leg_b roller
  roller_bottom  free body: slides on ground_rail, leg_b pivots on it
  leg_a          bar, bottom pinned to ground_pin (revolute)
  leg_b          bar, both ends ride rollers (revolute onto each roller)
  roller_top     free body: leg_b pivots on it, slides on the platform
  platform       horizontal top bar

Joints (7 total, batched as 5 revolute + 2 prismatic):
  J1  ground_pin    <-> leg_a         revolute  (base pin)
  J2  ground_rail   <-> roller_bottom prismatic (base rail, axis = world X)
  J2b roller_bottom <-> leg_b         revolute  (bottom roller pivot)
  J3  leg_a         <-> leg_b         revolute  (scissor pivot -- at both
                                                 legs' own centers, since
                                                 the legs are equal length
                                                 and cross at their
                                                 midpoints by construction)
  J4  leg_a         <-> platform      revolute  (platform pin)
  J5b leg_b         <-> roller_top    revolute  (top roller pivot)
  J5  roller_top    <-> platform      prismatic (platform rail, axis = world X)

Because the scissor pivot sits at each leg's exact midpoint, the
bottom-half and top-half leg lengths are equal, so the base span and
platform span open and close in lockstep as the legs rotate -- the
platform stays level as an emergent consequence of that symmetry, not
because anything constrains its rotation directly. Height varies with
the legs' pivot angle: this one-pair mechanism has a single DOF.

Scene layout (no gravity, no contact): legs are 1.0 m bars set at
+/-53.13 deg from vertical so a 0.6 m horizontal base span produces a
0.8 m rise. All bodies start unrotated except leg_a/leg_b, so every
prismatic axis is trivially world-X in the sliding body's own local
frame -- no inverse-rotation bookkeeping needed (unlike a version that
tried to prismatic-joint the tilted leg directly).
"""

import argparse

import numpy as np
import polyscope as ps
import uipc.builtin as builtin
import uipc.unit as unit
from asset_dir import AssetDir
from polyscope import imgui
from uipc import AngleAxis, Engine, Logger, Scene, Transform, Vector3, World, view
from uipc.constitution import AffineBodyConstitution, AffineBodyPrismaticJoint, AffineBodyRevoluteJoint
from uipc.geometry import (
    SimplicialComplex,
    SimplicialComplexIO,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.gui import SceneGUI

Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)

config = Scene.default_config()
config["dt"] = 0.01
config["contact"]["enable"] = False
config["line_search"]["report_energy"] = True
config["sanity_check"]["enable"] = 0
scene = Scene(config)

# ---------------------------------------------------------------------------
# Geometry: two 1.0 m legs crossing at (0, 0.4, 0), a 0.6 m platform on top,
# two tiny fixed anchors, and two tiny free roller blocks.
# ---------------------------------------------------------------------------
LEG_LEN = 1.0
LEG_THICK = 0.06
PLATFORM_LEN = 0.6
PLATFORM_THICK = 0.06
ANCHOR_SIZE = 0.06
ROLLER_SIZE = 0.06

BASE_HALF_SPAN = 0.3
TOP_HEIGHT = float(np.sqrt(LEG_LEN**2 - (2.0 * BASE_HALF_SPAN) ** 2))  # 0.8

THETA_A = float(np.arctan2(TOP_HEIGHT, 2.0 * BASE_HALF_SPAN))  # leg_a tilt, ~53.13 deg
THETA_B = float(np.pi - THETA_A)  # leg_b tilt (mirrored), ~126.87 deg

abd = AffineBodyConstitution()


def process_surface(sc: SimplicialComplex) -> SimplicialComplex:
    label_surface(sc)
    label_triangle_orient(sc)
    return flip_inward_triangles(sc)


def make_io(scale):
    pre = Transform.Identity()
    pre.scale(scale)
    return SimplicialComplexIO(pre)


io_leg = make_io([LEG_LEN, LEG_THICK, LEG_THICK])
io_platform = make_io([PLATFORM_LEN, PLATFORM_THICK, PLATFORM_THICK])
io_anchor = make_io([ANCHOR_SIZE, ANCHOR_SIZE, ANCHOR_SIZE])
io_roller = make_io([ROLLER_SIZE, ROLLER_SIZE, ROLLER_SIZE])


def make_abd_body(io: SimplicialComplexIO, position, rotation: AngleAxis = AngleAxis.Identity(), fixed: bool = False):
    mesh = process_surface(io.read(f"{AssetDir.tetmesh_path()}/cube.msh"))
    abd.apply_to(mesh, 100.0 * unit.MPa)
    mesh.instances().resize(1)

    t = Transform.Identity()
    t.translate(position)
    t.rotate(rotation)
    view(mesh.transforms())[0] = t.matrix()
    if fixed:
        view(mesh.instances().find(builtin.is_fixed))[0] = 1
    return mesh


_unit_x = np.asarray(Vector3.UnitX(), dtype=np.float64).reshape(3)
_unit_z = np.asarray(Vector3.UnitZ(), dtype=np.float64).reshape(3)

# Bodies -----------------------------------------------------------------
ground_pin_obj = scene.objects().create("ground_pin")
ground_pin_mesh = make_abd_body(io_anchor, [-BASE_HALF_SPAN, 0.0, 0.0], fixed=True)
ground_pin_slot, _ = ground_pin_obj.geometries().create(ground_pin_mesh)

ground_rail_obj = scene.objects().create("ground_rail")
ground_rail_mesh = make_abd_body(io_anchor, [BASE_HALF_SPAN, 0.0, 0.0], fixed=True)
ground_rail_slot, _ = ground_rail_obj.geometries().create(ground_rail_mesh)

roller_bottom_obj = scene.objects().create("roller_bottom")
roller_bottom_mesh = make_abd_body(io_roller, [BASE_HALF_SPAN, 0.0, 0.0])
roller_bottom_slot, _ = roller_bottom_obj.geometries().create(roller_bottom_mesh)

leg_a_obj = scene.objects().create("leg_a")
leg_a_mesh = make_abd_body(io_leg, [0.0, TOP_HEIGHT / 2.0, 0.0], AngleAxis(THETA_A, Vector3.UnitZ()))
leg_a_slot, _ = leg_a_obj.geometries().create(leg_a_mesh)

leg_b_obj = scene.objects().create("leg_b")
leg_b_mesh = make_abd_body(io_leg, [0.0, TOP_HEIGHT / 2.0, 0.0], AngleAxis(THETA_B, Vector3.UnitZ()))
leg_b_slot, _ = leg_b_obj.geometries().create(leg_b_mesh)

roller_top_obj = scene.objects().create("roller_top")
roller_top_mesh = make_abd_body(io_roller, [-BASE_HALF_SPAN, TOP_HEIGHT, 0.0])
roller_top_slot, _ = roller_top_obj.geometries().create(roller_top_mesh)

platform_obj = scene.objects().create("platform")
platform_mesh = make_abd_body(io_platform, [0.0, TOP_HEIGHT, 0.0])
platform_slot, _ = platform_obj.geometries().create(platform_mesh)

# Local attachment points -------------------------------------------------
# Leg local frame: -X end = bottom, +X end = top, origin = center (which
# coincides with the world scissor-pivot point for both legs).
leg_bottom_local = np.array([-LEG_LEN / 2.0, 0.0, 0.0])
leg_top_local = np.array([LEG_LEN / 2.0, 0.0, 0.0])
leg_center_local = np.array([0.0, 0.0, 0.0])

platform_left_local = np.array([-PLATFORM_LEN / 2.0, 0.0, 0.0])
platform_right_local = np.array([PLATFORM_LEN / 2.0, 0.0, 0.0])

anchor_local = np.array([0.0, 0.0, 0.0])
roller_local = np.array([0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Revolute joints: J1 (base pin), J2b (bottom roller pivot), J3 (scissor
# pivot), J4 (platform pin), J5b (top roller pivot)
# ---------------------------------------------------------------------------
revolute = AffineBodyRevoluteJoint()

rev_l_pos0 = np.array([anchor_local, roller_local, leg_center_local, leg_top_local, leg_top_local])
rev_l_pos1 = rev_l_pos0 + _unit_z
rev_r_pos0 = np.array([leg_bottom_local, leg_bottom_local, leg_center_local, platform_right_local, roller_local])
rev_r_pos1 = rev_r_pos0 + _unit_z

rev_l_slots = [ground_pin_slot, roller_bottom_slot, leg_a_slot, leg_a_slot, leg_b_slot]
rev_r_slots = [leg_a_slot, leg_b_slot, leg_b_slot, platform_slot, roller_top_slot]
rev_l_instance_ids = np.zeros(5, dtype=np.int32)
rev_r_instance_ids = np.zeros(5, dtype=np.int32)
rev_strength_ratios = np.full(5, 100.0, dtype=np.float64)

revolute_mesh = revolute.create_geometry(
    rev_l_pos0, rev_l_pos1, rev_r_pos0, rev_r_pos1,
    rev_l_slots, rev_l_instance_ids, rev_r_slots, rev_r_instance_ids,
    rev_strength_ratios,
)
revolute_object = scene.objects().create("revolute_joints")
revolute_geo_slot, _ = revolute_object.geometries().create(revolute_mesh)
REVOLUTE_LABELS = ["J1  ground_pin<->leg_a (base pin)",
                    "J2b roller_bottom<->leg_b (bottom roller pivot)",
                    "J3  leg_a<->leg_b (scissor pivot)",
                    "J4  leg_a<->platform (platform pin)",
                    "J5b leg_b<->roller_top (top roller pivot)"]

# ---------------------------------------------------------------------------
# Prismatic joints: J2 (base rail), J5 (platform rail) -- both slide along
# world +X; every body involved starts unrotated, so each local axis is
# trivially world X, no inverse-rotation transform needed.
# ---------------------------------------------------------------------------
prismatic = AffineBodyPrismaticJoint()

pris_l_pos0 = np.array([anchor_local, roller_local])
pris_l_pos1 = pris_l_pos0 + _unit_x
pris_r_pos0 = np.array([roller_local, platform_left_local])
pris_r_pos1 = pris_r_pos0 + _unit_x

pris_l_slots = [ground_rail_slot, roller_top_slot]
pris_r_slots = [roller_bottom_slot, platform_slot]
pris_l_instance_ids = np.zeros(2, dtype=np.int32)
pris_r_instance_ids = np.zeros(2, dtype=np.int32)
pris_strength_ratios = np.full(2, 100.0, dtype=np.float64)

prismatic_mesh = prismatic.create_geometry(
    pris_l_pos0, pris_l_pos1, pris_r_pos0, pris_r_pos1,
    pris_l_slots, pris_l_instance_ids, pris_r_slots, pris_r_instance_ids,
    pris_strength_ratios,
)
prismatic_object = scene.objects().create("prismatic_joints")
prismatic_geo_slot, _ = prismatic_object.geometries().create(prismatic_mesh)
PRISMATIC_LABELS = ["J2 ground_rail<->roller_bottom (base rail)",
                     "J5 roller_top<->platform (platform rail)"]


def _read_transform(slot):
    return np.asarray(view(slot.geometry().transforms())[0]).reshape(4, 4)


def to_world(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ p4)[:3]


def print_joint_state() -> None:
    rev_geo = revolute_geo_slot.geometry()
    angles = rev_geo.edges().find("angle")
    if angles is None:
        print(f"frame={world.frame()} revolute angle attribute not ready yet")
    else:
        v = view(angles)
        for i, label in enumerate(REVOLUTE_LABELS):
            rad = float(v[i])
            print(f"frame={world.frame()} {label}: angle={rad:+.4f} rad ({np.degrees(rad):+.2f} deg)")

    pris_geo = prismatic_geo_slot.geometry()
    distances = pris_geo.edges().find("distance")
    if distances is None:
        print(f"frame={world.frame()} prismatic distance attribute not ready yet")
    else:
        v = view(distances)
        for i, label in enumerate(PRISMATIC_LABELS):
            d = float(v[i])
            print(f"frame={world.frame()} {label}: distance={d:+.4f} m")


def platform_height() -> float:
    return float(_read_transform(platform_slot)[1, 3])


def check_revolute_attachments(atol: float = 1e-3) -> bool:
    """T_L * l_pos ~= T_R * r_pos for every revolute joint."""
    checks = [
        ("J1", ground_pin_slot, leg_a_slot, anchor_local, leg_bottom_local),
        ("J2b", roller_bottom_slot, leg_b_slot, roller_local, leg_bottom_local),
        ("J3", leg_a_slot, leg_b_slot, leg_center_local, leg_center_local),
        ("J4", leg_a_slot, platform_slot, leg_top_local, platform_right_local),
        ("J5b", leg_b_slot, roller_top_slot, leg_top_local, roller_local),
    ]
    for name, l_slot, r_slot, lp, rp in checks:
        TL, TR = _read_transform(l_slot), _read_transform(r_slot)
        lhs, rhs = to_world(TL, lp), to_world(TR, rp)
        assert np.allclose(lhs, rhs, atol=atol), (
            f"frame={world.frame()} {name} mismatch: T_L*l={lhs} != T_R*r={rhs} "
            f"(|diff|={np.linalg.norm(lhs - rhs):.6e}, atol={atol})"
        )
    return True


def check_prismatic_axes_collinear(atol: float = 5e-3) -> bool:
    checks = [
        ("J2", ground_rail_slot, roller_bottom_slot, anchor_local, anchor_local + _unit_x, roller_local, roller_local + _unit_x),
        ("J5", roller_top_slot, platform_slot, roller_local, roller_local + _unit_x, platform_left_local, platform_left_local + _unit_x),
    ]
    for name, l_slot, r_slot, lp0, lp1, rp0, rp1 in checks:
        TL, TR = _read_transform(l_slot), _read_transform(r_slot)
        dir_L = to_world(TL, lp1) - to_world(TL, lp0)
        dir_R = to_world(TR, rp1) - to_world(TR, rp0)
        cross = np.cross(dir_L, dir_R)
        assert np.linalg.norm(cross) < atol, (
            f"frame={world.frame()} {name} axes not parallel: dir_L={dir_L} dir_R={dir_R} |cross|={np.linalg.norm(cross):.6e}"
        )
        offset = to_world(TR, rp0) - to_world(TL, lp0)
        axis = dir_L / max(np.linalg.norm(dir_L), 1e-12)
        perp = offset - np.dot(offset, axis) * axis
        assert np.linalg.norm(perp) < atol, (
            f"frame={world.frame()} {name} axes not collinear: |perp|={np.linalg.norm(perp):.6e}"
        )
    return True


# ---------------------------------------------------------------------------
# Init & GUI
# ---------------------------------------------------------------------------
sgui = SceneGUI(scene)
world.init(scene)

print_joint_state()
check_revolute_attachments()
check_prismatic_axes_collinear()
print(f"frame={world.frame()} platform height={platform_height():.4f} m")

cli = argparse.ArgumentParser()
cli.add_argument("--steps", type=int, default=0, help="advance N frames headlessly, skip the GUI, then exit")
args = cli.parse_args()

if args.steps > 0:
    for _ in range(args.steps):
        world.advance()
        world.retrieve()
        print_joint_state()
        check_revolute_attachments()
        check_prismatic_axes_collinear()
        print(f"frame={world.frame()} platform height={platform_height():.4f} m")
    raise SystemExit(0)

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
    imgui.Text(f"Platform height: {platform_height():.4f} m")

    if run:
        world.advance()
        world.retrieve()
        sgui.update()
        print_joint_state()
        check_revolute_attachments()
        check_prismatic_axes_collinear()
        print(f"frame={world.frame()} platform height={platform_height():.4f} m")


ps.set_user_callback(on_update)
ps.show()
