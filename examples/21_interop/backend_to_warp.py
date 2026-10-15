import numpy as np
import uipc
import uipc.adapter.warp
import warp as wp
from asset_dir import AssetDir
from uipc import Logger, Timer
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.core import (
    AffineBodyStateAccessorFeature,
    Engine,
    FiniteElementStateAccessorFeature,
    Scene,
    World,
)
from uipc.geometry import flip_inward_triangles, label_surface, label_triangle_orient, tetmesh
from uipc.unit import GPa, MPa, kPa

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine("cuda", workspace)
world = World(engine)
config = Scene.default_config()
dt = 0.02
config["dt"] = dt
config["gravity"] = [[0.0], [-9.8], [0.0]]
config["contact"]["enable"] = False
scene = Scene(config)

# create constitution and contact model
abd = AffineBodyConstitution()
snh = StableNeoHookean()

# friction ratio and contact resistance
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create a regular tetrahedron
Vs = np.array([[0, 1, 0], [0, 0, 1], [-np.sqrt(3) / 2, 0, -0.5], [np.sqrt(3) / 2, 0, -0.5]])
Ts = np.array([[0, 1, 2, 3]])

# setup a base mesh to reduce the later work
base_mesh = tetmesh(Vs, Ts)
# apply the constitution and contact model to the base mesh
abd.apply_to(base_mesh, 100 * MPa)
# apply the default contact model to the base mesh
default_element.apply_to(base_mesh)

# label the surface, enable the contact
label_surface(base_mesh)
# label the triangle orientation to export the correct surface mesh
label_triangle_orient(base_mesh)
# flip the triangles inward for better rendering
base_mesh = flip_inward_triangles(base_mesh)

# ABD mesh: upper falling body
abd_mesh = base_mesh.copy()
pos_view = uipc.view(abd_mesh.positions())
pos_view += uipc.Vector3.UnitY() * 1.5

# ABD mesh: fixed lower body
abd_mesh_fixed = base_mesh.copy()
is_fixed = abd_mesh_fixed.instances().find(uipc.builtin.is_fixed)
is_fixed_view = uipc.view(is_fixed)
is_fixed_view[:] = 1

# FEM mesh: soft deformable body
fem_mesh = tetmesh(Vs, Ts)
snh.apply_to(fem_mesh, ElasticModuli.youngs_poisson(kPa * 20, 0.49))
default_element.apply_to(fem_mesh)
label_surface(fem_mesh)
label_triangle_orient(fem_mesh)
fem_mesh = flip_inward_triangles(fem_mesh)
fem_pos_view = uipc.view(fem_mesh.positions())
fem_pos_view += uipc.Vector3.UnitX() * 2.0  # offset to the side

# create objects
object1 = scene.objects().create("abd_upper")
object1.geometries().create(abd_mesh)

object2 = scene.objects().create("abd_lower")
object2.geometries().create(abd_mesh_fixed)

object3 = scene.objects().create("fem_soft")
object3.geometries().create(fem_mesh)

world.init(scene)

# --- ABD state accessor ---
abd_state_accessor: AffineBodyStateAccessorFeature = world.features().find(AffineBodyStateAccessorFeature)  # ty:ignore[invalid-assignment]
assert abd_state_accessor is not None, "ABD state accessor not found"
abd_body_count = 2  # abd_upper + abd_lower
abd_transform_wp = uipc.adapter.warp.buffer(abd_body_count, dtype=wp.mat44d, device="cuda")
abd_velocity_wp = uipc.adapter.warp.buffer(abd_body_count, dtype=wp.mat44d, device="cuda")
abd_state_accessor.copy_transform_to(abd_transform_wp.buffer_view())
abd_state_accessor.copy_velocity_to(abd_velocity_wp.buffer_view())

print("ABD transform:", abd_transform_wp.warp(), abd_transform_wp.warp().device)
print("ABD velocity:", abd_velocity_wp.warp(), abd_velocity_wp.warp().device)

# --- FEM state accessor ---
fem_state_accessor: FiniteElementStateAccessorFeature = world.features().find(FiniteElementStateAccessorFeature)  # ty:ignore[invalid-assignment]
assert fem_state_accessor is not None, "FEM state accessor not found"
fem_body_count = len(Vs)  # 4 vertices
fem_position_wp = uipc.adapter.warp.buffer(fem_body_count, dtype=wp.vec3d, device="cuda")
fem_velocity_wp = uipc.adapter.warp.buffer(fem_body_count, dtype=wp.vec3d, device="cuda")
fem_state_accessor.copy_position_to(fem_position_wp.buffer_view())
fem_state_accessor.copy_velocity_to(fem_velocity_wp.buffer_view())
print("FEM position:", fem_position_wp.warp(), fem_position_wp.warp().device)
print("FEM velocity:", fem_velocity_wp.warp(), fem_velocity_wp.warp().device)
