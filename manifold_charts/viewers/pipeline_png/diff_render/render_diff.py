# Renders the differential in ../diff.blend, following the setup used in
# kirigami_test / surface-filling-curve-render (Cycles, transparent film,
# shadow catcher only, three-point lights + ambient, alpha shadow threshold).
#
# usage: blender -b --python render_diff.py -- <overview|motion|torus|all> [--fast]
import sys, os, math
import numpy as np
import bpy, bmesh
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BLEND = os.path.join(HERE, '..', 'diff.blend')

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
FAST = '--fast' in argv
modes = [a for a in argv if not a.startswith('--')] or ['all']
if 'all' in modes:
    modes = ['overview', 'configs', 'motion', 'torus', 'samples', 'tangent', 'dense', 'unfold']

# ---------------------------------------------------------------- palette
def srgb(r, g, b, a=1.0):
    f = lambda c: (c / 255.0) / 12.92 if c / 255.0 <= 0.04045 else ((c / 255.0 + 0.055) / 1.055) ** 2.4
    return (f(r), f(g), f(b), a)

TOP_GEAR = srgb(240, 135, 95)      # soft orange
BOTTOM_GEAR = srgb(105, 155, 210)  # soft blue
PINION = srgb(245, 205, 105)       # soft yellow
CARRIER = srgb(125, 133, 145)      # slate grey
GREEN = srgb(135, 215, 115)        # light green: side-gear (coupled) rotation
PURPLE = srgb(150, 105, 210)       # purple: pinions walking around the center
TORUS = srgb(200, 206, 216)

# side gear : pinion tooth counts (measured from the meshes)
N_SIDE, N_PINION = 16, 12
RATIO = N_SIDE / N_PINION

# ---------------------------------------------------------------- scene
def init_scene(res=(1080, 1080)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 50 if FAST else 100
    s.render.film_transparent = True
    s.cycles.samples = 64 if FAST else 1024
    s.cycles.max_bounces = 8
    s.cycles.film_exposure = 1.0
    s.cycles.use_denoising = True
    s.view_settings.view_transform = 'Standard'
    s.view_settings.look = 'None'
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGBA'
    s.render.image_settings.color_depth = '16'

    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == 'OPTIX'
    s.cycles.device = 'GPU'

    # ambient light (bt.setLight_ambient)
    w = bpy.data.worlds.new('world')
    s.world = w
    w.use_nodes = True
    # diff.blend world: uniform white dome for lighting rays -> soft, AO-like shading
    nt = w.node_tree
    bg = nt.nodes['Background']
    bg.inputs['Strength'].default_value = 1.0
    lp = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (1, 1, 1, 1)
    mix.inputs['B'].default_value = (0.5, 0.5, 0.5, 1)
    nt.links.new(lp.outputs['Is Camera Ray'], mix.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], bg.inputs['Color'])
    return s


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def add_camera(target, azimuth, elevation, dist, lens=50):
    az, el = math.radians(azimuth), math.radians(elevation)
    t = Vector(target)
    loc = t + dist * Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    cd = bpy.data.cameras.new('cam')
    cd.lens = lens
    cam = bpy.data.objects.new('cam', cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    look_at(cam, t)
    bpy.context.scene.camera = cam
    return cam


def add_area(name, loc, target, energy, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.energy, ld.size, ld.color = energy, size, color
    ld.shape = 'DISK'
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    look_at(o, target)
    return o


def three_point_lights(target=(0, 0, 0), scale=1.0):
    # the three weak square area lights from diff.blend (the world dome does most of the work)
    for name, loc, rot, energy in (
            ('key', (-3.1781, 3.5753, 2.8426), (0.8778, 0.0, -2.415), 1.8),
            ('fill', (3.5753, 1.1918, 2.8426), (0.7591, 0.0, 1.8925), 8.6),
            ('rim', (0.0, 4.3699, 0.8563), (1.1442, 0.0, 3.1416), 2.0)):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy, ld.size, ld.shape = energy * scale ** 2, 3.3105 * scale, 'SQUARE'
        o = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(o)
        o.location = Vector(loc) * scale + Vector(target)
        o.rotation_euler = rot


def shadow_catcher(z):
    me = bpy.data.meshes.new('ground')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=50)
    bm.to_mesh(me)
    o = bpy.data.objects.new('ground', me)
    bpy.context.scene.collection.objects.link(o)
    o.location.z = z - GROUND_GAP  # objects float slightly -> diffused contact shadow
    o.is_shadow_catcher = True
    m = bpy.data.materials.new('ground')
    m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (1, 1, 1, 1)
    m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1.0
    o.data.materials.append(m)
    return o


# ---------------------------------------------------------------- materials
def plastic(name, color, rough=0.7, coat=0.0, emission=0.0):
    # diff.blend's material: plain Principled, roughness 0.7, specular 0.5
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = color
    p.inputs['Roughness'].default_value = rough
    p.inputs['Coat Weight'].default_value = coat
    p.inputs['Coat Roughness'].default_value = 0.08
    p.inputs['Specular IOR Level'].default_value = 0.5
    if emission > 0:
        p.inputs['Emission Color'].default_value = color
        p.inputs['Emission Strength'].default_value = emission
    return m


def set_mat(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------- geometry helpers
def mesh_obj(name, verts, faces, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if smooth:
        me.shade_smooth()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def tube_along(points, radius, nring=24, closed=False):
    """Sweep a circle along a polyline (parallel-transport frames)."""
    P = np.asarray(points, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(T[0], [1.0, 0, 0])
    N = [a / np.linalg.norm(a)]
    for i in range(1, n):
        v = N[-1] - np.dot(N[-1], T[i]) * T[i]
        N.append(v / np.linalg.norm(v))
    N = np.array(N)
    B = np.cross(T, N)
    th = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    V = (P[:, None, :] + radius * (np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :])).reshape(-1, 3)
    F = []
    segs = n if closed else n - 1
    for i in range(segs):
        i2 = (i + 1) % n
        for j in range(nring):
            j2 = (j + 1) % nring
            F.append((i * nring + j, i * nring + j2, i2 * nring + j2, i2 * nring + j))
    if not closed:  # end caps
        V = np.vstack([V, P[0], P[-1]])
        c0, c1 = len(V) - 2, len(V) - 1
        for j in range(nring):
            j2 = (j + 1) % nring
            F.append((c0, j2, j))
            F.append((c1, (n - 1) * nring + j, (n - 1) * nring + j2))
    return V, F


def cone(base, tip, radius, nseg=32):
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    ax = tip - base
    ax /= np.linalg.norm(ax)
    a = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(ax, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(ax, a)
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ring = base + radius * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
    V = np.vstack([ring, tip, base])
    F = [(j, (j + 1) % nseg, nseg) for j in range(nseg)] + [((j + 1) % nseg, j, nseg + 1) for j in range(nseg)]
    return V, F


def arc_arrow(name, center, e1, e2, radius, t0, t1, tube_r, head_len, head_r, mat, n=200):
    """Curved 3D arrow along c + R(cos t e1 + sin t e2), t: t0 -> t1 (head at t1)."""
    c, e1, e2 = (np.asarray(x, float) for x in (center, e1, e2))
    sgn = np.sign(t1 - t0)
    t_head = t1 - sgn * head_len / radius
    ts = np.linspace(t0, t_head + sgn * 0.15 * head_len / radius, n)
    P = c + radius * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2)
    V1, F1 = tube_along(P, tube_r)
    base = c + radius * (np.cos(t_head) * e1 + np.sin(t_head) * e2)
    # tip follows the arc chord so the head sits on the circle
    tip = c + radius * (np.cos(t1) * e1 + np.sin(t1) * e2)
    V2, F2 = cone(base, tip, head_r)
    F2 = [tuple(i + len(V1) for i in f) for f in F2]
    o = mesh_obj(name, np.vstack([V1, V2]), F1 + F2)
    set_mat(o, mat)
    return o


# ---------------------------------------------------------------- differential
def load_differential():
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in ('carrier', 'left_side_gear', 'right_side_gear', 'spider')]
    objs = {}
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
        o.animation_data_clear()
        o.rotation_mode = 'XYZ'  # appended objects are in QUATERNION mode; pose() drives Euler angles
        objs[o.name] = o

    # the -x pinion is baked into the carrier mesh: split it out by loose parts
    car = objs['carrier']
    bm = bmesh.new()
    bm.from_mesh(car.data)
    bm.verts.ensure_lookup_table()
    seen, parts = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            x = stack.pop()
            comp.append(x)
            for e in x.link_edges:
                y = e.other_vert(x)
                if y.index not in seen:
                    seen.add(y.index)
                    stack.append(y)
        parts.append(comp)
    pin_verts = set()
    for comp in parts:
        P = np.array([tuple(v.co) for v in comp])
        if P[:, 0].max() < -0.25 and P[:, 0].min() > -0.71:  # left pinion body, hub, teeth
            pin_verts.update(v.index for v in comp)
    bm2 = bm.copy()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index in pin_verts], context='VERTS')
    bm.to_mesh(car.data)
    bm2.verts.ensure_lookup_table()
    bmesh.ops.delete(bm2, geom=[v for v in bm2.verts if v.index not in pin_verts], context='VERTS')
    me = bpy.data.meshes.new('pinion_left')
    bm2.to_mesh(me)
    left = bpy.data.objects.new('pinion_left', me)
    bpy.context.scene.collection.objects.link(left)

    parts = {
        'carrier': car,
        'top': objs['left_side_gear'],      # z > 0
        'bottom': objs['right_side_gear'],  # z < 0
        'pin_r': objs['spider'],            # x > 0
        'pin_l': left,                      # x < 0
    }
    set_mat(parts['carrier'], plastic('carrier', CARRIER))
    set_mat(parts['top'], plastic('top_gear', TOP_GEAR))
    set_mat(parts['bottom'], plastic('bottom_gear', BOTTOM_GEAR))
    pm = plastic('pinion', PINION)
    set_mat(parts['pin_r'], pm)
    set_mat(parts['pin_l'], pm)

    # carrier pivot: carrier + both pinions orbit the z axis together
    piv = bpy.data.objects.new('carrier_pivot', None)
    bpy.context.scene.collection.objects.link(piv)
    for k in ('carrier', 'pin_r', 'pin_l'):
        parts[k].parent = piv
    parts['pivot'] = piv
    return parts


def pose(parts, theta_c, theta_d):
    """theta_c: carrier angle about z; theta_d: side-gear angle relative to carrier
    (top +theta_d, bottom -theta_d). Pinion spin follows from the tooth ratio."""
    parts['pivot'].rotation_euler = (0, 0, theta_c)
    parts['top'].rotation_euler = (0, 0, theta_c + theta_d)
    parts['bottom'].rotation_euler = (0, 0, theta_c - theta_d)
    parts['pin_r'].rotation_euler = (-theta_d * RATIO, 0, 0)
    parts['pin_l'].rotation_euler = (theta_d * RATIO, 0, 0)


# ---------------------------------------------------------------- post: soft shadow
# Render once with the shadow catcher and once without (object mask only), then
# post_shadow.py blurs and lightens just the ground shadow.
POST_PY = os.path.expanduser('~/miniconda3/bin/python')


def render(path, shadow_strength=0.55, shadow_blur=10.0, shadow_falloff=1.8):
    import subprocess
    s = bpy.context.scene
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=path.replace('.png', '.blend'))
    # object-only alpha mask: hide the ground, few samples are enough
    ground = bpy.data.objects['ground']
    ground.hide_render = True
    samples = s.cycles.samples
    s.cycles.samples = 64
    mask = path.replace('.png', '_mask.png')
    s.render.filepath = mask
    bpy.ops.render.render(write_still=True)
    ground.hide_render = False
    s.cycles.samples = samples
    blur = shadow_blur * s.render.resolution_percentage / 100
    subprocess.run([POST_PY, os.path.join(HERE, 'post_shadow.py'), path, mask,
                    str(shadow_strength), str(blur), str(shadow_falloff)], check=True)
    os.remove(mask)


GROUND_GAP = 0.04
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
GROUND_Z = -1.08
CAM_TARGET = (0, 0, -0.08)
CAM_AZ, CAM_EL, CAM_DIST = -32, 20, 5.0

# ---------------------------------------------------------------- 1. overview
if 'overview' in modes:
    init_scene()
    parts = load_differential()
    pose(parts, math.radians(0), math.radians(0))
    add_camera(CAM_TARGET, CAM_AZ, CAM_EL, CAM_DIST)
    three_point_lights()
    shadow_catcher(GROUND_Z)
    render(os.path.join(OUT, '1_overview.png'))

# ---------------------------------------------------------------- 1b. four carrier configurations
# Bottom gear held fixed: the pinions (and the carrier rods) walk around it,
# so the carrier turns theta_c, the top gear 2 theta_c, the pinions spin.
if 'configs' in modes:
    for i, deg in enumerate((0, 30, 60, 90)):
        init_scene()
        parts = load_differential()
        th = math.radians(deg)
        pose(parts, th, th)
        add_camera(CAM_TARGET, CAM_AZ, CAM_EL, CAM_DIST)
        three_point_lights()
        shadow_catcher(GROUND_Z)
        render(os.path.join(OUT, f'1b_config_{i + 1}_{deg:03d}deg.png'))

# ---------------------------------------------------------------- 2. rotating + arrows
if 'motion' in modes:
    s = init_scene()
    parts = load_differential()
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    w_c = math.radians(12)  # carrier (purple) per frame
    w_d = math.radians(30)  # side gears relative to carrier (green) per frame
    for f in (0, 1, 2):
        pose(parts, w_c * f, w_d * f)
        for k in ('pivot', 'top', 'bottom', 'pin_r', 'pin_l'):
            parts[k].keyframe_insert('rotation_euler', frame=f)
    s.frame_set(1)
    s.render.use_motion_blur = True
    s.render.motion_blur_position = 'CENTER'
    s.render.motion_blur_shutter = 1.0

    green = plastic('arrow_green', GREEN)
    purple = plastic('arrow_purple', PURPLE)
    # green: rides on the side gears (top +theta_d, bottom -theta_d coupled)
    arc_arrow('arrow_green', (0, 0, 0.66), (1, 0, 0), (0, 1, 0), 0.74,
              math.radians(-215), math.radians(-70), 0.04, 0.26, 0.1, green)
    # purple: pinions walk around the center axis together with the carrier
    arc_arrow('arrow_purple', (0, 0, 0.0), (1, 0, 0), (0, 1, 0), 1.18,
              math.radians(-215), math.radians(-55), 0.04, 0.26, 0.1, purple)

    add_camera(CAM_TARGET, CAM_AZ, CAM_EL, CAM_DIST)
    three_point_lights()
    shadow_catcher(GROUND_Z)
    render(os.path.join(OUT, '2_rotating.png'))

# ---------------------------------------------------------------- 3. SO(2) x SO(2) torus
# ---------------------------------------------------------------- torus helpers
R_TOR, r_TOR = 1.0, 0.42
GRID_COLOR = srgb(160, 166, 178)


def grid_material(name, base=None, attr=None, nu=24, nv=12, width=0.035, uv_name='uv'):
    """Matte material with a faint (theta1, theta2) grid from a UV map; base color from a
    constant or a color attribute."""
    m = plastic(name, base if base is not None else (1, 1, 1, 1))
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeUVMap')
    tc.uv_map = uv_name
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['UV'], sep.inputs['Vector'])

    def grid_line(inp, count):
        m1 = nt.nodes.new('ShaderNodeMath'); m1.operation = 'MULTIPLY'; m1.inputs[1].default_value = count
        nt.links.new(inp, m1.inputs[0])
        m2 = nt.nodes.new('ShaderNodeMath'); m2.operation = 'PINGPONG'; m2.inputs[1].default_value = 0.5
        nt.links.new(m1.outputs[0], m2.inputs[0])  # distance to nearest integer in [0, 0.5]
        m4 = nt.nodes.new('ShaderNodeMapRange'); m4.inputs['From Min'].default_value = 0; m4.inputs['From Max'].default_value = width
        m4.inputs['To Min'].default_value = 1; m4.inputs['To Max'].default_value = 0
        nt.links.new(m2.outputs[0], m4.inputs['Value'])
        return m4.outputs['Result']

    mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'
    nt.links.new(grid_line(sep.outputs['X'], nu), mx.inputs[0])
    nt.links.new(grid_line(sep.outputs['Y'], nv), mx.inputs[1])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    if attr:
        at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = attr
        nt.links.new(at.outputs['Color'], mix.inputs['A'])
    else:
        mix.inputs['A'].default_value = base
    mix.inputs['B'].default_value = GRID_COLOR
    nt.links.new(mx.outputs[0], mix.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], p.inputs['Base Color'])
    return m


def torus_point(u, v, lift=0.0):
    u, v = np.asarray(u, float), np.asarray(v, float)
    rr = r_TOR + lift
    return np.stack([(R_TOR + rr * np.cos(v)) * np.cos(u), (R_TOR + rr * np.cos(v)) * np.sin(u), rr * np.sin(v)], -1)


def torus_frame(u, v):
    """unit normal and the two principal directions (d/du: major circle, d/dv: minor circle)."""
    n = np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    eu = np.array([-np.sin(u), np.cos(u), 0.0])
    ev = np.array([-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)])
    return n, eu, ev


def torus_project(q):
    """closest point on the torus -> (u, v)."""
    u = math.atan2(q[1], q[0])
    c = np.array([R_TOR * math.cos(u), R_TOR * math.sin(u), 0])
    d = q - c
    v = math.atan2(d[2], np.dot(d, [math.cos(u), math.sin(u), 0]))
    return u, v


def make_torus(nu=256, nv=96, color_fn=None):
    """color_fn(U, V) -> (N, 4) per-vertex colors (else plain TORUS color)."""
    R, r = R_TOR, r_TOR
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    verts = torus_point(U, Vv).reshape(-1, 3)
    faces = [(i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv)
             for i in range(nu) for j in range(nv)]
    tor = mesh_obj('torus', verts, faces)
    # faint (theta1, theta2) grid on the surface
    uv = tor.data.uv_layers.new(name='uv')
    loop_uv = []
    for poly in tor.data.polygons:
        for li in poly.loop_indices:
            vi = tor.data.loops[li].vertex_index
            i, j = divmod(vi, nv)
            loop_uv.append((i, j))
    uvs = np.array(loop_uv, float)
    # fix wrap-around seams per polygon
    k = 0
    for poly in tor.data.polygons:
        n = poly.loop_total
        blk = uvs[k:k + n]
        for c, N in ((0, nu), (1, nv)):
            if blk[:, c].max() - blk[:, c].min() > N / 2:
                blk[blk[:, c] < N / 2, c] += N
        k += n
    uvs[:, 0] /= nu
    uvs[:, 1] /= nv
    uv.data.foreach_set('uv', uvs.ravel())
    if color_fn is None:
        set_mat(tor, grid_material('torus', TORUS))
    else:
        col = tor.data.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        col.data.foreach_set('color', np.asarray(color_fn(U.ravel(), Vv.ravel()), np.float32).ravel())
        set_mat(tor, grid_material('torus', attr='col', width=0.025))
    return tor


# sample colors: warm ramp along theta1 (mod 2pi), brightness along theta2 (mod 2pi);
# avoids the green / purple used for the two directions
DOT_RAMP = [srgb(250, 205, 70), srgb(245, 140, 50), srgb(220, 65, 55), srgb(165, 35, 50)]


def dot_color(u, v):
    t = (u % (2 * np.pi)) / (2 * np.pi) * (len(DOT_RAMP) - 1)
    i = min(int(t), len(DOT_RAMP) - 2)
    f = t - i
    c = (1 - f) * np.array(DOT_RAMP[i]) + f * np.array(DOT_RAMP[i + 1])
    c[:3] *= 0.8 + 0.2 * (0.5 + 0.5 * math.cos(v))
    return tuple(c)


def dots(name, centers, colors, radius, subdiv=3):
    """many spheres in one mesh, colored through a 'col' point attribute."""
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    sv = np.array([tuple(v.co) for v in bm.verts])
    sf = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    centers = np.asarray(centers, float)
    radius = np.broadcast_to(np.asarray(radius, float), (len(centers),))
    V = (centers[:, None, :] + radius[:, None, None] * sv[None]).reshape(-1, 3)
    F = [tuple(k * len(sv) + i for i in f) for k in range(len(centers)) for f in sf]
    o = mesh_obj(name, V, F)
    col = o.data.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
    C = np.repeat(np.asarray(colors, float), len(sv), axis=0)
    col.data.foreach_set('color', C.ravel())
    m = plastic(name + '_mat', (1, 1, 1, 1))
    at = m.node_tree.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'col'
    m.node_tree.links.new(at.outputs['Color'], m.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    set_mat(o, m)
    return o


def straight_arrow(name, p0, p1, tube_r, head_len, head_r, mat):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    base = p1 - head_len * d
    V1, F1 = tube_along(np.linspace(p0, base + 0.2 * head_len * d, 16), tube_r)
    V2, F2 = cone(base, p1, head_r)
    F2 = [tuple(i + len(V1) for i in f) for f in F2]
    o = mesh_obj(name, np.vstack([V1, V2]), F1 + F2)
    set_mat(o, mat)
    return o


if 'torus' in modes:
    init_scene()
    R, r = R_TOR, r_TOR
    make_torus()

    green = plastic('arrow_green', GREEN)
    purple = plastic('arrow_purple', PURPLE)
    lift = 0.012
    tube = 0.03
    # green: "bagel cut" -- the major circle (outer equator, slicing plane z = 0)
    rr = R + r + lift
    ts = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    Vg, Fg = tube_along(np.stack([rr * np.cos(ts), rr * np.sin(ts), 0 * ts], -1), tube, closed=True)
    set_mat(mesh_obj('major_circle', Vg, Fg), green)
    # arrowhead on the visible front of the major circle
    th0 = math.radians(-100)
    arc_arrow('major_head', (0, 0, 0), (1, 0, 0), (0, 1, 0), rr, th0 - 0.05, th0 + 0.2, tube, 0.24, 0.09, green, n=8)

    # purple: "donut piece cut" -- a minor circle (meridian) around the tube
    u0 = math.radians(-165)
    cu, su = math.cos(u0), math.sin(u0)
    c = np.array([R * cu, R * su, 0])
    e1 = np.array([cu, su, 0])
    e2 = np.array([0, 0, 1.0])
    rm = r + lift
    Vp, Fp = tube_along(c + rm * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2), tube, closed=True)
    set_mat(mesh_obj('minor_circle', Vp, Fp), purple)
    a0 = math.radians(40)
    arc_arrow('minor_head', c, e1, e2, rm, a0 - 0.1, a0 + 0.6, tube, 0.24, 0.09, purple, n=8)

    s = bpy.context.scene
    add_camera((0, 0, -0.05), -30, 32, 4.6)
    three_point_lights()
    shadow_catcher(-r)
    render(os.path.join(OUT, '3_torus.png'))

# ---------------------------------------------------------------- 4. random valid configurations on the torus
TORUS_CAM = ((0, 0, -0.05), -30, 32, 4.6)
rng = np.random.default_rng(7)

if 'samples' in modes:
    init_scene()
    make_torus()
    n = 70
    U, V = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    dots('samples', torus_point(U, V, 0.02), [dot_color(a, b) for a, b in zip(U, V)], 0.045)
    add_camera(*TORUS_CAM)
    three_point_lights()
    shadow_catcher(-r_TOR)
    render(os.path.join(OUT, '4_samples.png'))

# ---------------------------------------------------------------- 5. tangent space at one sample
if 'tangent' in modes:
    init_scene()
    make_torus()
    u0, v0 = math.radians(-112), math.radians(38)
    p0 = torus_point(u0, v0)
    n0, eu, ev = torus_frame(u0, v0)
    # neighbors (the local cloud the SVD is computed from)
    k = 16
    d = np.sqrt(rng.uniform(0.25, 1.0, k)) * 0.3
    phi = rng.uniform(0, 2 * np.pi, k)
    Un = u0 + d * np.cos(phi) / (R_TOR + r_TOR * math.cos(v0))
    Vn = v0 + d * np.sin(phi) / r_TOR
    dots('neighbors', torus_point(Un, Vn, 0.008), [srgb(120, 124, 132)] * k, 0.017)
    dots('current', [p0 + 0.012 * n0], [srgb(45, 45, 55)], 0.03)
    # tangent plane
    hs = 0.24
    corners = [p0 + a * hs * eu + b * hs * ev + 0.004 * n0 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    plane = mesh_obj('tangent_plane', corners, [(0, 1, 2, 3)], smooth=False)
    pm = plastic('tangent_plane', srgb(215, 228, 245))
    pm.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.3
    set_mat(plane, pm)
    plane.visible_shadow = False
    frame = np.array(corners + [corners[0]])
    pts = np.concatenate([np.linspace(frame[i], frame[i + 1], 30) for i in range(4)])
    Vf, Ff = tube_along(pts, 0.0035)
    set_mat(mesh_obj('plane_frame', Vf, Ff), plastic('frame', srgb(110, 125, 150)))
    # principal directions (= SVD directions of the cloud) and the next samples
    green = plastic('arrow_green', GREEN)
    purple = plastic('arrow_purple', PURPLE)
    h = 0.27
    nxt = []
    for e, mat, nm in ((eu, green, 'dir_major'), (ev, purple, 'dir_minor')):
        straight_arrow(nm, p0 + 0.03 * e, p0 + h * e, 0.011, 0.06, 0.027, mat)
        straight_arrow(nm + '_neg', p0 - 0.03 * e, p0 - 0.55 * h * e, 0.007, 0.04, 0.019, mat)
        nxt.append(torus_point(*torus_project(p0 + h * e), 0.01))
    nuv = [torus_project(q) for q in nxt]
    dots('next', nxt, [dot_color(*uv) for uv in nuv], 0.028)
    cam = add_camera(tuple(p0), 0, 0, 1)  # placeholder, aimed below
    cam.location = Vector(tuple(p0 + 1.3 * (0.62 * n0 - 0.62 * ev + 0.3 * eu)))
    look_at(cam, Vector(tuple(p0 + 0.06 * ev)))
    three_point_lights()
    shadow_catcher(-r_TOR)
    render(os.path.join(OUT, '5_tangent.png'))

# ---------------------------------------------------------------- 6a. many samples on the torus
# walking along the principal directions fills the torus with a lattice of samples
NU_S, NV_S = 30, 16
if 'dense' in modes:
    init_scene()
    make_torus()
    I, J = np.meshgrid(np.arange(NU_S), np.arange(NV_S), indexing='ij')
    U = (I.ravel() + 0.5) * 2 * np.pi / NU_S
    V = (J.ravel() + 0.5) * 2 * np.pi / NV_S
    dots('samples', torus_point(U, V, 0.012), [dot_color(a, b) for a, b in zip(U, V)], 0.026)
    add_camera(*TORUS_CAM)
    three_point_lights()
    shadow_catcher(-r_TOR)
    render(os.path.join(OUT, '6a_dense.png'))

# ---------------------------------------------------------------- 6b. "UV" unfolding with overlaps
if 'unfold' in modes:
    init_scene()
    TWO_PI = 2 * np.pi
    ou, ov = 1.25, 1.0                      # how far the walk runs past one period
    W, H = TWO_PI + ou, TWO_PI + ov
    xs = np.unique(np.concatenate([np.linspace(0, W, 120), [ou, TWO_PI]]))
    ys = np.unique(np.concatenate([np.linspace(0, H, 120), [ov, TWO_PI]]))
    xs = xs[xs <= W]; ys = ys[ys <= H]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    verts = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    ny = len(ys)
    faces = [(i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1)
             for i in range(len(xs) - 1) for j in range(ny - 1)]
    sheet = mesh_obj('chart', verts, faces, smooth=False)
    uvl = sheet.data.uv_layers.new(name='uv')
    loop_v = np.array([sheet.data.loops[li].vertex_index for poly in sheet.data.polygons for li in poly.loop_indices])
    uvl.data.foreach_set('uv', (verts[loop_v, :2] / TWO_PI).ravel())
    # overlap regions (covered twice) are tinted; corners covered four times are darker
    col = sheet.data.color_attributes.new('col', 'FLOAT_COLOR', 'CORNER')
    base, tint, tint2 = np.array(TORUS), np.array(srgb(160, 182, 222)), np.array(srgb(125, 150, 200))
    cols = []
    for poly in sheet.data.polygons:
        cx, cy = poly.center.x, poly.center.y
        k = int(cx < ou or cx > TWO_PI) + int(cy < ov or cy > TWO_PI)
        c = (base, tint, tint2)[k]
        cols += [c] * poly.loop_total
    col.data.foreach_set('color', np.array(cols).ravel())
    set_mat(sheet, grid_material('chart', attr='col'))
    # fundamental domain outline
    sq = np.array([(0, 0, 0), (TWO_PI, 0, 0), (TWO_PI, TWO_PI, 0), (0, TWO_PI, 0), (0, 0, 0)], float) + [0, 0, 0.005]
    pts = np.concatenate([np.linspace(sq[i], sq[i + 1], 40) for i in range(4)])
    Vf, Ff = tube_along(pts, 0.03)
    set_mat(mesh_obj('domain', Vf, Ff), plastic('domain', srgb(70, 76, 90)))
    # samples in chart coordinates; the ones past 2pi duplicate the ones near 0 (same color)
    du, dv = TWO_PI / NU_S, TWO_PI / NV_S
    U = (np.arange(int(W / du)) + 0.5) * du
    V = (np.arange(int(H / dv)) + 0.5) * dv
    U, V = [a.ravel() for a in np.meshgrid(U, V, indexing='ij')]
    keep = (U < W) & (V < H)
    U, V = U[keep], V[keep]
    rad = 0.075
    dots('samples', np.stack([U, V, np.full_like(U, rad)], -1), [dot_color(a, b) for a, b in zip(U, V)], rad)
    # axes: theta1 (major, green) along x, theta2 (minor, purple) along y
    green = plastic('arrow_green', GREEN)
    purple = plastic('arrow_purple', PURPLE)
    straight_arrow('axis_major', (0, -0.45, 0.05), (W + 0.3, -0.45, 0.05), 0.05, 0.38, 0.14, green)
    straight_arrow('axis_minor', (-0.45, 0, 0.05), (-0.45, H + 0.3, 0.05), 0.05, 0.38, 0.14, purple)
    s = bpy.context.scene
    add_camera((W / 2, H / 2 - 0.4, 0), -15, 55, 16.5)
    three_point_lights(target=(W / 2, H / 2, 0), scale=2.5)
    shadow_catcher(-0.01)
    render(os.path.join(OUT, '6b_unfold.png'), shadow_blur=6.0)

# ================================================================ method: sampling, then atlas
SAMPLE_DOT = srgb(70, 76, 92)
EXPLORE = srgb(35, 120, 235)  # exploration direction (blue)
SEED_DOT = srgb(30, 30, 38)
U_SEED, V_SEED = math.radians(-110), math.radians(35)
DU, DV = 2 * np.pi / NU_S, 2 * np.pi / NV_S


def lattice_from_seed(kmax):
    """BFS growth from the seed: each step moves one lattice step along +-e_u or +-e_v.
    returns the (i, j) offsets reached within kmax steps (wrapping on the torus)."""
    out = []
    for i in range(-NU_S // 2, NU_S // 2 + 1):
        for j in range(-NV_S // 2, NV_S // 2 + 1):
            if abs(i) + abs(j) <= kmax:
                out.append((i, j))
    # unique modulo the lattice period
    seen, uniq = set(), []
    for i, j in out:
        key = (i % NU_S, j % NV_S)
        if key not in seen:
            seen.add(key); uniq.append((i, j))
    return uniq


def tangent_frame_glyph(u0, v0, hs=0.3, arrow=0.32, explore=True):
    p0 = torus_point(u0, v0)
    n0, eu, ev = torus_frame(u0, v0)
    corners = [p0 + a * hs * eu + b * hs * ev + 0.006 * n0 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    plane = mesh_obj('tangent_plane', corners, [(0, 1, 2, 3)], smooth=False)
    pm = plastic('tangent_plane', srgb(215, 228, 245))
    pm.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 0.35
    set_mat(plane, pm)
    plane.visible_shadow = False
    frame = np.array(corners + [corners[0]])
    pts = np.concatenate([np.linspace(frame[i], frame[i + 1], 30) for i in range(4)])
    Vf, Ff = tube_along(pts, 0.004)
    set_mat(mesh_obj('plane_frame', Vf, Ff), plastic('frame', srgb(110, 125, 150)))
    green, purple = plastic('arrow_green', GREEN), plastic('arrow_purple', PURPLE)
    lift = 0.012 * n0
    # tangent basis (principal directions) ...
    straight_arrow('basis_u', p0 + lift, p0 + lift + arrow * eu, 0.012, 0.065, 0.03, green)
    straight_arrow('basis_v', p0 + lift, p0 + lift + arrow * ev, 0.012, 0.065, 0.03, purple)
    if explore:  # ... and the exploration direction: a random normalized combination of the basis
        d = -0.78 * eu + 0.62 * ev
        d /= np.linalg.norm(d)
        straight_arrow('explore', p0 + lift, p0 + lift + 0.95 * arrow * d, 0.013, 0.07, 0.032, plastic('arrow_blue', EXPLORE))
    return p0


def sampling_scene(kmax, frontier_arrows, plane, name):
    init_scene()
    make_torus()
    offs = lattice_from_seed(kmax)
    # slightly perturbed lattice (same jitter per lattice cell in every figure); the seed stays put
    jit = np.random.default_rng(3).uniform(-0.28, 0.28, (NU_S, NV_S, 2))
    jit[0, 0] = 0
    U = np.array([U_SEED + (i + jit[i % NU_S, j % NV_S, 0]) * DU for i, j in offs])
    V = np.array([V_SEED + (j + jit[i % NU_S, j % NV_S, 1]) * DV for i, j in offs])
    others = [(a, b) for a, b, (i, j) in zip(U, V, offs) if (i, j) != (0, 0)]
    if others:
        oU, oV = np.array(others).T
        dots('samples', torus_point(oU, oV, 0.012), [SAMPLE_DOT] * len(oU), 0.024)
    dots('seed', [torus_point(U_SEED, V_SEED, 0.015)], [SEED_DOT], 0.038)
    if plane:
        tangent_frame_glyph(U_SEED, V_SEED)
    if frontier_arrows:
        # exploration on the front: random (outward) normalized combinations of the local basis
        blue = plastic('arrow_blue', EXPLORE)
        rd = np.random.default_rng(11)
        for (i, j), a, b in zip(offs, U, V):
            if abs(i) + abs(j) != kmax:
                continue
            p = torus_point(a, b, 0.014)
            n, eu, ev = torus_frame(a, b)
            sx = np.sign(i) if i else rd.choice([-1, 1])
            sy = np.sign(j) if j else rd.choice([-1, 1])
            cu = sx * rd.uniform(0.2, 1.0) * (1.0 if i else 0.4)
            cv = sy * rd.uniform(0.2, 1.0) * (1.0 if j else 0.4)
            d = cu * eu + cv * ev
            d /= np.linalg.norm(d)
            straight_arrow(f'fe{i}_{j}', p, p + 0.15 * d, 0.008, 0.045, 0.02, blue)
    add_camera(*TORUS_CAM)
    three_point_lights()
    shadow_catcher(-r_TOR)
    render(os.path.join(OUT, name))


if 'grow' in modes:
    sampling_scene(0, False, True, '7a_seed_tangent.png')
    sampling_scene(4, True, False, '7b_growing.png')
    sampling_scene(NU_S, False, False, '7c_dataset.png')

# ---------------------------------------------------------------- atlas: 8 particles -> patches -> merge -> 3
# high-contrast palette (Trubetskoy's distinct colors)
PATCH_COLORS = {
    'red': srgb(230, 25, 75), 'blue': srgb(67, 99, 216), 'yellow': srgb(255, 200, 25),
    'teal': srgb(0, 170, 160), 'orange': srgb(245, 130, 48), 'magenta': srgb(240, 50, 230),
    'navy': srgb(0, 0, 117), 'lime': srgb(150, 210, 40),
}
SEEDS = {  # name: (u, v) in degrees -- placed where the camera can see them
    'red': (250, 15), 'orange': (295, 45), 'navy': (338, 60),
    'blue': (205, 25), 'teal': (160, 55), 'yellow': (118, 78),
    'lime': (35, 80), 'magenta': (80, 128),
}
_rs = np.random.default_rng(5)
SEEDS = {k: (u + _rs.uniform(-9, 9), v + _rs.uniform(-9, 9)) for k, (u, v) in SEEDS.items()}
# merge history: patch -> members, per stage
FINAL = {'red': ['red', 'orange', 'navy'], 'blue': ['blue', 'teal', 'yellow'], 'lime': ['lime', 'magenta']}
STAGES = [
    ('8a_particles', 0.0, {k: [k] for k in SEEDS}),
    ('8b_patches', 0.36, {k: [k] for k in SEEDS}),
    ('8c_merge', 0.6, {'red': ['red', 'orange'], 'navy': ['navy'], 'blue': ['blue'],
                       'teal': ['teal', 'yellow'], 'lime': ['lime', 'magenta']}),
    ('8d_merge', 0.72, FINAL),
    ('8e_cover', None, FINAL),
]


def torus_dist(U, V, u0, v0):
    du = (U - u0 + np.pi) % (2 * np.pi) - np.pi
    dv = (V - v0 + np.pi) % (2 * np.pi) - np.pi
    vb = v0 + dv / 2
    return np.hypot((R_TOR + r_TOR * np.cos(vb)) * du, r_TOR * dv)


def patch_colors(patches, rho, U, V, margin=0.22, edge=0.014):
    names = list(patches)
    D = np.stack([np.min([torus_dist(U, V, *np.radians(SEEDS[m])) for m in patches[k]], 0) for k in names])
    thr = np.broadcast_to((D.min(0) + margin)[None] if rho is None else rho, D.shape)
    cover = D < thr
    cols = np.array([0.95 * np.array(PATCH_COLORS[k]) + 0.05 for k in names])
    out = np.tile(np.array(TORUS), (len(U), 1))
    # overlaps: diagonal stripes cycling through the covering patches
    stripe = np.floor((R_TOR * U + r_TOR * V) / 0.075).astype(int)
    ncov = cover.sum(0)
    rank = np.cumsum(cover, 0) - 1                       # rank of each covering patch
    want = np.where(ncov > 0, stripe % np.maximum(ncov, 1), -1)
    pick = cover & (rank == want[None])
    has = pick.any(0)
    out[has] = cols[np.argmax(pick, 0)[has]]
    # soft dark outline at each patch boundary
    f = np.clip(1 - np.abs(D - thr) / edge, 0, 1).max(0)
    out[:, :3] *= (1 - 0.7 * f)[:, None]
    return out


if 'atlas' in modes:
    for name, rho, patches in STAGES:
        init_scene()
        make_torus(1024, 384, lambda U, V: patch_colors(patches, rho, U, V))
        su = np.radians([SEEDS[k][0] for k in SEEDS]); sv = np.radians([SEEDS[k][1] for k in SEEDS])
        # each particle keeps the color of the patch it currently belongs to
        owner = {m: k for k, ms in patches.items() for m in ms}
        dots('particles', torus_point(su, sv, 0.02), [tuple(np.array(PATCH_COLORS[owner[k]]) * [0.5, 0.5, 0.5, 1]) for k in SEEDS], 0.045)
        add_camera(*TORUS_CAM)
        three_point_lights()
        shadow_catcher(-r_TOR)
        render(os.path.join(OUT, name + '.png'))
