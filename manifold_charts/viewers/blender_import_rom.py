"""viewers/blender_import_rom.py -- build an animated Blender scene from a `viewers.export_rom_blender` output folder and save `scene.blend` in it.

Run headless (this is what the exporter does)::

    blender -b -P viewers/blender_import_rom.py -- <export_dir> [--camera auto|ortho|persp] [--render out.png --frame N] [--zsep REL] [--auto-camera] [--fast] [--display parts.blend]

or open it in Blender's Scripting tab, set EXPORT_DIR below, and press Run.

Optional layers are built only when the export carries their channels (balls: ball_counts/ball_xy; pen trails: pens_*; traced curve: curve_pts), so a plain pose export gives just the links.
Scene: one mesh per link (keyframed 4x4 transform every exported frame), a pool of UV spheres for the balls (keyed per frame, constant interpolation, hidden while the ball does not
exist), pen trails + the traced curve as bevelled curves, a camera (orthographic along -z with image up = +y for a planar mechanism, else a perspective three-quarter view; `--camera` forces one), three-point area lights and a white backdrop plane behind the mechanism.
Rendering follows viewers/render_utility's setup_scene: Cycles, 1000 samples (`--fast`: 100), exposure 1.25, denoise, grey ambient, the paper's palette.  Its BlenderToolbox is absent, so this is plain bpy.  Frame f of the export is Blender frame f + 1; the scene frame rate is set to the export's.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

_ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _opt(name, default=None):
    return _ARGV[_ARGV.index(name) + 1] if name in _ARGV and _ARGV.index(name) + 1 < len(_ARGV) else default


FAST = "--fast" in _ARGV                                       # `-- <dir> --fast`: 10x fewer samples, for iterating
ZSEP = float(_opt("--zsep", 1e-4))                              # z-fighting fix: z step, relative to the largest link, between mesh pieces that share a face plane (0 = off)
AUTO_CAMERA = "--auto-camera" in _ARGV                          # recompute the camera even if the existing scene.blend has one
CAMERA = _opt("--camera", "auto")                              # auto | ortho | persp
CASE_COLORS = "--case-colors" in _ARGV                          # colour bodies by the case's declared rgb (meshes.npz `color`) instead of LINK_COLORS
GROUP_COLORS = "--group-colors" in _ARGV                        # one colour per body group: the name suffix after "@" (e.g. one leg of an array case)
RENDER, FRAME = _opt("--render"), int(_opt("--frame", 0))      # render this 0-based exported frame to a PNG
DISPLAY = _opt("--display")                                     # detailed render-only parts (.blend, rest-pose world frame) hung on the links by name; see attach_display
HIDE = [_ARGV[i + 1] for i, a in enumerate(_ARGV[:-1]) if a == "--hide"]   # object-name patterns (fnmatch) hidden in viewport and render
FLOOR_Z = _opt("--floor-z")                                  # world z of the shadow-catcher floor (default: just under the lowest geometry); lights keep their default placement
TEMPLATE = _opt("--template")                                 # a saved scene to take the camera, render settings and object visibility from; see read_template
EXPORT_DIR = ""                                              # used only when run from the Scripting tab
# palette copied from viewers/render_utility/definitions.py (the paper's colours)
_c = lambda r, g, b: (r / 255.0, g / 255.0, b / 255.0, 1.0)
BLUE, ORANGE, DARK, PINK = _c(58, 119, 157), _c(233, 76, 38), _c(24, 37, 77), _c(221, 52, 151)
GREEN_2, PURPLE_2, YELLOW_2 = _c(51, 160, 44), _c(106, 61, 154), _c(225, 214, 35)
LINK_COLORS = [BLUE, ORANGE, DARK, GREEN_2, PURPLE_2, YELLOW_2]
BALL_COLOR, TRAIL_COLORS = PINK, [BLUE, ORANGE, GREEN_2]
GROUP_PALETTE = LINK_COLORS + [PINK, _c(23, 190, 207), _c(127, 127, 127), _c(140, 86, 75)]
SAMPLES, EXPOSURE = 1000, 0.0                                # paper: exposure 1.25 (Blender 3.x Filmic); in 5.0 with the Standard transform it blows the scene out, so 0


def material(name, rgba, emit=0.0, roughness=0.7):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgba
    try:
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = rgba
            bsdf.inputs["Roughness"].default_value = roughness
            if emit and "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = rgba
                bsdf.inputs["Emission Strength"].default_value = emit
    except Exception:
        pass
    return m


def set_interp(kind):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = kind


def curve_object(name, pts3, rgba, closed, radius):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts3) - 1)
    for p, q in zip(sp.points, pts3):
        p.co = (float(q[0]), float(q[1]), float(q[2]), 1.0)
    sp.use_cyclic_u = closed
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(material(name, rgba, emit=0.5))   # lines glow at half strength in their own colour
    bpy.context.scene.collection.objects.link(ob)
    return ob


def z_separation(pose0, me, nb, rel):
    """Vertex displacements that remove z-fighting between coplanar mesh PIECES -- pieces of one link (a handle built as two flush boxes, a crank web and its
    counterweight disc) as well as pieces of different links.  Done independently along each world axis (x, y, z): a piece's "levels" along axis a are the
    world-a planes of its a-facing faces at the first frame; two pieces with overlapping faces on a shared level are neighbours, and a greedy colouring of that
    graph gives each piece a colour c_a.  Piece p is displaced by sum_a c_a * eps * e_a (as a local-frame displacement, so it stays put relative to its link).
    eps = rel * the largest link's local size: pose-independent, hence identical for every export of a case.
    Returns (list of (nv, 3) displacements per link, number of pieces, largest number of colours over the axes)."""
    dV, n_layers, npieces = None, 1, 0
    for axis in (0, 1, 2):
        d_axis, npieces, ncol = _axis_separation(pose0, me, nb, rel, axis)
        dV = d_axis if dV is None else [a + b for a, b in zip(dV, d_axis)]
        n_layers = max(n_layers, ncol)
    return dV, npieces, n_layers


def _axis_separation(pose0, me, nb, rel, axis):
    """``z_separation`` along one world axis (``axis`` 0/1/2): the faces facing that axis, levels by that coordinate, overlap tested in the other two."""
    other = [k for k in (0, 1, 2) if k != axis]
    diag = max(float(np.linalg.norm(np.ptp(np.asarray(me[f"V_{i}"], float), axis=0))) for i in range(nb))
    eps, tol = rel * diag, max(1e-4 * diag, 1e-9)
    pieces, vpiece = [], []                                     # pieces: (link, set(levels)); vpiece[i]: piece index per vertex of link i
    for i in range(nb):
        V, Fc = np.asarray(me[f"V_{i}"], float), np.asarray(me[f"F_{i}"], int)
        _, inv = np.unique(np.round(V / max(diag, 1e-9), 6), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        parent = list(range(int(inv.max()) + 1))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for t in inv[Fc]:
            for k in (1, 2):
                parent[find(t[0])] = find(t[k])
        root = np.array([find(int(inv[v])) for v in range(len(V))])
        W = (pose0[i, :, :3] @ V.T).T + pose0[i, :, 3]
        P = W[Fc]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        a = np.linalg.norm(n, axis=1)
        zf = (a > 1e-12) & (np.abs(n[:, axis]) > 0.999 * a)
        ids = {int(r): len(pieces) + k for k, r in enumerate(np.unique(root))}
        levels = {j: {} for j in ids.values()}                   # piece -> {level: [in-plane triangles of its axis-facing faces at that level]}
        for f in np.nonzero(zf)[0]:
            levels[ids[int(root[Fc[f, 0]])]].setdefault(int(round(P[f, 0, axis] / tol)), []).append(P[f][:, other])
        pieces += [(i, levels[j]) for j in sorted(levels)]
        vpiece.append(np.array([ids[int(r)] for r in root]))
    by_level = {}
    for j, (_i, L) in enumerate(pieces):
        for z in L:
            by_level.setdefault(z, []).append(j)
    def overlap(A, B):
        """True when some triangle of A and some triangle of B overlap with positive area (separating-axis test over both triangles' unit edge normals; touching edges do not count)."""
        A, B = np.asarray(A, float), np.asarray(B, float)
        def axes(T):
            e = np.roll(T, -1, axis=1) - T
            nrm = np.stack([-e[..., 1], e[..., 0]], -1)
            return nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-18)
        def clear(own, other, N):                                          # overlap length on every normal of `own`, for every (own, other) pair: (n_own, 3, n_other)
            po = np.einsum("nvd,nkd->nkv", own, N)
            pt = np.einsum("mvd,nkd->nkmv", other, N)
            return np.minimum(po.max(2)[:, :, None], pt.max(3)) - np.maximum(po.min(2)[:, :, None], pt.min(3))
        t = 1e-6 * diag
        return bool(((clear(A, B, axes(A)) > t).all(1) & (clear(B, A, axes(B)) > t).all(1).T).any())

    colour = np.zeros(len(pieces), int)
    for j, (_i, L) in enumerate(pieces):
        used = set()
        for z, tj in L.items():
            for k in by_level[z]:
                if k >= j or colour[k] in used:
                    continue
                if overlap(tj, pieces[k][1][z]):
                    used.add(colour[k])
        c = 0
        while c in used:
            c += 1
        colour[j] = c
    dV = []
    for i in range(nb):
        d = np.zeros((len(vpiece[i]), 3))
        R = pose0[i, :, :3]
        for j in np.unique(vpiece[i]):
            if colour[j]:
                step = np.zeros(3)
                step[axis] = colour[j] * eps
                d[vpiece[i] == j] = np.linalg.solve(R, step)
        dV.append(d)
    return dV, len(pieces), int(colour.max()) + 1


def read_existing_camera(path):
    """The camera of an already-built scene.blend (so a hand-placed camera survives a rebuild), as plain values; None if the file or its camera is missing."""
    if not os.path.exists(path):
        return None
    try:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            if "camera" not in src.objects:
                return None
            dst.objects = ["camera"]
        ob = dst.objects[0]
        d = ob.data
        loc, rot, _ = ob.matrix_basis.decompose()                 # its own transform (matrix_world is not evaluated for an object outside a scene)
        cam = dict(loc=tuple(loc), rot=tuple(rot.to_euler()), type=d.type, lens=d.lens, sensor=d.sensor_width, ortho=d.ortho_scale, clip=(d.clip_start, d.clip_end), shift=(d.shift_x, d.shift_y))
        cd = d
        bpy.data.objects.remove(ob)
        bpy.data.cameras.remove(cd)
        return cam
    except Exception as ex:
        print(f"could not read the existing camera: {ex}")
        return None


def read_template(path):
    """Camera, render settings and object visibility of a saved scene, as plain values (the file is opened, read, and
    replaced by the factory scene again in ``main``).  An object hidden there in any way -- render toggle, viewport
    toggle or the eye -- is hidden in the new scene's viewport AND render: the template is a picture of what to show."""
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(path))
    sc = bpy.context.scene
    out = {"objects": {o.name: bool(o.hide_render or o.hide_viewport or o.hide_get()) for o in sc.objects}}
    co = sc.camera
    if co is not None:
        d = co.data
        loc, rot, _ = co.matrix_world.decompose()
        out["camera"] = dict(loc=tuple(loc), rot=tuple(rot.to_euler()), type=d.type, lens=d.lens, sensor=d.sensor_width, ortho=d.ortho_scale,
                             clip=(d.clip_start, d.clip_end), shift=(d.shift_x, d.shift_y))
    r = sc.render
    out["render"] = dict(resolution_x=r.resolution_x, resolution_y=r.resolution_y, resolution_percentage=r.resolution_percentage,
                         film_transparent=r.film_transparent, engine=r.engine)
    out["cycles"] = dict(samples=sc.cycles.samples, use_denoising=sc.cycles.use_denoising) if r.engine == "CYCLES" else {}
    v = sc.view_settings
    out["view"] = dict(view_transform=v.view_transform, look=v.look, exposure=v.exposure, gamma=v.gamma)
    print(f"template {path}: camera {'yes' if co else 'none'}, {r.resolution_x}x{r.resolution_y}, "
          f"hidden {sorted(n for n, h in out['objects'].items() if h)}")
    return out


def animate_actuator_colors(ob, active, names):
    """A display part with custom properties ``active_when`` (an actuator name) and ``active_color`` (rgb) shows its own
    material colour while that actuator is idle and ``active_color`` on the frames the export recorded it as engaged
    (``frames.npz`` ``actuator_active``), switching hard between frames."""
    want = ob.get("active_when")
    if want is None or active is None:
        return False
    if want not in names:                                # a display kit shared by cases with different actuators
        print(f"display part {ob.name}: active_when={want!r} names no actuator of this export {names}; it keeps its idle colour")
        return False
    k = names.index(want)
    on_rgb = tuple(float(c) for c in ob.get("active_color", (0.72, 0.06, 0.05)))
    set_interp("CONSTANT")
    for slot in ob.material_slots:                       # own copy: the idle colour of every other part sharing the material stays put
        if slot.material is None:
            continue
        m = slot.material = slot.material.copy()
        bsdf = m.node_tree.nodes.get("Principled BSDF") if m.use_nodes else None
        if bsdf is None:
            continue
        sock = bsdf.inputs["Base Color"]
        idle = tuple(sock.default_value)
        colour = lambda on: (*on_rgb, 1.0) if on else idle
        prev = None
        for f, row in enumerate(np.asarray(active, bool)):
            if row[k] != prev:
                if prev is not None:                     # hold the old colour through the frame before the switch, so every whole frame is exact
                    sock.default_value = colour(prev)
                    sock.keyframe_insert("default_value", frame=f)
                sock.default_value = colour(row[k])
                sock.keyframe_insert("default_value", frame=f + 1)
                prev = row[k]
        for fc in _fcurves(m.node_tree):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
    set_interp("LINEAR")
    return True


def _fcurves(id_data):
    """F-curves of an ID's action: Blender 5 keeps them in the action slot's channelbag (``Action.fcurves`` is gone)."""
    ad = id_data.animation_data
    if ad is None or ad.action is None:
        return []
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
        return list(bag.fcurves) if bag is not None else []
    except ImportError:
        return list(ad.action.fcurves)


def attach_display(path, link_objects, sc, active=None, act_names=()):
    """Append a display .blend's objects and hang each on the body it names.

    Display parts are modelled in the case's world frame AT THE REST POSE, the
    frame the exported link meshes are baked in, so a part parented to its link
    with an identity parent-inverse follows the link's keyframed transform
    exactly.  Naming convention (the name as stored in the display file):

        <body>          replaces that body's functional mesh (which is hidden)
        <body>.<part>   detail added to that body; the functional mesh stays
        anything else   static scenery, left where it was modelled

    A part may also carry ``active_when`` / ``active_color`` (see animate_actuator_colors).
    """
    with bpy.data.libraries.load(os.path.abspath(path), link=False) as (src, dst):
        names = [str(n) for n in src.objects]
        dst.objects = list(names)          # Blender swaps loaded objects into THIS list in place
    attached, replaced, static, corners, coloured = 0, set(), 0, [], 0
    for name, ob in zip(names, dst.objects):
        if ob is None:
            continue
        sc.collection.objects.link(ob)
        coloured += animate_actuator_colors(ob, active, list(act_names))
        rest = ob.matrix_world.copy()
        corners += [list(rest @ Vector(c)) for c in ob.bound_box]
        body = name if name in link_objects else name.split(".", 1)[0]
        link = link_objects.get(body)
        if link is None:
            static += 1
            continue
        ob.parent = link
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = rest
        attached += 1
        if name == body:
            replaced.add(body)
    for body in replaced:
        link_objects[body].hide_render = link_objects[body].hide_viewport = True
    print(f"display {path}: {attached} parts on bodies ({len(replaced)} functional meshes replaced), "
          f"{static} static scenery objects, {coloured} coloured by actuator activity")
    return np.asarray(corners, float).reshape(-1, 3)


def main(export_dir):
    fr = np.load(os.path.join(export_dir, "frames.npz"))
    me = np.load(os.path.join(export_dir, "meshes.npz"))
    pose = fr["pose"]
    counts = fr["ball_counts"] if "ball_counts" in fr.files else np.zeros(len(pose), np.int64)
    ball_xy = fr["ball_xy"] if "ball_xy" in fr.files else np.zeros((0, 2))
    F, nb = pose.shape[:2]
    fps, radius = float(fr["fps"]), float(fr["ball_radius"]) if "ball_radius" in fr.files else 0.04
    body_order = [str(b) for b in fr["body_order"]]

    active = fr["actuator_active"] if "actuator_active" in fr.files else None
    act_names = [str(n) for n in fr["actuator_names"]] if "actuator_names" in fr.files else []

    old_cam = None if AUTO_CAMERA else read_existing_camera(os.path.abspath(os.path.join(export_dir, "scene.blend")))
    template = read_template(TEMPLATE) if TEMPLATE else None
    if template and template.get("camera"):
        old_cam = template["camera"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, F
    sc.render.fps, sc.render.fps_base = 100, 100.0 / fps

    # ---- links -----------------------------------------------------------------------------------------------------------------------------------------
    set_interp("LINEAR")
    dV = [np.zeros_like(np.asarray(me[f"V_{i}"], float)) for i in range(nb)]
    if ZSEP > 0.0:
        dV, npieces, ncol = z_separation(pose[0], me, nb, ZSEP)
        print(f"z-fighting: {npieces} mesh pieces in {nb} links, {ncol} z layers; displaced {sum(int((np.abs(d).sum(1) > 0).sum()) for d in dV)} vertices")
    links, allV = [], []
    for i in range(nb):
        V, Fc = np.asarray(me[f"V_{i}"], float), np.asarray(me[f"F_{i}"], int)
        mesh = bpy.data.meshes.new(body_order[i])
        mesh.from_pydata((V + dV[i]).tolist(), [], Fc.tolist())
        mesh.update()
        ob = bpy.data.objects.new(body_order[i], mesh)
        if GROUP_COLORS and "@" in body_order[i]:
            groups = list(dict.fromkeys(b.split("@", 1)[1] for b in body_order if "@" in b))   # first-appearance order
            rgba = GROUP_PALETTE[groups.index(body_order[i].split("@", 1)[1]) % len(GROUP_PALETTE)]
        else:
            rgba = (*map(float, me["color"][i][:3]), 1.0) if CASE_COLORS and "color" in me.files else LINK_COLORS[i % len(LINK_COLORS)]
        ob.data.materials.append(material(body_order[i], rgba))
        ob.rotation_mode = "QUATERNION"
        sc.collection.objects.link(ob)
        links.append((ob, V))
    prev_q = [None] * nb
    for f in range(F):
        for i, (ob, V) in enumerate(links):
            M = np.eye(4)
            M[:3, :4] = pose[f, i]
            loc, quat, scl = Matrix(M.tolist()).decompose()
            if prev_q[i] is not None:
                quat.make_compatible(prev_q[i])
            prev_q[i] = quat
            ob.location, ob.rotation_quaternion, ob.scale = loc, quat, scl
            for path in ("location", "rotation_quaternion", "scale"):
                ob.keyframe_insert(path, frame=f + 1)
    if DISPLAY:                                                  # rest-pose extents: floor and camera must clear the display parts too
        allV.append(attach_display(DISPLAY, {ob.name: ob for ob, _V in links}, sc, active, act_names))
    for f in range(0, F, max(1, F // 40)):
        for i, (_ob, V) in enumerate(links):
            allV.append((pose[f, i, :, :3] @ V.T).T + pose[f, i, :, 3])
    allV = np.vstack(allV)

    # ---- balls (pool indexed by position in the exported array; constant interpolation, so each frame shows exactly the recorded state) -------------------
    set_interp("CONSTANT")
    nmax = int(counts.max()) if len(counts) else 0
    if nmax:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=24, ring_count=12)
        proto = bpy.context.active_object
        bmesh = proto.data
        bmesh.materials.append(material("ball", BALL_COLOR))
        bpy.data.objects.remove(proto)
        balls = []
        for k in range(nmax):
            ob = bpy.data.objects.new(f"ball_{k:03d}", bmesh)
            sc.collection.objects.link(ob)
            ob.hide_viewport = ob.hide_render = True
            balls.append(ob)
        offs = np.concatenate([[0], np.cumsum(counts)])
        z0 = float(allV[:, 2].mean())
        alive = np.zeros(nmax, bool)
        for f in range(F):
            xy = ball_xy[offs[f]:offs[f + 1]]
            for k in range(nmax):
                ex = k < len(xy)
                if ex:
                    balls[k].location = (float(xy[k, 0]), float(xy[k, 1]), z0)
                    balls[k].keyframe_insert("location", frame=f + 1)
                if ex != alive[k] or f == 0:
                    balls[k].hide_viewport = balls[k].hide_render = not ex
                    balls[k].keyframe_insert("hide_viewport", frame=f + 1)
                    balls[k].keyframe_insert("hide_render", frame=f + 1)
                    alive[k] = ex
        if len(ball_xy):
            allV = np.vstack([allV, np.column_stack([ball_xy, np.full(len(ball_xy), z0)])])

    # ---- trails + traced curve ---------------------------------------------------------------------------------------------------------------------------
    trail_r = 0.008
    for j, key in enumerate([k for k in fr.files if k.startswith("pens_")]):
        pts = np.asarray(fr[key], float)
        ob = curve_object("trail_" + key[5:], pts, TRAIL_COLORS[j % len(TRAIL_COLORS)], False, trail_r)
        ob.data.bevel_factor_mapping_end = "SPLINE"                  # fraction by arclength, not by point index
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        frac = cum / max(cum[-1], 1e-12)
        set_interp("LINEAR")
        for f in range(F):                                           # the trail grows with the pen: visible length up to frame f
            ob.data.bevel_factor_end = float(frac[f])
            ob.data.keyframe_insert("bevel_factor_end", frame=f + 1)
    if "curve_pts" in fr.files:
        cp = np.asarray(fr["curve_pts"], float)
        curve_object("traced_curve", np.column_stack([cp, np.full(len(cp), allV[:, 2].mean())]), (0.9, 0.9, 0.2, 1), True, trail_r)

    # ---- backdrop, light, camera --------------------------------------------------------------------------------------------------------------------------
    lo, hi = allV[:, :2].min(0), allV[:, :2].max(0)
    ctr, ext = (lo + hi) / 2.0, float((hi - lo).max()) * 1.15 + 4 * radius
    zback = float(allV[:, 2].min()) - 0.05
    bpy.ops.mesh.primitive_plane_add(size=ext * 6.0, location=(ctr[0], ctr[1], float(FLOOR_Z) if FLOOR_Z is not None else zback))
    bpy.context.active_object.name = "backdrop"
    bpy.context.active_object.data.materials.append(material("backdrop", (0.92, 0.92, 0.92, 1.0), roughness=1.0))
    bpy.context.active_object.is_shadow_catcher = True           # the floor shows only its shadows over the (white) background, like the paper's invisibleGround
    # three-point area lights (render_utility.setup_scene: setLight_threePoints radius 2, height 3, intensity 1500, softness 6, key on the left), scaled to this scene's size
    d = 3.6 * ext / 2.0                                          # the paper's scene is ~2 units across; keep the same distance-to-size ratio
    # With Normalize off a light's output grows with its AREA, and the lights here scale with the scene (size ~ ext, distance ~ ext), so the illumination is already scale-invariant at a fixed `energy`:
    # the user's hand-tuned powers (key 1.8, fill 8.6, rim 2.0) therefore apply unchanged to every case -- every case gets the same lighting.
    for nm, off, energy in (("key", (-0.8, 0.9, 1.0), 1.8), ("fill", (0.9, 0.3, 1.0), 8.6), ("rim", (0.0, 1.1, 0.5), 2.0)):
        bpy.ops.object.light_add(type="AREA", location=(ctr[0] + off[0] * d, ctr[1] + off[1] * d, zback + off[2] * d))
        lt = bpy.context.active_object
        lt.name = "light_" + nm
        lt.data.normalize = False                                # "Normalize" off, as the user set it (the powers below were tuned that way)
        lt.data.energy, lt.data.size = energy, 6.0 * ext / 4.0       # softness 6 (paper units) ~ light size relative to the scene
        lt.rotation_euler = (Vector((ctr[0], ctr[1], zback)) - lt.location).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("world")                         # setLight_ambient(color=(0.5, 0.5, 0.5, 1))
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is not None:                                           # ambient light stays grey 0.5, but camera rays see pure white (the figure goes on a white page)
        lp, mix = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs[6].default_value = (0.5, 0.5, 0.5, 1.0)       # A: seen by every ray except the camera's -> lights the scene
        mix.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)       # B: seen by camera rays -> the background
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(mix.outputs[2], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
    sc.world = world
    sc.render.engine = "CYCLES"                                  # render_utility.setup_scene: Cycles, denoised
    sc.cycles.samples = SAMPLES // (10 if FAST else 1)
    sc.cycles.device = "GPU"                                      # GPU Cycles by default: enable the first working GPU backend (the device type is a user preference, not stored in the .blend)
    try:
        cp = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = kind
                cp.get_devices()
            except Exception:
                continue
            gpus = [d for d in cp.devices if d.type == kind]
            if gpus:
                for d in cp.devices:
                    d.use = d.type == kind
                print(f"cycles GPU: {kind} -> {', '.join(d.name for d in gpus)}")
                break
        else:
            sc.cycles.device = "CPU"
            print("cycles GPU: none found, using CPU")
    except Exception as ex:
        print(f"cycles GPU setup skipped: {ex}")
    sc.cycles.use_denoising = True
    sc.cycles.denoising_use_gpu = True                            # OpenImageDenoise on the GPU: ~3.6x faster per frame than on the CPU (1.85 s vs 6.7 s at 1080p / 1000 samples)
    sc.view_settings.view_transform = "Standard"                  # Blender 5 defaults to AgX, which washes the palette out
    sc.view_settings.exposure = EXPOSURE
    cam = bpy.data.cameras.new("cam")
    zext = float(allV[:, 2].max() - allV[:, 2].min())
    ortho = CAMERA == "ortho" or (CAMERA == "auto" and zext < 0.25 * ext)          # planar mechanism: look straight down its plane
    co = bpy.data.objects.new("camera", cam)
    sc.collection.objects.link(co)
    if ortho:
        cam.type, cam.ortho_scale = "ORTHO", ext
        co.location = (ctr[0], ctr[1], float(allV[:, 2].max()) + 3.0 * ext)
    else:                                                        # three-quarter view from the +z side, so the backdrop stays behind the mechanism
        cam.type, cam.lens = "PERSP", 50.0
        aim = Vector((ctr[0], ctr[1], float(allV[:, 2].mean())))
        co.location = aim + Vector((0.35, -0.55, 1.0)).normalized() * (max(ext, 1.15 * zext) * 1.55 * cam.lens / cam.sensor_width)
        co.rotation_euler = (aim - co.location).to_track_quat("-Z", "Y").to_euler()
    if old_cam is not None:                                      # keep the camera already in scene.blend (hand-placed); --auto-camera recomputes it
        cam.type, cam.lens, cam.sensor_width, cam.ortho_scale = old_cam["type"], old_cam["lens"], old_cam["sensor"], old_cam["ortho"]
        cam.clip_start, cam.clip_end = old_cam["clip"]
        cam.shift_x, cam.shift_y = old_cam["shift"]
        co.location, co.rotation_euler = old_cam["loc"], old_cam["rot"]
        print("camera: from --template" if template and template.get("camera") else "camera: kept the one already in scene.blend (--auto-camera to recompute)")
    sc.camera = co
    sc.render.resolution_x = sc.render.resolution_y = 1080
    if template:
        for k, v in template["render"].items():
            setattr(sc.render, k, v)
        for k, v in template["cycles"].items():
            setattr(sc.cycles, k, v)
        for k, v in template["view"].items():
            setattr(sc.view_settings, k, v)
        for name, hidden in template["objects"].items():
            ob = bpy.data.objects.get(name)
            if ob is not None and hidden:
                ob.hide_render = ob.hide_viewport = True
    if HIDE:
        import fnmatch
        gone = [ob for ob in bpy.data.objects if any(fnmatch.fnmatchcase(ob.name, p) for p in HIDE)]
        for ob in gone:
            ob.hide_render = ob.hide_viewport = True
        print(f"hidden by --hide {HIDE}: {sorted(ob.name for ob in gone)}")

    out = os.path.join(export_dir, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print(f"saved {out}: {F} frames, {nb} links, {nmax} ball objects, {'ortho' if cam.type == 'ORTHO' else 'perspective'} camera")
    if RENDER:
        sc.frame_set(min(max(FRAME, 0), F - 1) + 1)
        sc.render.filepath = os.path.abspath(RENDER)
        sc.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        print(f"rendered frame {sc.frame_current} -> {sc.render.filepath}")


if __name__ == "__main__":
    pos = [a for i, a in enumerate(_ARGV) if not a.startswith("--") and (i == 0 or _ARGV[i - 1] not in ("--camera", "--render", "--frame", "--zsep", "--display", "--template", "--floor-z", "--hide"))]
    main(pos[0] if pos else EXPORT_DIR)
