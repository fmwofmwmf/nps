"""geometry.py — declarative meshes -> vertices/faces -> physical inertia.

One place turns a `MeshSpec` into `(V, F)`, and one place turns the resulting body
meshes into the `(mass, inertia, com_local)` triple the reduced mass needs.  This
replaces `viz_cases.CASES[name]` (a per-case geometry function) and
`chart_inertia.build_inertia(case)` (a per-case inertia function) with a rule
driven entirely by the spec.

Primitives are vertex-identical to `viz_cases` / `libuipc_oracle._make_box` /
`_make_sphere`, so a ROM built from a spec renders and simulates the same geometry
the oracle explored.

Mass is the physical mass libuipc's ABD integrates: `density * volume`, with the
FULL 3x3 inertia tensor about the body CoM (see the mass-calibration note — a
declared `mass=1` is wrong by orders of magnitude).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import trimesh as tm

# unit cube surface — identical to viz_cases._CUBE_V / libuipc_oracle._CUBE_V
_CUBE_V = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)
_CUBE_F = np.array([
    [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
], dtype=np.int32)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # atlas_rom/
_ROOT = os.path.dirname(_HERE)                              # testing/manifold_charts
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_ASSET_ROOTS = (os.path.dirname(_ROOT),                     # testing/
                _REPO,                                      # repo root
                _ROOT,
                os.path.join(_REPO, "testing_old", "testing"))  # legacy mesh assets


# ─── primitives ───────────────────────────────────────────────────────────────

def _box(center=(0, 0, 0), half=1.0):
    half = np.broadcast_to(np.asarray(half, float), (3,))
    return _CUBE_V * half + np.asarray(center, float), _CUBE_F.copy()


def _sphere(radius=1.0, center=(0, 0, 0), subdiv=2):
    s = tm.creation.icosphere(subdivisions=int(subdiv), radius=float(radius))
    return (np.ascontiguousarray(s.vertices, float) + np.asarray(center, float),
            np.ascontiguousarray(s.faces, np.int32))


def _cylinder(radius=1.0, height=1.0, center=(0, 0, 0), axis=(0, 0, 1), sections=24):
    c = tm.creation.cylinder(radius=float(radius), height=float(height),
                             sections=int(sections))
    V = np.ascontiguousarray(c.vertices, float)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    s, dot = np.linalg.norm(v), float(z @ axis)
    if s > 1e-12:                                    # Rodrigues onto the declared axis
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - dot) / s**2)
        V = V @ R.T
    elif dot < 0:
        V = -V
    return V + np.asarray(center, float), np.ascontiguousarray(c.faces, np.int32)


def _orient(V, axis):
    """Rotate local (+z) geometry onto `axis`, matching `_cylinder`'s Rodrigues."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis)
    s, dot = np.linalg.norm(v), float(z @ axis)
    if s > 1e-12:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - dot) / s**2)
        return V @ R.T
    if dot < 0:
        return -V
    return V


def _torus(radius=0.3, tube=0.04, center=(0, 0, 0), axis=(0, 0, 1), sections=40):
    """A ring: circle of `radius` swept by a tube of `tube`, normal along `axis`."""
    t = tm.creation.torus(float(radius), float(tube))
    V = _orient(np.ascontiguousarray(t.vertices, float), axis) + np.asarray(center, float)
    return V, np.ascontiguousarray(t.faces, np.int32)


def _bevel_gear(pitch_angle=45.0, toe=0.28, heel=0.68, teeth=18, tooth_phase=0.0,
                addendum=0.05, backlash=0.16, clearance=0.0, hub_radius=0.10, hub_len=0.28,
                center=(0, 0, 0), axis=(0, 0, 1), flip=False):
    """A bevel gear whose pitch cone has its APEX AT THE LOCAL ORIGIN.

    `pitch_angle` (deg) is the pitch-cone half-angle from the local +z axis;
    `toe`/`heel` bound the toothed blank as 3-D distances from the apex.  Two
    such gears mesh cleanly for every pose when they (a) share this apex, (b)
    have pitch angles summing to the shaft angle -- 90 deg for crossed axes --
    and (c) carry teeth counts in the same ratio as the analytic gear ratio, so
    one tooth-pitch of one gear is exactly one tooth-pitch of the other and the
    engagement never drifts.  `tooth_phase` (rad) spins the tooth ring to drop a
    tooth into the mate's gap; `backlash` thins each tooth by that fraction of
    its pitch.  Local axis +z with the cone opening toward +z; `flip` opens it
    toward -z.  `_orient`/`center` then place it like `_cylinder`.
    """
    g = np.radians(float(pitch_angle))
    sg, cg = np.sin(g), np.cos(g)
    toe, heel = float(toe), float(heel)
    N = max(int(teeth), 1)
    ded = 1.35 * float(addendum)                      # tooth space cut below the pitch line
    n = max(N * 2, 48)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    d0, d1 = toe + 0.05 * (heel - toe), heel - 0.06 * (heel - toe)
    z_t, z_h = toe * cg, heel * cg
    z_hb = z_h + 0.32 * heel * sg                     # the back cone stands proud of the heel rim
    # frustum wall: smooth to d0, recessed by `ded` through the toothed band
    # (so a mate tooth has a real space to enter), smooth again to the heel.
    rings_zr = [(z_t, toe * sg), (d0 * cg, d0 * sg - ded),
                (d1 * cg, d1 * sg - ded), (z_h, heel * sg)]
    V, F, off = [], [], 0
    prev = None
    for z, r in rings_zr:
        ring = np.c_[r * ca, r * sa, np.full(n, z)]
        V.append(ring)
        if prev is not None:
            for k in range(n):
                j = (k + 1) % n
                F += [[off - n + k, off - n + j, off + j], [off - n + k, off + j, off + k]]
        prev, off = ring, off + n
    ring_hb = np.c_[0.38 * heel * sg * ca, 0.38 * heel * sg * sa, np.full(n, z_hb)]
    V.append(ring_hb)
    hi = off                                          # first index of ring_hb
    ci_t, ci_h = hi + n, hi + n + 1
    V.append(np.array([[0.0, 0.0, z_t], [0.0, 0.0, z_hb]]))
    for k in range(n):
        j = (k + 1) % n
        F += [[hi - n + k, hi - n + j, hi + j], [hi - n + k, hi + j, hi + k],   # back cone
              [ci_t, j, k], [ci_h, hi + k, hi + j]]                              # toe / back caps
    V = [np.vstack(V)]
    Vs, Fs, off = V, [np.asarray(F, np.int32)], len(V[0])

    hv, hf = _cylinder(radius=float(hub_radius), height=float(hub_len))
    hv = hv.copy(); hv[:, 2] += z_hb + float(hub_len) / 2.0 - 0.04     # from the back cone outward
    Vs.append(hv); Fs.append(hf + off); off += len(hv)

    u = np.array([sg, 0.0, cg])                       # slant direction (apex -> heel) at azimuth 0
    nrm = np.array([cg, 0.0, -sg])                    # cone outward normal at azimuth 0
    tang = np.array([0.0, 1.0, 0.0])
    half_w = 0.5 * (np.pi / N) * (1.0 - float(backlash))   # tooth half-width: half the pitch, less backlash
    quads = ([0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3])
    tris = np.array([[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads], np.int32)
    for k in range(N):
        a = 2.0 * np.pi * k / N + float(tooth_phase)
        corners = []
        for d in (d0, d1):
            w = d * sg * half_w
            for dt in (-w, w):
                for dn in (-ded, float(addendum)):    # root sits on the recessed floor
                    corners.append(d * u + tang * dt + nrm * dn)
        C = np.asarray(corners)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        Vs.append(C @ Rz.T); Fs.append(tris + off); off += len(C)

    V = np.vstack(Vs)
    F = np.vstack(Fs).astype(np.int32)
    if clearance:                       # pull the whole blank in toward its axis
        V = V.copy(); V[:, :2] *= (1.0 - float(clearance))
    if flip:
        V = V * np.array([1.0, 1.0, -1.0])
    return _orient(V, axis) + np.asarray(center, float), np.ascontiguousarray(F, np.int32)


def _spur_gear(pitch_radius=0.2, teeth=24, width=0.05, tooth_phase=0.0,
               center=(0, 0, 0), axis=(0, 0, 1)):
    """A closed, extruded spur gear with trapezoidal teeth.

    Module `m = 2 * pitch_radius / teeth`: addendum `m`, dedendum `1.25 m`, so
    two gears mesh visually when their centre distance is the sum of their
    pitch radii and their radii are in the ratio of their tooth counts (equal
    module).  Tooth 0 is centred at local azimuth `tooth_phase` (rad); to drop a
    tooth into the mate's gap, give the mate `phase_to_mate + pi + pi / teeth`.
    The profile is star-shaped about the axis, so both caps fan from the centre
    and the solid is watertight.  `_orient`/`center` place it like `_cylinder`.
    """
    N = max(int(teeth), 3)
    R = float(pitch_radius)
    m = 2.0 * R / N
    r_tip, r_root = R + m, max(R - 1.25 * m, 0.25 * R)
    p = 2.0 * np.pi / N
    # per tooth: root, tip, tip, root (CCW); the remaining 0.4 p is the gap
    offsets = np.array([-0.30, -0.17, 0.17, 0.30]) * p
    radii = np.array([r_root, r_tip, r_tip, r_root])
    ang = (np.arange(N)[:, None] * p + float(tooth_phase) + offsets[None, :]).ravel()
    rad = np.tile(radii, N)
    n = ang.size
    h = 0.5 * float(width)
    ring = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    V = np.vstack([np.c_[ring, np.full(n, -h)], np.c_[ring, np.full(n, h)],
                   [[0.0, 0.0, -h], [0.0, 0.0, h]]])
    cb, ct = 2 * n, 2 * n + 1
    k = np.arange(n)
    j = (k + 1) % n
    F = np.vstack([np.c_[np.full(n, ct), n + k, n + j],          # top cap (+z)
                   np.c_[np.full(n, cb), j, k],                  # bottom cap (-z)
                   np.c_[k, j, n + j], np.c_[k, n + j, n + k]])  # flanks
    return _orient(V, axis) + np.asarray(center, float), np.ascontiguousarray(F, np.int32)


def _link(p0, p1, thick=0.1):
    """A thin bar spanning p0->p1 in the xy-plane, verts baked at rest world pose
    (so the body starts at transform = identity, matching viz_cases._link_box)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    ang = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    V = _CUBE_V * np.array([L / 2.0, thick / 2.0, thick / 2.0])
    return np.ascontiguousarray((Rz @ V.T).T + 0.5 * (p0 + p1)), _CUBE_F.copy()


def _welded(mesh):
    """Close a mesh enough that `volume`/`moment_inertia` are meaningful."""
    mesh = mesh.copy()
    for digits in (6, 4, 3):
        mesh.merge_vertices(digits_vertex=digits)
        if mesh.is_watertight:
            break
    if not mesh.is_watertight:
        tm.repair.fill_holes(mesh)
    return mesh


def resolve_asset(path: str) -> str:
    """Find a mesh file relative to the spec, testing/, or the repo root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for root in _ASSET_ROOTS:
        candidate = os.path.normpath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"mesh file {path!r} not found under {_ASSET_ROOTS}")


_BLEND_EXPORT = ("import bpy, sys; bpy.ops.export_scene.gltf("
                 "filepath=sys.argv[sys.argv.index('--') + 1], export_format='GLB', "
                 "export_yup=True, export_apply=True)")


def _blender_exe() -> str:
    """``BLENDER_EXE`` if set, else the newest installed Blender (as export_rom_blender)."""
    import glob
    exe = os.environ.get("BLENDER_EXE") or next(iter(sorted(
        glob.glob("C:/Program Files/Blender Foundation/Blender */blender.exe"), reverse=True)), None)
    if not exe or not os.path.exists(exe):
        raise FileNotFoundError("a .blend mesh needs Blender: set BLENDER_EXE or install it under "
                                "C:/Program Files/Blender Foundation")
    return exe


def _blend_as_glb(path: str) -> str:
    """Export a .blend to a content-hashed .glb in ``.blend_cache/`` beside it (headless
    Blender, +Y up so file coordinates round-trip); object names become node names."""
    import hashlib, subprocess
    with open(path, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(os.path.dirname(path), ".blend_cache")
    out = os.path.join(cache, f"{stem}.{digest}.glb")
    if os.path.exists(out):
        return out
    os.makedirs(cache, exist_ok=True)
    tmp = out + ".tmp.glb"
    proc = subprocess.run([_blender_exe(), "-b", path, "--python-expr", _BLEND_EXPORT, "--", tmp],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"Blender export of {path} failed (exit {proc.returncode}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    os.replace(tmp, out)
    for old in os.listdir(cache):                       # keep only the current export
        if old.startswith(stem + ".") and old.endswith(".glb") and os.path.join(cache, old) != out:
            os.remove(os.path.join(cache, old))
    return out


@lru_cache(maxsize=32)
def _load_scene(path: str):
    if path.lower().endswith(".blend"):
        path = _blend_as_glb(path)
    return tm.load(path, force="scene")


def _from_file(path, node=None, weld=True, scale=1.0):
    """Load one node from a mesh file, with its scene world matrix baked in — the
    same convention `load_fbx.py` uses when handing verts to libuipc."""
    resolved = resolve_asset(path)
    scene = _load_scene(resolved)
    candidates = list(scene.graph.nodes_geometry)
    if node is not None:
        if node not in candidates:
            raise KeyError(f"{path}: no node {node!r}; have {candidates}")
        candidates = [node]
    matrix, geom_name = scene.graph[candidates[0]]
    mesh = scene.geometry[geom_name]
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError(f"{path}:{candidates[0]} is not a triangle mesh")
    if weld:
        mesh = _welded(mesh)
    V = np.ascontiguousarray(mesh.vertices, float)
    if node is not None:
        h = np.hstack([V, np.ones((len(V), 1))])
        V = np.ascontiguousarray((np.asarray(matrix, float) @ h.T).T[:, :3])
    scale = np.broadcast_to(np.asarray(scale, float), (3,))
    if np.any(scale <= 0.0):
        raise ValueError(f"file mesh scale must be positive, got {scale.tolist()}")
    return np.ascontiguousarray(V * scale, float), np.ascontiguousarray(mesh.faces, np.int32)


def _compound(parts):
    """Concatenate sub-meshes into one body: a welded triangle of bars, a link with
    a boss, any body that is several primitives rigidly attached."""
    Vs, Fs, offset = [], [], 0
    for part in parts:
        V, F = build_mesh(part)
        Vs.append(V)
        Fs.append(F + offset)
        offset += len(V)
    return np.vstack(Vs), np.vstack(Fs).astype(np.int32)


_BUILDERS = {"box": _box, "sphere": _sphere, "cylinder": _cylinder,
             "link": _link, "file": _from_file, "compound": _compound,
             "bevel_gear": _bevel_gear, "torus": _torus, "spur_gear": _spur_gear}


def build_mesh(mesh_spec) -> tuple[np.ndarray, np.ndarray]:
    """MeshSpec -> (V (n,3) float64, F (m,3) int32)."""
    kind = mesh_spec.kind if hasattr(mesh_spec, "kind") else mesh_spec["kind"]
    params = dict(mesh_spec.params if hasattr(mesh_spec, "params")
                  else mesh_spec.get("params", {}))
    try:
        V, F = _BUILDERS[kind](**params)
    except TypeError as exc:
        raise TypeError(f"mesh kind {kind!r} got bad params {sorted(params)}: {exc}") from exc
    return np.ascontiguousarray(V, float), np.ascontiguousarray(F, np.int32)


# ─── physical properties ──────────────────────────────────────────────────────

def body_inertia(V, F, density=1000.0):
    """(mass, inertia_about_com (3,3), com_local (3,)) for one mesh.

    Degenerate or unclosable meshes fall back to their bounding box, which keeps
    the reduced mass finite and SPD rather than silently zero.
    """
    mesh = tm.Trimesh(V, F, process=False)
    volume = float(mesh.volume)
    if volume <= 1e-12 or not mesh.is_watertight:
        extents = mesh.extents if (mesh.extents is not None
                                   and np.min(mesh.extents) > 0) else (1.0, 1.0, 1.0)
        proxy = tm.creation.box(extents=extents)
        proxy.apply_translation(np.asarray(mesh.bounds).mean(axis=0))
        mesh, volume = proxy, float(proxy.volume)
    return (float(density) * volume,
            float(density) * np.asarray(mesh.moment_inertia, float),
            np.asarray(mesh.center_mass, float))


def build_bodies(mechanism):
    """All driven bodies' geometry + inertia, in config-packing order.

    Returns a dict of parallel arrays::

        names   [n]            driven body names
        V, F    [n] lists      local meshes
        masses  (n,)           density * volume
        inertia (n,3,3)        about each body's CoM, body frame
        com     (n,3)          CoM in body-local coordinates
    """
    names, Vs, Fs, masses, inertias, coms = [], [], [], [], [], []
    for body in mechanism.driven_bodies:
        V, F = build_mesh(body.mesh)
        m, I, com = body_inertia(V, F, body.density)
        names.append(body.name)
        Vs.append(V)
        Fs.append(F)
        masses.append(m)
        inertias.append(I)
        coms.append(com)
    return {"names": names, "V": Vs, "F": Fs,
            "masses": np.asarray(masses, float),
            "inertia": np.asarray(inertias, float),
            "com": np.asarray(coms, float)}


def build_static(mechanism):
    """Fixed / render-only bodies, already in world frame (verts are rest-baked)."""
    out = []
    for body in mechanism.static_bodies:
        V, F = build_mesh(body.mesh)
        if not np.allclose(body.initial_transform, np.eye(4)):
            h = np.hstack([V, np.ones((len(V), 1))])
            V = (body.initial_transform @ h.T).T[:, :3]
        out.append({"name": body.name, "V": np.ascontiguousarray(V, float), "F": F,
                    "color": body.color})
    return out


def apply_transform(V_local, T):
    """Place local verts by a 4x4 (or 3x4) transform."""
    T = np.asarray(T, float)
    if T.shape == (3, 4):
        full = np.eye(4)
        full[:3, :4] = T
        T = full
    h = np.hstack([V_local, np.ones((len(V_local), 1))])
    return (T @ h.T).T[:, :3]
