"""atlas_rom/view_pantograph_drag.py — interactive `pantograph_2d` viewer.

Click near the smaller, inner pen `C` (the scaled-copy output) and drag it
with the mouse. Each frame the cursor's world position (found by casting a ray
from the camera through the mouse and intersecting the mechanism's z=0 plane)
becomes a virtual spring target; `gain * (target - C)` is mapped through the
Jacobian transpose `dC/d[theta1, theta_rel]` into a generalized force on the
mechanism's two actual joints (`O`, `A`) — the same "force on a coordinate,
not a direct pose" idea `case_spec.ActuatorDef` uses for `joint_torque`
actuators — and damped-integrated forward. The mechanism never leaves its real
configuration space (there is nothing to snap or clamp; every reachable
`(theta1, theta_rel)` is valid, per `pantograph_2d`'s module docstring), so the
drag can never "break" the linkage.

Both pens leave a trail (the dragged pen `C` in blue, the outer pen `P` it
drives in orange) and each carries a short vertical marker post so its live
position stays visible against the linkage. `pantograph_2d.MAGNIFICATION` is
the ratio `|OP|/|OC|` this mechanism holds constant everywhere; the status
line reports the live ratio next to it as a running check.

While a drag is active, Polyscope's default trackball camera control is
disabled (`ps.set_do_default_mouse_interaction(False)`) so orbiting the camera
doesn't fight with dragging the pen; it's restored on mouse release.

This is a one-off, case-specific script (not a generic sample/dataset viewer
like `view_samples.py`): it needs the case's closed-form forward kinematics,
which only `pantograph_2d` exposes.

A second mode drives the same drag through the TRAINED chart atlas
(``work/<work>/patch_atlas.pt``) instead of the analytic forward kinematics:
the mechanism's pose comes from a chart's learned decoder ``D(z)`` for a 2-D
latent ``z``. The trained decoder is not guaranteed injective or joint-
consistent (measured separately: `distortion_ok=False`), so re-solving `z`
from scratch each frame (a global nearest-anchor search, or an instantaneous
exact IK re-solve) can teleport to a disconnected, physically-torn "other"
reconstruction with no warning. Latent-mode dragging instead takes ONE
damped-least-squares Newton step per frame -- same well-conditioned direction
`track_point_latent` uses -- but caps its length to `--latent-max-speed *
dt`: a bounded-speed step can never cross a gap in one frame, so it can't
jump onto an unrelated sheet. (A naive spring force through the Jacobian
TRANSPOSE was tried first and measured unusable: this decoder's Jacobian is
far more anisotropic than the analytic case's ever was -- singular-value
ratio ~130:1 vs ~2.6:1 -- so `J^T @ error` nearly freezes the compliant
direction, the same "slip" pathology the analytic mode hit before switching
to DLS.) Which chart owns the current pose is still re-ranked every frame by
the trained ``PhysicalRBFSelector`` (the same responsibility ranker
`competitive_cae` uses at fit time); a hand-off re-encodes the just-decoded
pose into the new chart's own (unrelated) latent so POSE stays continuous
even though the raw ``z`` numbers reset.

Usage::

    python -m viewers.view_pantograph_drag
    python -m viewers.view_pantograph_drag --gain 10.0 --damping 8.0
    python -m viewers.view_pantograph_drag --work work/pantograph_2d
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from atlas_rom.cases import load_spec  # noqa: E402
from atlas_rom.cases.pantograph_2d import (  # noqa: E402
    A as A_REST, AB, B as B_REST, C as C_REST, D as D_REST, MAGNIFICATION,
    O as O_REST, OD, P as P_REST, THETA1_REST, THETA_REL_REST, _forward,
)
from viewers.mesh_view import MeshAdapter  # noqa: E402

# Driven-body order matches `pantograph_2d.CASE.mechanism.driven_bodies`
# (ground is fixed, excluded); each entry names the two rest-baked endpoints
# `_forward` returns under those same keys.
_BODY_ENDPOINTS = {
    "crank": (("O", O_REST), ("A", A_REST)),
    "arm": (("A", A_REST), ("P", P_REST)),
    "link_b": (("B", B_REST), ("C", C_REST)),
    "link_d": (("D", D_REST), ("C", C_REST)),
}
_BODY_ORDER = ["crank", "arm", "link_b", "link_d"]


def _rigid_from_two_points(p0_rest, p1_rest, p0_new, p1_new) -> np.ndarray:
    """The unique planar rigid transform taking a rod's baked endpoints to its
    current ones (2-point alignment: same rotation the rod's own direction
    vector underwent, then translate the first endpoint into place)."""
    d_rest = p1_rest[:2] - p0_rest[:2]
    d_new = p1_new[:2] - p0_new[:2]
    angle = np.arctan2(d_new[1], d_new[0]) - np.arctan2(d_rest[1], d_rest[0])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = p0_new - R @ p0_rest
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T


def transforms_at(theta1: float, theta_rel: float) -> np.ndarray:
    pts = _forward(theta1, theta_rel)
    out = []
    for name in _BODY_ORDER:
        (k0, p0_rest), (k1, p1_rest) = _BODY_ENDPOINTS[name]
        out.append(_rigid_from_two_points(p0_rest, p1_rest, pts[k0], pts[k1]))
    return np.stack(out)


def jacobian_point(key: str, theta1: float, theta_rel: float,
                   eps: float = 1e-5) -> np.ndarray:
    """d(point `key`)/d[theta1, theta_rel], by central difference of the
    closed-form FK. `key` is any of the keys `_forward` returns ("C", "P", ...)."""
    p0x = _forward(theta1 - eps, theta_rel)[key][:2]
    p0y = _forward(theta1, theta_rel - eps)[key][:2]
    p1x = _forward(theta1 + eps, theta_rel)[key][:2]
    p1y = _forward(theta1, theta_rel + eps)[key][:2]
    return np.column_stack(((p1x - p0x) / (2 * eps), (p1y - p0y) / (2 * eps)))


def _ray_plane_hit(origin: np.ndarray, direction: np.ndarray,
                   plane_z: float = 0.0) -> np.ndarray | None:
    """World xy where a camera ray crosses the mechanism's z=`plane_z` plane."""
    if abs(direction[2]) < 1e-8:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0.0:
        return None
    hit = origin + t * direction
    return hit[:2]


def track_point(key: str, theta1: float, theta_rel: float, target: np.ndarray,
               damping_lambda: float = 0.05, iters: int = 8,
               tol: float = 1e-5) -> tuple[float, float]:
    """Damped-least-squares Newton solve: the (theta1, theta_rel) that puts
    point `key` exactly at `target`, starting from the current angles.

    A spring force mapped through the Jacobian transpose only nudges toward
    the target, and since the Jacobian's two columns are neither orthogonal
    nor equal length, that nudge is anisotropic: dragging along one screen
    direction visibly bleeds into the other joint ("slip"). Solving the
    inverse kinematics exactly each frame removes that entirely — whatever
    direction the cursor moves, the tracked point follows it directly. The
    Levenberg damping term keeps this well-behaved through the (rare, since
    both angles are unrestricted per the case's own docstring) near-singular
    configurations where the plain Jacobian would blow up.
    """
    for _ in range(iters):
        current = _forward(theta1, theta_rel)[key][:2]
        error = target - current
        if np.linalg.norm(error) < tol:
            break
        J = jacobian_point(key, theta1, theta_rel)
        JJt = J @ J.T + (damping_lambda ** 2) * np.eye(2)
        delta = J.T @ np.linalg.solve(JJt, error)
        theta1 += delta[0]
        theta_rel += delta[1]
    return theta1, theta_rel


class LatentAtlas:
    """Loads a trained ``patch_atlas.pt`` and exposes per-chart encode/decode
    plus the trained selector, reusing the real training-time classes
    (`atlas_rom.charts.physical_chart_autoencoder`, `atlas_rom.charts.
    competitive_cae`) rather than re-deriving the architecture here."""

    def __init__(self, checkpoint_path: str) -> None:
        import torch
        from atlas_rom.charts.physical_chart_autoencoder import _module
        from atlas_rom.charts.competitive_cae import PhysicalRBFSelector

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ca = ckpt["chart_atlas"]
        cls = _module(ca["n_bodies"], width=96, d=2, encoder_kind=ca["encoder_kind"])
        self.models = []
        for state in ca["states"]:
            model = cls(state["centre"].numpy(), state["scale"].numpy())
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)
        self.selector = PhysicalRBFSelector.from_payload(ca["selector"])
        self.n_charts = len(self.models)
        self.n_bodies = ca["n_bodies"]
        # Real training-time latent anchors per chart -- used to WARM START
        # the drag IK from an actual sample instead of pure local Newton
        # refinement, since `distortion_ok=False` (measured separately) means
        # the decoder folds: a from-scratch local solve can converge onto the
        # wrong sheet or crawl through a near-singular direction.
        self.anchors = [np.asarray(a, float) for a in ca["anchors"]]

    def decode(self, chart_idx: int, z: np.ndarray) -> np.ndarray:
        """z (2,) -> per-body (n_bodies, 3, 4) world transforms."""
        import torch
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(z, float).reshape(1, 2))
            return self.models[chart_idx].transforms(u)[0].numpy()

    def decode_batch(self, chart_idx: int, Z: np.ndarray) -> np.ndarray:
        """Z (N, 2) -> (N, n_bodies, 3, 4) world transforms, one torch call."""
        import torch
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(Z, float))
            return self.models[chart_idx].transforms(u).numpy()

    def encode(self, chart_idx: int, pose_flat: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            q = torch.as_tensor(np.asarray(pose_flat, float).reshape(1, -1))
            return self.models[chart_idx].encode(q)[0].numpy()

    def best_chart(self, pose_flat: np.ndarray) -> int:
        return int(np.argmax(self.selector.score(pose_flat)))

    def best_reconstructing_chart(self, pose_flat: np.ndarray) -> int:
        """Which chart actually reconstructs `pose_flat` best (encode then
        decode, measured error), rather than which one the trained selector
        (separately measured `rbf_assignment_calibration_ok=False`, ~coin
        flip) ranks highest. At the pantograph's own rest pose the selector's
        top pick reconstructs with error 2.97 while the true best chart
        reconstructs at 0.017 -- ~175x worse, not a rounding difference."""
        best_idx, best_err = 0, np.inf
        for idx in range(self.n_charts):
            z = self.encode(idx, pose_flat)
            recon = self.decode(idx, z).reshape(-1)
            err = float(np.linalg.norm(recon - np.asarray(pose_flat, float)))
            if err < best_err:
                best_idx, best_err = idx, err
        return best_idx


def _point_from_transforms(T: np.ndarray, body_name: str, p_rest: np.ndarray) -> np.ndarray:
    """World position of a body's rest-baked local point `p_rest` under its
    current (R, t) -- valid because every body here is rest-world-baked
    (identity transform at rest), same convention `_rigid_from_two_points`
    relies on for the analytic path."""
    idx = _BODY_ORDER.index(body_name)
    R, t = T[idx, :, :3], T[idx, :, 3]
    return R @ np.asarray(p_rest, float) + t


def latent_pen_points(atlas: LatentAtlas, chart_idx: int, z: np.ndarray) -> dict[str, np.ndarray]:
    T = atlas.decode(chart_idx, z)
    return {"T": T, "P": _point_from_transforms(T, "arm", P_REST),
           "C": _point_from_transforms(T, "link_b", C_REST)}


def jacobian_point_latent(atlas: LatentAtlas, chart_idx: int, z: np.ndarray,
                          body_name: str, p_rest: np.ndarray,
                          eps: float = 1e-4) -> np.ndarray:
    z = np.asarray(z, float)
    ex, ey = np.array([eps, 0.0]), np.array([0.0, eps])

    def pt(zz: np.ndarray) -> np.ndarray:
        return _point_from_transforms(atlas.decode(chart_idx, zz), body_name, p_rest)[:2]

    return np.column_stack(((pt(z + ex) - pt(z - ex)) / (2 * eps),
                            (pt(z + ey) - pt(z - ey)) / (2 * eps)))


_anchor_point_cache: dict[tuple[int, int, str], np.ndarray] = {}
_anchor_pose_cache: dict[tuple[int, int], np.ndarray] = {}


def _anchor_positions_and_poses(atlas: LatentAtlas, chart_idx: int,
                                body_name: str, p_rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_key, pose_key = (id(atlas), chart_idx, body_name), (id(atlas), chart_idx)
    poses = _anchor_pose_cache.get(pose_key)
    if poses is None:
        T = atlas.decode_batch(chart_idx, atlas.anchors[chart_idx])
        poses = T.reshape(len(T), -1)
        _anchor_pose_cache[pose_key] = poses
    positions = _anchor_point_cache.get(pos_key)
    if positions is None:
        T = poses.reshape(len(poses), atlas.n_bodies, 3, 4)
        idx = _BODY_ORDER.index(body_name)
        R, t = T[:, idx, :, :3], T[:, idx, :, 3]
        positions = np.einsum("nij,j->ni", R, np.asarray(p_rest, float)) + t
        _anchor_point_cache[pos_key] = positions
    return positions, poses


def warm_start_z(atlas: LatentAtlas, chart_idx: int, body_name: str,
                 p_rest: np.ndarray, target: np.ndarray,
                 current_pose: np.ndarray | None = None,
                 branch_weight: float = 1.0) -> np.ndarray:
    """The chart's own training anchor whose decoded `body_name` point lands
    closest to `target` -- a coarse global search that sidesteps a folded
    decoder's wrong-sheet local optima, before `track_point_latent` polishes
    the result with a few Newton steps.

    Pen position alone does NOT pick out a unique configuration: even the
    real pantograph has a two-fold "elbow" ambiguity reaching a given pen
    position (fixed `theta1` puts D at a fixed point; C sweeps a full circle
    of radius AB around it as `theta_rel` varies, and two different `theta1`
    generically place D at two points equidistant from a reachable C -- the
    same character as `five_bar`'s branch ambiguity, just not from a
    coupler-point circle intersection). So scoring anchors by pen distance
    alone can land on a totally different overall configuration that merely
    happens to put the pen in the same place. When `current_pose` (the FULL
    48-dim pose, not just the tracked point) is given, it is weighted into
    the search so the chosen anchor stays close to the configuration we were
    already in, not just to the target point.
    """
    positions, poses = _anchor_positions_and_poses(atlas, chart_idx, body_name, p_rest)
    cost = np.square(np.linalg.norm(positions[:, :2] - target, axis=1))
    if current_pose is not None:
        cost = cost + branch_weight * np.square(
            np.linalg.norm(poses - np.asarray(current_pose, float).reshape(1, -1), axis=1))
    nearest = int(np.argmin(cost))
    return atlas.anchors[chart_idx][nearest].copy()


def track_point_latent(atlas: LatentAtlas, chart_idx: int, z: np.ndarray,
                       body_name: str, p_rest: np.ndarray, target: np.ndarray,
                       damping_lambda: float = 0.05, iters: int = 8,
                       tol: float = 1e-5) -> np.ndarray:
    """Same damped-least-squares IK as `track_point`, through the trained
    decoder's own Jacobian instead of the analytic `_forward`'s."""
    z = np.asarray(z, float).copy()
    for _ in range(iters):
        current = _point_from_transforms(atlas.decode(chart_idx, z), body_name, p_rest)[:2]
        error = target - current
        if np.linalg.norm(error) < tol:
            break
        J = jacobian_point_latent(atlas, chart_idx, z, body_name, p_rest)
        JJt = J @ J.T + (damping_lambda ** 2) * np.eye(2)
        z = z + J.T @ np.linalg.solve(JJt, error)
    return z


def pick_chart(atlas: LatentAtlas, pose_flat: np.ndarray, selection: str) -> int:
    """`selection="reconstruction"`: which chart actually reconstructs this
    pose best (measured, reliable). `selection="learned"`: the trained
    `PhysicalRBFSelector`'s own top pick (separately measured
    `rbf_assignment_calibration_ok=False`, ~coin flip -- kept as a toggle so
    the learned selector's behavior stays inspectable, not because it's
    recommended)."""
    if selection == "learned":
        return atlas.best_chart(pose_flat)
    return atlas.best_reconstructing_chart(pose_flat)


def maybe_switch_chart(atlas: LatentAtlas, chart_idx: int, z: np.ndarray,
                       body_name: str, p_rest: np.ndarray,
                       chart_selection: str = "reconstruction") -> tuple[int, np.ndarray]:
    """Re-rank the just-decoded pose against every chart; hand off to
    whichever one `chart_selection` prefers. The new chart's own `encode()`
    is only a feedforward approximation and can land far from the actual pen
    position given the measured distortion, so warm-start into the new
    chart from its OWN nearest real anchor (by pen position) instead of
    trusting the raw encode -- a quick, chart-local search, not the global
    every-frame one that broke continuity."""
    T = atlas.decode(chart_idx, z)
    pose_flat = T.reshape(-1)
    point = _point_from_transforms(T, body_name, p_rest)[:2]
    best = pick_chart(atlas, pose_flat, chart_selection)
    if best == chart_idx:
        return chart_idx, z
    return best, warm_start_z(atlas, best, body_name, p_rest, point, current_pose=pose_flat)


def step_latent_dynamics(atlas: LatentAtlas, chart_idx: int, z: np.ndarray,
                         z_vel: np.ndarray, body_name: str, p_rest: np.ndarray,
                         target: np.ndarray, dt: float, max_latent_speed: float = 60.0,
                         damping_lambda: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """One rate-limited damped-least-squares step in latent `z`, instead of
    re-solving `z` from scratch (an instantaneous global solve can teleport
    across a fold in a non-injective decoder; a bounded-speed integrator
    physically cannot) AND instead of a plain mass-spring through the
    Jacobian TRANSPOSE (measured unusable here: the decoder's Jacobian is far
    more anisotropic than the analytic case ever was -- singular-value ratio
    ~130:1 vs ~2.6:1 -- so `J^T @ error` intrinsically underdrives the
    compliant direction almost to a standstill, the same "slip" pathology
    this file's analytic mode hit before switching to DLS). `z_vel` still
    carries the CLAMPED per-frame step as a velocity purely for the coasting
    behaviour on release; the direction itself is exact damped-least-squares,
    same as `track_point_latent`'s single iteration."""
    current = _point_from_transforms(atlas.decode(chart_idx, z), body_name, p_rest)[:2]
    error = target - current
    J = jacobian_point_latent(atlas, chart_idx, z, body_name, p_rest)
    JJt = J @ J.T + (damping_lambda ** 2) * np.eye(2)
    delta = J.T @ np.linalg.solve(JJt, error)
    max_step = max_latent_speed * dt
    step_norm = float(np.linalg.norm(delta))
    if step_norm > max_step:
        delta = delta * (max_step / step_norm)
    z_vel = delta / dt
    return z + delta, z_vel


def run(grab_radius: float = 0.35, damping_lambda: float = 0.05, ik_iters: int = 8,
       trail_length: int = 2500, trail_min_step: float = 0.004,
       work: str = "work/pantograph_2d", latent_max_speed: float = 20.0,
       latent_damping: float = 10.0, latent_dt: float = 1.0 / 60.0) -> None:
    import polyscope as ps
    from polyscope import imgui

    case = load_spec("pantograph_2d")
    adapter = MeshAdapter(case.mechanism)

    ps.init()
    ps.set_up_dir(adapter.up_dir)
    ps.set_ground_plane_mode("shadow_only")
    adapter.register(ps, transforms_at(THETA1_REST, THETA_REL_REST))

    def _register_circle(name: str, radius: float, color: tuple[float, float, float],
                         n: int = 96) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        nodes = np.column_stack((radius * np.cos(angles), radius * np.sin(angles),
                                 np.zeros(n))).astype(np.float32)
        edges = np.column_stack((np.arange(n), (np.arange(n) + 1) % n)).astype(np.int32)
        net = ps.register_curve_network(name, nodes, edges)
        net.set_color(color)
        net.set_radius(0.003, relative=False)

    # Pen C's true reachable region is the annulus |OD - AB| <= |C| <= OD + AB
    # (see pantograph_2d's FEASIBILITY docstring); static, drawn once.
    purple = (0.55, 0.25, 0.85)
    _register_circle("pen_C boundary (inner)", abs(OD - AB), purple)
    _register_circle("pen_C boundary (outer)", OD + AB, purple)

    checkpoint_path = os.path.join(work, "patch_atlas.pt")
    atlas: LatentAtlas | None = None
    atlas_error: str | None = None
    if os.path.isfile(checkpoint_path):
        try:
            atlas = LatentAtlas(checkpoint_path)
        except Exception as exc:  # noqa: BLE001 -- report in-UI, never crash the viewer
            atlas_error = f"{type(exc).__name__}: {exc}"
    else:
        atlas_error = f"no checkpoint at {checkpoint_path}"

    rest = _forward(THETA1_REST, THETA_REL_REST)
    state = {
        "mode": "physical", "chart_selection": "reconstruction",
        "theta1": THETA1_REST, "theta_rel": THETA_REL_REST,
        "chart_idx": 0, "z": np.zeros(2), "z_vel": np.zeros(2),
        "dragging": False,
        "trail_p": [rest["P"].copy()], "trail_c": [rest["C"].copy()],
    }
    if atlas is not None:
        rest_pose = transforms_at(THETA1_REST, THETA_REL_REST).reshape(-1)
        state["chart_idx"] = pick_chart(atlas, rest_pose, state["chart_selection"])
        state["z"] = atlas.encode(state["chart_idx"], rest_pose)

    def _post(name: str, point: np.ndarray, color: tuple[float, float, float]) -> None:
        h = 0.15
        nodes = np.array([[point[0], point[1], -h], [point[0], point[1], h]],
                         dtype=np.float32)
        net = ps.register_curve_network(name, nodes, np.array([[0, 1]], dtype=np.int32))
        net.set_color(color)
        net.set_radius(0.012, relative=False)

    def _trail(name: str, points: list[np.ndarray],
              color: tuple[float, float, float]) -> None:
        if len(points) < 2:
            # Registering fewer than 2 points is a no-op below, which would
            # leave a stale (long) curve network on screen from before a
            # "Clear trails" click -- explicitly drop it instead.
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)
            return
        nodes = np.asarray(points, dtype=np.float32)
        edges = np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))])
        net = ps.register_curve_network(name, nodes, edges.astype(np.int32))
        net.set_color(color)
        net.set_radius(0.004, relative=False)

    def _reset() -> None:
        state["theta1"], state["theta_rel"] = THETA1_REST, THETA_REL_REST
        state["trail_p"] = [rest["P"].copy()]
        state["trail_c"] = [rest["C"].copy()]
        if atlas is not None:
            rest_pose = transforms_at(THETA1_REST, THETA_REL_REST).reshape(-1)
            state["chart_idx"] = pick_chart(atlas, rest_pose, state["chart_selection"])
            state["z"] = atlas.encode(state["chart_idx"], rest_pose)
            state["z_vel"] = np.zeros(2)

    def callback() -> None:
        if atlas is not None:
            label = "Switch to latent mode" if state["mode"] == "physical" else "Switch to physical mode"
            if imgui.Button(label):
                if state["mode"] == "physical":
                    # Continuity: hand the CURRENT physical pose to whichever
                    # chart `chart_selection` prefers for it.
                    pose_flat = transforms_at(state["theta1"], state["theta_rel"]).reshape(-1)
                    state["chart_idx"] = pick_chart(atlas, pose_flat, state["chart_selection"])
                    state["z"] = atlas.encode(state["chart_idx"], pose_flat)
                    state["z_vel"] = np.zeros(2)
                    state["mode"] = "latent"
                else:
                    # No general inverse from an arbitrary reconstructed pose
                    # back to (theta1, theta_rel) -- land on rest instead of
                    # faking continuity here.
                    state["theta1"], state["theta_rel"] = THETA1_REST, THETA_REL_REST
                    state["mode"] = "physical"
            imgui.SameLine()
            if imgui.Button(f"Chart selection: {state['chart_selection']}"):
                state["chart_selection"] = ("learned" if state["chart_selection"] == "reconstruction"
                                            else "reconstruction")
        elif atlas_error is not None:
            imgui.TextWrapped(f"Latent mode unavailable: {atlas_error}")

        if imgui.Button("Reset"):
            _reset()
        imgui.SameLine()
        if imgui.Button("Clear trails"):
            state["trail_p"] = []
            state["trail_c"] = []

        if state["mode"] == "physical":
            pts_now = _forward(state["theta1"], state["theta_rel"])
        else:
            pts_now = latent_pen_points(atlas, state["chart_idx"], state["z"])

        io = imgui.GetIO()
        origin = ps.get_view_camera_parameters().get_position()
        direction = ps.screen_coords_to_world_ray(imgui.GetMousePos())
        hit = _ray_plane_hit(np.asarray(origin, float), np.asarray(direction, float))

        if (not io.WantCaptureMouse) and imgui.IsMouseClicked(0) and hit is not None:
            if np.linalg.norm(hit - pts_now["C"][:2]) < grab_radius:
                state["dragging"] = True
                ps.set_do_default_mouse_interaction(False)
        if state["dragging"] and imgui.IsMouseReleased(0):
            state["dragging"] = False
            state["z_vel"] = np.zeros(2)
            ps.set_do_default_mouse_interaction(True)

        if state["dragging"] and hit is not None:
            if state["mode"] == "physical":
                state["theta1"], state["theta_rel"] = track_point(
                    "C", state["theta1"], state["theta_rel"], hit,
                    damping_lambda=damping_lambda, iters=ik_iters)
            else:
                state["z"], state["z_vel"] = step_latent_dynamics(
                    atlas, state["chart_idx"], state["z"], state["z_vel"],
                    "link_b", C_REST, hit, dt=latent_dt,
                    max_latent_speed=latent_max_speed, damping_lambda=damping_lambda)
                state["chart_idx"], state["z"] = maybe_switch_chart(
                    atlas, state["chart_idx"], state["z"], "link_b", C_REST,
                    chart_selection=state["chart_selection"])
        elif state["mode"] == "latent":
            # Not actively dragged: let residual velocity settle rather than
            # freezing dead (a released spring should coast, not snap still).
            state["z"] = state["z"] + state["z_vel"] * latent_dt
            state["z_vel"] = state["z_vel"] * max(0.0, 1.0 - latent_damping * latent_dt)

        if state["mode"] == "physical":
            pts = _forward(state["theta1"], state["theta_rel"])
            adapter.update(transforms_at(state["theta1"], state["theta_rel"]))
        else:
            pts = latent_pen_points(atlas, state["chart_idx"], state["z"])
            adapter.update(pts["T"])

        if not state["trail_c"] or np.linalg.norm(pts["C"] - state["trail_c"][-1]) > trail_min_step:
            state["trail_p"].append(pts["P"].copy())
            state["trail_c"].append(pts["C"].copy())
            if len(state["trail_c"]) > trail_length:
                state["trail_p"].pop(0)
                state["trail_c"].pop(0)

        _post("pen_C (drag me)", pts["C"], (0.20, 0.55, 0.90))
        _post("pen_P (driven)", pts["P"], (0.90, 0.35, 0.15))
        _trail("pen_C trail", state["trail_c"], (0.20, 0.55, 0.90))
        _trail("pen_P trail", state["trail_p"], (0.90, 0.35, 0.15))

        ratio = np.linalg.norm(pts["P"]) / max(1e-9, np.linalg.norm(pts["C"]))
        imgui.Separator()
        imgui.Text(f"mode={state['mode']}  dragging={state['dragging']}  grab_radius={grab_radius}")
        if state["mode"] == "physical":
            imgui.Text(f"theta1={state['theta1']:.3f}  theta_rel={state['theta_rel']:.3f}")
        else:
            imgui.Text(f"chart={state['chart_idx']}/{atlas.n_charts}  z=({state['z'][0]:+.3f}, {state['z'][1]:+.3f})"
                      f"  |z_vel|={np.linalg.norm(state['z_vel']):.3f}")
        imgui.Text(f"|OP|/|OC|={ratio:.4f}  (fixed magnification={MAGNIFICATION:.4f})")

    ps.set_user_callback(callback)
    print("pantograph drag viewer: click-drag the blue pen C in the viewport; "
         "pen P follows the fixed-magnification scaled relation."
         + ("" if atlas is not None else f"  (latent mode unavailable: {atlas_error})"))
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grab-radius", type=float, default=0.35,
                        help="world-space distance from pen C a click must land within to grab it")
    parser.add_argument("--damping-lambda", type=float, default=0.05,
                        help="Levenberg damping for the IK solve near singular configurations")
    parser.add_argument("--ik-iters", type=int, default=8,
                        help="Newton iterations per frame to track the cursor exactly")
    parser.add_argument("--trail-length", type=int, default=2500,
                        help="max trail points kept per pen")
    parser.add_argument("--work", default="work/pantograph_2d",
                        help="directory holding a trained patch_atlas.pt for latent mode "
                             "(if absent, latent mode is disabled with an in-UI message)")
    parser.add_argument("--latent-max-speed", type=float, default=20.0,
                        help="latent-mode cap on |dz|/dt: bounds how far z can move in one "
                             "frame so a drag can never teleport across a decoder fold")
    parser.add_argument("--latent-damping", type=float, default=10.0,
                        help="latent-mode coast-to-stop rate after releasing the drag")
    parser.add_argument("--latent-dt", type=float, default=1.0 / 60.0,
                        help="latent-mode integration step per frame")
    args = parser.parse_args()
    run(grab_radius=args.grab_radius, damping_lambda=args.damping_lambda,
       ik_iters=args.ik_iters, trail_length=args.trail_length, work=args.work,
       latent_max_speed=args.latent_max_speed, latent_damping=args.latent_damping,
       latent_dt=args.latent_dt)


if __name__ == "__main__":
    main()
