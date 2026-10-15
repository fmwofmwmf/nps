"""ideal_oracle.py — the projection oracle for mechanisms that are ALGEBRAIC.

`oracle.py` drives libuipc's stiff affine-body solver and reads back where it
settles.  That is the right and only tool when a case's manifold is defined by
CONTACT: there is no closed form for "the nearest configuration where these
meshes touch but do not overlap", so the manifold has to be discovered by
simulation.

A case with `contact_model="none"` and none but ideal joints is not that case.
Its configuration manifold is the zero set of a small polynomial system — per
body `RᵀR = I`, and per joint a pivot coincidence plus zero to three axis
alignments — so `project()` is a Gauss-Newton root-find and the tangent space is
the null space of the same Jacobian, both in milliseconds on the CPU.  Paying a
GPU IPC solve for that is paying to *rediscover* constraints the spec states.

Measured on `three_rru` (10 bodies, 12 revolute joints), per frontier pop:

    libuipc oracle    ~15 s     (8 probe projections for the tangent frame)
    this module      ~0.2 s     (one SVD, exact)

and the two agree: the tangent subspaces differ by at most 0.26 degrees, and
libuipc's own `project()` leaves this module's output exactly fixed with a
closure residual of 2e-16.  It is the same manifold, found analytically.

RANK BY SPECTRAL GAP
--------------------
The one subtlety.  The Jacobian's singular values fall into two populations —
the genuine constraints (bottoming out around 0.13 for this mechanism) and the
null directions, which are 1e-16 at a tightly converged point but drift to 1e-7
as `project()` leaves residual behind.  Any fixed cutoff eventually lands
between them and reports the wrong dimension: a relative `s[0]*1e-8` threshold
gave dim 3, 3, 1, 3 at four configurations of the SAME 3-DOF mechanism.  The gap
between the two populations is five or more orders of magnitude, so the rank is
read off the FIRST ratio gap wider than `MIN_GAP_RATIO`, scanning down from the
top.  That is still a policy constant — a smaller one than a fixed cutoff, and
one chosen against a property of the spectrum rather than of the residual, but a
constant.  It is backed up rather than trusted alone: `tangent_frame()` compares
each rank against the case's declared dimension and reports the disagreements,
so a rank that changes where the spec says it should not becomes a visible
singularity event instead of a silently varying atlas dimension.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from atlas_rom.simulate.oracle_base import ManifoldOracle
from atlas_rom.spec.physical_domain import violation as physical_domain_violation

try:
    from scipy.linalg import lstsq as scipy_lstsq
except ImportError:  # pragma: no cover - NumPy-only installations
    scipy_lstsq = None

# Upper-triangular index pairs of RᵀR - I, in one fixed order so the residual
# and the Jacobian cannot drift apart.
_ORTHONORMALITY_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# Joint kinds whose constraint this module can write down exactly.  A kind not
# listed here is not "unsupported forever" — it just has to be added as a
# residual + Jacobian block before a case using it can take the analytic path.
SUPPORTED_JOINTS = ("revolute", "prismatic", "cylindrical", "spherical",
                    "universal", "fixed", "phase_lock", "gear_ratio", "bevel_gear")

# Kinds whose anchor is confined to a LINE rather than to a point: the two
# transverse components of the anchor separation are constrained and the axial
# one is the joint's slide.
_SLIDING_JOINTS = ("cylindrical", "prismatic")

# A rank cut is only trusted when the two singular-value populations are this
# far apart.  Below it the mechanism is at (or numerically near) a singularity
# and the dimension is genuinely ambiguous, which is worth saying out loud
# rather than silently resolving one way.
MIN_GAP_RATIO = 1e3

# A response whose constraint residual exceeds this did not reach the manifold
# (`project` converges to 1e-11 when it converges at all).
RESPONSE_RESIDUAL_TOL = 1e-8

# How many dimension disagreements to name individually before summarising.
_MISMATCH_REPORTS = 5


def supports(spec) -> bool:
    """Is this case's manifold algebraic, so the analytic oracle is valid?"""
    if spec.contact_model != "none":
        return False
    if not spec.mechanism.joints:
        return False
    return all(j.kind in SUPPORTED_JOINTS for j in spec.mechanism.joints)


class IdealJointOracle(ManifoldOracle):
    """Analytic `project()` / `tangent_frame()` over packed (n_bodies, 3, 4) poses.

    Interface-compatible with `LibuipcConfigOracle` for everything the explorer
    uses: `seed()`, `project(q)`, `tangent_frame(q)`, `pack()`.
    """

    def __init__(self, spec, log=print):
        super().__init__()
        if not supports(spec):
            raise ValueError(
                f"{spec.name}: not an ideal-joint case "
                f"(contact_model={spec.contact_model!r}, kinds "
                f"{sorted({j.kind for j in spec.mechanism.joints})})")
        self.spec = spec
        mech = spec.mechanism
        self.order = mech.body_order                 # DRIVEN bodies, packing order
        self.n_bodies = len(self.order)
        self.dim_ambient = 12 * self.n_bodies
        index = {name: i for i, name in enumerate(self.order)}
        rest = {b.name: np.asarray(b.initial_transform, float) for b in mech.bodies}

        def to_local(body, vector, is_direction=False):
            """World pivot/axis -> that body's frame at ITS rest pose.

            A body not in the packing order is ground: it never moves, so its
            image of the pivot stays the world pivot and contributes no columns.
            """
            if body is None or body not in index:
                return None
            T = rest[body]
            if is_direction:
                return T[:3, :3].T @ vector
            return (np.linalg.inv(T) @ np.append(vector, 1.0))[:3]

        self.joints, self.n_rows = [], 6 * self.n_bodies
        for j in mech.joints:
            if j.kind == "gear_ratio":
                # An ideal open differential constrains side-gear spin relative
                # to its carrier: theta_a + ratio * theta_b =
                # (1 + ratio) * theta_carrier.  This is a physical gear law,
                # not a chart coordinate; the atlas observes only poses.
                #
                # The angles are absolute and wrapped to (-pi, pi], so the law
                # is carried as  p*theta_a + q*theta_b - (p+q)*theta_c  with
                # ratio = q/p in lowest terms (tooth counts).  Integer
                # multiples of a wrapped angle stay continuous across +-pi;
                # a non-integer ratio times one jumps by ratio*2pi there and
                # tears the constraint set.  The zero set is then exactly the
                # physical mesh: p pinion poses per wheel pose when q/p is not
                # an integer, because the mesh repeats only every p turns.
                ratio = Fraction(float(j.ratio)).limit_denominator(1000)
                if abs(float(ratio) - float(j.ratio)) > 1e-9 * max(1.0, abs(float(j.ratio))):
                    raise ValueError(f"{j.name}: gear ratio {j.ratio!r} is not a ratio "
                                     f"of tooth counts (denominator <= 1000)")
                axis = np.asarray(j.axis, float)
                trial = (np.array([0.0, 1.0, 0.0])
                         if abs(float(axis[0])) > 0.9 else np.array([1.0, 0.0, 0.0]))
                phase = np.cross(axis, trial)
                phase /= np.linalg.norm(phase)
                self.joints.append({
                    "name": j.name, "kind": j.kind,
                    "ia": index.get(j.body_a), "ib": index.get(j.body_b),
                    "ic": index.get(j.body_c), "axis": axis, "phase": phase,
                    "phase_a": to_local(j.body_a, phase, True),
                    "phase_b": to_local(j.body_b, phase, True),
                    "phase_c": to_local(j.body_c, phase, True),
                    "p": float(ratio.denominator), "q": float(ratio.numerator),
                })
                self.n_rows += 1
                continue
            if j.kind == "bevel_gear":
                # Rolling of a pinion (body_a) on a mating gear (body_b), both
                # measured relative to the carrier (body_c): the pinion spin
                # about its carrier-borne axle equals -ratio times the mating
                # gear's carrier-relative spin about `axis2`.  A companion
                # revolute keeps the pinion axle following the carrier, so the
                # axle image may be taken from the carrier pose alone.
                axle = np.asarray(j.axis, float)
                trial = (np.array([0.0, 1.0, 0.0])
                         if abs(float(axle[0])) > 0.9 else np.array([1.0, 0.0, 0.0]))
                p_w = np.cross(axle, trial); p_w /= np.linalg.norm(p_w)
                ax2 = np.asarray(j.axis2, float); ax2 /= np.linalg.norm(ax2)
                trial2 = (np.array([0.0, 1.0, 0.0])
                          if abs(float(ax2[0])) > 0.9 else np.array([1.0, 0.0, 0.0]))
                u2 = np.cross(ax2, trial2); u2 /= np.linalg.norm(u2)
                self.joints.append({
                    "name": j.name, "kind": j.kind,
                    "ia": index.get(j.body_a), "ib": index.get(j.body_b),
                    "ic": index.get(j.body_c),
                    "axle_c": to_local(j.body_c, axle, True),
                    "p_c": to_local(j.body_c, p_w, True),
                    "p_pin": to_local(j.body_a, p_w, True),
                    "qd_side": to_local(j.body_b, u2, True),
                    "qd_c2": to_local(j.body_c, u2, True),
                    "ax2": ax2, "u2": u2, "w2": np.cross(ax2, u2),
                    "ratio": float(j.ratio),
                })
                self.n_rows += 1
                continue
            if j.kind == "phase_lock":
                # A shaft lock has no pivot coincidence: each crank keeps its
                # own axle.  It only keeps their incremental spin equal about
                # the declared common axis.  Carrying one transverse rest-frame
                # direction is rank two and deliberately leaves axial rotation
                # of the entire assembly unconstrained.
                axis = np.asarray(j.axis, float)
                trial = (np.array([0.0, 1.0, 0.0])
                         if abs(float(axis[0])) > 0.9 else np.array([1.0, 0.0, 0.0]))
                transverse = np.cross(axis, trial)
                transverse /= np.linalg.norm(transverse)
                entry = {
                    "name": j.name, "kind": j.kind,
                    "ia": index.get(j.body_a), "ib": index.get(j.body_b),
                    # These distinct local directions map to the SAME world
                    # transverse direction at rest.  Thus a phase offset in
                    # the declared rest transforms is retained, not erased.
                    "phase_a": to_local(j.body_a, transverse, True),
                    "phase_b": to_local(j.body_b, transverse, True),
                }
                self.joints.append(entry)
                self.n_rows += 3
                continue
            entry = {
                "name": j.name,
                "kind": j.kind,
                "ia": index.get(j.body_a), "ib": index.get(j.body_b),
                "pa": to_local(j.body_a, np.asarray(j.pivot, float)),
                "pb": to_local(j.body_b, np.asarray(j.pivot, float)),
                "world_pivot": np.asarray(j.pivot, float),
                # Every kind pins the anchor; they differ only in how much of
                # the relative ORIENTATION they also hold, and each held
                # direction is one more alignment row.
                #
                #   spherical  no directions   -> 3 rotational freedoms left
                #   revolute   the hinge axis  -> 1 (spin about the axis)
                #   fixed      a whole frame   -> 0
                #
                # One direction cannot weld a frame: holding a single axis
                # parallel still permits spin about it, which is a revolute.
                # Three independent directions pin R_b relative to R_a
                # completely, so `fixed` carries the world basis.
                "axes": {"spherical": (),
                         # A universal joint carries its one perpendicularity
                         # row separately (a scalar, not a vector equality), so
                         # it contributes no full-direction rows here.
                         "universal": (),
                         "revolute": (np.asarray(j.axis, float),),
                         # The axis equality has rank two (both directions are
                         # unit vectors).  Together with the two transverse
                         # anchor separations below it leaves axial slide and
                         # axial spin: a true cylindrical pair.
                         "cylindrical": (np.asarray(j.axis, float),),
                         # A slider is a cylindrical pair with the spin taken
                         # away, so it holds the WHOLE relative frame (rank 3,
                         # as `fixed` does) over a line-confined anchor (rank 2):
                         # five constraints, one freedom.
                         "prismatic": tuple(np.eye(3)),
                         "fixed": tuple(np.eye(3))}[j.kind],
            }
            entry["dirs"] = [(to_local(j.body_a, a, True),
                              to_local(j.body_b, a, True), a)
                             for a in entry["axes"]]
            if j.kind in _SLIDING_JOINTS:
                # The rail direction, and the two directions across it.  For
                # `prismatic` these cannot come from `axes` — that slot carries
                # the frame lock, not the slide — so the slide axis is kept
                # separately for both sliding kinds.
                axis = np.asarray(j.axis, float)
                trial = (np.array([0.0, 1.0, 0.0])
                         if abs(float(axis[0])) > 0.9 else np.array([1.0, 0.0, 0.0]))
                e1 = np.cross(axis, trial)
                e1 /= np.linalg.norm(e1)
                entry["transverse_world"] = (e1, np.cross(axis, e1))
                # ...and the same two directions carried in body_a's frame, so
                # the constraint follows the rail when the body carrying it
                # rotates.  Ground contributes no rotation, and then this is
                # identical to the world pair.
                entry["transverse_local"] = tuple(
                    to_local(j.body_a, t, True) for t in entry["transverse_world"])
            extra = 0
            if j.kind == "universal":
                # One scalar row: the pin fixed in body_a stays perpendicular to
                # the pin fixed in body_b.  `axis` and `axis2` are given in world
                # frame at rest; carry each in its body's frame (world for
                # ground) so the constraint rotates with the bodies.
                na = np.asarray(j.axis, float)
                nb = np.asarray(j.axis2, float)
                entry["perp_a"] = to_local(j.body_a, na, True)
                entry["perp_b"] = to_local(j.body_b, nb, True)
                entry["perp_world"] = (na, nb)
                extra = 1
            self.joints.append(entry)
            self.n_rows += ((2 if j.kind in _SLIDING_JOINTS else 3)
                            + 3 * len(entry["dirs"]) + extra)

        self._rest_packed = np.concatenate(
            [np.asarray(rest[name], float)[:3, :4].reshape(-1) for name in self.order])
        self.n_projects = 0
        self._log = log
        self.expected_dim = spec.build.expect.dimension
        self.dim_mismatches = 0
        log(f"ideal oracle: {self.n_bodies} bodies, {len(self.joints)} joints, "
            f"{self.n_rows} constraint rows over {self.dim_ambient} ambient dims")

    # -- packing ---------------------------------------------------------------
    def _poses(self, q):
        return np.asarray(q, float).reshape(self.n_bodies, 3, 4)

    def pack(self):
        return self._rest_packed.copy()

    def seed(self):
        """The rest assembly, projected — a spec whose rest pose is already
        assembled (the normal case) gets it back unchanged."""
        return self.project(self._rest_packed.copy())

    def response_validity(self, result):
        """Keep algebraic responses within declared physical travel/workspace.

        These are simulator-side inequalities: they determine whether a command
        has a physical response, while the advancing front still observes only
        the resulting pose and validity flag.  Chart-coordinate bounds are not
        consulted here.

        A projection that stopped short of the constraint set (its residual
        above RESPONSE_RESIDUAL_TOL) is not a physical response either: the
        explorer otherwise takes any finite result as a sample.
        """
        residual = float(np.linalg.norm(self.residual(result), np.inf))
        if residual > RESPONSE_RESIDUAL_TOL:
            return False, {"projection_unconverged": True, "residual": residual}
        diagnostics = physical_domain_violation(self.spec, self._poses(result))
        return diagnostics is None, diagnostics or {}

    # -- the algebraic system --------------------------------------------------
    def _transverse(self, e, T):
        """The two world directions ACROSS a slider's rail at the current pose.

        The rail belongs to body_a, so when body_a rotates the rail rotates with
        it and the pair has to be carried along; holding the rest-pose world
        directions instead silently constrains the slide to a rail that is no
        longer there.  On a rail bolted to ground (`body_a` fixed, or ground
        itself) body_a's rotation is the identity and this returns exactly the
        rest-pose pair.
        """
        i = e["ia"]
        if i is None:
            return e["transverse_world"]
        R = T[i, :, :3]
        return tuple(R @ t for t in e["transverse_local"])

    def _bevel_row(self, e, T):
        """Signed carrier-relative rolling residual for one bevel-gear mesh."""
        R_c, R_pin, R_side = T[e["ic"], :, :3], T[e["ia"], :, :3], T[e["ib"], :, :3]
        a_w = R_c @ e["axle_c"]
        ref_c = R_c @ e["p_c"]
        ref_pin = R_pin @ e["p_pin"]
        # Inline the a_w x ref_c cross product: numpy's np.cross spends ~15x its
        # arithmetic on moveaxis/axis-normalisation dispatch, and this row is
        # differenced 144x per Jacobian.
        cross_dot = ((a_w[1] * ref_c[2] - a_w[2] * ref_c[1]) * ref_pin[0]
                     + (a_w[2] * ref_c[0] - a_w[0] * ref_c[2]) * ref_pin[1]
                     + (a_w[0] * ref_c[1] - a_w[1] * ref_c[0]) * ref_pin[2])
        phi = np.arctan2(float(cross_dot), float(ref_c @ ref_pin))
        d_side = R_side @ e["qd_side"]
        d_c2 = R_c @ e["qd_c2"]
        theta_side = np.arctan2(float(e["w2"] @ d_side), float(e["u2"] @ d_side))
        theta_c2 = np.arctan2(float(e["w2"] @ d_c2), float(e["u2"] @ d_c2))
        rel = np.arctan2(np.sin(theta_side - theta_c2), np.cos(theta_side - theta_c2))
        g = phi + e["ratio"] * rel
        return float(np.arctan2(np.sin(g), np.cos(g)))

    def _bevel_row_batched(self, e, Tb):
        """Vectorised :meth:`_bevel_row` over a stack ``Tb`` of packed poses,
        shape ``(M, n_bodies, 3, 4)`` -> ``(M,)``.  Used to evaluate all 72
        central-difference perturbations of one bevel row in a single pass."""
        R_c = Tb[:, e["ic"], :, :3]
        R_pin = Tb[:, e["ia"], :, :3]
        R_side = Tb[:, e["ib"], :, :3]
        a_w = R_c @ e["axle_c"]
        ref_c = R_c @ e["p_c"]
        ref_pin = R_pin @ e["p_pin"]
        cross = np.cross(a_w, ref_c)                      # one vectorised cross over M rows
        phi = np.arctan2(np.einsum("mi,mi->m", cross, ref_pin),
                         np.einsum("mi,mi->m", ref_c, ref_pin))
        d_side = R_side @ e["qd_side"]
        d_c2 = R_c @ e["qd_c2"]
        theta_side = np.arctan2(d_side @ e["w2"], d_side @ e["u2"])
        theta_c2 = np.arctan2(d_c2 @ e["w2"], d_c2 @ e["u2"])
        rel = np.arctan2(np.sin(theta_side - theta_c2), np.cos(theta_side - theta_c2))
        g = phi + e["ratio"] * rel
        return np.arctan2(np.sin(g), np.cos(g))

    def residual(self, q):
        T = self._poses(q)
        out = np.empty(self.n_rows)
        row = 0
        for i in range(self.n_bodies):
            R = T[i, :, :3]
            gram = R.T @ R - np.eye(3)
            for k, l in _ORTHONORMALITY_PAIRS:
                out[row] = gram[k, l]
                row += 1
        for e in self.joints:
            if e["kind"] == "gear_ratio":
                def phase_angle(side, local):
                    if e[side] is None:     # ground never turns: its phase is 0
                        return 0.0
                    direction = T[e[side], :, :3] @ e[local]
                    x = float(e["phase"] @ direction)
                    y = float(np.cross(e["axis"], e["phase"]) @ direction)
                    return np.arctan2(y, x)
                p, q = e["p"], e["q"]
                phase = (p * phase_angle("ia", "phase_a") + q * phase_angle("ib", "phase_b")
                         - (p + q) * phase_angle("ic", "phase_c"))
                out[row] = np.arctan2(np.sin(phase), np.cos(phase))
                row += 1
                continue
            if e["kind"] == "bevel_gear":
                out[row] = self._bevel_row(e, T)
                row += 1
                continue
            if e["kind"] == "phase_lock":
                out[row:row + 3] = (T[e["ia"], :, :3] @ e["phase_a"]
                                     - T[e["ib"], :, :3] @ e["phase_b"])
                row += 3
                continue
            def anchor(side, local):
                i = e[side]
                if i is None:
                    return e["world_pivot"]
                return T[i, :, :3] @ local + T[i, :, 3]
            delta = anchor("ia", e["pa"]) - anchor("ib", e["pb"])
            if e["kind"] in _SLIDING_JOINTS:
                for k, t in enumerate(self._transverse(e, T)):
                    out[row + k] = float(np.dot(delta, t))
                row += 2
            else:
                out[row:row + 3] = delta
                row += 3
            for local_a, local_b, world in e["dirs"]:
                def direction(side, local):
                    i = e[side]
                    if i is None:
                        return world
                    return T[i, :, :3] @ local
                out[row:row + 3] = direction("ia", local_a) - direction("ib", local_b)
                row += 3
            if e["kind"] == "universal":
                va = (T[e["ia"], :, :3] @ e["perp_a"] if e["ia"] is not None
                      else e["perp_world"][0])
                vb = (T[e["ib"], :, :3] @ e["perp_b"] if e["ib"] is not None
                      else e["perp_world"][1])
                out[row] = float(va @ vb)
                row += 1
        return out

    def jacobian(self, q):
        """Closed form — every residual is quadratic in the packed poses."""
        T = self._poses(q)
        J = np.zeros((self.n_rows, self.dim_ambient))
        row = 0

        def column(body, r, c):
            return 12 * body + 4 * r + c

        for i in range(self.n_bodies):
            R = T[i, :, :3]
            for k, l in _ORTHONORMALITY_PAIRS:
                # d(RᵀR)_kl / dR_mc = delta(c,k) R_ml + delta(c,l) R_mk
                for m in range(3):
                    J[row, column(i, m, k)] += R[m, l]
                    J[row, column(i, m, l)] += R[m, k]
                row += 1
        q_flat = np.asarray(q, float).reshape(-1)
        for e in self.joints:
            if e["kind"] == "bevel_gear":
                # The rolling residual mixes two atan2 terms bilinearly across
                # three body frames; a central difference over just the involved
                # columns is exact to ~1e-7 and far less error-prone to carry
                # than the hand Jacobian.
                h = 1e-6
                cols = np.fromiter(
                    (12 * b + cc for b in (e["ia"], e["ib"], e["ic"]) for cc in range(12)),
                    dtype=int, count=36)
                perturb = np.zeros((2 * len(cols), q_flat.size))
                perturb[np.arange(len(cols)) * 2, cols] = h
                perturb[np.arange(len(cols)) * 2 + 1, cols] = -h
                configs = (q_flat + perturb).reshape(-1, self.n_bodies, 3, 4)
                vals = self._bevel_row_batched(e, configs)
                J[row, cols] = (vals[0::2] - vals[1::2]) / (2.0 * h)
                row += 1
                continue
            if e["kind"] == "gear_ratio":
                # For theta = atan2(y, x), dtheta/dd = (x dy - y dx)/(x^2+y^2).
                # Each phase direction is R @ local_phase, so this remains an
                # exact analytic Jacobian of the carrier-relative gear law.
                for side, coefficient, local in (
                        ("ia", e["p"], "phase_a"), ("ib", e["q"], "phase_b"),
                        ("ic", -(e["p"] + e["q"]), "phase_c")):
                    i = e[side]
                    if i is None:           # ground carrier: constant, no columns
                        continue
                    direction = T[i, :, :3] @ e[local]
                    x = float(e["phase"] @ direction)
                    dy = np.cross(e["axis"], e["phase"])
                    y = float(dy @ direction)
                    grad = (x * dy - y * e["phase"]) / max(x * x + y * y, 1e-15)
                    for r in range(3):
                        for c in range(3):
                            J[row, column(i, r, c)] += coefficient * grad[r] * e[local][c]
                row += 1
                continue
            if e["kind"] == "phase_lock":
                for side, sign, direction in (("ia", 1.0, e["phase_a"]),
                                              ("ib", -1.0, e["phase_b"])):
                    i = e[side]
                    for r in range(3):
                        for c in range(3):
                            J[row + r, column(i, r, c)] += sign * direction[c]
                row += 3
                continue
            if e["kind"] in _SLIDING_JOINTS:
                transverse = self._transverse(e, T)
                # res_k = (anchor_a - anchor_b) . t_k, and t_k = R_a t_k^local
                # now depends on R_a too, so body_a picks up a second term.
                for side, sign, local in (("ia", 1.0, "pa"), ("ib", -1.0, "pb")):
                    i = e[side]
                    if i is None:
                        continue
                    p = e[local]
                    for k, direction in enumerate(transverse):
                        for r in range(3):
                            for c in range(3):
                                J[row + k, column(i, r, c)] += sign * direction[r] * p[c]
                            J[row + k, column(i, r, 3)] += sign * direction[r]
                ia = e["ia"]
                if ia is not None:
                    # d/dR_a[r,c] of (delta . R_a t^local) = delta[r] * t^local[c]
                    def anchor_now(side, local):
                        i = e[side]
                        if i is None:
                            return e["world_pivot"]
                        return T[i, :, :3] @ e[local] + T[i, :, 3]
                    delta = anchor_now("ia", "pa") - anchor_now("ib", "pb")
                    for k, t_local in enumerate(e["transverse_local"]):
                        for r in range(3):
                            for c in range(3):
                                J[row + k, column(ia, r, c)] += delta[r] * t_local[c]
                row += 2
            else:
                for side, sign, local in (("ia", 1.0, "pa"), ("ib", -1.0, "pb")):
                    i = e[side]
                    if i is None:
                        continue
                    p = e[local]
                    for r in range(3):
                        for c in range(3):
                            J[row + r, column(i, r, c)] += sign * p[c]
                        J[row + r, column(i, r, 3)] += sign        # translation
                row += 3
            for local_a, local_b, _world in e["dirs"]:
                for side, sign, a in (("ia", 1.0, local_a), ("ib", -1.0, local_b)):
                    i = e[side]
                    if i is None:
                        continue
                    for r in range(3):
                        for c in range(3):
                            J[row + r, column(i, r, c)] += sign * a[c]
                row += 3
            if e["kind"] == "universal":
                # res = (R_a n_a) . (R_b n_b);  d/dR_a[r,c] = (R_b n_b)[r] n_a[c]
                ia, ib = e["ia"], e["ib"]
                va = (T[ia, :, :3] @ e["perp_a"] if ia is not None
                      else e["perp_world"][0])
                vb = (T[ib, :, :3] @ e["perp_b"] if ib is not None
                      else e["perp_world"][1])
                if ia is not None:
                    for r in range(3):
                        for c in range(3):
                            J[row, column(ia, r, c)] += vb[r] * e["perp_a"][c]
                if ib is not None:
                    for r in range(3):
                        for c in range(3):
                            J[row, column(ib, r, c)] += va[r] * e["perp_b"][c]
                row += 1
        return J

    # -- the oracle interface --------------------------------------------------
    def project(self, q, tol=1e-11, max_iterations=60, hold=None):
        """The closest configuration on the manifold to `q`, by Gauss-Newton.

        Every iterate keeps the pull toward the input: with z = q_k - q, each
        step takes the minimum-norm z solving the linearised constraints
        J z = -C + J (q_k - q), so the fixed point satisfies the closest-point
        KKT condition (q* - q in the row space of J).  Chaining minimum-norm
        corrections from the current iterate instead (the previous scheme)
        does not: each correction is short, but their sum can slide far along
        the manifold.  Measured on lazy_tongs_pantograph, a 0.04 random nudge
        landed a median 2.0x and up to 160x the nudge away that way, against
        0.12x (the tangent fraction sqrt(d/n)) and at most 0.3x for this
        scheme, which converged on all probes in a median of 5 iterations.
        Call it with a `q` that is already close: from far away the closest
        point can still lie on another branch.

        `hold` names (or indices, into `self.order`) bodies to exclude from the
        correction entirely, leaving their block of `q` exactly as given.  Plain
        least-norm correction picks whichever nearby point costs the least
        ambient motion, and on a mechanism with more than one assembly branch
        through the same region that can be a DIFFERENT branch than the one the
        candidate `q` was meant to stay on (measured on `rotate`: an infinitesimal
        step along a genuine tangent direction gets undone back to the branch it
        started from, because reverting it is cheaper than following the curve).
        `hold` names exactly the bodies a stepping direction's own tangent block
        says are stationary there, so the correction can't drift onto a branch
        where they aren't.
        """
        q = np.asarray(q, float).copy()
        self.n_projects += 1
        free_cols = None
        if hold:
            held = {self.order.index(b) if isinstance(b, str) else int(b)
                    for b in hold}
            free_cols = [c for i in range(self.n_bodies) if i not in held
                        for c in range(12 * i, 12 * i + 12)]
        target = q.copy()
        for _ in range(max_iterations):
            res = self.residual(q)
            if np.linalg.norm(res, np.inf) <= tol:
                break
            J = self.jacobian(q)
            if free_cols is not None:
                J = J[:, free_cols]
            offset = (q - target) if free_cols is None else (q - target)[free_cols]
            rhs = -res + J @ offset
            try:
                z, *_ = np.linalg.lstsq(J, rhs, rcond=None)
            except np.linalg.LinAlgError:
                if scipy_lstsq is None:
                    raise
                # QR with pivoting avoids the SVD failure mode that can
                # occur when an exploratory affine chart is temporarily
                # close to a rank-changing configuration.
                z = scipy_lstsq(J, rhs, lapack_driver="gelsy")[0]
            if free_cols is None:
                q = target + z
            else:
                q = target.copy()
                q[free_cols] = target[free_cols] + z
        return q

    def stationary_bodies(self, tangent, rel_tol=1e-6):
        """Bodies whose block of `tangent` is (numerically) zero.

        A stepping direction that only moves a subset of bodies to first order
        is a signal of which bodies must be held fixed during the correction —
        see `project`'s `hold`.  Empty on a mechanism where every direction
        moves every body (the ordinary case), so this is a no-op there.
        """
        tangent = np.asarray(tangent, float)
        norms = np.array([np.linalg.norm(tangent[12 * i:12 * i + 12])
                          for i in range(self.n_bodies)])
        floor = norms.max() * rel_tol
        return [self.order[i] for i in range(self.n_bodies) if norms[i] <= floor]

    def tangent_measurement(self, q, k=None, eps=None, n_probe=None,
                            rel_tol=None, rng=None, scales=(1.0,),
                            witnessed_secants=None, proposal_directions=None):
        """Exact analytic tangent — the override the FD paired probe would do.

        `ManifoldOracle.tangent_measurement` estimates the tangent by finite
        differences of `project()` in random directions and refuses ("scale or
        batch uncertain") wherever the h vs h/2 spectra disagree — which a
        random coarse walking step can trigger near a mechanism singularity even
        though the rank there is perfectly well defined.  This case's Jacobian
        is closed form, so take its nullspace directly: one SVD, deterministic,
        no probe size to tune, and `_rank_by_gap` degrades to a conservative
        absolute cut at a genuine singularity instead of aborting the explore.
        Every probe kwarg is accepted for signature compatibility and ignored.
        """
        _u, singular, vt = np.linalg.svd(self.jacobian(q))
        rank = self._rank_by_gap(singular)
        dim = self.dim_ambient - rank
        self._note_dimension(dim, singular)
        gap = float("inf")
        if 0 < rank < singular.size and singular[rank] > 0.0:
            gap = float(singular[rank - 1] / singular[rank])
        return vt[rank:].T, singular, dim, {
            "probe_method": "exact_analytic_jacobian_nullspace",
            "probe_confidence": 1.0,
            "probe_scale_disagreement": 0.0,
            "probe_rank_gap_ratio": gap,
        }

    def tangent_frame(self, q, eps=None, rel_tol=None):
        """(basis (ambient, dim), singular values, dim) — exact, by SVD.

        `eps` and `rel_tol` exist only to match the probing oracle's signature;
        nothing here is sampled, so there is no probe size and no noise
        threshold to tune.
        """
        frame, singular, dim, _diag = self.tangent_measurement(q)
        return frame, singular, dim

    def _note_dimension(self, dim, singular):
        """Cross-check every rank against the declared dimension.

        `MIN_GAP_RATIO` is a policy constant and a spectral gap is a local read,
        so a rank is a measurement, not a certificate.  The spec's expected
        dimension is an independent statement about the same quantity: where the
        two disagree the mechanism is at a singularity, near one, or the spec is
        wrong — all three worth surfacing.  Counted rather than raised, because
        a genuine singular set is measure-zero and legitimately gets hit; the
        tally is what tells you whether it was a set or a systematic error.
        """
        if self.expected_dim is None or dim == self.expected_dim:
            return
        self.dim_mismatches += 1
        if self.dim_mismatches <= _MISMATCH_REPORTS:
            gap = "n/a"
            r = self.dim_ambient - dim
            if 0 < r < singular.size and singular[r] > 0.0:
                gap = f"{singular[r - 1] / singular[r]:.1e}"
            self._log(f"ideal oracle: dim {dim}, expected {self.expected_dim} "
                      f"(gap at the cut {gap}): singular configuration, or a "
                      f"wrong expect.dimension")
        elif self.dim_mismatches == _MISMATCH_REPORTS + 1:
            self._log("ideal oracle: further dimension mismatches silenced; "
                      "final count is in the sampling summary")

    @staticmethod
    def _rank_by_gap(singular):
        """Rank from the FIRST large ratio gap, scanning down from the top.

        The genuine constraints form a contiguous block at the top of the
        spectrum and the null directions sit orders of magnitude below it, so
        the first big drop is where the rank ends.  Taking the LARGEST gap
        instead does not work: the null values are not all the same size — they
        land around sqrt(residual), because a body whose RᵀR = I holds only to
        1e-14 perturbs those singular values by ~1e-7, while directions that are
        null for structural reasons stay at 1e-15.  The biggest ratio in the
        spectrum is then *inside* the null block (1.8e-7 / 1.1e-15 = 1.7e8,
        against 1.4e-1 / 1.8e-7 = 7.6e5 at the true cut), which reported dim 1
        for this 3-DOF mechanism at 28 of 40 configurations.
        """
        s = np.asarray(singular, float)
        if s.size == 0 or s[0] <= 0.0:
            return 0
        floor = s[0] * np.finfo(float).eps
        safe = np.maximum(s, floor)
        ratios = safe[:-1] / safe[1:]
        hits = np.flatnonzero(ratios >= MIN_GAP_RATIO)
        if hits.size:
            return int(hits[0]) + 1
        # No clean separation anywhere: at or near a singularity the dimension is
        # genuinely ambiguous.  Fall back to a conservative absolute cut and let
        # the caller's rank check see it.
        return int((s > floor).sum())


def build_ideal_oracle(spec, log=print):
    return IdealJointOracle(spec, log=log)
