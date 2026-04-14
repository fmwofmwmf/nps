import math

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import igl


def load_mesh(path):
    """
    Load a mesh file using libigl.
    Supports .obj, .off, .stl, .ply, .mesh and more.

    Returns:
        {'vertices': (N, 3) np.ndarray float64, 'faces': (F, 3) np.ndarray int32}
    """
    v, f = igl.read_triangle_mesh(path)
    return {
        'vertices': np.array(v, dtype=np.float64),
        'faces': np.array(f, dtype=np.int32),
    }

def line_intersection(p1, p2, p3, p4):
    """
    վերադարձ intersection point of lines (p1,p2) and (p3,p4)
    assumes they are not parallel
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-8:
        raise ValueError("Lines are parallel or degenerate")

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py])


def normalized_pos_on_bar(p, a, b):
    c = 0.5 * (a + b)
    d = b - a
    return np.dot(p - c, d) / np.dot(d, d)

def parse_mechanism(file_path):
    points = []
    bars_idx = []
    joints_idx = []
    fixed = []
    gears = []

    section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line in ["POINTS", "BARS", "JOINTS", "FIXED", "GEARS"]:
                section = line
                continue

            vals = list(map(float, line.split()))

            if section == "POINTS":
                points.append(vals)

            elif section == "BARS":
                bars_idx.append([int(vals[0]), int(vals[1])])

            elif section == "JOINTS":
                joints_idx.append([int(vals[0]), int(vals[1])])

            elif section == "FIXED":
                fixed.append([int(vals[0]), vals[1]])

            elif section == "GEARS":
                # idx idy ratio
                gears.append([int(vals[0]), int(vals[1]), vals[2]])

    points = np.array(points)

    # --- Build bars ---
    bars = []
    for i1, i2 in bars_idx:
        p1 = points[i1]
        p2 = points[i2]
        bars.append([p1[0], p1[1], p2[0], p2[1]])
    bars = np.array(bars)

    # --- Build joints ---
    joints = []
    for b1_idx, b2_idx in joints_idx:
        b1 = bars[b1_idx]
        b2 = bars[b2_idx]

        p1 = b1[:2]
        p2 = b1[2:]
        p3 = b2[:2]
        p4 = b2[2:]

        # intersection point
        p = line_intersection(p1, p2, p3, p4)

        # normalized positions
        t1 = normalized_pos_on_bar(p, p1, p2)
        t2 = normalized_pos_on_bar(p, p3, p4)

        joints.append([b1_idx, b2_idx, t1, t2])

    joints = np.array(joints)

    # --- Build fixed joints ---
    fixed_joints = []
    for bar_idx, dist in fixed:
        b = bars[bar_idx]
        p1 = b[:2]
        p2 = b[2:]

        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-12:
            raise ValueError("Degenerate bar in FIXED")

        c = 0.5 * (p1 + p2)
        dir = (p2 - p1) / np.linalg.norm(p2 - p1)
        pos = c + dir * dist * length

        fixed_joints.append([pos[0], pos[1], bar_idx])

    fixed_joints = np.array(fixed_joints)

    gears = np.array(gears) if len(gears) > 0 else np.zeros((0, 3))

    return bars, joints, fixed_joints, gears

def convert_to_solver_format(
    bars,
    joints,
    fixed_joints,
    gears,
    dtype=torch.float64
):
    bars = np.asarray(bars)
    joints = np.asarray(joints)
    fixed_joints = np.asarray(fixed_joints)

    # --- bars → center, angle, length ---
    centers = []
    angles = []
    lengths = []

    for b in bars:
        p1 = b[:2]
        p2 = b[2:]

        c = 0.5 * (p1 + p2)
        d = p2 - p1
        L = np.linalg.norm(d)

        angle = math.atan2(d[1], d[0])

        centers.append(c)
        angles.append(angle)
        lengths.append(L)

    centers = torch.tensor(centers, dtype=dtype)
    angles = torch.tensor(angles, dtype=dtype)

    # --- helpers ---
    def centered_t(p, p1, p2):
        c = 0.5 * (p1 + p2)
        d = p2 - p1
        return np.dot(p - c, d) / np.dot(d, d)

    def intersect(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-8:
            raise ValueError("Parallel bars")

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

        return np.array([px, py])

    # --- joints: recompute centered t directly ---
    solver_joints = []

    for b1, b2, _, _ in joints:
        b1 = int(b1)
        b2 = int(b2)

        p1 = bars[b1][:2]
        p2 = bars[b1][2:]
        p3 = bars[b2][:2]
        p4 = bars[b2][2:]

        p = intersect(p1, p2, p3, p4)

        t1c = centered_t(p, p1, p2)
        t2c = centered_t(p, p3, p4)

        solver_joints.append([b1, b2, t1c, t2c])

    # --- fixed joints ---
    fixed_positions = []
    fixed_entries = []

    for k, (x, y, bar_idx) in enumerate(fixed_joints):
        bar_idx = int(bar_idx)

        p1 = bars[bar_idx][:2]
        p2 = bars[bar_idx][2:]

        p = np.array([x, y])

        t = centered_t(p, p1, p2)

        fixed_positions.append([x, y])
        fixed_entries.append([-(k+1), bar_idx, 0.0, t])

    gear_constraints = []

    if gears is not None and len(gears) > 0:
        for (a, b, ratio) in gears:
            a = int(a)
            b = int(b)

            theta_a0 = angles[a].item()
            theta_b0 = angles[b].item()

            # want: theta_a - ratio * theta_b - phase = 0 at init
            phase = theta_a0 - ratio * theta_b0

            gear_constraints.append({
                'bar_a': a,
                'bar_b': b,
                'ratio': ratio,
                'phase': phase
            })

    # --- combine ---
    all_joints = solver_joints + fixed_entries

    all_joints = all_joints
    fixed_positions = torch.tensor(fixed_positions, dtype=dtype)

    return centers, angles, lengths, all_joints, fixed_positions, gear_constraints

class BarJoint2DSystem:
    """
    2D bar-joint system where rigid bars are connected by rotational joints.
    Each bar has 3 DOF (x, y position of center, rotation angle).
    Joints can be placed anywhere along bars and have spring penalties.

    NEW: Object rendering
        system.model_meshes  — list of mesh dicts {'vertices': (N,3), 'faces': (M,3)}
                               in local bar frame (+X along bar, origin at center, scaled
                               so ±0.5 in X = bar endpoints).
        system.bar_models    — list of ints, one per bar.
                               -1  = default line rendering.
                               k   = use model_meshes[k].
        API:
            obj_id = system.register_mesh(vertices, faces)
            system.assign_bar_model(bar_idx, obj_id)

    NEW: Sliding joints
        A point on bar A is constrained to lie on the infinite axis of bar B.
        Two penalties:
          - Perpendicular distance from the point to bar B's axis  (k_perp)
          - sin²(θ_a - θ_b) angle misalignment                     (k_angle, optional)
        API:
            system.add_sliding_joint(bar_a_idx, pos_norm_a, bar_b_idx,
                                     stiffness_perp, stiffness_angle=0.0)
    """

    @staticmethod
    def construct(problem_name, config, dtype=torch.float64):
        system_def = {}
        system = BarJoint2DSystem()

        system.system_name = "BarJoint2D"
        system.problem_name = str(problem_name)

        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,), dtype=dtype)
        system.cond_dim = 0

        # Object rendering — populated after construction via register_mesh / assign_bar_model
        system.model_meshes = []   # list of {'vertices': np (N,3), 'faces': np (M,3)}
        system.bar_models   = []   # filled to [-1]*num_bars after num_bars is known

        # Sliding joints: list of [bar_a_idx, pos_norm_a, bar_b_idx, k_perp, k_angle]
        system.sliding_joints = []
        HALF = math.pi/2
        if problem_name == 'single':
            num_bars = 1
            bar_lengths = [1.0]
            bar_masses = [1.0]
            bar_inertias = [1.0 / 12.0]

            joints = [[-1, 0, 0.0, -0.5]]
            joint_stiffness = [config["system"]["joint_stiffness"]]

            fixed_joint_positions = torch.tensor([[0.0, 2.0]], dtype=dtype)
            init_bar_positions = torch.tensor([[0.5, 2.0]], dtype=dtype)
            init_bar_angles = torch.tensor([0.0], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'double':
            num_bars = 2
            bar_lengths = [1.0, 0.8]
            bar_masses = [1.0, 0.8]
            bar_inertias = [1.0 / 12.0, 0.64 / 12.0]

            joints = [
                [-1, 0, 0.0, -0.5],
                [0, 1, 0.5, -0.5]
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * 2

            fixed_joint_positions = torch.tensor([[0.0, 2.5]], dtype=dtype)
            init_bar_positions = torch.tensor([[0.0, 2.0], [0.0, 1.1]], dtype=dtype)
            init_bar_angles = torch.tensor([0.2, -0.3], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'chain':
            num_bars = config["system"]["num_bars"]
            bar_length = config["system"]["bar_length"]
            bar_mass = config["system"]["bar_mass"]

            bar_lengths = [bar_length] * num_bars
            bar_masses = [bar_mass] * num_bars
            bar_inertias = [bar_mass * bar_length ** 2 / 12.0] * num_bars

            joints = [[-1, 0, 0.0, -0.5]]
            joint_stiffness = [config["system"]["joint_stiffness"]]
            for i in range(num_bars - 1):
                joints.append([i, i + 1, 0.5, -0.5])
                joint_stiffness.append(config["system"]["joint_stiffness"])

            fixed_joint_positions = torch.tensor([[0.0, 3.0]], dtype=dtype)
            init_bar_positions = []
            y_pos = 3.0
            for i in range(num_bars):
                y_pos -= bar_length * 0.5
                init_bar_positions.append([0.0, y_pos])
                y_pos -= bar_length * 0.5
            init_bar_positions = torch.tensor(init_bar_positions, dtype=dtype)
            init_bar_angles = torch.zeros(num_bars, dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'bridge':
            num_bars = 7
            bar_lengths = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.0]
            bar_masses = [0.8] * num_bars
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            joints = [
                [-1, 0, 0.0, -0.5], [-2, 3, 0.0, 0.5],
                [0, 1, 0.5, -0.5], [1, 2, 0.5, -0.5], [2, 3, 0.5, -0.5],
                [0, 4, 0.0, -0.5], [4, 1, 0.5, 0.0],
                [2, 5, 0.0, -0.5], [5, 3, 0.5, 0.0],
                [1, 6, 0.0, -0.5], [6, 2, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([[0.0, 2.0], [3.0, 2.0]], dtype=dtype)
            init_bar_positions = torch.tensor([
                [0.5, 2.0], [1.5, 2.0], [2.5, 2.0], [3.5, 2.0],
                [0.85, 1.3], [2.15, 1.3], [2.0, 1.5],
            ], dtype=dtype)
            init_bar_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.785, -0.785, 1.57], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'linkage':
            num_bars = 3
            bar_lengths = [2, 1, 1]
            bar_masses = [2, 1, 1]
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            joints = [
                [-1, 1, 0.0, -0.5], [-2, 2, 0.0, -0.5],
                [0, 1, -0.5, 0.5],  [0, 2, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([[0.0, 0], [2.0, 0]], dtype=dtype)
            init_bar_positions = torch.tensor([[1, 1], [0, 0.5], [2, 0.5]], dtype=dtype)
            init_bar_angles = torch.tensor([0.0, math.pi / 2, math.pi / 2], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'stack':
            num_bars = 6
            bar_lengths = [2, 1, 1, 2, 1, 1]
            bar_masses = [2, 1, 1, 2, 1, 1]
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            joints = [
                [-1, 1, 0.0, -0.5], [-2, 2, 0.0, -0.5],
                [0, 1, -0.5, 0.5],  [0, 2, 0.5, 0.5],
                [0, 4, -0.5, -0.5], [0, 5, 0.5, -0.5],
                [3, 4, -0.5, 0.5],  [3, 5, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([[0.0, 0], [2.0, 0]], dtype=dtype)
            init_bar_positions = torch.tensor([
                [1, 1], [0, 0.5], [2, 0.5],
                [1, 2], [0, 1.5], [2, 1.5],
            ], dtype=dtype)
            init_bar_angles = torch.tensor(
                [0.0, math.pi / 2, math.pi / 2, 0.0, math.pi / 2, math.pi / 2], dtype=dtype
            )

            system_def['gravity'] = 9.8

        elif problem_name == 'scissor_lift':
            n_stages = config["system"].get("n_stages", 3)
            L        = config["system"].get("bar_length", 2.0)
            bar_mass = config["system"].get("bar_mass", 1.0)
            k        = config["system"]["joint_stiffness"]
            symmetric_horse = config["system"]["symmetric"]
            weight = config["system"]["horse_weight"]

            system.model_meshes = [
                load_mesh("objects/scissor_lift.obj"),
                load_mesh("objects/scissor_lift_top_symm.obj") if symmetric_horse else load_mesh("objects/scissor_lift_top.obj"),
            ]

            system.bar_models = [0] * (2 * n_stages) + [1]

            angle_a = 2 * math.pi / 3
            angle_b =     math.pi / 3
            cos_a, sin_a = math.cos(angle_a), math.sin(angle_a)
            cos_b, sin_b = math.cos(angle_b), math.sin(angle_b)

            hy_b         = 0.5 * L * sin_b
            stage_height = 2 * hy_b
            cx, cy0      = 1.0, hy_b

            init_bar_positions, init_bar_angles = [], []
            for i in range(n_stages):
                cy_i = cy0 + i * stage_height
                init_bar_positions += [[cx, cy_i], [cx, cy_i]]
                init_bar_angles    += [angle_a, angle_b]
            init_bar_positions = torch.tensor(init_bar_positions, dtype=dtype)
            init_bar_angles    = torch.tensor(init_bar_angles,    dtype=dtype)

            anchor_x = cx + (-0.5) * L * cos_a
            anchor_y = cy0 + (-0.5) * L * sin_a
            fixed_joint_positions = torch.tensor([[anchor_x, anchor_y]], dtype=dtype)

            joints, joint_stiffness = [], []
            for i in range(n_stages):
                a, b = 2 * i, 2 * i + 1
                joints.append([a, b, 0.0, 0.0]);  joint_stiffness.append(k)
                if i == 0:
                    pass
                    #joints.append([-1, a, 0.0, -0.5]); joint_stiffness.append(k)
                else:
                    pa, pb = 2*(i-1), 2*(i-1)+1
                    joints.append([pb, a, 0.5, -0.5]); joint_stiffness.append(k)
                    joints.append([pa, b, 0.5, -0.5]); joint_stiffness.append(k)

            # ---- top platform bar ----
            # Sits horizontally across the top of the scissor.
            # Right end pinned to top-right of last bar_b (hard joint).
            # Left end slides along top-left of last bar_a (sliding joint, free to move along it).

            top_bar_idx = 2 * n_stages  # next available bar index

            last_a = 2 * (n_stages - 1)
            last_b = 2 * (n_stages - 1) + 1

            # Compute top-right and top-left positions to place the bar correctly
            top_right_x = cx + 0.5 * L * cos_b  # = cx + 0.5*L*0.5
            top_right_y = cy0 + (n_stages - 1) * stage_height + 0.5 * L * sin_b
            top_left_x = cx + 0.5 * L * cos_a  # = cx - 0.5*L*0.5
            top_left_y = cy0 + (n_stages - 1) * stage_height + 0.5 * L * sin_a

            top_cx = L/2#(top_right_x - 1)
            top_cy = (top_right_y + top_left_y) / 2.0

            num_bars = 2 * n_stages + 1
            bar_lengths = [L] * (2 * n_stages + 1)
            bar_masses = [bar_mass] * num_bars
            bar_masses[num_bars - 1] = weight
            bar_inertias = [bar_mass * l ** 2 / 12.0 for l in bar_lengths]

            init_bar_positions = torch.cat([
                init_bar_positions,
                torch.tensor([[top_cx, top_cy]], dtype=dtype)
            ])
            init_bar_angles = torch.cat([
                init_bar_angles,
                torch.tensor([0.0], dtype=dtype)  # horizontal
            ])

            # Hard pin: top-right end of platform (pos=+0.5) to top-right end of last bar_b (pos=+0.5)
            #joints.append([last_b, top_bar_idx, 0.5, 0.5])
            #joint_stiffness.append(k)

            # Sliding joint: left end of platform slides along last bar_a's axis
            # bar_a is the right-side bar (angle=2pi/3), so the platform left end
            # can slide up/down along it as the scissor extends
            if 'sliding_joints' not in system_def:
                system_def['sliding_joints'] = [{
                'bar_a': last_a,
                'pos_norm_a': 0.5,  # left end of platform
                'bar_b': top_bar_idx,  # slides along last bar_a's axis
                'stiffness': k,
                'angle_stiffness': 0.0,  # platform stays horizontal, no angle coupling
            },
                    {
                    'bar_a': last_b,
                    'pos_norm_a': 0.5,  # left end of platform
                    'bar_b': top_bar_idx,  # slides along last bar_a's axis
                    'stiffness': k,
                    'angle_stiffness': 0.0,  # platform stays horizontal, no angle coupling
                }
                ]

            system_def['y_bar_constraints'] = [{
                'bar_idx': 1, 'pos_norm': -0.5, 'target_y': 0.0, 'stiffness': k,
            },
                {
                    'bar_idx': 0, 'pos_norm': -0.5, 'target_y': 0.0, 'stiffness': k,
                }
            ]
            system_def['gravity'] = 9.8
            system_def['external_forces']['torque_strength']  = 0.0
            system_def['external_forces']['torque_joint_idx'] = 0

        elif problem_name == 'scissor_lift_piston':
            n_stages = config["system"].get("n_stages", 3)
            L        = config["system"].get("bar_length", 2.0)
            bar_mass = config["system"].get("bar_mass", 1.0)
            k        = config["system"]["joint_stiffness"]

            angle_a = 2 * math.pi / 3
            angle_b =     math.pi / 3
            cos_a, sin_a = math.cos(angle_a), math.sin(angle_a)
            cos_b, sin_b = math.cos(angle_b), math.sin(angle_b)

            hy_b         = 0.5 * L * sin_b
            stage_height = 2 * hy_b
            cx, cy0      = 1.0, hy_b

            sl_left_x = cx + (-0.5) * L * cos_b
            sl_left_y = cy0 + (-0.5) * L * sin_b
            anchor_x  = cx + (-0.5) * L * cos_a
            anchor_y  = cy0 + (-0.5) * L * sin_a

            Lp = config["system"].get("piston_crank_length", .3)
            Lr = config["system"].get("piston_rod_length",   1)
            pivot_x, pivot_y = sl_left_x - Lp - Lr, 0.0

            p0_idx = 2 * n_stages
            p1_idx = 2 * n_stages + 1

            num_bars     = 2 * n_stages + 2
            bar_lengths  = [L] * (2 * n_stages) + [Lp, Lr]
            bar_masses   = [bar_mass] * num_bars
            bar_inertias = [bar_mass * l ** 2 / 12.0 for l in bar_lengths]

            init_bar_positions, init_bar_angles = [], []
            for i in range(n_stages):
                cy_i = cy0 + i * stage_height
                init_bar_positions += [[cx, cy_i], [cx, cy_i]]
                init_bar_angles    += [angle_a, angle_b]
            init_bar_positions += [[pivot_x + 0.5*Lp, 0.0], [pivot_x + Lp + 0.5*Lr, 0.0]]
            init_bar_angles    += [0.0, 0.0]
            init_bar_positions = torch.tensor(init_bar_positions, dtype=dtype)
            init_bar_angles    = torch.tensor(init_bar_angles,    dtype=dtype)

            fixed_joint_positions = torch.tensor(
                [[pivot_x, pivot_y], [anchor_x, anchor_y]], dtype=dtype
            )

            joints, joint_stiffness = [], []
            joints.append([-1,     p0_idx, 0.0, -0.5]); joint_stiffness.append(k)
            joints.append([p0_idx, p1_idx, 0.5, -0.5]); joint_stiffness.append(k)
            joints.append([p1_idx, 1,      0.5, -0.5]); joint_stiffness.append(k)

            for i in range(n_stages):
                a, b = 2*i, 2*i+1
                joints.append([a, b, 0.0, 0.0]); joint_stiffness.append(k)
                if i == 0:
                    joints.append([-2, a, 0.0, -0.5]); joint_stiffness.append(k)
                else:
                    pa, pb = 2*(i-1), 2*(i-1)+1
                    joints.append([pb, a, 0.5, -0.5]); joint_stiffness.append(k)
                    joints.append([pa, b, 0.5, -0.5]); joint_stiffness.append(k)

            system_def['y_bar_constraints'] = [{
                'bar_idx': 1, 'pos_norm': -0.5, 'target_y': 0.0, 'stiffness': k,
            }]
            system_def['gravity'] = 0.0
            system_def['external_forces']['torque_strength']  = 0.0
            system_def['external_forces']['torque_joint_idx'] = 0
        elif problem_name == 'platform':
            system.model_meshes = [
                load_mesh("objects/scissor_lift.obj"),
                load_mesh("objects/scissor_lift_top.obj"),
            ]
            k = config["system"]["joint_stiffness"]
            #system.bar_models = [0] * (2 * n_stages) + [1]

            init_bar_positions = [(0,0), (-1, 0), (1.5, 0), (-1, 1.5), (1.5, 1.5), (1.5, 2), (1.5, 2)]
            init_bar_angles = [0, 0, 0, HALF, HALF, 0, 0]
            init_bar_positions = torch.tensor(init_bar_positions, dtype=dtype)
            init_bar_angles    = torch.tensor(init_bar_angles,    dtype=dtype)

            fixed_joint_positions = torch.tensor([(0,0), (-1, 0), (1.5, 0)], dtype=dtype)

            joints  = [
                (-1, 0, 0, 0),
                (-2, 1, 0, 0),
                (-3, 2, 0, 0),
                (3, 1, -0.5, -0.5),
                (4, 2, -0.5, 0.5),
            ]
            joint_stiffness = [k] * len(joints)

            bar_lengths = [1, 1, 2, 2, 2, 1, 2]
            num_bars = len(bar_lengths)
            bar_masses = [1] * num_bars
            bar_inertias = [1 * l ** 2 / 12.0 for l in bar_lengths]

            # if 'sliding_joints' not in system_def:
            #     system_def['sliding_joints'] = []
            # system_def['sliding_joints'].append({
            #     'bar_a': last_a,
            #     'pos_norm_a': 0.5,  # left end of platform
            #     'bar_b': top_bar_idx,  # slides along last bar_a's axis
            #     'stiffness': k,
            #     'angle_stiffness': 0.0,  # platform stays horizontal, no angle coupling
            # })

            system_def['gear_constraints'] = [{
                'bar_a': 0,
                'bar_b': 1,
                'ratio': 1,
                'stiffness': k,
                'phase': init_bar_angles[0] - 0.5 * init_bar_angles[1],
            },
                {
                    'bar_a': 0,
                    'bar_b': 2,
                    'ratio': 2,
                    'stiffness': k,
                    'phase': init_bar_angles[0] - 0.5 * init_bar_angles[2],
                }
            ]

            system_def['x_bar_constraints'] = [
                {
                    'bar_idx': 3, 'pos_norm': 0.5, 'target_x': -1.0, 'stiffness': k,
                },
                {
                    'bar_idx': 4, 'pos_norm': 0.5, 'target_x': 1.5, 'stiffness': k,
                }
            ]
            system_def['gravity'] = 0.0
            system_def['external_forces']['torque_strength']  = 0.0
            system_def['external_forces']['torque_joint_idx'] = 0
        elif problem_name == 'grabber':
            # Grabber mechanism: two gears driving articulated arms that close like a claw.
            #
            # Bar layout:
            #   0: G_right  — right gear bar,  pivot at (1,0),   tip at (2,2)
            #   1: G_left   — left gear bar,   pivot at (-1,0),  tip at (-2,2)
            #   2: Driver   — driver gear bar, pivot at center,  radius 0.5
            #   3: A_right  — right upper arm, fixed base at (0.5,1.5),  elbow at (1,4)
            #   4: B_right  — right lower arm, elbow at (1,4),   tip at gear (2,2)
            #   5: A_left   — left upper arm,  fixed base at (-0.5,1.5), elbow at (-1,4)
            #   6: B_left   — left lower arm,  elbow at (-1,4),  tip at gear (-2,2)
            #
            # Gear constraints (external gears, opposite rotation):
            #   Driver -> G_right:  ratio = -r_driver/r_G = -0.5
            #   G_right -> G_left:  ratio = -r_G/r_G      = -1.0  (they mesh directly)
            #
            # Torquing joint 4 (driver pivot) opens/closes the grabber.

            k = config["system"]["joint_stiffness"]

            A_len = math.sqrt(0.5 ** 2 + 2.5 ** 2)  # (0.5->1.0, 1.5->4.0) = sqrt(6.5) ~ 2.5495
            B_len = math.sqrt(5)  # (1->2, 4->2) = sqrt(1+4)
            G_len = 2 * math.sqrt(5)  # pivot(1,0) to tip(2,2)
            D_len = 1.0  # driver stub, 2 * radius

            G_right_angle = math.atan2(2.0, 1.0)  # 63.4 deg
            G_left_angle = math.atan2(2.0, -1.0)  # 116.6 deg
            A_right_angle = math.atan2(2.5, 0.5)  # 78.7 deg
            A_left_angle = math.atan2(2.5, -0.5)  # 101.3 deg
            B_right_angle = math.atan2(-2.0, 1.0)  # -63.4 deg
            B_left_angle = math.atan2(-2.0, -1.0)  # -116.6 deg
            Driver_angle = 0.0

            num_bars = 7
            bar_lengths = [G_len, G_len, D_len, A_len, B_len, A_len, B_len]
            bar_masses = [1.0] * num_bars
            bar_inertias = [m * l ** 2 / 12.0 for m, l in zip(bar_masses, bar_lengths)]

            system.model_meshes = [
                load_mesh("objects/gear_big.obj"),
                load_mesh("objects/gear_small.obj"),
            ]

            system.bar_models = [0, 0, 1, -1, -1, -1, -1]

            init_bar_positions = torch.tensor([
                [1.0, 0.0],  # G_right center = midpoint of (1,0)-(2,2)
                [-1.0, 0.0],  # G_left  center = midpoint of (-1,0)-(-2,2)
                [1.0, -1.5],  # Driver  center = pivot (pinned at center)
                [0.75, 2.75],  # A_right center = midpoint of (0.5,1.5)-(1,4)
                [1.5, 3.0],  # B_right center = midpoint of (1,4)-(2,2)
                [-0.75, 2.75],  # A_left  center = midpoint of (-0.5,1.5)-(-1,4)
                [-1.5, 3.0],  # B_left  center = midpoint of (-1,4)-(-2,2)
            ], dtype=dtype)

            init_bar_angles = torch.tensor([
                G_right_angle, G_left_angle, Driver_angle,
                A_right_angle, B_right_angle,
                A_left_angle, B_left_angle,
            ], dtype=dtype)

            # Fixed points — pinned ends / pivots
            # [0] = A_right base (0.5, 1.5)  -> bar1_idx = -1
            # [1] = A_left  base (-0.5,1.5)  -> bar1_idx = -2
            # [2] = G_right pivot (1, 0)     -> bar1_idx = -3
            # [3] = G_left  pivot (-1, 0)    -> bar1_idx = -4
            # [4] = Driver  pivot (1,-1.5)   -> bar1_idx = -5
            fixed_joint_positions = torch.tensor([
                [0.5, 1.5],
                [-0.5, 1.5],
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, -1.5],
            ], dtype=dtype)

            joints = [
                [-1, 3, 0.0, -0.5],  # [0] A_right base pinned
                [-2, 5, 0.0, -0.5],  # [1] A_left  base pinned
                [-3, 0, 0.0, -0.0],  # [2] G_right pivot pinned
                [-4, 1, 0.0, -0.0],  # [3] G_left  pivot pinned
                [-5, 2, 0.0, 0.0],  # [4] Driver  center pinned  <- TORQUE JOINT
                [3, 4, 0.5, -0.5],  # [5] A_right elbow: A_right(+0.5) <-> B_right(-0.5)
                [0, 4, 0.5, 0.5],  # [6] G_right tip:   G_right(+0.5) <-> B_right(+0.5)
                [5, 6, 0.5, -0.5],  # [7] A_left  elbow: A_left(+0.5)  <-> B_left(-0.5)
                [1, 6, 0.5, 0.5],  # [8] G_left  tip:   G_left(+0.5)  <-> B_left(+0.5)
            ]
            joint_stiffness = [k] * len(joints)

            # Gear constraints (external gears -> opposite rotation -> negative ratio)
            system_def['gear_constraints'] = [
                {  # Driver (bar 2) drives G_right (bar 0): r_driver=0.5, r_G=1.0
                    'bar_a': 0,  # G_right
                    'bar_b': 2,  # Driver
                    'ratio': 0.5,  # theta_G = -0.5 * theta_D
                    'stiffness': k,
                    'phase': float(init_bar_angles[0]) + 0.5 * float(init_bar_angles[2]),  # = G_right_angle
                },
                {  # G_right (bar 0) drives G_left (bar 1): same radius, opposite spin
                    'bar_a': 1,  # G_left
                    'bar_b': 0,  # G_right
                    'ratio': 1.0,  # theta_GL = -theta_GR
                    'stiffness': k,
                    'phase': float(init_bar_angles[1]) + 1.0 * float(init_bar_angles[0]),  # = pi
                },
            ]

            system_def['rotation_constraints'] = [{
                'bar_idx': 2,  # Driver bar
                'min_angle': -math.pi/4,  # half turn back
                'max_angle': math.pi/4,  # half turn forward
                'stiffness': k,
            }]

            system_def['gravity'] = 0.0
            system_def['external_forces']['torque_strength'] = 0.0
            system_def['external_forces']['torque_joint_idx'] = 4  # driver pivot
        elif problem_name == 'file':
            bars, joints, fixed_joints, gears = (parse_mechanism(config["system"]["mechanism_name"]))
            init_bar_positions, init_bar_angles, bar_lengths, joints, fixed_joint_positions, gear_constraints = convert_to_solver_format(bars, joints, fixed_joints, gears)
            system_def['gear_constraints'] = gear_constraints
            bar_masses = [m/10 for m in bar_lengths]
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]
            num_bars = len(bars)

            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)
            system_def['gravity'] = 9.8
            system_def['external_forces']['torque_strength'] = 0.0
            system_def['external_forces']['torque_joint_idx'] = 0  # first crank pin joint
        elif problem_name == 'file_array':
            # Load base mechanism from file
            base_bars, base_joints_raw, base_fixed_joints, base_gears = parse_mechanism(
                config["system"]["mechanism_name"]
            )
            (base_positions, base_angles, base_lengths,
             base_joints, base_fixed_pos, base_gear_constraints) = convert_to_solver_format(
                base_bars, base_joints_raw, base_fixed_joints, base_gears
            )

            instances = config["system"]["instances"]
            # instances = [{"x": 0, "y": 0, "rotation": 0, "flipped": False, "driver_bar": 0}, ...]

            n_base_bars = len(base_bars)
            n_base_fixed = len(base_fixed_pos)

            k = config["system"]["joint_stiffness"]
            driver_length = config["system"].get("driver_length", 0.5)

            def transform_point(p, x_off, y_off, rotation, flipped):
                """Transform a 2D point: flip (optional) -> rotate -> translate."""
                px, py = float(p[0]), float(p[1])
                if flipped:
                    px = -px
                cos_r, sin_r = math.cos(rotation), math.sin(rotation)
                rx = px * cos_r - py * sin_r
                ry = px * sin_r + py * cos_r
                return torch.tensor([rx + x_off, ry + y_off], dtype=dtype)

            def transform_angle(a, rotation, flipped):
                """Transform an angle: flip (optional) -> rotate."""
                angle = float(a)
                if flipped:
                    angle = math.pi - angle
                return angle + rotation

            # === Central driver bar (bar index 0) ===
            all_positions = [torch.tensor([0.0, 0.0], dtype=dtype)]
            all_angles = [torch.tensor(0.0, dtype=dtype)]
            all_lengths = [driver_length]
            all_masses = [1.0]

            # Fixed joint 0: driver bar pivot at origin
            all_fixed_positions = [torch.tensor([0.0, 0.0], dtype=dtype)]

            # Joint: driver bar center pinned to fixed joint 0
            all_joints = [[-1, 0, 0.0, 0.0]]
            all_joint_stiffness = [k]

            gear_constraints = []

            bar_offset = 1
            fixed_offset = 1

            for inst_idx, inst in enumerate(instances):
                x_off = inst["x"]
                y_off = inst["y"]
                rotation = inst.get("rotation", 0.0)
                flipped = inst.get("flipped", False)
                driver_bar_local = inst["driver_bar"]

                # Add bars
                for i in range(n_base_bars):
                    pos = transform_point(base_positions[i], x_off, y_off, rotation, flipped)
                    all_positions.append(pos)

                    angle = transform_angle(base_angles[i], rotation, flipped)
                    all_angles.append(torch.tensor(angle, dtype=dtype))

                    all_lengths.append(base_lengths[i])
                    all_masses.append(base_lengths[i] / 10.0)

                # Add fixed joints
                for fj in base_fixed_pos:
                    pos = transform_point(fj, x_off, y_off, rotation, flipped)
                    all_fixed_positions.append(pos)

                # Add joints (remap indices, pos_norm unchanged)
                for (bar1_idx, bar2_idx, pos1_norm, pos2_norm) in base_joints:
                    if bar1_idx >= 0:
                        new_bar1 = bar1_idx + bar_offset
                    else:
                        old_fixed_idx = -1 - bar1_idx
                        new_fixed_idx = old_fixed_idx + fixed_offset
                        new_bar1 = -1 - new_fixed_idx

                    new_bar2 = bar2_idx + bar_offset

                    # pos_norm stays the same (it's in bar-local coordinates)
                    all_joints.append([new_bar1, new_bar2, pos1_norm, pos2_norm])
                    all_joint_stiffness.append(k)

                # Add gear constraints from base mechanism
                for gc in base_gear_constraints:
                    bar_a_global = gc['bar_a'] + bar_offset
                    bar_b_global = gc['bar_b'] + bar_offset
                    theta_a = float(all_angles[bar_a_global])
                    theta_b = float(all_angles[bar_b_global])

                    # Flip inverts gear ratio (rotation direction reverses)
                    ratio = -gc['ratio'] if flipped else gc['ratio']
                    # Phase must satisfy: theta_a + ratio * theta_b - phase = 0
                    # (because potential_energy_batch negates the stored ratio)
                    new_phase = theta_a + ratio * theta_b

                    gear_constraints.append({
                        'bar_a': bar_a_global,
                        'bar_b': bar_b_global,
                        'ratio': ratio,
                        'stiffness': gc.get('stiffness', k),
                        'phase': new_phase,
                    })

                # Gear link: instance driver bar -> central driver bar
                global_driver_bar = driver_bar_local + bar_offset
                driver_angle = float(all_angles[global_driver_bar])
                central_angle = float(all_angles[0])

                # ratio: +1 if not flipped, -1 if flipped
                ratio = -1.0 if flipped else 1.0
                phase = driver_angle - ratio * central_angle

                gear_constraints.append({
                    'bar_a': global_driver_bar,
                    'bar_b': 0,
                    'ratio': ratio,
                    'stiffness': k,
                    'phase': phase,
                })

                bar_offset += n_base_bars
                fixed_offset += n_base_fixed

            # === Finalize ===
            num_bars = len(all_positions)
            bar_lengths = all_lengths
            bar_masses = all_masses
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            init_bar_positions = torch.stack(all_positions)
            init_bar_angles = torch.stack(all_angles)

            fixed_joint_positions = torch.stack(all_fixed_positions)

            joints = all_joints
            joint_stiffness = all_joint_stiffness

            system_def['gear_constraints'] = gear_constraints
            system_def['gravity'] = config["system"].get("gravity", 9.8)
            system_def['external_forces']['torque_strength'] = 0.0
            system_def['external_forces']['torque_joint_idx'] = 0
        else:
            raise ValueError(f"Unrecognized problem_name: {problem_name}")

        # ------------------------------------------------------------------ #
        # Finalise system properties
        # ------------------------------------------------------------------ #
        system.sliding_joints = [
            (sj['bar_a'], sj['pos_norm_a'], sj['bar_b'], sj['stiffness'], sj.get('angle_stiffness', 0.0))
            for sj in system_def.get('sliding_joints', [])
        ]
        system.num_bars      = num_bars
        system.bar_lengths   = torch.tensor(bar_lengths,  dtype=dtype)
        system.bar_masses    = torch.tensor(bar_masses,   dtype=dtype)
        system.bar_inertias  = torch.tensor(bar_inertias, dtype=dtype)
        system.joints        = joints
        system.joint_stiffness = torch.tensor(joint_stiffness, dtype=dtype)
        system.num_joints    = len(joints)

        # Default: all bars use line rendering
        if len(system.bar_models) == 0:
            system.bar_models = [-1] * num_bars

        system.dim            = num_bars * 3
        system_def['dim']     = system.dim

        init_state = torch.zeros(system.dim, dtype=dtype)
        for i in range(num_bars):
            init_state[i * 3:i * 3 + 2] = init_bar_positions[i]
            init_state[i * 3 + 2]        = init_bar_angles[i]

        system_def['rest_pos']        = init_state.clone()
        system_def['init_pos']        = init_state.clone()
        system_def['fixed_joint_pos'] = fixed_joint_positions.flatten()

        system_def['external_forces'].setdefault('torque_strength',  0.0)
        system_def['external_forces'].setdefault('torque_joint_idx', 0)

        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _get_bar_state_batch(self, state_flat_batch):
        batch_shape = state_flat_batch.shape[:-1]
        state     = state_flat_batch.view(*batch_shape, self.num_bars, 3)
        positions = state[..., :2]
        angles    = state[..., 2]
        return positions, angles

    def _get_joint_positions_batch(self, positions_batch, angles_batch, system_def):
        device, dtype = positions_batch.device, positions_batch.dtype
        batch_shape   = positions_batch.shape[:-2]

        bar_lengths  = self.bar_lengths.to(device=device, dtype=dtype)
        fixed_joints = system_def['fixed_joint_pos'].to(device=device, dtype=dtype).view(-1, 2)

        pos1_list, pos2_list = [], []
        for (bar1_idx, bar2_idx, pos1_norm, pos2_norm) in self.joints:
            if bar1_idx >= 0:
                c1 = positions_batch[..., bar1_idx, :]
                a1 = angles_batch[..., bar1_idx]
                l1 = bar_lengths[bar1_idx]
                off1 = torch.stack([pos1_norm * l1 * torch.cos(a1),
                                    pos1_norm * l1 * torch.sin(a1)], dim=-1)
                p1 = c1 + off1
            else:
                p1 = fixed_joints[-1 - bar1_idx].expand(*batch_shape, 2)

            c2 = positions_batch[..., bar2_idx, :]
            a2 = angles_batch[..., bar2_idx]
            l2 = bar_lengths[bar2_idx]
            off2 = torch.stack([pos2_norm * l2 * torch.cos(a2),
                                pos2_norm * l2 * torch.sin(a2)], dim=-1)
            p2 = c2 + off2

            pos1_list.append(p1)
            pos2_list.append(p2)

        return torch.stack(pos1_list, dim=-2), torch.stack(pos2_list, dim=-2)

    # ---------------------------------------------------------------------- #
    # Energy
    # ---------------------------------------------------------------------- #

    def potential_energy_batch(self, system_def, pos_flat_batch, shape=None):
        """
        Compute potential energy for a batch of configurations.

        Args:
            pos_flat_batch: (B, num_bars * 3)
            shape: unused (API compatibility)

        Returns:
            energy: (B,)
        """
        B      = pos_flat_batch.shape[0]
        device = pos_flat_batch.device
        dtype  = pos_flat_batch.dtype

        positions, angles = self._get_bar_state_batch(pos_flat_batch)
        bar_lengths = self.bar_lengths.to(device=device, dtype=dtype)

        # --- Gravity ---
        masses      = self.bar_masses.to(device=device, dtype=dtype)
        grav_energy = torch.sum(masses[None, :] * system_def['gravity'] * positions[:, :, 1], dim=1)

        # --- Regular joint spring penalties ---
        jp1, jp2      = self._get_joint_positions_batch(positions, angles, system_def)
        disp          = jp1 - jp2
        dist          = torch.sqrt((disp ** 2).sum(dim=2) + 1e-8)
        stiffness     = self.joint_stiffness.to(device=device, dtype=dtype)
        joint_energy  = 0.5 * torch.sum(stiffness[None, :] * dist ** 2, dim=1)

        # --- Y-bar constraints ---
        for c in system_def.get('y_bar_constraints', []):
            bi   = c['bar_idx']
            pn   = c['pos_norm']
            end_y = (positions[:, bi, 1]
                     + pn * bar_lengths[bi] * torch.sin(angles[:, bi]))
            joint_energy = joint_energy + 0.5 * c['stiffness'] * (end_y - c['target_y']) ** 2
        for c in system_def.get('x_bar_constraints', []):
            bi   = c['bar_idx']
            pn   = c['pos_norm']
            end_x = (positions[:, bi, 0]
                     + pn * bar_lengths[bi] * torch.cos(angles[:, bi]))
            joint_energy = joint_energy + 0.5 * c['stiffness'] * (end_x - c['target_x']) ** 2

        # --- Sliding joint penalties ---
        for (bar_a, pos_a, bar_b, k_perp, k_angle) in self.sliding_joints:
            # Point P on bar A
            ca = positions[:, bar_a, :]                              # (B, 2)
            aa = angles[:, bar_a]                                    # (B,)
            P  = ca + pos_a * bar_lengths[bar_a] * torch.stack(
                [torch.cos(aa), torch.sin(aa)], dim=-1)              # (B, 2)

            # Bar B axis: unit direction through center_b
            cb  = positions[:, bar_b, :]                             # (B, 2)
            ab  = angles[:, bar_b]                                   # (B,)
            d_b = torch.stack([torch.cos(ab), torch.sin(ab)], dim=-1)  # (B, 2)

            # Perpendicular displacement of P from bar B's axis
            v       = P - cb                                         # (B, 2)
            v_along = (v * d_b).sum(dim=-1, keepdim=True) * d_b     # (B, 2)
            v_perp  = v - v_along                                    # (B, 2)
            joint_energy = joint_energy + 0.5 * k_perp * (v_perp ** 2).sum(dim=-1)

            # Angle alignment penalty: 0 when bars are parallel / anti-parallel
            if k_angle != 0.0:
                joint_energy = joint_energy + 0.5 * k_angle * torch.sin(aa - ab) ** 2

        # In potential_energy_batch, add a gear constraints section:

        for gc in system_def.get('gear_constraints', []):
            bar_a = gc['bar_a']
            bar_b = gc['bar_b']
            ratio = -gc['ratio']
            k_gear = gc.get('stiffness', self.joint_stiffness[0])
            phase = gc.get('phase', 0.0)

            theta_a = angles[:, bar_a]
            theta_b = angles[:, bar_b]

            violation = theta_a - ratio * theta_b - phase
            # Wrap to [-π, π]
            violation = torch.remainder(violation + math.pi, 2 * math.pi) - math.pi
            joint_energy = joint_energy + 0.5 * k_gear * violation ** 2

        for rc in system_def.get('rotation_constraints', []):
            bar_idx = rc['bar_idx']
            k_rot = rc['stiffness']
            min_angle = rc.get('min_angle', None)
            max_angle = rc.get('max_angle', None)

            theta = angles[:, bar_idx]  # (B,)

            if min_angle is not None:
                violation_min = torch.clamp(min_angle - theta, min=0.0)  # > 0 when theta < min
                joint_energy = joint_energy + 0.5 * k_rot * violation_min ** 2

            if max_angle is not None:
                violation_max = torch.clamp(theta - max_angle, min=0.0)  # > 0 when theta > max
                joint_energy = joint_energy + 0.5 * k_rot * violation_max ** 2

        # --- External torque ---
        torque_energy = torch.zeros(B, device=device, dtype=dtype)
        ext           = system_def.get('external_forces', {})
        if ext.get('torque_strength', 0.0) != 0.0:
            j_idx = ext.get('torque_joint_idx', 0)
            if j_idx < self.num_joints:
                b1, b2, _, _ = self.joints[j_idx]
                rel_angle    = (angles[:, b2] - angles[:, b1]) if b1 >= 0 else angles[:, b2]
                torque_energy = -ext['torque_strength'] * rel_angle

        return grav_energy + joint_energy + torque_energy

    def potential_energy(self, system_def, pos_flat, shape=None):
        return self.potential_energy_batch(system_def, pos_flat.unsqueeze(0), shape)[0]

    # ---------------------------------------------------------------------- #
    # Cubature / gravity-split helpers
    # ---------------------------------------------------------------------- #

    def gravity_gradient(self, system_def, q):
        """Constant gravity gradient — does not depend on q.

        Returns (dim,) tensor: ∂E_grav/∂q[3i+1] = m_i * g, all others 0.
        """
        grad = torch.zeros_like(q)
        masses = self.bar_masses.to(dtype=q.dtype, device=q.device)
        grad[1::3] = masses * system_def['gravity']
        return grad

    def energy_per_joint_batch(self, system_def, pos_flat_batch):
        """Per-joint spring energy (regular joints only).

        Returns (B, n_joints) — each entry is 0.5 * k_j * dist_j^2.
        """
        device = pos_flat_batch.device
        dtype  = pos_flat_batch.dtype
        positions, angles = self._get_bar_state_batch(pos_flat_batch)
        jp1, jp2  = self._get_joint_positions_batch(positions, angles, system_def)
        disp      = jp1 - jp2
        dist      = torch.sqrt((disp ** 2).sum(dim=2) + 1e-8)
        stiffness = self.joint_stiffness.to(device=device, dtype=dtype)
        return 0.5 * stiffness[None, :] * dist ** 2  # (B, n_joints)

    def _special_energy_batch(self, system_def, pos_flat_batch):
        """Energy from all non-gravity, non-regular-joint terms.

        Includes: y_bar / x_bar constraints, sliding joints, gear constraints,
        rotation constraints, external torque.  Returns (B,).
        """
        B      = pos_flat_batch.shape[0]
        device = pos_flat_batch.device
        dtype  = pos_flat_batch.dtype

        positions, angles = self._get_bar_state_batch(pos_flat_batch)
        bar_lengths = self.bar_lengths.to(device=device, dtype=dtype)

        special = torch.zeros(B, device=device, dtype=dtype)

        for c in system_def.get('y_bar_constraints', []):
            bi    = c['bar_idx']
            pn    = c['pos_norm']
            end_y = (positions[:, bi, 1]
                     + pn * bar_lengths[bi] * torch.sin(angles[:, bi]))
            special = special + 0.5 * c['stiffness'] * (end_y - c['target_y']) ** 2
        for c in system_def.get('x_bar_constraints', []):
            bi    = c['bar_idx']
            pn    = c['pos_norm']
            end_x = (positions[:, bi, 0]
                     + pn * bar_lengths[bi] * torch.cos(angles[:, bi]))
            special = special + 0.5 * c['stiffness'] * (end_x - c['target_x']) ** 2

        for (bar_a, pos_a, bar_b, k_perp, k_angle) in self.sliding_joints:
            ca  = positions[:, bar_a, :]
            aa  = angles[:, bar_a]
            P   = ca + pos_a * bar_lengths[bar_a] * torch.stack(
                      [torch.cos(aa), torch.sin(aa)], dim=-1)
            cb  = positions[:, bar_b, :]
            ab  = angles[:, bar_b]
            d_b = torch.stack([torch.cos(ab), torch.sin(ab)], dim=-1)
            v       = P - cb
            v_along = (v * d_b).sum(dim=-1, keepdim=True) * d_b
            v_perp  = v - v_along
            special = special + 0.5 * k_perp * (v_perp ** 2).sum(dim=-1)
            if k_angle != 0.0:
                special = special + 0.5 * k_angle * torch.sin(aa - ab) ** 2

        for gc in system_def.get('gear_constraints', []):
            bar_a = gc['bar_a']
            bar_b = gc['bar_b']
            ratio = -gc['ratio']
            k_gear = gc.get('stiffness', self.joint_stiffness[0])
            phase = gc.get('phase', 0.0)
            theta_a   = angles[:, bar_a]
            theta_b   = angles[:, bar_b]
            violation = theta_a - ratio * theta_b - phase
            violation = torch.remainder(violation + math.pi, 2 * math.pi) - math.pi
            special   = special + 0.5 * k_gear * violation ** 2

        for rc in system_def.get('rotation_constraints', []):
            bar_idx   = rc['bar_idx']
            k_rot     = rc['stiffness']
            theta     = angles[:, bar_idx]
            if rc.get('min_angle') is not None:
                v = torch.clamp(rc['min_angle'] - theta, min=0.0)
                special = special + 0.5 * k_rot * v ** 2
            if rc.get('max_angle') is not None:
                v = torch.clamp(theta - rc['max_angle'], min=0.0)
                special = special + 0.5 * k_rot * v ** 2

        ext = system_def.get('external_forces', {})
        if ext.get('torque_strength', 0.0) != 0.0:
            j_idx = ext.get('torque_joint_idx', 0)
            if j_idx < self.num_joints:
                b1, b2, _, _ = self.joints[j_idx]
                rel_angle = (angles[:, b2] - angles[:, b1]) if b1 >= 0 else angles[:, b2]
                special   = special - ext['torque_strength'] * rel_angle

        return special

    def energy_springs_batch(self, system_def, pos_flat_batch):
        """All non-gravity energy (regular joints + special constraints + torque).

        Use this instead of potential_energy_batch when gravity gradient is
        handled separately (precomputed constant).  Returns (B,).
        """
        joint_e   = self.energy_per_joint_batch(system_def, pos_flat_batch).sum(dim=1)
        special_e = self._special_energy_batch(system_def, pos_flat_batch)
        return joint_e + special_e

    def potential_energy_cubature_batch(self, system_def, pos_flat_batch, cubature):
        """Cubature approximation of non-gravity energy.

        Regular joint springs are replaced by a weighted subset; special
        constraints and torque are kept exact.  Returns (B,).

        cubature: dict with keys 'joint_indices' (LongTensor, m) and
                  'joint_weights' (FloatTensor, m).
        """
        per_joint = self.energy_per_joint_batch(system_def, pos_flat_batch)  # (B, n_joints)
        idx       = cubature['joint_indices']
        w         = cubature['joint_weights'].to(dtype=pos_flat_batch.dtype,
                                                  device=pos_flat_batch.device)
        idx       = idx.to(device=pos_flat_batch.device)
        cub_e     = (per_joint[:, idx] * w[None, :]).sum(dim=1)
        special_e = self._special_energy_batch(system_def, pos_flat_batch)
        return cub_e + special_e

    def prepare_cubature_sparse(self, system_def, cubature):
        """Precompute sparse cubature metadata for dimension-free Newton solves.

        Identifies the minimal set of bar DOFs touched by the cubature joints
        plus all special constraints (torque, gear, etc.), so the Newton loop
        can differentiate only through those DOFs rather than the full q.

        Returns a dict to be stored in system_def['cubature_sparse'].
        """
        device = self.bar_lengths.device
        dtype  = self.bar_lengths.dtype

        joint_indices = cubature['joint_indices'].tolist()
        joint_weights = cubature['joint_weights'].tolist()

        # ── Collect active bars ──────────────────────────────────────────────
        active_bar_set = set()
        for j in joint_indices:
            b1, b2, _, _ = self.joints[j]
            if int(b1) >= 0:
                active_bar_set.add(int(b1))
            active_bar_set.add(int(b2))

        # Special constraints
        for c in system_def.get('y_bar_constraints', []):
            active_bar_set.add(int(c['bar_idx']))
        for c in system_def.get('x_bar_constraints', []):
            active_bar_set.add(int(c['bar_idx']))
        for (bar_a, _, bar_b, _, _) in self.sliding_joints:
            active_bar_set.add(int(bar_a)); active_bar_set.add(int(bar_b))
        for gc in system_def.get('gear_constraints', []):
            active_bar_set.add(int(gc['bar_a'])); active_bar_set.add(int(gc['bar_b']))
        for rc in system_def.get('rotation_constraints', []):
            active_bar_set.add(int(rc['bar_idx']))
        ext = system_def.get('external_forces', {})
        if ext.get('torque_strength', 0.0) != 0.0:
            j_idx = ext.get('torque_joint_idx', 0)
            if j_idx < self.num_joints:
                b1t, b2t, _, _ = self.joints[j_idx]
                if int(b1t) >= 0:
                    active_bar_set.add(int(b1t))
                active_bar_set.add(int(b2t))

        active_bars  = sorted(active_bar_set)
        bar_to_local = {b: i for i, b in enumerate(active_bars)}
        n_active     = len(active_bars)

        dof_indices = torch.tensor(
            [3 * b + d for b in active_bars for d in range(3)],
            dtype=torch.long, device=device,
        )
        active_bar_lengths = self.bar_lengths[active_bars].to(device=device, dtype=dtype)

        # ── Cubature joint specs ─────────────────────────────────────────────
        fixed_joint_pos = (system_def['fixed_joint_pos']
                           .to(device=device, dtype=dtype).view(-1, 2))
        joint_specs = []
        for ji, j in enumerate(joint_indices):
            b1g, b2g, t1, t2 = self.joints[j]
            b1g, b2g = int(b1g), int(b2g)
            spec = {
                'b2_local': bar_to_local[b2g],
                'b2_t':     float(t2),
                'b2_len':   float(self.bar_lengths[b2g]),
                'stiffness': float(self.joint_stiffness[j]),
                'weight':    float(joint_weights[ji]),
            }
            if b1g >= 0:
                spec['b1_type']  = 'bar'
                spec['b1_local'] = bar_to_local[b1g]
                spec['b1_t']     = float(t1)
                spec['b1_len']   = float(self.bar_lengths[b1g])
            else:
                spec['b1_type']      = 'fixed'
                spec['b1_fixed_pos'] = fixed_joint_pos[-1 - b1g].clone()
            joint_specs.append(spec)

        # ── Torque spec — always store bar indices; strength is read live ────────
        torque_spec = None
        j_idx = ext.get('torque_joint_idx', None)
        if j_idx is not None and j_idx < self.num_joints:
            b1t, b2t, _, _ = self.joints[j_idx]
            b1t, b2t = int(b1t), int(b2t)
            torque_spec = {
                'b1_local': bar_to_local[b1t] if b1t >= 0 else None,
                'b2_local': bar_to_local[b2t],
            }

        # ── Special constraint specs (y_bar, x_bar, sliding, gear, rotation) ─
        special = {
            'y_bar':    [(c, bar_to_local[int(c['bar_idx'])])
                         for c in system_def.get('y_bar_constraints', [])
                         if int(c['bar_idx']) in bar_to_local],
            'x_bar':    [(c, bar_to_local[int(c['bar_idx'])])
                         for c in system_def.get('x_bar_constraints', [])
                         if int(c['bar_idx']) in bar_to_local],
            'sliding':  [(int(bar_a), pos_a, int(bar_b), k_perp, k_ang,
                          bar_to_local[int(bar_a)], bar_to_local[int(bar_b)])
                         for (bar_a, pos_a, bar_b, k_perp, k_ang) in self.sliding_joints
                         if int(bar_a) in bar_to_local and int(bar_b) in bar_to_local],
            'gear':     [(gc, bar_to_local[int(gc['bar_a'])], bar_to_local[int(gc['bar_b'])])
                         for gc in system_def.get('gear_constraints', [])
                         if int(gc['bar_a']) in bar_to_local and int(gc['bar_b']) in bar_to_local],
            'rotation': [(rc, bar_to_local[int(rc['bar_idx'])])
                         for rc in system_def.get('rotation_constraints', [])
                         if int(rc['bar_idx']) in bar_to_local],
        }

        n_total = self.num_bars
        print(f"Cubature sparse: {n_active}/{n_total} active bars, "
              f"{len(dof_indices)}/{n_total * 3} active DOFs")

        # ── Vectorised joint data (for batched autograd-friendly evaluation) ───
        # Split joints into bar-bar and bar-fixed groups; skip zero-weight joints.
        # Precomputed here once so energy_cubature_sparse has O(1) graph nodes
        # regardless of joint count.
        bb_b1, bb_b2, bb_t1l1, bb_t2l2, bb_ws = [], [], [], [], []
        bf_b2, bf_t2l2, bf_p1_list, bf_ws      = [], [], [], []
        for sp in joint_specs:
            if sp['weight'] == 0.0:
                continue
            ws = sp['weight'] * 0.5 * sp['stiffness']
            if sp['b1_type'] == 'bar':
                bb_b1.append(sp['b1_local'])
                bb_b2.append(sp['b2_local'])
                bb_t1l1.append(sp['b1_t'] * sp['b1_len'])
                bb_t2l2.append(sp['b2_t'] * sp['b2_len'])
                bb_ws.append(ws)
            else:
                bf_b2.append(sp['b2_local'])
                bf_t2l2.append(sp['b2_t'] * sp['b2_len'])
                bf_p1_list.append(sp['b1_fixed_pos'])
                bf_ws.append(ws)

        def _lt(lst): return torch.tensor(lst, dtype=torch.long,  device=device)
        def _ft(lst): return torch.tensor(lst, dtype=dtype,        device=device)

        vec = {}
        if bb_b1:
            vec['bb_b1']   = _lt(bb_b1)
            vec['bb_b2']   = _lt(bb_b2)
            vec['bb_t1l1'] = _ft(bb_t1l1)
            vec['bb_t2l2'] = _ft(bb_t2l2)
            vec['bb_ws']   = _ft(bb_ws)
        if bf_b2:
            vec['bf_b2']   = _lt(bf_b2)
            vec['bf_t2l2'] = _ft(bf_t2l2)
            vec['bf_p1']   = torch.stack(bf_p1_list).to(device=device, dtype=dtype)
            vec['bf_ws']   = _ft(bf_ws)
        if special['y_bar']:
            vec['yb_loc']    = _lt([loc for _, loc in special['y_bar']])
            vec['yb_pnorm']  = _ft([c['pos_norm']  for c, _ in special['y_bar']])
            vec['yb_stiff']  = _ft([c['stiffness'] for c, _ in special['y_bar']])
            vec['yb_target'] = _ft([c['target_y']  for c, _ in special['y_bar']])
        if special['x_bar']:
            vec['xb_loc']    = _lt([loc for _, loc in special['x_bar']])
            vec['xb_pnorm']  = _ft([c['pos_norm']  for c, _ in special['x_bar']])
            vec['xb_stiff']  = _ft([c['stiffness'] for c, _ in special['x_bar']])
            vec['xb_target'] = _ft([c['target_x']  for c, _ in special['x_bar']])
        if special['gear']:
            vec['gear_la']    = _lt([la for _, la, _  in special['gear']])
            vec['gear_lb']    = _lt([lb for _, _,  lb in special['gear']])
            vec['gear_ratio'] = _ft([-gc['ratio']                              for gc, _, _ in special['gear']])
            vec['gear_kg']    = _ft([gc.get('stiffness', float(self.joint_stiffness[0])) for gc, _, _ in special['gear']])
            vec['gear_phase'] = _ft([gc.get('phase', 0.0)                      for gc, _, _ in special['gear']])

        return {
            'n_active':           n_active,
            'active_bars':        active_bars,
            'bar_to_local':       bar_to_local,
            'dof_indices':        dof_indices,
            'active_bar_lengths': active_bar_lengths,
            'joint_specs':        joint_specs,
            'torque_spec':        torque_spec,
            'special':            special,
            'vec':                vec,
        }

    def energy_cubature_sparse(self, system_def, q_active, sparse_info):
        """All non-gravity energy using only active-bar DOFs — no full-space tensors.

        q_active: (n_active * 3,) — active bar DOFs only (x, y, theta per bar).
        Returns scalar energy.

        Uses precomputed index/weight tensors from sparse_info['vec'] so that all
        joint springs are evaluated in O(1) graph nodes (batched) rather than
        O(n_joints) chained scalar nodes.
        """
        device = q_active.device
        dtype  = q_active.dtype
        n      = sparse_info['n_active']
        vec    = sparse_info['vec']
        lens   = sparse_info['active_bar_lengths'].to(device=device, dtype=dtype)

        st  = q_active.view(n, 3)
        pos = st[:, :2]   # (n, 2)
        ang = st[:, 2]    # (n,)

        E = q_active.new_zeros(())

        # ── Bar-to-bar cubature joints (batched) ─────────────────────────────
        # One IndexBackward + one cos/sin + one stack per group, regardless of
        # joint count.  Also skips sqrt since dist² = ||p1-p2||² directly.
        if 'bb_b1' in vec:
            a1 = ang[vec['bb_b1'].to(device)]          # (m,)
            a2 = ang[vec['bb_b2'].to(device)]          # (m,)
            tl1 = vec['bb_t1l1'].to(device=device, dtype=dtype)
            tl2 = vec['bb_t2l2'].to(device=device, dtype=dtype)
            p1 = pos[vec['bb_b1'].to(device)] + tl1[:, None] * torch.stack([torch.cos(a1), torch.sin(a1)], dim=1)
            p2 = pos[vec['bb_b2'].to(device)] + tl2[:, None] * torch.stack([torch.cos(a2), torch.sin(a2)], dim=1)
            dist2 = ((p1 - p2) ** 2).sum(dim=1) + 1e-8
            E = E + (vec['bb_ws'].to(device=device, dtype=dtype) * dist2).sum()

        # ── Bar-to-fixed cubature joints (batched) ───────────────────────────
        if 'bf_b2' in vec:
            a2  = ang[vec['bf_b2'].to(device)]
            tl2 = vec['bf_t2l2'].to(device=device, dtype=dtype)
            p2  = pos[vec['bf_b2'].to(device)] + tl2[:, None] * torch.stack([torch.cos(a2), torch.sin(a2)], dim=1)
            p1  = vec['bf_p1'].to(device=device, dtype=dtype)
            dist2 = ((p1 - p2) ** 2).sum(dim=1) + 1e-8
            E = E + (vec['bf_ws'].to(device=device, dtype=dtype) * dist2).sum()

        # ── Torque ───────────────────────────────────────────────────────────
        ts = sparse_info['torque_spec']
        if ts is not None:
            strength = system_def.get('external_forces', {}).get('torque_strength', 0.0)
            if strength != 0.0:
                a2 = ang[ts['b2_local']]
                rel = a2 - ang[ts['b1_local']] if ts['b1_local'] is not None else a2
                E = E - strength * rel

        # ── Y-bar constraints (batched) ──────────────────────────────────────
        if 'yb_loc' in vec:
            loc  = vec['yb_loc'].to(device)
            a    = ang[loc]
            end_y = pos[loc, 1] + vec['yb_pnorm'].to(device=device, dtype=dtype) * lens[loc] * torch.sin(a)
            E = E + (0.5 * vec['yb_stiff'].to(device=device, dtype=dtype) *
                     (end_y - vec['yb_target'].to(device=device, dtype=dtype)) ** 2).sum()

        # ── X-bar constraints (batched) ──────────────────────────────────────
        if 'xb_loc' in vec:
            loc  = vec['xb_loc'].to(device)
            a    = ang[loc]
            end_x = pos[loc, 0] + vec['xb_pnorm'].to(device=device, dtype=dtype) * lens[loc] * torch.cos(a)
            E = E + (0.5 * vec['xb_stiff'].to(device=device, dtype=dtype) *
                     (end_x - vec['xb_target'].to(device=device, dtype=dtype)) ** 2).sum()

        # ── Sliding joints (kept as loop — small count, complex projection) ──
        for (_ba, pos_a, _bb, k_perp, k_ang, la, lb) in sparse_info['special']['sliding']:
            ca, aa = pos[la], ang[la]
            P   = ca + pos_a * lens[la] * torch.stack([torch.cos(aa), torch.sin(aa)])
            cb, ab = pos[lb], ang[lb]
            d_b    = torch.stack([torch.cos(ab), torch.sin(ab)])
            v      = P - cb
            v_perp = v - torch.dot(v, d_b) * d_b
            E = E + 0.5 * k_perp * (v_perp ** 2).sum()
            if k_ang != 0.0:
                E = E + 0.5 * k_ang * torch.sin(aa - ab) ** 2

        # ── Gear constraints (batched) ───────────────────────────────────────
        if 'gear_la' in vec:
            la    = vec['gear_la'].to(device)
            lb    = vec['gear_lb'].to(device)
            ratio = vec['gear_ratio'].to(device=device, dtype=dtype)
            kg    = vec['gear_kg'].to(device=device, dtype=dtype)
            phase = vec['gear_phase'].to(device=device, dtype=dtype)
            viol  = ang[la] - ratio * ang[lb] - phase
            viol  = torch.remainder(viol + math.pi, 2 * math.pi) - math.pi
            E = E + (0.5 * kg * viol ** 2).sum()

        # ── Rotation constraints (kept as loop — per-constraint clamp logic) ─
        for rc, loc in sparse_info['special']['rotation']:
            theta, k_r = ang[loc], rc['stiffness']
            if rc.get('min_angle') is not None:
                E = E + 0.5 * k_r * torch.clamp(rc['min_angle'] - theta, min=0.0) ** 2
            if rc.get('max_angle') is not None:
                E = E + 0.5 * k_r * torch.clamp(theta - rc['max_angle'], min=0.0) ** 2

        return E

    def kinetic_energy_batch(self, system_def, pos_flat_batch, vel_flat_batch, shape=None):
        B      = vel_flat_batch.shape[0]
        device = vel_flat_batch.device
        dtype  = vel_flat_batch.dtype

        vel_state   = vel_flat_batch.view(B, self.num_bars, 3)
        lin_vel     = vel_state[:, :, :2]
        ang_vel     = vel_state[:, :, 2]

        masses   = self.bar_masses.to(device=device, dtype=dtype)
        inertias = self.bar_inertias.to(device=device, dtype=dtype)

        trans_ke = 0.5 * torch.sum(masses[None, :] * (lin_vel ** 2).sum(dim=2), dim=1)
        rot_ke   = 0.5 * torch.sum(inertias[None, :] * ang_vel ** 2, dim=1)
        return trans_ke + rot_ke

    def kinetic_energy(self, system_def, pos_flat, vel_flat, shape=None):
        return self.kinetic_energy_batch(
            system_def, pos_flat.unsqueeze(0), vel_flat.unsqueeze(0), shape)[0]

    def physical_mass_matrix(self, system_def, q):
        if not hasattr(self, '_M_diag_cache'):
            masses, inertias = self.bar_masses, self.bar_inertias
            M_diag = []
            for i in range(self.num_bars):
                M_diag.extend([masses[i], masses[i], inertias[i]])
            self._M_diag_cache = torch.tensor(M_diag)
        diag = self._M_diag_cache.to(dtype=q.dtype, device=q.device)
        return torch.diag(diag)

    # ---------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------- #

    def _apply_bar_transform(self, vertices_local, center, angle, length):
        """
        Transform (N,3) local-frame vertices to world space.
        Local frame: ±0.5 in X = bar endpoints; scaled by length.
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Build 2D rotation + scale matrix
        # world_xy = R * (scale * local_xy) + center
        scale = length
        R = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]])
        verts = vertices_local.copy()
        verts[:, 0] *= scale          # scale X (along bar) by bar length
        # Y/Z are not scaled — they stay in physical units
        verts_world = verts.copy()
        verts_world[:, :2] = (R @ verts[:, :2].T).T + center
        return verts_world

    def visualize(
            self,
            system_def,
            pos_flat,
            shape=None,
            return_transforms=False,
            name_prefix="",
            offset=None,
            main_color=(0.2, 0.5, 0.8)
    ):
        """
        Visualize the bar-joint system.

        Args:
            name_prefix: string added to all polyscope object names
            offset: optional (3,) translation applied to visualization
            main_color: RGB tuple for bar rendering
        """
        device, dtype = pos_flat.device, pos_flat.dtype

        if offset is None:
            offset = np.zeros(3)

        positions, angles = self._get_bar_state_batch(pos_flat.unsqueeze(0))
        positions = positions[0]
        angles = angles[0]

        fixed_joints = system_def['fixed_joint_pos'].to(device=device, dtype=dtype).view(-1, 2)
        positions_np = positions.detach().cpu().numpy()
        angles_np = angles.detach().cpu().numpy()
        bar_lengths_np = self.bar_lengths.detach().cpu().numpy()
        fixed_joints_np = fixed_joints.detach().cpu().numpy()

        # if return_transforms:
        #     body_data = []
        #     for i in range(self.num_bars):
        #         body_data.append({
        #             'name': f"bar{i}",
        #             'position': positions_np[i],
        #             'angle': float(angles_np[i]),
        #             'length': float(bar_lengths_np[i]),
        #             'mass': float(self.bar_masses[i]),
        #             'mesh_id': self.bar_models[i],
        #         })
        #     return body_data

        # ---- bars ----
        bar_nodes = []
        bar_edges = []
        node_idx = 0

        for i in range(self.num_bars):
            center = positions_np[i]
            angle = float(angles_np[i])
            length = float(bar_lengths_np[i])
            mesh_id = self.bar_models[i]
            meshes = self.model_meshes
            use_mesh = (mesh_id >= 0 and meshes is not None and mesh_id < len(meshes))
            if use_mesh:
                mesh = meshes[mesh_id]
                verts = np.array(mesh['vertices'], dtype=np.float64).copy()
                faces = np.array(mesh['faces'], dtype=np.int32)

                # Scale along X to bar length (mesh authored at unit length)
                # verts[:, 0] *= length

                # Rotate in XY by bar angle
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]], dtype=np.float64)
                verts = (R @ verts.T).T

                # Translate to bar center + offset
                verts[:, 0] += center[0]
                verts[:, 1] += center[1]
                verts += offset

                ps_mesh = ps.register_surface_mesh(f"{name_prefix}bar_mesh_{i}", verts, faces)
                # if (name_prefix): print(name_prefix)
                ps_mesh.set_color(main_color)
            else:
                half = length / 2.0
                dx = half * np.cos(angle)
                dy = half * np.sin(angle)
                p1 = np.array([center[0] - dx, center[1] - dy, 0.0]) + offset
                p2 = np.array([center[0] + dx, center[1] + dy, 0.0]) + offset
                bar_nodes.append(p1)
                bar_nodes.append(p2)
                bar_edges.append([node_idx, node_idx + 1])
                node_idx += 2

        if bar_nodes:
            ps_bars = ps.register_curve_network(
                f"{name_prefix}bars", np.array(bar_nodes), np.array(bar_edges))
            ps_bars.set_radius(0.03, relative=False)
            ps_bars.set_color(main_color)

        # ---- joints ----
        joint_pos1, joint_pos2 = self._get_joint_positions_batch(
            positions.unsqueeze(0), angles.unsqueeze(0), system_def)
        joint_pos1 = joint_pos1[0].detach().cpu().numpy()
        joint_pos2 = joint_pos2[0].detach().cpu().numpy()

        joint_points = []
        for i in range(self.num_joints):
            joint_points.append(np.array([joint_pos1[i, 0], joint_pos1[i, 1], 0.0]) + offset)
            joint_points.append(np.array([joint_pos2[i, 0], joint_pos2[i, 1], 0.0]) + offset)

        ext = system_def.get('external_forces', {})
        torque_idx = ext.get('torque_joint_idx', None)
        torque_strength = ext.get('torque_strength', 0.0)

        if torque_strength != 0.0 and torque_idx is not None and torque_idx < len(joint_points):
            # Create a temporary point cloud with only the torqued joint
            ps_torque = ps.register_point_cloud(f"{name_prefix}torque_joint", np.array([joint_points[torque_idx]]))
            ps_torque.set_radius(0.06, relative=False)  # bigger radius
            ps_torque.set_color((1.0, 0.8, 0.0))  # highlight color


        if joint_points:
            ps_joints = ps.register_point_cloud(f"{name_prefix}joint_points", np.array(joint_points))
            ps_joints.set_radius(0.02, relative=False)
            ps_joints.set_color((0.9, 0.3, 0.3))

        # ---- fixed joints ----
        if len(fixed_joints_np) > 0:
            fixed_3d = np.zeros((len(fixed_joints_np), 3))
            fixed_3d[:, :2] = fixed_joints_np
            fixed_3d += offset
            ps_fixed = ps.register_point_cloud(f"{name_prefix}fixed_joints", fixed_3d)
            ps_fixed.set_radius(0.04, relative=False)
            ps_fixed.set_color((0.2, 0.2, 0.2))

        # ---- joint constraint springs ----
        sn, se, si = [], [], 0
        for i in range(self.num_joints):
            sn.append(np.array([joint_pos1[i, 0], joint_pos1[i, 1], 0.0]) + offset)
            sn.append(np.array([joint_pos2[i, 0], joint_pos2[i, 1], 0.0]) + offset)
            se.append([si, si + 1])
            si += 2
        if sn:
            cn = ps.register_curve_network(
                f"{name_prefix}joint_constraints", np.array(sn), np.array(se))
            cn.set_radius(0.008, relative=False)
            cn.set_color((0.9, 0.7, 0.2))

        # ---- sliding joints ----
        for s_idx, sj in enumerate(system_def.get('sliding_joints', [])):
            a = sj['bar_a']
            pos_norm_a = sj['pos_norm_a']
            b = sj['bar_b']

            ca = positions_np[a]
            ang_a = float(angles_np[a])
            la = float(bar_lengths_np[a])
            P = np.array([
                ca[0] + pos_norm_a * la * np.cos(ang_a),
                ca[1] + pos_norm_a * la * np.sin(ang_a),
                0.0
            ]) + offset

            cb = positions_np[b]
            ang_b = float(angles_np[b])
            d_b = np.array([np.cos(ang_b), np.sin(ang_b)])
            v = (P - offset)[:2] - cb
            t = np.dot(v, d_b)
            foot = np.array([cb[0] + t * d_b[0], cb[1] + t * d_b[1], 0.0]) + offset

            gap = ps.register_curve_network(
                f"{name_prefix}sliding_gap_{s_idx}", np.array([P, foot]), np.array([[0, 1]]))
            gap.set_radius(0.006, relative=False)
            gap.set_color((1.0, 0.8, 0.0))

            pt = ps.register_point_cloud(f"{name_prefix}sliding_pt_{s_idx}", P[np.newaxis])
            pt.set_radius(0.025, relative=False)
            pt.set_color((1.0, 0.5, 0.0))

    def build_system_ui(self, system_def):
        if psim.TreeNode("Bar-Joint System UI"):
            psim.TextUnformatted(f"Bars: {self.num_bars}  |  "
                                 f"Joints: {self.num_joints}  |  "
                                 f"Sliding joints: {len(self.sliding_joints)}  |  "
                                 f"Meshes: {len(self.model_meshes)}")

            _, new_g = psim.SliderFloat("Gravity", float(system_def['gravity']), 0.0, 20.0)
            system_def['gravity'] = float(new_g)

            if "torque_strength" in system_def["external_forces"]:
                v = system_def['external_forces']['torque_strength']

                slider_val = int(math.copysign(math.log2(abs(v)) if v != 0 else 0, v))

                _, new_t = psim.SliderInt(
                    "External Torque",
                    slider_val,
                    -15, 15
                )

                system_def['external_forces']['torque_strength'] = (
                    0 if new_t == 0 else math.copysign(2 ** abs(new_t), new_t)
                )
                _, new_j = psim.InputInt(
                    "Torque Joint Index",
                    int(system_def['external_forces']['torque_joint_idx'])
                )
                system_def['external_forces']['torque_joint_idx'] = max(
                    0, min(self.num_joints - 1, new_j)
                )

            if psim.TreeNode("Joint Stiffness"):
                for i in range(self.num_joints):
                    _, new_k = psim.SliderFloat(
                        f"Joint {i}", float(self.joint_stiffness[i]), 0.0, 500.0
                    )
                    self.joint_stiffness[i] = float(new_k)
                psim.TreePop()

            if self.sliding_joints and psim.TreeNode("Sliding Joints"):
                for s_idx, sj in enumerate(self.sliding_joints):
                    psim.TextUnformatted(
                        f"[{s_idx}] bar{int(sj[0])}@{sj[1]:.2f} slides on bar{int(sj[2])}  "
                        f"k_perp={sj[3]:.1f}  k_angle={sj[4]:.1f}"
                    )
                psim.TreePop()

            if self.model_meshes and psim.TreeNode("Bar Models"):
                for i in range(self.num_bars):

                    obj = self.bar_models[i] if i < len(self.bar_models) else -1
                    label = f"bar{i}: {'line' if obj == -1 else f'mesh #{obj}'}"
                    psim.TextUnformatted(label)
                psim.TreePop()

            psim.TreePop()

    def visualize_set_nice_view(self, system_def, x):
        # Bigger view (camera farther back)
        ps.look_at((0.0, 3.0, 12.0), (0.0, 1.5, 0.0))
        ps.set_vertical_fov_degrees(130)
        # Switch to orthographic projection
        ps.set_view_projection_mode("orthographic")

    def export(self, system_def, x, prefix=""):
        pass