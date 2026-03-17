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

            system.model_meshes = [
                load_mesh("objects/scissor_lift.obj"),
                load_mesh("objects/scissor_lift_top.obj"),
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
                    joints.append([-1, a, 0.0, -0.5]); joint_stiffness.append(k)
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

            top_cx = (top_right_x - 1)
            top_cy = (top_right_y + top_left_y) / 2.0

            num_bars = 2 * n_stages + 1
            bar_lengths = [L] * (2 * n_stages + 1)
            bar_masses = [bar_mass] * num_bars
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
            joints.append([last_b, top_bar_idx, 0.5, 0.5])
            joint_stiffness.append(k)

            # Sliding joint: left end of platform slides along last bar_a's axis
            # bar_a is the right-side bar (angle=2pi/3), so the platform left end
            # can slide up/down along it as the scissor extends
            if 'sliding_joints' not in system_def:
                system_def['sliding_joints'] = []
            system_def['sliding_joints'].append({
                'bar_a': last_a,
                'pos_norm_a': 0.5,  # left end of platform
                'bar_b': top_bar_idx,  # slides along last bar_a's axis
                'stiffness': k,
                'angle_stiffness': 0.0,  # platform stays horizontal, no angle coupling
            })

            system_def['y_bar_constraints'] = [{
                'bar_idx': 1, 'pos_norm': -0.5, 'target_y': 0.0, 'stiffness': k,
            }]
            system_def['gravity'] = 0.0
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
            ratio = -gc['ratio']  # r_a / r_b  — e.g. 2.0 means bar_a turns half as fast
            k_gear = gc['stiffness']

            theta_a = angles[:, bar_a]  # (B,)
            theta_b = angles[:, bar_b]  # (B,)

            # Violation: θ_a - ratio * θ_b should be constant (we penalize deviation from initial)
            # Simpler: penalize d/dt violation by penalizing θ_a * r_b - θ_b * r_a = 0
            # i.e. enforce θ_a = ratio * θ_b up to a phase offset
            violation = theta_a - ratio * theta_b
            joint_energy = joint_energy + 0.5 * k_gear * violation ** 2

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
        masses, inertias = self.bar_masses, self.bar_inertias
        M_diag = []
        for i in range(self.num_bars):
            M_diag.extend([masses[i], masses[i], inertias[i]])
        return torch.diag(torch.tensor(M_diag, dtype=q.dtype, device=q.device))

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

        if joint_points:
            ps_joints = ps.register_point_cloud(
                f"{name_prefix}joint_points", np.array(joint_points))
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
        ps.look_at((0., 1.5, 5.0), (0., 1.5, 0.))

    def export(self, system_def, x, prefix=""):
        pass