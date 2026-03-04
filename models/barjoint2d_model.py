import math

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class BarJoint2DSystem:
    """
    2D bar-joint system where rigid bars are connected by rotational joints.
    Each bar has 3 DOF (x, y position of center, rotation angle).
    Joints can be placed anywhere along bars and have spring penalties.
    """

    @staticmethod
    def construct(problem_name, config, dtype=torch.float64):
        system_def = {}
        system = BarJoint2DSystem()

        system.system_name = "BarJoint2D"
        system.problem_name = str(problem_name)

        # Defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,), dtype=dtype)
        system.cond_dim = 0

        if problem_name == 'single':
            # Single bar with one end fixed (pendulum)
            num_bars = 1
            bar_lengths = [1.0]
            bar_masses = [1.0]
            bar_inertias = [1.0 / 12.0]  # I = mL^2/12 for uniform bar

            # Joints: [bar1_idx, bar2_idx, position_on_bar1, position_on_bar2]
            # -1 for bar index means fixed point in space
            # position is normalized: -0.5 = left end, 0.0 = center, 0.5 = right end
            joints = [
                [-1, 0, 0.0, -0.5]  # Fixed point to left end of bar 0
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]]

            fixed_joint_positions = torch.tensor([[0.0, 2.0]], dtype=dtype)
            init_bar_positions = torch.tensor([[0.5, 2.0]], dtype=dtype)
            init_bar_angles = torch.tensor([0.0], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'double':
            # Double pendulum (two bars)
            num_bars = 2
            bar_lengths = [1.0, 0.8]
            bar_masses = [1.0, 0.8]
            bar_inertias = [1.0 / 12.0, 0.64 / 12.0]

            joints = [
                [-1, 0, 0.0, -0.5],  # Fixed point to left end of bar 0
                [0, 1, 0.5, -0.5]  # Right end of bar 0 to left end of bar 1
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * 2

            fixed_joint_positions = torch.tensor([[0.0, 2.5]], dtype=dtype)
            init_bar_positions = torch.tensor([
                [0.0, 2.0],
                [0.0, 1.1]
            ], dtype=dtype)
            init_bar_angles = torch.tensor([0.2, -0.3], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'chain':
            # Chain of bars
            num_bars = config["system"]["num_bars"]
            bar_length = config["system"]["bar_length"]
            bar_mass = config["system"]["bar_mass"]

            bar_lengths = [bar_length] * num_bars
            bar_masses = [bar_mass] * num_bars
            bar_inertias = [bar_mass * bar_length ** 2 / 12.0] * num_bars

            # Connect bars in series
            joints = [[-1, 0, 0.0, -0.5]]  # Fix first bar
            joint_stiffness = [config["system"]["joint_stiffness"]]

            for i in range(num_bars - 1):
                joints.append([i, i + 1, 0.5, -0.5])
                joint_stiffness.append(config["system"]["joint_stiffness"])

            fixed_joint_positions = torch.tensor([[0.0, 3.0]], dtype=dtype)

            # Initialize bars vertically
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
            # Bridge-like structure
            num_bars = 7
            bar_lengths = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.0]  # Diagonals longer
            bar_masses = [0.8] * num_bars
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            # Create a truss bridge pattern
            # Bars: 0-3 horizontal, 4-5 diagonal supports, 6 center
            joints = [
                [-1, 0, 0.0, -0.5],  # Left support
                [-2, 3, 0.0, 0.5],  # Right support
                [0, 1, 0.5, -0.5],  # Connect horizontal bars
                [1, 2, 0.5, -0.5],
                [2, 3, 0.5, -0.5],
                [0, 4, 0.0, -0.5],  # Left diagonal
                [4, 1, 0.5, 0.0],  # Diagonal to middle
                [2, 5, 0.0, -0.5],  # Right diagonal
                [5, 3, 0.5, 0.0],
                [1, 6, 0.0, -0.5],  # Center vertical
                [6, 2, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([
                [0.0, 2.0],  # Left support
                [3.0, 2.0]  # Right support
            ], dtype=dtype)

            # Initialize in bridge configuration
            init_bar_positions = torch.tensor([
                [0.5, 2.0],  # Bar 0: left horizontal
                [1.5, 2.0],  # Bar 1: center-left horizontal
                [2.5, 2.0],  # Bar 2: center-right horizontal
                [3.5, 2.0],  # Bar 3: right horizontal
                [0.85, 1.3],  # Bar 4: left diagonal
                [2.15, 1.3],  # Bar 5: right diagonal
                [2.0, 1.5],  # Bar 6: center vertical
            ], dtype=dtype)
            init_bar_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.785, -0.785, 1.57], dtype=dtype)

            system_def['gravity'] = 9.8

        elif problem_name == 'linkage':
            num_bars = 3
            bar_lengths = [2, 1, 1]
            bar_masses = [2, 1, 1]
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            joints = [
                [-1, 1, 0.0, -0.5],
                [-2, 2, 0.0, -0.5],
                [0, 1, -0.5, 0.5],
                [0, 2, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([
                [0.0, 0],
                [2.0, 0]
            ], dtype=dtype)

            init_bar_positions = torch.tensor([
                [1, 1],
                [0, 0.5],
                [2, 0.5],
            ], dtype=dtype)
            init_bar_angles = torch.tensor([0.0, math.pi/2, math.pi/2], dtype=dtype)

            system_def['gravity'] = 9.8
        elif problem_name == 'stack':
            num_bars = 6
            bar_lengths = [2, 1, 1, 2, 1, 1]
            bar_masses = [2, 1, 1, 2, 1, 1]
            bar_inertias = [m * L ** 2 / 12.0 for m, L in zip(bar_masses, bar_lengths)]

            joints = [
                [-1, 1, 0.0, -0.5],
                [-2, 2, 0.0, -0.5],
                [0, 1, -0.5, 0.5],
                [0, 2, 0.5, 0.5],

                [0, 4, -0.5, -0.5],
                [0, 5, 0.5, -0.5],
                [3, 4, -0.5, 0.5],
                [3, 5, 0.5, 0.5],
            ]
            joint_stiffness = [config["system"]["joint_stiffness"]] * len(joints)

            fixed_joint_positions = torch.tensor([
                [0.0, 0],
                [2.0, 0]
            ], dtype=dtype)

            init_bar_positions = torch.tensor([
                [1, 1],
                [0, 0.5],
                [2, 0.5],
                [1, 2],
                [0, 1.5],
                [2, 1.5],
            ], dtype=dtype)
            init_bar_angles = torch.tensor([0.0, math.pi/2, math.pi/2, 0.0, math.pi/2, math.pi/2], dtype=dtype)

            system_def['gravity'] = 9.8
        else:
            raise ValueError(f"Unrecognized problem_name: {problem_name}")

        # Store system properties
        system.num_bars = num_bars
        system.bar_lengths = torch.tensor(bar_lengths, dtype=dtype)
        system.bar_masses = torch.tensor(bar_masses, dtype=dtype)
        system.bar_inertias = torch.tensor(bar_inertias, dtype=dtype)

        system.joints = joints  # List of [bar1_idx, bar2_idx, pos1, pos2]
        system.joint_stiffness = torch.tensor(joint_stiffness, dtype=dtype)
        system.num_joints = len(joints)

        # DOF: 3 per bar (x, y, theta)
        system.dim = num_bars * 3
        system_def['dim'] = system.dim

        # Flatten state: [x1, y1, theta1, x2, y2, theta2, ...]
        init_state = torch.zeros(system.dim, dtype=dtype)
        for i in range(num_bars):
            init_state[i * 3:i * 3 + 2] = init_bar_positions[i]
            init_state[i * 3 + 2] = init_bar_angles[i]

        system_def['rest_pos'] = init_state.clone()
        system_def['init_pos'] = init_state.clone()
        system_def['fixed_joint_pos'] = fixed_joint_positions.flatten()

        # External forces
        system_def['external_forces']['torque_strength'] = 0.0
        system_def['external_forces']['torque_joint_idx'] = 0

        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    def _get_bar_state_batch(self, state_flat_batch):
        """
        Extract bar positions and angles from flattened state.

        Args:
            state_flat_batch: (..., num_bars * 3)

        Returns:
            positions: (..., num_bars, 2) - center positions
            angles: (..., num_bars) - rotation angles
        """
        batch_shape = state_flat_batch.shape[:-1]
        state = state_flat_batch.view(*batch_shape, self.num_bars, 3)
        positions = state[..., :2]  # (..., num_bars, 2)
        angles = state[..., 2]  # (..., num_bars)
        return positions, angles

    def _get_joint_positions_batch(self, positions_batch, angles_batch, system_def):
        """
        Compute actual positions of all joints in world space.

        Args:
            positions_batch: (..., num_bars, 2) - bar centers
            angles_batch: (..., num_bars) - bar angles

        Returns:
            joint_pos1: (..., num_joints, 2) - position of joint on first bar/fixed point
            joint_pos2: (..., num_joints, 2) - position of joint on second bar
        """
        device, dtype = positions_batch.device, positions_batch.dtype
        batch_shape = positions_batch.shape[:-2]

        bar_lengths = self.bar_lengths.to(device=device, dtype=dtype)
        fixed_joints = system_def['fixed_joint_pos'].to(device=device, dtype=dtype).view(-1, 2)

        joint_pos1_list = []
        joint_pos2_list = []

        for joint_idx, (bar1_idx, bar2_idx, pos1_norm, pos2_norm) in enumerate(self.joints):
            # Position on first bar/fixed point
            if bar1_idx >= 0:
                # Joint on a bar
                center1 = positions_batch[..., bar1_idx, :]  # (..., 2)
                angle1 = angles_batch[..., bar1_idx]  # (...)
                length1 = bar_lengths[bar1_idx]

                # Offset along bar in local coordinates
                offset_local = torch.stack([
                    pos1_norm * length1 * torch.cos(angle1),
                    pos1_norm * length1 * torch.sin(angle1)
                ], dim=-1)

                pos1 = center1 + offset_local
            else:
                # Fixed point
                fixed_idx = -1 - bar1_idx
                pos1 = fixed_joints[fixed_idx].view((1,) * len(batch_shape) + (2,))
                pos1 = pos1.expand(*batch_shape, 2)

            # Position on second bar
            center2 = positions_batch[..., bar2_idx, :]
            angle2 = angles_batch[..., bar2_idx]
            length2 = bar_lengths[bar2_idx]

            offset_local = torch.stack([
                pos2_norm * length2 * torch.cos(angle2),
                pos2_norm * length2 * torch.sin(angle2)
            ], dim=-1)

            pos2 = center2 + offset_local

            joint_pos1_list.append(pos1)
            joint_pos2_list.append(pos2)

        joint_pos1 = torch.stack(joint_pos1_list, dim=-2)  # (..., num_joints, 2)
        joint_pos2 = torch.stack(joint_pos2_list, dim=-2)  # (..., num_joints, 2)

        return joint_pos1, joint_pos2

    def potential_energy_batch(self, system_def, pos_flat_batch, shape=None):
        """
        Compute potential energy for batch of configurations.

        Args:
            pos_flat_batch: (B, num_bars * 3) - flattened state
            shape: unused (for API compatibility)

        Returns:
            energy: (B,) - potential energies
        """
        B = pos_flat_batch.shape[0]
        device, dtype = pos_flat_batch.device, pos_flat_batch.dtype

        positions, angles = self._get_bar_state_batch(pos_flat_batch)

        # Gravitational potential energy (based on center of mass of each bar)
        g = system_def['gravity']
        masses = self.bar_masses.to(device=device, dtype=dtype)
        grav_energy = torch.sum(masses[None, :] * g * positions[:, :, 1], dim=1)  # (B,)

        # Joint constraint penalty (spring energy)
        joint_pos1, joint_pos2 = self._get_joint_positions_batch(positions, angles, system_def)

        displacements = joint_pos1 - joint_pos2  # (B, num_joints, 2)
        distances = torch.sqrt((displacements ** 2).sum(dim=2) + 1e-8)  # (B, num_joints)

        stiffness = self.joint_stiffness.to(device=device, dtype=dtype)  # (num_joints,)
        joint_energy = 0.5 * torch.sum(stiffness[None, :] * distances ** 2, dim=1)  # (B,)

        # External torque potential applied to a joint
        ext_forces = system_def.get('external_forces', {})
        torque_energy = torch.zeros(B, device=device, dtype=dtype)

        if 'torque_strength' in ext_forces and ext_forces['torque_strength'] != 0:
            joint_idx = ext_forces.get('torque_joint_idx', 0)
            if joint_idx < self.num_joints:
                # Get the two bars connected by this joint
                bar1_idx, bar2_idx, _, _ = self.joints[joint_idx]

                if bar1_idx >= 0:
                    # Relative angle between two bars
                    angle1 = angles[:, bar1_idx]
                    angle2 = angles[:, bar2_idx]
                    relative_angle = angle2 - angle1
                else:
                    # Bar attached to fixed point - use absolute angle
                    relative_angle = angles[:, bar2_idx]

                # Torque does work: -T * theta
                torque_work = -ext_forces['torque_strength'] * relative_angle
                torque_energy = torque_work

        return grav_energy + joint_energy + torque_energy

    def potential_energy(self, system_def, pos_flat, shape=None):
        """Single configuration version"""
        return self.potential_energy_batch(system_def, pos_flat.unsqueeze(0), shape)[0]

    def kinetic_energy_batch(self, system_def, pos_flat_batch, vel_flat_batch, shape=None):
        """
        Compute kinetic energy for batch of configurations.

        Args:
            pos_flat_batch: (B, num_bars * 3) - positions (unused but kept for API)
            vel_flat_batch: (B, num_bars * 3) - velocities
            shape: unused (for API compatibility)

        Returns:
            energy: (B,) - kinetic energies
        """
        B = vel_flat_batch.shape[0]
        device, dtype = vel_flat_batch.device, vel_flat_batch.dtype

        # Extract velocities
        vel_state = vel_flat_batch.view(B, self.num_bars, 3)
        linear_vel = vel_state[:, :, :2]  # (B, num_bars, 2)
        angular_vel = vel_state[:, :, 2]  # (B, num_bars)

        masses = self.bar_masses.to(device=device, dtype=dtype)
        inertias = self.bar_inertias.to(device=device, dtype=dtype)

        # Translational KE: 0.5 * m * v^2
        v_squared = torch.sum(linear_vel ** 2, dim=2)  # (B, num_bars)
        trans_energy = 0.5 * torch.sum(masses[None, :] * v_squared, dim=1)  # (B,)

        # Rotational KE: 0.5 * I * omega^2
        omega_squared = angular_vel ** 2  # (B, num_bars)
        rot_energy = 0.5 * torch.sum(inertias[None, :] * omega_squared, dim=1)  # (B,)

        return trans_energy + rot_energy

    def kinetic_energy(self, system_def, pos_flat, vel_flat, shape=None):
        """Single configuration version"""
        return self.kinetic_energy_batch(system_def, pos_flat.unsqueeze(0), vel_flat.unsqueeze(0), shape)[0]

    def physical_mass_matrix(self, system_def, q):
        """
        Build mass matrix for the system.
        Each bar has mass m and moment of inertia I.
        """
        masses = self.bar_masses
        inertias = self.bar_inertias

        M_diag = []
        for i in range(self.num_bars):
            M_diag.extend([masses[i], masses[i], inertias[i]])

        M_phys = torch.diag(torch.tensor(M_diag, dtype=q.dtype, device=q.device))
        return M_phys

    def build_system_ui(self, system_def):
        """Build ImGui interface for system parameters"""
        if psim.TreeNode("Bar-Joint System UI"):
            psim.TextUnformatted(f"Number of bars: {self.num_bars}")
            psim.TextUnformatted(f"Number of joints: {self.num_joints}")

            # Gravity control
            _, new_g = psim.SliderFloat("Gravity", float(system_def['gravity']), 0.0, 20.0)
            system_def['gravity'] = float(new_g)

            # External torque
            if "torque_strength" in system_def["external_forces"]:
                _, new_torque = psim.SliderInt("External Torque",
                                                 int(system_def['external_forces']['torque_strength'] * 2),
                                                 -10, 10)
                new_torque /= 2
                system_def['external_forces']['torque_strength'] = new_torque

                _, new_joint = psim.InputInt("Torque Joint Index",
                                             int(system_def['external_forces']['torque_joint_idx']))
                system_def['external_forces']['torque_joint_idx'] = max(0, min(self.num_joints - 1, new_joint))

            # Joint stiffness
            if psim.TreeNode("Joint Stiffness"):
                for i in range(self.num_joints):
                    _, new_k = psim.SliderFloat(f"Joint {i}", float(self.joint_stiffness[i]), 0.0, 500.0)
                    self.joint_stiffness[i] = float(new_k)
                psim.TreePop()

            psim.TreePop()

    def visualize_with_gradients(self, system, system_def, pos_flat, model, space,
                                 show_gradient=True, show_pca=True, show_model=True,
                                 n_pca_samples=100, arrow_scale=0.1, arrow_rad=0.005, k=5):
        """
        Visualize the bar-joint system with gradient information.

        Args:
            system_def: System definition
            pos_flat: (num_bars * 3,) - flattened state
            model: Trained gradient basis field model
            space: Space parameter
            show_gradient: If True, show actual gradient direction
            show_pca: If True, show PCA basis directions
            show_model: If True, show model's predicted basis directions
            n_pca_samples: Number of samples for computing local PCA
            arrow_scale: Scale factor for arrow lengths
        """
        device, dtype = pos_flat.device, pos_flat.dtype

        # First, do normal visualization
        self.visualize(system_def, pos_flat)

        positions, angles = self._get_bar_state_batch(pos_flat.unsqueeze(0))
        positions = positions[0]  # (num_bars, 2)
        angles = angles[0]  # (num_bars,)

        # Compute center of system for arrow placement
        center_pos = positions.mean(dim=0).detach().cpu().numpy()
        center_3d = np.array([center_pos[0], center_pos[1], 0.0])

        # === 1. Compute actual gradient ===
        if show_gradient:
            q_grad = pos_flat.clone().requires_grad_(True)
            E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0),
                                              space.unsqueeze(0))[0]
            grad_full = torch.autograd.grad(E, q_grad)[0]  # (dim,)

            # Visualize gradient as arrow from center
            grad_np = grad_full.detach().cpu().numpy()

            # Extract just the position components (ignore angle components)
            # For bar-joint: state is [x1, y1, theta1, x2, y2, theta2, ...]
            grad_positions = []
            for i in range(self.num_bars):
                grad_positions.append(grad_np[i * 3:i * 3 + 2])  # x, y components

            # Average gradient direction in position space
            grad_pos_avg = np.mean(grad_positions, axis=0)
            grad_norm = np.linalg.norm(grad_pos_avg)

            if grad_norm > 1e-8:
                grad_dir = grad_pos_avg / grad_norm * arrow_scale
                grad_arrow_end = center_3d + np.array([grad_dir[0], grad_dir[1], 0.0])

                # Draw arrow
                arrow_nodes = np.array([center_3d, grad_arrow_end])
                arrow_edges = np.array([[0, 1]])
                ps_grad = ps.register_curve_network("gradient", arrow_nodes, arrow_edges)
                ps_grad.set_radius(arrow_rad, relative=False)
                ps_grad.set_color((1.0, 0.0, 0.0))  # Red

        # === 2. Compute PCA basis ===
        if show_pca:
            # Sample gradients near current configuration
            perturbations = torch.randn(n_pca_samples, pos_flat.shape[0],
                                        device=device, dtype=dtype) * 0.01
            q_samples = pos_flat.unsqueeze(0) + perturbations

            # Compute gradients at samples
            gradients_list = []
            for i in range(n_pca_samples):
                q_sample = q_samples[i].requires_grad_(True)
                E_sample = system.potential_energy_batch(system_def, q_sample.unsqueeze(0),
                                                         space.unsqueeze(0))[0]
                grad_sample = torch.autograd.grad(E_sample, q_sample)[0]
                gradients_list.append(grad_sample)

            gradients_samples = torch.stack(gradients_list, dim=0)

            # Compute PCA
            mean_grad = gradients_samples.mean(dim=0)
            centered = gradients_samples - mean_grad
            cov = (centered.T @ centered) / (n_pca_samples - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Sort descending
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Visualize top PCA directions
            k_viz = min(k, eigenvectors.shape[1])  # Show top 3
            colors = [(0.0, 1.0, 0.0)]  # Green shades

            for k in range(k_viz):
                eigvec = eigenvectors[:, k].detach().cpu().numpy()
                eigval = eigenvalues[k].item()

                # Extract position components
                eigvec_positions = []
                for i in range(self.num_bars):
                    eigvec_positions.append(eigvec[i * 3:i * 3 + 2])

                eigvec_pos_avg = np.mean(eigvec_positions, axis=0)
                eigvec_norm = np.linalg.norm(eigvec_pos_avg)

                if eigvec_norm > 1e-8:
                    # Scale by sqrt(eigenvalue) to show relative importance
                    scale = arrow_scale * np.sqrt(eigval / eigenvalues[0].item())
                    eigvec_dir = eigvec_pos_avg / eigvec_norm * scale

                    # Offset slightly to avoid overlap
                    offset = np.array([0.05 * k, 0.05 * k, 0.0])
                    pca_start = center_3d + offset
                    pca_end = pca_start + np.array([eigvec_dir[0], eigvec_dir[1], 0.0])

                    arrow_nodes = np.array([pca_start, pca_end])
                    arrow_edges = np.array([[0, 1]])
                    ps_pca = ps.register_curve_network(f"pca_basis_{k}", arrow_nodes, arrow_edges)
                    ps_pca.set_radius(arrow_rad, relative=False)
                    ps_pca.set_color(colors[0])

        # === 3. Visualize model's predicted basis ===
        if show_model:
            B = model(pos_flat.unsqueeze(0)).squeeze(0)  # (dim, k)

            colors = [(0.0, 0.0, 1.0)]  # Blue shades

            for i in range(min(k, B.shape[1])):  # Show up to 3 basis vectors
                basis_vec = B[:, i].detach().cpu().numpy()

                # Extract position components
                basis_positions = []
                for j in range(self.num_bars):
                    basis_positions.append(basis_vec[j * 3:j * 3 + 2])

                basis_pos_avg = np.mean(basis_positions, axis=0)
                basis_norm = np.linalg.norm(basis_pos_avg)

                if basis_norm > 1e-8:
                    basis_dir = basis_pos_avg / basis_norm * arrow_scale

                    # Offset to avoid overlap
                    offset = np.array([0.05 * i, 0.05 * i, 0.0])
                    model_start = center_3d + offset
                    model_end = model_start + np.array([basis_dir[0], basis_dir[1], 0.0])

                    arrow_nodes = np.array([model_start, model_end])
                    arrow_edges = np.array([[0, 1]])
                    ps_model = ps.register_curve_network(f"model_basis_{i}",
                                                         arrow_nodes, arrow_edges)
                    ps_model.set_radius(arrow_rad, relative=False)
                    ps_model.set_color(colors[0])

    def visualize(self, system_def, pos_flat, shape=None, return_transforms=False):
        """
        Visualize the bar-joint system.

        Args:
            pos_flat: (num_bars * 3,) - flattened state
            shape: unused (for API compatibility)
            return_transforms: if True, return mesh data instead of registering
        """
        device, dtype = pos_flat.device, pos_flat.dtype

        positions, angles = self._get_bar_state_batch(pos_flat.unsqueeze(0))
        positions = positions[0]  # (num_bars, 2)
        angles = angles[0]  # (num_bars,)

        fixed_joints = system_def['fixed_joint_pos'].to(device=device, dtype=dtype).view(-1, 2)

        # Convert to numpy
        positions_np = positions.detach().cpu().numpy()
        angles_np = angles.detach().cpu().numpy()
        bar_lengths_np = self.bar_lengths.detach().cpu().numpy()
        fixed_joints_np = fixed_joints.detach().cpu().numpy()

        if return_transforms:
            body_data = []
            for i in range(self.num_bars):
                body_data.append({
                    'name': f"bar{i}",
                    'position': positions_np[i],
                    'angle': float(angles_np[i]),
                    'length': float(bar_lengths_np[i]),
                    'mass': float(self.bar_masses[i])
                })
            return body_data

        # Visualize bars as line segments
        bar_nodes = []
        bar_edges = []
        node_idx = 0

        for i in range(self.num_bars):
            center = positions_np[i]
            angle = angles_np[i]
            length = bar_lengths_np[i]

            # Endpoints of bar
            half_len = length / 2.0
            dx = half_len * np.cos(angle)
            dy = half_len * np.sin(angle)

            p1 = np.array([center[0] - dx, center[1] - dy, 0.0])
            p2 = np.array([center[0] + dx, center[1] + dy, 0.0])

            bar_nodes.append(p1)
            bar_nodes.append(p2)
            bar_edges.append([node_idx, node_idx + 1])
            node_idx += 2

        if len(bar_nodes) > 0:
            bar_nodes = np.array(bar_nodes)
            bar_edges = np.array(bar_edges)
            ps_bars = ps.register_curve_network("bars", bar_nodes, bar_edges)
            ps_bars.set_radius(0.03, relative=False)
            ps_bars.set_color((0.2, 0.5, 0.8))

        # Visualize joints
        joint_pos1, joint_pos2 = self._get_joint_positions_batch(
            positions.unsqueeze(0), angles.unsqueeze(0), system_def
        )
        joint_pos1 = joint_pos1[0].detach().cpu().numpy()  # (num_joints, 2)
        joint_pos2 = joint_pos2[0].detach().cpu().numpy()

        # Check which joint has torque applied
        torque_joint_idx = system_def.get('external_forces', {}).get('torque_joint_idx', -1)
        has_torque = (system_def.get('external_forces', {}).get('torque_strength', 0.0) != 0.0
                      and 0 <= torque_joint_idx < self.num_joints)

        # Show joint positions (color differently if torque is applied)
        all_joint_points = []
        torqued_joint_points = []

        for i in range(self.num_joints):
            p1 = np.array([joint_pos1[i, 0], joint_pos1[i, 1], 0.0])
            p2 = np.array([joint_pos2[i, 0], joint_pos2[i, 1], 0.0])

            if has_torque and i == torque_joint_idx:
                # This joint has torque - visualize separately
                torqued_joint_points.append(p1)
                torqued_joint_points.append(p2)
            else:
                all_joint_points.append(p1)
                all_joint_points.append(p2)

        if len(all_joint_points) > 0:
            all_joint_points = np.array(all_joint_points)
            ps_joints = ps.register_point_cloud("joint_points", all_joint_points)
            ps_joints.set_radius(0.02, relative=False)
            ps_joints.set_color((0.9, 0.3, 0.3))

        if len(torqued_joint_points) > 0:
            torqued_joint_points = np.array(torqued_joint_points)
            ps_torqued = ps.register_point_cloud("torqued_joint", torqued_joint_points)
            ps_torqued.set_radius(0.035, relative=False)  # Larger radius
            ps_torqued.set_color((0.1, 0.9, 0.1))  # Bright green

        # Visualize fixed points
        if len(fixed_joints_np) > 0:
            fixed_3d = np.zeros((len(fixed_joints_np), 3))
            fixed_3d[:, :2] = fixed_joints_np
            ps_fixed = ps.register_point_cloud("fixed_joints", fixed_3d)
            ps_fixed.set_radius(0.04, relative=False)
            ps_fixed.set_color((0.2, 0.2, 0.2))

        # Draw joint constraint violations (springs)
        joint_springs = []
        joint_spring_edges = []
        spring_node_idx = 0

        for i in range(self.num_joints):
            p1 = np.array([joint_pos1[i, 0], joint_pos1[i, 1], 0.0])
            p2 = np.array([joint_pos2[i, 0], joint_pos2[i, 1], 0.0])

            joint_springs.append(p1)
            joint_springs.append(p2)
            joint_spring_edges.append([spring_node_idx, spring_node_idx + 1])
            spring_node_idx += 2

        if len(joint_springs) > 0:
            joint_springs = np.array(joint_springs)
            joint_spring_edges = np.array(joint_spring_edges)
            ps_springs = ps.register_curve_network("joint_constraints", joint_springs, joint_spring_edges)
            ps_springs.set_radius(0.008, relative=False)
            ps_springs.set_color((0.9, 0.7, 0.2))

    def visualize_set_nice_view(self, system_def, x):
        """Set a nice camera view for the bar-joint system"""
        ps.look_at((0., 1.5, 5.0), (0., 1.5, 0.))

    def export(self, system_def, x, prefix=""):
        """Export system state (optional)"""
        pass