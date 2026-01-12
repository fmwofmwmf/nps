import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class Pendulum2DSystem:
    """
    2D pendulum chain system where each pendulum is a single DOF (angle).
    Each pendulum's angle is relative to its parent pendulum.
    """

    @staticmethod
    def construct(problem_name, dtype=torch.float64):
        system_def = {}
        system = Pendulum2DSystem()

        system.system_name = "Pendulum2D"
        system.problem_name = str(problem_name)
        system.point_mass = True
        # Defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,), dtype=dtype)
        system.cond_dim = 0

        if problem_name == 'single':
            # Single pendulum
            num_pendulums = 1
            lengths = [1.0]
            masses = [1.0]
            system_def['gravity'] = 9.8

        elif problem_name == 'double':
            num_pendulums = 2
            lengths = [1.0, 1.0]
            masses = [1.0, 1.0]
            system_def['gravity'] = 9.8

        elif problem_name == 'chain':
            # Chain of pendulums
            num_pendulums = 3
            lengths = [0.4] * num_pendulums
            masses = [0.5] * num_pendulums
            system_def['gravity'] = 9.8

        elif problem_name == 'variable':
            # Variable length chain
            num_pendulums = 4
            lengths = [1.2, 0.8, 1.0, 0.6]
            masses = [1.0, 0.8, 0.6, 0.4]
            system_def['gravity'] = 9.8

        else:
            raise ValueError(f"Unrecognized problem_name: {problem_name}")

        # Store pendulum properties
        system.num_pendulums = num_pendulums
        system.lengths = torch.tensor(lengths, dtype=dtype)
        system.masses = torch.tensor(masses, dtype=dtype)

        # DOF: one angle per pendulum
        system.dim = num_pendulums
        system_def['dim'] = num_pendulums

        # Initial configuration: all pendulums hanging down
        init_angles = torch.zeros(num_pendulums, dtype=dtype)
        # Add small perturbation to break symmetry
        init_angles[0] = torch.pi / 2

        system_def['rest_pos'] = init_angles
        system_def['init_pos'] = init_angles.clone()
        system_def['fixed_pos'] = torch.tensor([], dtype=dtype)

        # Mass matrix (for kinetic energy computation)
        # For simplicity, we'll compute it on-the-fly in kinetic_energy
        system_def['mass'] = None  # Not used directly in this formulation

        # External forces
        system_def['external_forces']['torque_strength_minmax'] = (-5.0, 5.0)
        system_def['external_forces']['torque_strength'] = 0.0
        system_def['external_forces']['applied_link'] = 0  # Which pendulum to apply torque to

        # Visualization parameters
        system_def['anchor_point'] = torch.tensor([0.0, 0.0], dtype=dtype)

        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    def _compute_positions_batch(self, angles_batch, anchor):
        """
        Compute Cartesian positions for a batch of configurations (vmap-compatible).

        This function handles a batch dimension but is designed to be vmappable
        over additional outer dimensions.

        Args:
            angles_batch: (B, num_pendulums) - angles relative to parent
            anchor: (2,) - anchor point

        Returns:
            positions: (B, num_pendulums+1, 2) - positions of anchor + all endpoints
        """
        device = angles_batch.device
        dtype = angles_batch.dtype
        B = angles_batch.shape[0]

        # Cumulative angles (global angles from vertical)
        cumulative_angles = torch.cumsum(angles_batch, dim=1)  # (B, num_pendulums)

        # Compute displacements for all pendulums at once
        lengths = self.lengths.to(device=device, dtype=dtype)  # (num_pendulums,)
        dx = lengths.unsqueeze(0) * torch.sin(cumulative_angles)  # (B, num_pendulums)
        dy = -lengths.unsqueeze(0) * torch.cos(cumulative_angles)  # (B, num_pendulums)

        # Stack displacements
        displacements = torch.stack([dx, dy], dim=2)  # (B, num_pendulums, 2)

        # Cumulative sum to get absolute positions of endpoints relative to anchor
        relative_positions = torch.cumsum(displacements, dim=1)  # (B, num_pendulums, 2)

        # Create positions array by concatenating anchor with endpoint positions
        anchor_expanded = anchor.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, 2)
        endpoint_positions = anchor_expanded + relative_positions  # (B, num_pendulums, 2)
        positions = torch.cat([anchor_expanded, endpoint_positions], dim=1)  # (B, num_pendulums+1, 2)

        return positions

    def _compute_positions(self, angles, anchor):
        """Non-batch version for single configuration"""
        return self._compute_positions_batch(angles.unsqueeze(0), anchor)[0]

    def potential_energy_batch(self, system_def, angles_batch, shape=None):
        """
        Compute potential energy for batch of configurations.

        Args:
            angles_batch: (B, num_pendulums) - angles
            shape: unused (for API compatibility)

        Returns:
            energy: (B,) - potential energies
        """
        B = angles_batch.shape[0]
        device = angles_batch.device
        dtype = angles_batch.dtype

        anchor = system_def['anchor_point'].to(device=device, dtype=dtype)
        g = system_def['gravity']

        # Get positions of all endpoints
        positions = self._compute_positions_batch(angles_batch, anchor)  # (B, num_pendulums+1, 2)

        # Potential energy = sum of m_i * g * h_i for each pendulum
        # Center of mass is at midpoint of each pendulum
        # Vectorize: compute all COM y-coordinates at once
        if self.point_mass:
            # Point mass at tip: COM is at endpoint
            com_y = positions[:, 1:, 1]  # (B, num_pendulums)
        else:
            # Uniform rod: COM is at midpoint
            com_y = (positions[:, :-1, 1] + positions[:, 1:, 1]) / 2.0  # (B, num_pendulums)

        # Vectorize mass multiplication: masses * g * com_y, then sum over pendulums
        masses = self.masses.to(device=device, dtype=dtype).unsqueeze(0)  # (1, num_pendulums)
        energy = torch.sum(masses * g * com_y, dim=1)  # (B,)

        # External torque energy (torque * angle)
        ext_forces = system_def['external_forces']
        if 'torque_strength' in ext_forces and ext_forces['torque_strength'] != 0:
            link_idx = ext_forces.get('applied_link', 0)
            if link_idx < self.num_pendulums:
                # Vectorized selection without explicit indexing
                # Create a one-hot mask for the link
                link_mask = torch.zeros(self.num_pendulums, device=device, dtype=dtype)
                link_mask[link_idx] = 1.0
                link_mask = link_mask.unsqueeze(0)  # (1, num_pendulums)

                # Compute torque contribution
                torque_energy = ext_forces['torque_strength'] * torch.sum(link_mask * angles_batch, dim=1)
                energy -= torque_energy

        return energy

    def potential_energy(self, system_def, angles, shape=None):
        """Single configuration version"""
        return self.potential_energy_batch(system_def, angles.unsqueeze(0), shape)[0]

    def kinetic_energy_batch(self, system_def, angles_batch, angvel_batch, shape):
        """
        angles_batch: (B, n) relative angles per link
        angvel_batch: (B, n) relative angular velocities per link
        Returns: (B,) kinetic energies
        """
        # angles_batch = angles_batch.clone()
        # angles_batch[:, 1:] = 0
        B, n = angles_batch.shape
        device, dtype = angles_batch.device, angles_batch.dtype

        lengths = self.lengths.to(device=device, dtype=dtype)  # (n,)
        masses = self.masses.to(device=device, dtype=dtype)  # (n,)

        # Step 1: cumulative angles → absolute orientations
        cum_angles = torch.cumsum(angles_batch, dim=1)  # (B, n)
        cos = torch.cos(cum_angles)
        sin = torch.sin(cum_angles)

        # Step 2: build the upper-triangular matrix of contributions
        # Each link i gets contributions from joints 0..i
        # mask[i,j] = 1 if j <= i, else 0
        mask = torch.triu(torch.ones(n, n, device=device, dtype=dtype), diagonal=0).T  # (n,n)

        # Step 3: compute per-link Jacobian contributions
        # L: (1,n,1)
        L = lengths[None, :, None]  # broadcastable
        # cos/sin: (B,n,1)
        cos_ = cos[:, :, None]
        sin_ = sin[:, :, None]
        # Apply mask: mask[i,j] = 1 if j <= i else 0
        mask = mask[None, :, :]  # (1,n,n)
        contrib_x = L * cos_ * mask  # (B, n, n)
        contrib_y = L * sin_ * mask  # (B, n, n)

        # Step 4: multiply by angular velocities
        # angvel_batch: (B,n) → (B,1,n) to multiply along joints axis
        angvel = angvel_batch[:, None, :]  # (B,1,n)
        vel_x = (contrib_x * angvel).sum(-1)  # sum over joints → (B,n)
        vel_y = (contrib_y * angvel).sum(-1)  # (B,n)

        velocities = torch.stack([vel_x, vel_y], dim=-1)  # (B, n, 2)

        # Step 5: kinetic energy
        if self.point_mass:
            v2 = (velocities ** 2).sum(-1)  # (B,n)
            energy = 0.5 * (v2 * masses[None, :]).sum(-1)  # (B,)
        else:
            # Rods: center-of-mass velocity
            velocities_ext = torch.cat([torch.zeros(B, 1, 2, device=device, dtype=dtype), velocities], dim=1)
            v_com = 0.5 * (velocities_ext[:, :-1, :] + velocities_ext[:, 1:, :])
            v_com2 = (v_com ** 2).sum(-1)
            KE_trans = 0.5 * v_com2 * masses[None, :]
            I_com = masses * lengths ** 2 / 12.0
            KE_rot = 0.5 * I_com[None, :] * (angvel_batch ** 2)
            energy = (KE_trans + KE_rot).sum(-1)

        return energy
    def kinetic_energy(self, system_def, angles, angvel, shape=None):
        """Single configuration version"""
        return self.kinetic_energy_batch(system_def, angles.unsqueeze(0), angvel.unsqueeze(0), shape)[0]

    def build_system_ui(self, system_def):
        """Build ImGui interface for system parameters"""
        if psim.TreeNode("Pendulum System UI"):
            psim.TextUnformatted(f"Number of pendulums: {self.num_pendulums}")

            # Gravity control
            _, new_g = psim.SliderFloat("Gravity", float(system_def['gravity']), 0.0, 20.0)
            system_def['gravity'] = float(new_g)

            # External torque
            if "torque_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['torque_strength_minmax']
                _, new_val = psim.SliderFloat("Applied Torque",
                                              float(system_def['external_forces']['torque_strength']),
                                              low, high)
                system_def['external_forces']['torque_strength'] = float(new_val)

                _, new_link = psim.SliderInt("Torque Applied to Link",
                                             int(system_def['external_forces']['applied_link']),
                                             0, self.num_pendulums - 1)
                system_def['external_forces']['applied_link'] = int(new_link)

            psim.TreePop()

    def visualize(self, system_def, angles, shape=None, return_transforms=False):
        """
        Visualize the pendulum chain.

        Args:
            angles: (num_pendulums,) - angles
            shape: unused (for API compatibility)
            return_transforms: if True, return mesh data instead of registering
        """
        device = angles.device
        dtype = angles.dtype

        anchor = system_def['anchor_point'].to(device=device, dtype=dtype)
        positions = self._compute_positions(angles, anchor)

        # Convert to numpy for visualization
        positions_np = positions.detach().cpu().numpy()

        if return_transforms:
            body_data = []
            for i in range(self.num_pendulums):
                start = positions_np[i]
                end = positions_np[i + 1]
                body_data.append({
                    'name': f"pendulum{i}",
                    'start': start,
                    'end': end,
                    'length': float(self.lengths[i]),
                    'mass': float(self.masses[i])
                })
            return body_data

        # Create edges for the pendulum chain
        edges = []
        for i in range(self.num_pendulums):
            edges.append([i, i + 1])
        edges = np.array(edges)

        # Pad positions to 3D for polyscope
        positions_3d = np.zeros((self.num_pendulums + 1, 3))
        positions_3d[:, :2] = positions_np

        # Register as curve network
        ps_chain = ps.register_curve_network("pendulum_chain", positions_3d, edges)
        ps_chain.set_radius(0.02, relative=False)

        # Add spheres for masses
        for i in range(self.num_pendulums):
            pos_3d = np.zeros(3)
            pos_3d[:2] = positions_np[i + 1]
            ps_mass = ps.register_point_cloud(f"mass_{i}", pos_3d.reshape(1, 3))
            radius = 0.03 * np.sqrt(float(self.masses[i]))
            ps_mass.set_radius(radius, relative=False)

    def visualize_set_nice_view(self, system_def, x):
        """Set a nice camera view for the pendulum system"""
        # Calculate extent of pendulum chain
        total_length = torch.sum(self.lengths).item()
        ps.look_at((0., 0., 3.0 * total_length), (0., -total_length / 2, 0.))

    def export(self, system_def, x, prefix=""):
        """Export system state (optional)"""
        pass