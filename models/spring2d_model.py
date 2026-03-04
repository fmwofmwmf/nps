import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class MassSpring2DSystem:
    """
    2D mass-spring system where masses are connected by springs.
    Each mass has 2 DOF (x, y position).
    """

    @staticmethod
    def construct(problem_name, config, dtype=torch.float64):
        system_def = {}
        system = MassSpring2DSystem()

        system.system_name = "MassSpring2D"
        system.problem_name = str(problem_name)

        # Defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,), dtype=dtype)
        system.cond_dim = 0

        if problem_name == 'single':
            # Single mass on a spring (1D vertical)
            num_masses = 1
            masses = [1.0]
            spring_connections = [[0, -1]]  # -1 represents fixed anchor
            spring_stiffness = [10.0]
            spring_rest_lengths = [1.0]
            fixed_points = torch.tensor([[0.0, 2.0]], dtype=dtype)
            init_positions = torch.tensor([[0.0, 1.0]], dtype=dtype)
            system_def['gravity'] = 9.8

        elif problem_name == 'double':
            # Two masses connected in series
            num_masses = 2
            masses = [1.0, 1.0]
            spring_connections = [[0, -1], [1, 0]]  # mass 0 to anchor, mass 1 to mass 0
            spring_stiffness = [15.0, 15.0]
            spring_rest_lengths = [0.8, 0.8]
            fixed_points = torch.tensor([[0.0, 2.0]], dtype=dtype)
            init_positions = torch.tensor([[0.0, 1.2], [0.0, 0.4]], dtype=dtype)
            system_def['gravity'] = 9.8

        elif problem_name == 'chain':
            # Vertical chain of masses
            num_masses = config["system"]["num_masses"]
            masses = [config["system"]["mass"]] * num_masses
            masses[-1] = config["system"]["end_mass"]  # Heavier mass at bottom
            spring_connections = [[0, -1]]  # First mass to anchor
            for i in range(1, num_masses):
                spring_connections.append([i, i - 1])
            spring_stiffness = [config["system"]["spring_stiffness"]] * num_masses
            spring_rest_lengths = [config["system"]["spring_rest_lengths"]] * num_masses
            fixed_points = torch.tensor([[0.0, 2.5]], dtype=dtype)
            init_positions = torch.tensor([[0.0, 2.5 - 0.3 * (i + 1)] for i in range(num_masses)], dtype=dtype)
            system_def['gravity'] = 9.8

        elif problem_name == 'cloth':
            # Small cloth patch (grid of masses)
            grid_w, grid_h = 4, 4
            num_masses = grid_w * grid_h
            masses = [0.3] * num_masses

            # Initialize positions in a grid
            spacing = 0.4
            init_positions = []
            for j in range(grid_h):
                for i in range(grid_w):
                    init_positions.append([i * spacing - (grid_w - 1) * spacing / 2, 2.0 - j * spacing])
            init_positions = torch.tensor(init_positions, dtype=dtype)

            # Create spring connections (structural springs)
            spring_connections = []
            spring_stiffness = []
            spring_rest_lengths = []

            # Horizontal springs
            for j in range(grid_h):
                for i in range(grid_w - 1):
                    idx1 = j * grid_w + i
                    idx2 = j * grid_w + i + 1
                    spring_connections.append([idx2, idx1])
                    spring_stiffness.append(30.0)
                    spring_rest_lengths.append(spacing)

            # Vertical springs
            for j in range(grid_h - 1):
                for i in range(grid_w):
                    idx1 = j * grid_w + i
                    idx2 = (j + 1) * grid_w + i
                    spring_connections.append([idx2, idx1])
                    spring_stiffness.append(30.0)
                    spring_rest_lengths.append(spacing)

            # Fix top row
            fixed_points = init_positions[:grid_w].clone()
            for i in range(grid_w):
                spring_connections.append([i, -1 - i])  # Connect to fixed points
                spring_stiffness.append(50.0)
                spring_rest_lengths.append(0.0)

            system_def['gravity'] = 9.8

        elif problem_name == 'network':
            # Complex network of springs
            num_masses = 5
            masses = [1.0, 0.8, 0.8, 0.6, 1.2]

            # Pentagonal arrangement
            angles = [i * 2 * np.pi / 5 for i in range(5)]
            radius = 1.0
            init_positions = torch.tensor([[radius * np.cos(a), 1.0 + radius * np.sin(a)]
                                           for a in angles], dtype=dtype)

            # Connect masses in a pentagonal pattern + internal connections
            spring_connections = []
            spring_stiffness = []
            spring_rest_lengths = []

            # Pentagon edges
            for i in range(num_masses):
                j = (i + 1) % num_masses
                spring_connections.append([i, j])
                spring_stiffness.append(15.0)
                dist = torch.norm(init_positions[i] - init_positions[j]).item()
                spring_rest_lengths.append(dist)

            # Internal connections (star pattern)
            for i in range(num_masses):
                j = (i + 2) % num_masses
                spring_connections.append([i, j])
                spring_stiffness.append(10.0)
                dist = torch.norm(init_positions[i] - init_positions[j]).item()
                spring_rest_lengths.append(dist)

            # Fix one point at top
            fixed_points = torch.tensor([[0.0, 2.0]], dtype=dtype)
            spring_connections.append([0, -1])
            spring_stiffness.append(25.0)
            spring_rest_lengths.append(torch.norm(init_positions[0] - fixed_points[0]).item())

            system_def['gravity'] = 9.8

        else:
            raise ValueError(f"Unrecognized problem_name: {problem_name}")

        # Store system properties
        system.num_masses = num_masses
        system.masses = torch.tensor(masses, dtype=dtype)
        system.spring_connections = spring_connections  # List of [mass_i, mass_j] pairs
        system.spring_stiffness = torch.tensor(spring_stiffness, dtype=dtype)
        system.spring_rest_lengths = torch.tensor(spring_rest_lengths, dtype=dtype)
        system.num_springs = len(spring_connections)

        # DOF: 2 per mass (x, y)
        system.dim = num_masses * 2
        system_def['dim'] = system.dim

        # Flatten positions for state representation
        system_def['rest_pos'] = init_positions.flatten()
        system_def['init_pos'] = init_positions.flatten().clone()
        system_def['fixed_pos'] = fixed_points.flatten()

        # External forces
        system_def['external_forces']['wind_strength_minmax'] = (-10.0, 10.0)
        system_def['external_forces']['wind_strength'] = 0.0
        system_def['external_forces']['wind_direction'] = torch.tensor([1.0, 0.0], dtype=dtype)

        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    def _get_spring_endpoints_batch(self, positions_batch, system_def):
        """
        Vectorized and vmap-safe.
        positions_batch: (..., num_masses, 2)
        Returns:
            endpoint1, endpoint2: (..., num_springs, 2)
        """
        device, dtype = positions_batch.device, positions_batch.dtype

        # Build / cache spring index tensors
        if (not hasattr(self, "_spring_idx_i")
                or self._spring_idx_i.device != device):
            idx_i = torch.tensor([i for i, j in self.spring_connections],
                                 device=device, dtype=torch.long)
            idx_j = torch.tensor([j for i, j in self.spring_connections],
                                 device=device, dtype=torch.long)
            self._spring_idx_i = idx_i
            self._spring_idx_j = idx_j
        else:
            idx_i = self._spring_idx_i
            idx_j = self._spring_idx_j

        fixed_points = system_def['fixed_pos'].to(device=device, dtype=dtype).view(-1, 2)

        # Endpoint 1 is always a mass
        endpoint1 = positions_batch[..., idx_i, :]  # (..., S, 2)

        S = idx_j.shape[0]
        batch_shape = positions_batch.shape[:-2]

        # --- mass-to-mass candidate ---
        # Clamp idx_j so it's valid even when j < 0 (won't be used due to masking)
        idx_j_clamped = idx_j.clamp(min=0)
        endpoint2_mass = positions_batch[..., idx_j_clamped, :]  # (..., S, 2)

        # --- fixed-point candidate ---
        fixed_idx = (-1 - idx_j).clamp(min=0)
        fixed_sel = fixed_points[fixed_idx]  # (S, 2)

        # Broadcast fixed points across batch dims
        fixed_sel = fixed_sel.reshape((1,) * len(batch_shape) + fixed_sel.shape)
        endpoint2_fixed = fixed_sel.expand(*batch_shape, S, 2)

        # --- select based on whether j >= 0 ---
        mask = (idx_j >= 0).view((1,) * len(batch_shape) + (S, 1))

        endpoint2 = torch.where(mask, endpoint2_mass, endpoint2_fixed)

        return endpoint1, endpoint2

    def potential_energy_batch(self, system_def, pos_flat_batch, shape=None):
        """
        Compute potential energy for batch of configurations.

        Args:
            pos_flat_batch: (B, num_masses * 2) - flattened positions
            shape: unused (for API compatibility)

        Returns:
            energy: (B,) - potential energies
        """
        B = pos_flat_batch.shape[0]
        device, dtype = pos_flat_batch.device, pos_flat_batch.dtype

        # Reshape to (B, num_masses, 2)
        positions = pos_flat_batch.view(B, self.num_masses, 2)

        # Gravitational potential energy
        g = system_def['gravity']
        masses = self.masses.to(device=device, dtype=dtype)
        grav_energy = torch.sum(masses[None, :] * g * positions[:, :, 1], dim=1)  # (B,)

        # Spring potential energy
        endpoint1, endpoint2 = self._get_spring_endpoints_batch(positions, system_def)
        displacements = endpoint1 - endpoint2  # (B, num_springs, 2)
        distances = torch.sqrt((displacements**2).sum(dim=2) + 1e-8)  # (B, num_springs)

        rest_lengths = self.spring_rest_lengths.to(device=device, dtype=dtype)  # (num_springs,)
        stiffness = self.spring_stiffness.to(device=device, dtype=dtype)  # (num_springs,)

        extensions = distances - rest_lengths[None, :]  # (B, num_springs)
        spring_energy = 0.5 * torch.sum(stiffness[None, :] * extensions ** 2, dim=1)  # (B,)

        # External wind force (negative work done)
        ext_forces = system_def.get('external_forces', {})
        wind_energy = torch.zeros(B, device=device, dtype=dtype)

        if self.training:
            # Random wind vector per batch
            wind_scale = torch.tensor([0.3, 0], device=device, dtype=dtype)
            wind_vec = torch.randn(B, 2, device=device, dtype=dtype) * wind_scale
            wind_force = wind_vec[:, None, :]  # (B, 1, 2)

            # Wind work: -F · x
            wind_energy = -torch.sum(
                positions * wind_force, dim=(1, 2)
            )

        if 'wind_strength' in ext_forces and ext_forces['wind_strength'] != 0:
            wind_dir = ext_forces['wind_direction'].to(device=device, dtype=dtype)
            wind_dir = wind_dir / torch.norm(wind_dir)  # Normalize
            # Wind does work: -F·x
            wind_work = -ext_forces['wind_strength'] * torch.sum(
                positions * wind_dir[None, None, :], dim=(1, 2)
            )
            wind_energy = wind_work

        return grav_energy + spring_energy + wind_energy

    def potential_energy(self, system_def, pos_flat, shape=None):
        """Single configuration version"""
        return self.potential_energy_batch(system_def, pos_flat.unsqueeze(0), shape)[0]

    def kinetic_energy_batch(self, system_def, pos_flat_batch, vel_flat_batch, shape=None):
        """
        Compute kinetic energy for batch of configurations.

        Args:
            pos_flat_batch: (B, num_masses * 2) - positions (unused but kept for API)
            vel_flat_batch: (B, num_masses * 2) - velocities
            shape: unused (for API compatibility)

        Returns:
            energy: (B,) - kinetic energies
        """
        B = vel_flat_batch.shape[0]
        device, dtype = vel_flat_batch.device, vel_flat_batch.dtype

        # Reshape to (B, num_masses, 2)
        velocities = vel_flat_batch.view(B, self.num_masses, 2)

        masses = self.masses.to(device=device, dtype=dtype)

        # KE = 0.5 * m * v^2
        v_squared = torch.sum(velocities ** 2, dim=2)  # (B, num_masses)
        energy = 0.5 * torch.sum(masses[None, :] * v_squared, dim=1)  # (B,)

        return energy

    def kinetic_energy(self, system_def, pos_flat, vel_flat, shape=None):
        """Single configuration version"""
        return self.kinetic_energy_batch(system_def, pos_flat.unsqueeze(0), vel_flat.unsqueeze(0), shape)[0]

    def physical_mass_matrix(self, system_def, q):
        # Each particle has 2 DOF (x, y)
        masses = self.masses  # tensor of shape (num_masses,)
        M_phys = torch.diag(masses.repeat_interleave(2))
        return M_phys

    def build_system_ui(self, system_def):
        """Build ImGui interface for system parameters"""
        if psim.TreeNode("Mass-Spring System UI"):
            psim.TextUnformatted(f"Number of masses: {self.num_masses}")
            psim.TextUnformatted(f"Number of springs: {self.num_springs}")

            # Gravity control
            _, new_g = psim.SliderFloat("Gravity", float(system_def['gravity']), 0.0, 20.0)
            system_def['gravity'] = float(new_g)

            # External wind force
            if "wind_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['wind_strength_minmax']
                _, new_wind = psim.SliderFloat("Wind Strength",
                                               float(system_def['external_forces']['wind_strength']),
                                               low, high)
                system_def['external_forces']['wind_strength'] = float(new_wind)

                # Wind direction angle
                wind_dir = system_def['external_forces']['wind_direction']
                current_angle = float(torch.atan2(wind_dir[1], wind_dir[0]) * 180 / np.pi)
                _, new_angle = psim.SliderFloat("Wind Direction (deg)", current_angle, 0.0, 360.0)
                angle_rad = new_angle * np.pi / 180
                system_def['external_forces']['wind_direction'] = torch.tensor(
                    [np.cos(angle_rad), np.sin(angle_rad)], dtype=wind_dir.dtype
                )

            psim.TreePop()

    def visualize(self, system_def, pos_flat, shape=None, return_transforms=False):
        """
        Visualize the mass-spring system.

        Args:
            pos_flat: (num_masses * 2,) - flattened positions
            shape: unused (for API compatibility)
            return_transforms: if True, return mesh data instead of registering
        """
        device, dtype = pos_flat.device, pos_flat.dtype

        # Reshape to (num_masses, 2)
        positions = pos_flat.view(self.num_masses, 2)
        fixed_points = system_def['fixed_pos'].to(device=device, dtype=dtype).view(-1, 2)

        # Convert to numpy
        positions_np = positions.detach().cpu().numpy()
        fixed_points_np = fixed_points.detach().cpu().numpy()

        if return_transforms:
            body_data = []
            for i in range(self.num_masses):
                body_data.append({
                    'name': f"mass{i}",
                    'position': positions_np[i],
                    'mass': float(self.masses[i])
                })
            return body_data

        # Pad to 3D for polyscope
        positions_3d = np.zeros((self.num_masses, 3))
        positions_3d[:, :2] = positions_np

        fixed_3d = np.zeros((len(fixed_points_np), 3))
        fixed_3d[:, :2] = fixed_points_np

        # Register masses as point cloud
        ps_masses = ps.register_point_cloud("masses", positions_3d)
        radii = 0.05 * np.sqrt(self.masses.detach().cpu().numpy())
        ps_masses.add_scalar_quantity("mass", self.masses.detach().cpu().numpy())
        ps_masses.set_radius(0.05, relative=False)

        # Register fixed points
        if len(fixed_points_np) > 0:
            ps_fixed = ps.register_point_cloud("fixed_points", fixed_3d)
            ps_fixed.set_radius(0.03, relative=False)
            ps_fixed.set_color((0.5, 0.5, 0.5))

        # Draw springs as curve network
        spring_nodes = []
        spring_edges = []
        node_idx = 0

        for spring_idx, (i, j) in enumerate(self.spring_connections):
            if j >= 0:
                # Spring between two masses
                start = positions_3d[i]
                end = positions_3d[j]
            else:
                # Spring to fixed point
                fixed_idx = -1 - j
                start = positions_3d[i]
                end = fixed_3d[fixed_idx]

            spring_nodes.append(start)
            spring_nodes.append(end)
            spring_edges.append([node_idx, node_idx + 1])
            node_idx += 2

        if len(spring_nodes) > 0:
            spring_nodes = np.array(spring_nodes)
            spring_edges = np.array(spring_edges)
            ps_springs = ps.register_curve_network("springs", spring_nodes, spring_edges)
            ps_springs.set_radius(0.01, relative=False)
            ps_springs.set_color((0.3, 0.6, 0.9))

    def visualize_set_nice_view(self, system_def, x):
        """Set a nice camera view for the mass-spring system"""
        ps.look_at((0., 1., 4.0), (0., 1., 0.))

    def export(self, system_def, x, prefix=""):
        """Export system state (optional)"""
        pass