import torch
import numpy as np
import os

import polyscope as ps
import polyscope.imgui as psim

try:
    import igl
except Exception:
    print("WARNING: igl bindings not available")


def make_body(file, density, scale, dtype=torch.float64):
    v, f = igl.read_triangle_mesh(file)
    v = v @ scale

    vol = igl.massmatrix(v, f).data
    vol = np.nan_to_num(vol)  # massmatrix returns Nans in some stewart meshes

    c = np.sum(vol[:, None] * v, axis=0) / np.sum(vol)
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:, None] * W) * density

    # x0 is 4x3 in original: first 3 rows identity, last row is c
    x0 = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       c], dtype=dtype)

    body = {'v': v, 'f': f, 'W': W, 'x0': x0, 'mass': mass}
    return body


def make_joint(b0, b1, bodies, joint_pos_world, joint_vec_world):
    # Creates a joint between the specified bodies, assumes the bodies have zero rotation and are properly aligned in the world
    # TODO: Use rotation for joint initialization
    pb0 = joint_pos_world.clone()
    vb0 = joint_vec_world.clone()
    if b0 != -1:
        c0 = bodies[b0]['x0'][3, :].clone()
        pb0 = pb0 - c0
    pb1 = joint_pos_world.clone()
    vb1 = joint_vec_world.clone()
    if b1 != -1:
        c1 = bodies[b1]['x0'][3, :].clone()
        pb1 = pb1 - c1
    joint = {'body_id0': b0, 'body_id1': b1,
             'pos_body0': pb0, 'pos_body1': pb1,
             'vec_body0': vb0, 'vec_body1': vb1}
    return joint


def bodiesToStructOfArrays(bodies, dtype=torch.float64):
    v_arr = []
    f_arr = []
    W_arr = []
    x0_arr = []
    mass_arr = []
    for b in bodies:
        # keep original numpy arrays for v, f, W (Polyscope / igl expect numpy) but convert x0/mass to torch
        v_arr.append(torch.tensor(b['v'], dtype=dtype))
        f_arr.append(torch.tensor(b['f'], dtype=torch.long))
        W_arr.append(torch.tensor(b['W'], dtype=dtype))
        x0_arr.append(b['x0'].to(dtype=dtype))
        mass_arr.append(torch.tensor(b['mass'], dtype=dtype))

    out_struct = {
        'v': torch.stack(v_arr, dim=0),
        'f': torch.stack(f_arr, dim=0),
        'W': torch.stack(W_arr, dim=0),
        'x0': torch.stack(x0_arr, dim=0),
        'mass': torch.stack(mass_arr, dim=0),
    }

    n_bodies = len(v_arr)

    return out_struct, n_bodies


class Rigid3DSystem:

    @staticmethod
    def construct(problem_name, dtype=torch.float64):
        system_def = {}
        system = Rigid3DSystem()

        system.system_name = "Rigid3d"
        system.problem_name = str(problem_name)
        system.shape_param_names = []  # Will be set per-problem

        # set some defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,), dtype=dtype)
        system_def["contact_stiffness"] = 1e8
        system.cond_dim = 0
        system.body_ID = None

        bodies = []
        joint_list = []
        linkContactPairs = []
        numBodiesFixed = 0

        if problem_name == 'klann':
            bodies.append(make_body(os.path.join(".", "data", "klann-red.obj"), 1000, 1.0, dtype))
            bodies.append(make_body(os.path.join(".", "data", "klann-purple.obj"), 1000, 1.0, dtype))
            bodies.append(make_body(os.path.join(".", "data", "klann-brown.obj"), 1000, 1.0, dtype))
            bodies.append(make_body(os.path.join(".", "data", "klann-distal.obj"), 1000, 1.0, dtype))
            bodies.append(make_body(os.path.join(".", "data", "klann-top.obj"), 1000, 1.0, dtype))

            joint_list.append(make_joint(0, -1, bodies, torch.tensor([0, 0.08, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(0, 1, bodies,
                                         torch.tensor([-0.046622, 0.097594, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(1, 2, bodies,
                                         torch.tensor([-0.1736, 0.11205, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(1, 3, bodies,
                                         torch.tensor([-0.31194, 0.16654, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(4, -1, bodies,
                                         torch.tensor([-0.13, 0.1875, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(2, -1, bodies,
                                         torch.tensor([-0.13, 0.045, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(4, 3, bodies,
                                         torch.tensor([-0.21981, 0.25102, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))

            system_def["gravity"] = torch.tensor([0.0, -0.98, 0.0], dtype=dtype)
            system_def['external_forces']['force_strength_minmax'] = (-10, 10)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0

        elif problem_name == 'stewart':

            scale = 5.0

            bodies.append(make_body(os.path.join(".", "data", "stewart-base.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm1.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm2.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm3.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm4.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm5.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-arm6.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut1.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut2.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut3.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut4.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut5.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-strut6.obj"), 1000, scale, dtype))
            bodies.append(make_body(os.path.join(".", "data", "stewart-top.obj"), 1000, scale, dtype))

            numBodiesFixed = 1

            ang = np.pi * 2.0 / 3.0
            R = torch.tensor([[np.cos(ang), 0.0, np.sin(ang)],
                              [0.0, 1.0, 0.0],
                              [-np.sin(ang), 0.0, np.cos(ang)]],
                             dtype=dtype)
            Rh = torch.tensor([[np.cos(ang / 2), 0.0, np.sin(ang / 2)],
                               [0.0, 1.0, 0.0],
                               [-np.sin(ang / 2), 0.0, np.cos(ang / 2)]],
                              dtype=dtype)

            #### [x,z,-y] original comment preserved
            a = scale * torch.tensor([-0.018, 0.0215, -0.044856], dtype=dtype)
            b = scale * torch.tensor([-0.047847, 0.0215, 0.00684], dtype=dtype)
            v = torch.tensor([0, 0.0, 1.0], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb
            Rv = R @ v
            RRv = R @ Rv

            joint_list.append(make_joint(0, 1, bodies, a, v))
            joint_list.append(make_joint(0, 2, bodies, b, Rv))
            joint_list.append(make_joint(0, 3, bodies, Ra, Rv))
            joint_list.append(make_joint(0, 4, bodies, Rb, RRv))
            joint_list.append(make_joint(0, 5, bodies, RRa, RRv))
            joint_list.append(make_joint(0, 6, bodies, RRb, v))

            #### [x,z,-y]
            a = scale * torch.tensor([-0.003, 0.0215, -0.051856], dtype=dtype)
            b = scale * torch.tensor([-0.046409, 0.0215, 0.02333], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb

            joint_list.append(make_joint(1, 7, bodies, a, 0.01 * v))
            joint_list.append(make_joint(2, 8, bodies, b, 0.01 * Rv))
            joint_list.append(make_joint(3, 9, bodies, Ra, 0.01 * Rv))
            joint_list.append(make_joint(4, 10, bodies, Rb, 0.01 * RRv))
            joint_list.append(make_joint(5, 11, bodies, RRa, 0.01 * RRv))
            joint_list.append(make_joint(6, 12, bodies, RRb, 0.01 * v))

            #### [x,z,-y]
            a = scale * torch.tensor([-0.032159, 0.082222, -0.022686], dtype=dtype)
            b = scale * torch.tensor([-0.035712, 0.082222, -0.016488], dtype=dtype)
            v = Rh @ torch.tensor([0, 0.0, 1.0], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb
            Rv = R @ v
            RRv = R @ Rv

            joint_list.append(make_joint(7, 13, bodies, a, 0.01 * v))
            joint_list.append(make_joint(8, 13, bodies, b, 0.01 * v))
            joint_list.append(make_joint(9, 13, bodies, Ra, 0.01 * Rv))
            joint_list.append(make_joint(10, 13, bodies, Rb, 0.01 * Rv))
            joint_list.append(make_joint(11, 13, bodies, RRa, 0.01 * RRv))
            joint_list.append(make_joint(12, 13, bodies, RRb, 0.01 * RRv))

            ###
            system_def["gravity"] = torch.tensor([0.0, 0.98, 0.0], dtype=dtype)
            system_def['external_forces']['force_strength_minmax'] = (-300, 300)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0

            system.body_ID = np.array([2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2])

        elif problem_name == 'links':

            numLinks = 24
            link_separation = 0.0491

            system_def['link_le'] = 0.013
            system_def['link_r1'] = 0.02
            system_def['link_r2'] = 0.009

            for i in range(numLinks):
                scale = Rigid3DSystem.get_shape_transform(torch.tensor([1, 1, 1]))
                body = make_body(os.path.join(".", "data", "link.obj"), 1000, scale.cpu().numpy())
                if i % 2 == 1:
                    body['x0'] = torch.tensor([
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, -i * link_separation, 0]
                    ], dtype=torch.float32)
                else:
                    body['x0'] = torch.tensor([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0],
                        [0, -i * link_separation, 0]
                    ], dtype=torch.float32)
                bodies.append(body)

            numBodiesFixed = 1

            # Immediate neighbour contacts
            for i in range(numLinks):
                # Do both directions, since our computation is not symmetric
                if i - 1 > 0:
                    linkContactPairs.append((i, i - 1, 1))
                if i + 1 < numLinks:
                    linkContactPairs.append((i, i + 1, 1))

            # Neighbor-of-neighbor contacts
            for i in range(numLinks):
                if i - 2 > 0:
                    linkContactPairs.append((i, i - 2, 0))
                if i + 2 < numLinks:
                    linkContactPairs.append((i, i + 2, 0))

            system_def["gravity"] = torch.tensor([0.0, -0.5, 0.0], dtype=torch.float32)
            system_def['external_forces']['force_strength_minmax'] = (-200, 200)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0
            system_def['forcedBodyId'] = (numLinks - 1) // 2
            system.shape_param_names = ["Link Width", "Link Thickness", "Link Length", "Links"]

            system.bodies, system.n_bodies = bodiesToStructOfArrays(bodies)

        else:
            raise ValueError("unrecognized system problem_name")

        #
        posFixed = torch.cat(
            [body['x0'] for body in bodies[0:numBodiesFixed]],
            dim=0
        ).reshape(-1).to(dtype=dtype)

        pos = torch.cat(
            [body['x0'] for body in bodies[numBodiesFixed:]],
            dim=0
        ).reshape(-1).to(dtype=dtype)

        mass = torch.tensor(
            [body['mass'] for body in bodies[numBodiesFixed:]],
            dtype=dtype
        ).reshape(-1)
        #
        system.dim = pos.numel()

        system.bodiesRen = bodies
        system.n_bodies = len(bodies)

        #
        system.joints = joint_list
        system.linkContactPairs = torch.tensor(linkContactPairs, dtype=torch.float32)

        system_def['fixed_pos'] = posFixed
        system_def['rest_pos'] = pos
        system_def['init_pos'] = pos.clone()
        system_def['mass'] = mass
        system_def['dim'] = pos.numel()

        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    def to(self, device):
        """Move all tensors in the system to the specified device."""
        # Move bodies struct-of-arrays
        if hasattr(self, 'bodies') and self.bodies is not None:
            for key in self.bodies:
                if isinstance(self.bodies[key], torch.Tensor):
                    self.bodies[key] = self.bodies[key].to(device)
        
        # Move linkContactPairs
        if hasattr(self, 'linkContactPairs') and isinstance(self.linkContactPairs, torch.Tensor):
            self.linkContactPairs = self.linkContactPairs.to(device)
        
        # Move joints (each joint is a dict with tensors)
        if hasattr(self, 'joints'):
            for joint in self.joints:
                for key in joint:
                    if isinstance(joint[key], torch.Tensor):
                        joint[key] = joint[key].to(device)
        
        # Move bodiesRen (list of body dicts)
        if hasattr(self, 'bodiesRen'):
            for body in self.bodiesRen:
                for key in body:
                    if isinstance(body[key], torch.Tensor):
                        body[key] = body[key].to(device)
        
        return self

    @staticmethod
    def get_shape_transform(shape):
        return torch.diag(shape)

    @staticmethod
    def get_shape_transform_batch(shape):
        """
        Converts a batch of shape vectors into batch of diagonal transform matrices.
        Fully tensorized, no Python loops.

        shape: (B, 3) tensor
        Returns: (B, 3, 3) tensor
        """
        B = shape.shape[0]
        # Create a zeros tensor and fill diagonal
        diag_indices = torch.arange(3, device=shape.device)
        transforms = torch.zeros(B, 3, 3, dtype=shape.dtype, device=shape.device)
        transforms[:, diag_indices, diag_indices] = shape
        return transforms  # (B, 3, 3)

    def apply_shape_batch_shared_bodies(self, bodies, transforms):
        """
        More efficient version without clone.

        Args:
            bodies: dict with key 'W' -> (N, V, 4) tensor
            transforms: (B, 3, 3) tensor

        Returns:
            W_all: (B, N, V, 4) tensor
        """
        B = transforms.shape[0]
        W = bodies['W']  # (N, V, 4)
        N, V, _ = W.shape
        device = transforms.device
        dtype = transforms.dtype

        W = W.to(device=device, dtype=dtype)

        # Split into xyz and homogeneous coordinate
        W_xyz = W[..., :3]  # (N, V, 3)
        W_ones = W[..., 3:4]  # (N, V, 1)

        # Transform xyz: (1, N, V, 3) @ (B, 1, 3, 3) → (B, N, V, 3)
        W_xyz_batch = W_xyz.unsqueeze(0)  # (1, N, V, 3)
        transforms_broadcast = transforms.transpose(1, 2).unsqueeze(1)  # (B, 1, 3, 3)

        W_xyz_transformed = torch.matmul(W_xyz_batch, transforms_broadcast)  # (B, N, V, 3)

        # Append ones column
        W_ones_batch = W_ones.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, V, 1)
        W_all = torch.cat([W_xyz_transformed, W_ones_batch], dim=-1)  # (B, N, V, 4)

        return W_all

    def eval_link_contact_energy_batch(self, system_def, transformed_bodies, qRFull, shape_transforms):
        """
        Vectorized evaluation of link contact energies for a batch of transformed bodies.

        Args:
            transformed_bodies: (B, num_bodies, V, 4) - vertices already transformed by shape
            qRFull: (B, num_bodies, 4, 3) - world transforms
            shape_transforms: (B, 4, 4) - shape transformation matrices

        Returns: (B,) total contact energy
        """
        if not hasattr(self, 'linkContactPairs') or self.linkContactPairs.shape[0] == 0:
            return torch.zeros(qRFull.shape[0], device=qRFull.device, dtype=qRFull.dtype)

        device = qRFull.device
        B = qRFull.shape[0]
        pairs = self.linkContactPairs
        b0 = pairs[:, 0].long()
        b1 = pairs[:, 1].long()
        measure_dont_sep = pairs[:, 2]

        le_base = system_def['link_le']
        r1_base = system_def['link_r1']
        r2_base = system_def['link_r2']
        stiffness = system_def['contact_stiffness']

        # Extract per-axis scales from shape_transforms (B, 4, 4)
        shape_3x3 = shape_transforms[:, :3, :3]  # (B, 3, 3)
        scale_xy = torch.sqrt((shape_3x3[:, :, 0] ** 2).sum(dim=1))  # (B,) - x/y scale
        scale_z = torch.sqrt((shape_3x3[:, :, 2] ** 2).sum(dim=1))  # (B,) - z scale

        # Scale collision parameters (capsule aligned with z-axis)
        le = le_base * scale_z.view(B, 1)  # (B, 1) - half-length (z-direction)
        r1 = r1_base * scale_xy.view(B, 1)  # (B, 1) - major radius (xy-plane)
        r2 = r2_base * scale_xy.view(B, 1)  # (B, 1) - minor radius (xy-plane)

        # Gather transforms for all pairs
        q_b0 = qRFull[:, b0, :, :]  # (B, num_pairs, 4, 3)
        q_b1 = qRFull[:, b1, :, :]

        relT = q_b1[:, :, 3, :] - q_b0[:, :, 3, :]
        qRelT = torch.cat([q_b1[:, :, 0:3, :], relT.unsqueeze(2)], dim=2)
        qRel = torch.matmul(qRelT, q_b0[:, :, 0:3, :].transpose(2, 3))

        W1 = transformed_bodies[:, b1, :, :]  # (B, num_pairs, V, 4)
        v10 = torch.matmul(W1, qRel)  # (B, num_pairs, V, 3)

        # --- SDF term (with scaled parameters) ---
        ly = torch.clamp(torch.abs(v10[..., 2]) - le.unsqueeze(-1), min=0.0)
        lxy = torch.sqrt(v10[..., 0] ** 2 + ly ** 2 + 1e-6) - r1.unsqueeze(-1)
        l = torch.sqrt(v10[..., 1] ** 2 + lxy ** 2 + 1e-6) - r2.unsqueeze(-1)
        c = torch.minimum(l, torch.zeros_like(l))
        sdf_term = torch.mean(c ** 2, dim=2)  # (B, num_pairs)

        # --- Inner bbox term (with scaled parameters) ---
        good_bbox_x = r1 - 2 * r2
        good_bbox_y = r2
        good_bbox_z = le + r1 - 2 * r2
        good_bbox = torch.stack([good_bbox_x, good_bbox_y, good_bbox_z], dim=-1)  # (B, 1, 3)
        good_bbox = good_bbox + r2.unsqueeze(-1) / 2

        dist_bbox = torch.sum(torch.clamp(torch.abs(v10) - good_bbox.unsqueeze(1), min=0.0) ** 2, dim=-1)
        min_dist_bbox = torch.min(dist_bbox, dim=2).values
        min_dist_bbox = measure_dont_sep * min_dist_bbox

        pair_penalty = sdf_term + 10 * min_dist_bbox
        total_penalty = torch.sum(pair_penalty, dim=1)
        return stiffness * total_penalty

    #@torch.compile()
    def potential_energy_batch(self, system_def, q_batch, shape):
        shape = shape[:, :min(3, shape.shape[1])]
        B = q_batch.shape[0]
        dtype = q_batch.dtype
        device = q_batch.device

        num_bodies = system_def['mass'].numel() // (4 * 4)

        # reshape q_batch + fixed_pos: (B, num_bodies, 4, 3)
        fixed_pos = system_def['fixed_pos'].reshape(1, -1).float()
        q_full_batch = torch.cat([fixed_pos.expand(B, -1), q_batch], dim=1)
        qRFull = q_full_batch.reshape(B, -1, 4, 3).float()

        joint_energy = torch.zeros(B, dtype=dtype, device=device)

        ## joints
        num_joints = len(self.joints)
        joint_energy = 0
        if num_joints > 0:
            # stack joint info into tensors
            pb0s = torch.stack([j['pos_body0'] for j in self.joints], dim=0).to(dtype=dtype,
                                                                                device=device)  # (num_joints, 3)
            vb0s = torch.stack([j['vec_body0'] for j in self.joints], dim=0).to(dtype=dtype, device=device)
            pb1s = torch.stack([j['pos_body1'] for j in self.joints], dim=0).to(dtype=dtype, device=device)
            vb1s = torch.stack([j['vec_body1'] for j in self.joints], dim=0).to(dtype=dtype, device=device)
            b0ids = torch.tensor([j['body_id0'] for j in self.joints], device=device)
            b1ids = torch.tensor([j['body_id1'] for j in self.joints], device=device)

            one = torch.tensor([1.0], dtype=dtype, device=device)
            zero = torch.tensor([0.0], dtype=dtype, device=device)

            # compute body transforms
            # (B, num_joints, 4)
            vec4_pb0 = torch.cat([pb0s, one.expand(num_joints, 1)], dim=1)  # (num_joints, 4)
            vec4_vb0 = torch.cat([vb0s, zero.expand(num_joints, 1)], dim=1)
            vec4_pb1 = torch.cat([pb1s, one.expand(num_joints, 1)], dim=1)
            vec4_vb1 = torch.cat([vb1s, zero.expand(num_joints, 1)], dim=1)

            # expand to batch
            vec4_pb0 = vec4_pb0.unsqueeze(0).expand(B, -1, -1)  # (B, num_joints, 4)
            vec4_vb0 = vec4_vb0.unsqueeze(0).expand(B, -1, -1)
            vec4_pb1 = vec4_pb1.unsqueeze(0).expand(B, -1, -1)
            vec4_vb1 = vec4_vb1.unsqueeze(0).expand(B, -1, -1)

            # fetch body transforms for all joints
            qRFull_exp = qRFull.unsqueeze(1)  # (B,1,num_bodies,4,3)

            # gather body transforms
            b0ids_exp = b0ids.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, num_joints, 4, 3)
            b1ids_exp = b1ids.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, num_joints, 4, 3)

            qR_b0 = torch.gather(qRFull_exp.expand(-1, num_joints, -1, -1, -1), 2, b0ids_exp)  # (B,num_joints,4,3)
            qR_b1 = torch.gather(qRFull_exp.expand(-1, num_joints, -1, -1, -1), 2, b1ids_exp)  # (B,num_joints,4,3)

            # handle body_id = -1 by replacing with identity (or just keep vecs)
            mask_b0 = (b0ids != -1).unsqueeze(0).expand(B, num_joints)
            mask_b1 = (b1ids != -1).unsqueeze(0).expand(B, num_joints)

            pb0_b = torch.where(mask_b0.unsqueeze(-1), torch.matmul(vec4_pb0.unsqueeze(-2), qR_b0).squeeze(-2),
                                vec4_pb0[:, :, :3])
            vb0_b = torch.where(mask_b0.unsqueeze(-1), torch.matmul(vec4_vb0.unsqueeze(-2), qR_b0).squeeze(-2),
                                vec4_vb0[:, :, :3])
            pb1_b = torch.where(mask_b1.unsqueeze(-1), torch.matmul(vec4_pb1.unsqueeze(-2), qR_b1).squeeze(-2),
                                vec4_pb1[:, :, :3])
            vb1_b = torch.where(mask_b1.unsqueeze(-1), torch.matmul(vec4_vb1.unsqueeze(-2), qR_b1).squeeze(-2),
                                vec4_vb1[:, :, :3])

            # compute distances and alignment
            d = pb1_b - pb0_b  # (B, num_joints, 3)
            dist_squared = torch.sum(d ** 2, dim=-1)
            align = 1.0 - torch.sum(vb0_b * vb1_b, dim=-1)

            joint_energy = 0.5 * 300000.0 * torch.sum(dist_squared, dim=1) + 0.5 * 500.0 * torch.sum(align, dim=1)

        ##

        transform = self.get_shape_transform_batch(shape)

        new_bodies = self.apply_shape_batch_shared_bodies(self.bodies, transform)
        # Contact energy
        contact_energy = self.eval_link_contact_energy_batch(system_def, new_bodies, qRFull, transform)

        # External forces
        ext_force_energy = torch.zeros(B, dtype=dtype, device=device)
        external_forces = system_def['external_forces']
        forcedBodyId = system_def.get('forcedBodyId', 23 // 2)
        for axis, key in enumerate(['force_strength_x', 'force_strength_y', 'force_strength_z']):
            if key in external_forces:
                ext_force_energy += qRFull[:, forcedBodyId, 3, axis] * float(external_forces[key])

        # Gravity
        qR = q_batch.reshape(B, -1, 4, 3)
        massR = system_def['mass'].reshape(1, -1, 4, 4).expand(B, -1, -1, -1)
        gravity = system_def['gravity']
        c_weighted = massR[:, :, 3, 3].unsqueeze(2) * qR[:, :, 3, :]
        gravity_energy = -torch.sum(c_weighted * gravity.unsqueeze(0), dim=(1, 2))

        # Rigid rotation constraint
        rotT = qR[:, :, 0:3, :]
        ide = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        const = torch.matmul(rotT, rotT.transpose(2, 3)) - ide
        rigid_energy = 5000.0 * torch.sum(const ** 2, dim=(1, 2, 3))

        total_energy = joint_energy + gravity_energy + ext_force_energy + rigid_energy + contact_energy
        return total_energy

    def potential_energy(self, system_def, q, shape):
        # q is torch tensor flattened for non-fixed bodies
        qRFull = torch.cat((system_def['fixed_pos'], q), dim=0).reshape(-1, 4, 3)

        joint_energy = torch.tensor(0.0, dtype=q.dtype, device=q.device)

        for j in self.joints:
            pb0 = j['pos_body0'].to(dtype=q.dtype)
            vb0 = j['vec_body0'].to(dtype=q.dtype)

            b0id = j['body_id0']
            if b0id != -1:
                # transform point on body to point in world: append 1 and multiply by 4x3 transform
                vec4 = torch.cat((pb0, torch.tensor([1.0], dtype=q.dtype)))
                pb0 = torch.matmul(vec4, qRFull[b0id])
                vec4v = torch.cat((vb0, torch.tensor([0.0], dtype=q.dtype)))
                vb0 = torch.matmul(vec4v, qRFull[b0id])

            pb1 = j['pos_body1'].to(dtype=q.dtype)
            vb1 = j['vec_body1'].to(dtype=q.dtype)

            b1id = j['body_id1']
            if b1id != -1:
                vec4 = torch.cat((pb1, torch.tensor([1.0], dtype=q.dtype)))
                pb1 = torch.matmul(vec4, qRFull[b1id])
                vec4v = torch.cat((vb1, torch.tensor([0.0], dtype=q.dtype)))
                vb1 = torch.matmul(vec4v, qRFull[b1id])

            d = pb1 - pb0
            dist_squared = torch.sum(d * d)
            joint_stiffness = 300000.0

            align = 1.0 - torch.sum(vb0 * vb1)
            align_stiffness = 500.0

            joint_energy = joint_energy + 0.5 * joint_stiffness * dist_squared + 0.5 * align_stiffness * align

        ###########

        contact_energy = 0.0

        # helper to evaluate energy between a pair of links
        def eval_link_contact_energy(pair):

            le = system_def['link_le']
            r1 = system_def['link_r1']
            r2 = system_def['link_r2']

            b0id = int(pair[0].item() if isinstance(pair[0], torch.Tensor) else pair[0])
            b1id = int(pair[1].item() if isinstance(pair[1], torch.Tensor) else pair[1])
            measure_dont_sep_term = pair[2]

            relT = qRFull[b1id, 3, :] - qRFull[b0id, 3, :]

            qRelT = torch.cat((qRFull[b1id, 0:3, :], relT.unsqueeze(0)), dim=0)

            # transpose instead of inverse (as in your approximation)
            qRel = torch.matmul(qRelT, qRFull[b0id, 0:3, :].transpose(0, 1))

            W1 = self.bodies['W'][b1id]
            v10 = torch.matmul(W1, qRel)

            #### --- SDF TERM --- ####
            ly = torch.clamp(torch.abs(v10[:, 2]) - le, min=0.0)
            lxy = torch.sqrt(v10[:, 0] * v10[:, 0] + ly * ly + 1e-6) - r1
            l = torch.sqrt(v10[:, 1] * v10[:, 1] + lxy * lxy + 1e-6) - r2
            c = torch.minimum(l, torch.tensor(0.0, dtype=l.dtype, device=l.device))
            sdf_nocollision_dist = torch.mean(c * c)

            #### --- INNER BBOX TERM --- ####
            good_bbox = torch.tensor([r1 - 2 * r2, r2, le + r1 - 2 * r2],
                                     dtype=v10.dtype, device=v10.device)
            good_bbox = good_bbox + r2 / 2

            dist_from_bbox = torch.sum(torch.square(torch.clamp(torch.abs(v10) - good_bbox, min=0.0)), dim=-1)
            min_dist_from_bbox = torch.min(dist_from_bbox)

            # enable only if requested
            min_dist_from_bbox = measure_dont_sep_term * min_dist_from_bbox

            combined_penalty = sdf_nocollision_dist + 10 * min_dist_from_bbox

            return system_def['contact_stiffness'] * combined_penalty

        if self.linkContactPairs.shape[0] > 0:
            # torch equivalent of vmap → list comprehension
            link_energies = torch.stack([
                eval_link_contact_energy(pair) for pair in self.linkContactPairs
            ])
            contact_energy += torch.sum(link_energies)

        ###########

        contact_energy = torch.tensor(0.0, dtype=q.dtype, device=q.device)
        ext_force_energy = torch.tensor(0.0, dtype=q.dtype, device=q.device)

        external_forces = system_def['external_forces']
        forcedBodyId = 23 // 2

        if 'force_strength_x' in external_forces:
            ext_force_energy = ext_force_energy + torch.sum(qRFull[forcedBodyId, 3, 0] * float(external_forces['force_strength_x']))

        if 'force_strength_y' in external_forces:
            ext_force_energy = ext_force_energy + torch.sum(qRFull[forcedBodyId, 3, 1] * float(external_forces['force_strength_y']))

        if 'force_strength_z' in external_forces:
            ext_force_energy = ext_force_energy + torch.sum(qRFull[forcedBodyId, 3, 2] * float(external_forces['force_strength_z']))

        qR = q.reshape(-1, 4, 3)

        massR = system_def['mass'].reshape(-1, 4, 4)

        gravity = system_def["gravity"].to(dtype=q.dtype)
        c_weighted = massR[:, 3, 3].unsqueeze(1) * qR[:, 3, :]

        gravity_energy = -torch.sum(c_weighted * gravity.unsqueeze(0))

        rotT = qR[:, 0:3, :]
        # rotT expected shape (n,3,3) => rotT @ rotT.transpose(1,2)
        ide = torch.stack([torch.eye(3, dtype=q.dtype, device=q.device)] * rotT.shape[0], dim=0)

        const = torch.matmul(rotT, rotT.transpose(1, 2)) - ide
        rigid_energy = 5000.0 * torch.sum(const * const)

        return joint_energy + gravity_energy + ext_force_energy + rigid_energy + contact_energy

    # @torch.compile()
    def kinetic_energy_batch(self, system_def, qdot_batch, shape):
        B = qdot_batch.shape[0]
        num_bodies = system_def['mass'].numel() // (4 * 4)  # total number of bodies

        qdotR = qdot_batch.reshape(B, num_bodies, 4, 3)
        massR = system_def['mass'].reshape(1, num_bodies, 4, 4).expand(B, -1, -1, -1)

        A = torch.matmul(torch.matmul(qdotR.transpose(2, 3), massR), qdotR)
        tr = A.diagonal(dim1=-2, dim2=-1).sum(-1)

        energies = 0.5 * tr.sum(dim=1)

        return energies

    def kinetic_energy(self, system_def, q_dot):
        print(q_dot.shape)
        q_dotR = q_dot.reshape(-1, 4, 3)
        massR = system_def['mass'].reshape(-1, 4, 4)

        # A = q_dotR.transpose(1,2) @ massR @ q_dotR  -> batch multiplication
        A = torch.matmul(torch.matmul(q_dotR.transpose(1, 2), massR), q_dotR)
        # trace for each batch
        tr = torch.einsum('bii->b', A)
        return 0.5 * torch.sum(tr)

    def build_system_ui(self, system_def):
        if psim.TreeNode("system UI"):
            psim.TextUnformatted("External forces:")

            if "force_strength_x" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_x",
                                              float(system_def['external_forces']['force_strength_x']), low, high)
                system_def['external_forces']['force_strength_x'] = float(new_val)

            if "force_strength_y" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_y",
                                              float(system_def['external_forces']['force_strength_y']), low, high)
                system_def['external_forces']['force_strength_y'] = float(new_val)

            if "force_strength_z" in system_def["external_forces"]:
                low, high = system_def['external_forces']['force_strength_minmax']
                _, new_val = psim.SliderFloat("force_strength_z",
                                              float(system_def['external_forces']['force_strength_z']), low, high)
                system_def['external_forces']['force_strength_z'] = float(new_val)

            psim.TreePop()

    def visualize(self, system_def, x, shape):
        """
        x: non-fixed DOF positions
        shape: (3,) shape parameters for scaling/stretching the mesh
        """
        # full qR
        xr = torch.cat((system_def['fixed_pos'].cpu(), x.cpu()), dim=0).reshape(-1, 4, 3)

        # ---- get the shape transform matrix (3x3) ----
        # get_shape_transform_batch expects (B,3)
        shapeb = shape[:3].unsqueeze(0)  # (1,3)
        T = self.get_shape_transform_batch(shapeb)[0]  # (3,3)
        T_np = T.detach().cpu().numpy()

        k = int(shape[3].clamp(0, 1) * self.n_bodies)
        print(k)
        for bid in range(self.n_bodies):
            if bid >= k and bid != 0:
                try:
                    ps.get_surface_mesh(f"body{bid}").set_enabled(False)
                except Exception as e:
                    pass
                continue
            # original W is numpy, shape (V,4)
            W = self.bodiesRen[bid]['W']  # numpy (V,4)

            # full rigid transform for this body
            xr_bid = xr[bid].detach().cpu().numpy()  # (4,3)

            # ----- apply shape transform to the vertex positions -----
            # W[:, :3] is (V,3), apply T (3x3)
            W_xyz = W[:, :3] @ T_np.T  # (V,3)

            # recombine with homogeneous coord
            W_scaled = np.concatenate([W_xyz, W[:, 3:4]], axis=1)  # (V,4)

            # local → world
            v = W_scaled @ xr_bid  # (V,3)

            # faces
            f = np.array(self.bodiesRen[bid]['f'])

            # register mesh
            ps_body = ps.register_surface_mesh(f"body{bid}", v, f)
            ps_body.set_enabled(True)
            ps_body.set_transform(np.identity(4))

        return ps_body

    def export(self, system_def, x, prefix=""):
        pass

    def visualize_set_nice_view(self, system_def, x):
        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))
