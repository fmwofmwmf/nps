import torch
import numpy as np
import os
import copy

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
    vol = np.nan_to_num(vol)

    c = np.sum(vol[:, None] * v, axis=0) / np.sum(vol)
    v = v - c

    W = np.c_[v, np.ones(v.shape[0])]
    mass = np.matmul(W.T, vol[:, None] * W) * density

    x0 = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       c], dtype=dtype)

    body = {'v': v, 'f': f, 'W': W, 'x0': x0, 'mass': mass}
    return body


def make_joint(b0, b1, bodies, joint_pos_world, joint_vec_world):
    """Create joint with world-space reference for consistent transforms."""

    # Compute initial local positions
    pb0 = joint_pos_world.clone()
    vb0 = joint_vec_world.clone()
    pb1 = joint_pos_world.clone()
    vb1 = joint_vec_world.clone()

    if b0 != -1:
        c0 = bodies[b0]['x0'][3, :].clone()
        pb0 = pb0 - c0
    if b1 != -1:
        c1 = bodies[b1]['x0'][3, :].clone()
        pb1 = pb1 - c1

    joint = {
        'body_id0': b0, 'body_id1': b1,
        'pos_body0': pb0, 'pos_body1': pb1,
        'vec_body0': vb0, 'vec_body1': vb1,
        'pos_body0_original': pb0.clone(),
        'pos_body1_original': pb1.clone(),
    }
    return joint


def bodiesToStructOfArrays(bodies, dtype=torch.float64):
    x0_arr, mass_arr, W_arr = [], [], []
    for b in bodies:
        x0_arr.append(b['x0'].to(dtype=dtype))
        mass_arr.append(torch.tensor(b['mass'], dtype=dtype))
        W_arr.append(torch.tensor(b['W'], dtype=dtype))

    out_struct = {
        'x0': torch.stack(x0_arr, dim=0),
        'mass': torch.stack(mass_arr, dim=0),
        'W': W_arr,  # List
    }
    return out_struct, len(bodies)


class Rigid3DSystem:
    def __init__(self):
        self.system_name = "Rigid3d"
        self.problem_name = None
        self.shape_param_names = []
        self.shape_transform_body_indices = None
        self.per_body_scaling = False
        self.per_body_shape_indices = None
        self.joints = []
        self.original_joints = None
        self.bodies = None
        self.bodiesRen = None
        self.n_bodies = 0
        self.dim = 0
        self.linkContactPairs = None
        self.body_ID = None
        self.cond_dim = 0

    @staticmethod
    def construct(problem_name, dtype=torch.float64):
        system_def = {}
        system = Rigid3DSystem()

        system.problem_name = str(problem_name)
        system.shape_param_names = []
        system.shape_transform_body_indices = None
        system.per_body_scaling = False
        system.per_body_shape_indices = None

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
            # NEW: Per-body scaling configuration
            system.per_body_scaling = True
            system.per_body_shape_indices = [0, 2, 4]  # shape[i] scales body[i]
            system.shape_param_names = ["Body0 Scale", "Body2 Scale", "Body4 Scale"]

            # Load bodies without initial scaling (scale=1.0)
            for obj in ["klann-red.obj", "klann-purple.obj", "klann-brown.obj",
                        "klann-distal.obj", "klann-top.obj"]:
                bodies.append(make_body(os.path.join(".", "data", obj), 1000, np.identity(3), dtype))

            # Create joints (unchanged)
            joint_list.append(make_joint(0, -1, bodies, torch.tensor([0, 0.08, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(0, 1, bodies, torch.tensor([-0.046622, 0.097594, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(1, 2, bodies, torch.tensor([-0.1736, 0.11205, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(1, 3, bodies, torch.tensor([-0.31194, 0.16654, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(4, -1, bodies, torch.tensor([-0.13, 0.1875, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(2, -1, bodies, torch.tensor([-0.13, 0.045, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))
            joint_list.append(make_joint(4, 3, bodies, torch.tensor([-0.21981, 0.25102, 0.044], dtype=dtype),
                                         torch.tensor([0, 0.0, 1.0], dtype=dtype)))

            system_def["gravity"] = torch.tensor([0.0, -0.98, 0.0], dtype=dtype)
            system_def['external_forces']['force_strength_minmax'] = (-10, 10)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0

        elif problem_name == 'stewart':
            # Unified scaling for specific bodies (arms only)
            scale = system.get_shape_transform(torch.tensor([5, 5, 5], dtype=dtype))
            scalenp = scale.cpu().numpy()

            obj_files = ["stewart-base.obj", "stewart-arm1.obj", "stewart-arm2.obj",
                        "stewart-arm3.obj", "stewart-arm4.obj", "stewart-arm5.obj", "stewart-arm6.obj",
                        "stewart-strut1.obj", "stewart-strut2.obj", "stewart-strut3.obj",
                        "stewart-strut4.obj", "stewart-strut5.obj", "stewart-strut6.obj", "stewart-top.obj"]

            for obj in obj_files:
                bodies.append(make_body(os.path.join(".", "data", obj), 1000, scalenp, dtype))

            numBodiesFixed = 1
            system.shape_transform_body_indices = list(range(7, 13))  # Transform arms only
            system.shape_param_names = ["Length"]

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
            a = scale @ torch.tensor([-0.018, 0.0215, -0.044856], dtype=dtype)
            b = scale @ torch.tensor([-0.047847, 0.0215, 0.00684], dtype=dtype)
            v = torch.tensor([0, 0.0, 1.0], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb
            Rv = R @ v
            RRv = R @ Rv

            # base to strut
            joint_list.append(make_joint(0, 1, bodies, a, v))
            joint_list.append(make_joint(0, 2, bodies, b, Rv))
            joint_list.append(make_joint(0, 3, bodies, Ra, Rv))
            joint_list.append(make_joint(0, 4, bodies, Rb, RRv))
            joint_list.append(make_joint(0, 5, bodies, RRa, RRv))
            joint_list.append(make_joint(0, 6, bodies, RRb, v))

            #### [x,z,-y]
            a = scale @ torch.tensor([-0.003, 0.0215, -0.051856], dtype=dtype)
            b = scale @ torch.tensor([-0.046409, 0.0215, 0.02333], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb

            # strut to arm
            joint_list.append(make_joint(1, 7, bodies, a, 0.01 * v))
            joint_list.append(make_joint(2, 8, bodies, b, 0.01 * Rv))
            joint_list.append(make_joint(3, 9, bodies, Ra, 0.01 * Rv))
            joint_list.append(make_joint(4, 10, bodies, Rb, 0.01 * RRv))
            joint_list.append(make_joint(5, 11, bodies, RRa, 0.01 * RRv))
            joint_list.append(make_joint(6, 12, bodies, RRb, 0.01 * v))

            #### [x,z,-y]
            a = scale @ torch.tensor([-0.032159, 0.082222, -0.022686], dtype=dtype)
            b = scale @ torch.tensor([-0.035712, 0.082222, -0.016488], dtype=dtype)
            v = Rh @ torch.tensor([0, 0.0, 1.0], dtype=dtype)

            Ra = R @ a
            RRa = R @ Ra
            Rb = R @ b
            RRb = R @ Rb
            Rv = R @ v
            RRv = R @ Rv

            # arm to platform
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
            # Unified scaling for all bodies
            numLinks = 24
            link_separation = 0.0491

            system_def['link_le'] = 0.013
            system_def['link_r1'] = 0.02
            system_def['link_r2'] = 0.009

            for i in range(numLinks):
                scale = system.get_shape_transform(torch.tensor([1, 1, 1]))
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
            system.shape_transform_body_indices = None  # Transform all bodies

            system_def["gravity"] = torch.tensor([0.0, -0.5, 0.0], dtype=torch.float32)
            system_def['external_forces']['force_strength_minmax'] = (-200, 200)
            system_def['external_forces']['force_strength_x'] = 0.0
            system_def['external_forces']['force_strength_y'] = 0.0
            system_def['external_forces']['force_strength_z'] = 0.0
            system_def['forcedBodyId'] = (numLinks - 1) // 2
            system.shape_param_names = ["Link Width", "Link Thickness", "Link Length"]
        else:
            raise ValueError("unrecognized system problem_name")

        system.original_joints = copy.deepcopy(joint_list)
        system.joints = joint_list

        system.bodies, system.n_bodies = bodiesToStructOfArrays(bodies)

        if numBodiesFixed > 0:
            posFixed = torch.cat([body['x0'] for body in bodies[0:numBodiesFixed]], dim=0)
            posFixed = posFixed.reshape(-1).to(dtype=dtype)
        else:
            # create empty tensor of correct dtype
            posFixed = torch.empty(0, dtype=dtype)
        pos = torch.cat([b['x0'] for b in bodies[numBodiesFixed:]], dim=0).reshape(-1).to(dtype=dtype)
        mass = torch.tensor([b['mass'] for b in bodies[numBodiesFixed:]], dtype=dtype).reshape(-1)

        system.dim = pos.numel()
        system.bodiesRen = bodies

        system.linkContactPairs = torch.tensor(linkContactPairs, dtype=dtype)

        system_def['fixed_pos'] = posFixed
        system_def['rest_pos'] = pos
        system_def['init_pos'] = pos.clone()
        system_def['mass'] = mass
        system_def['dim'] = pos.numel()
        system_def['interesting_states'] = system_def['init_pos'].unsqueeze(0)

        return system, system_def

    def to(self, device):
        """Move all tensors to device."""
        if hasattr(self, 'bodies') and self.bodies is not None:
            for key in self.bodies:
                if isinstance(self.bodies[key], torch.Tensor):
                    self.bodies[key] = self.bodies[key].to(device)

        if hasattr(self, 'linkContactPairs') and isinstance(self.linkContactPairs, torch.Tensor):
            self.linkContactPairs = self.linkContactPairs.to(device)

        if hasattr(self, 'joints'):
            for joint in self.joints:
                for key in joint:
                    if isinstance(joint[key], torch.Tensor):
                        joint[key] = joint[key].to(device)

        if hasattr(self, 'original_joints'):
            for joint in self.original_joints:
                for key in joint:
                    if isinstance(joint[key], torch.Tensor):
                        joint[key] = joint[key].to(device)

        if hasattr(self, 'bodiesRen'):
            for body in self.bodiesRen:
                for key in body:
                    if isinstance(body[key], torch.Tensor):
                        body[key] = body[key].to(device)

        return self

    def get_shape_transform(self, shape):
        return torch.diag(shape)

    @staticmethod
    def get_shape_transform_batch(shape):
        """Convert batch of shape vectors to diagonal transform matrices."""
        B = shape.shape[0]
        diag_indices = torch.arange(3, device=shape.device)
        transforms = torch.zeros(B, 3, 3, device=shape.device)
        transforms[:, diag_indices, diag_indices] = shape
        return transforms

    def apply_shape_batch_shared_bodies(self, bodies, transforms, body_indices=None):
        """Apply shape transforms to specific bodies only.
        transforms can be:
            - single tensor (B, 3, 3): same transform for all bodies
            - list of tensors: different transform per body in body_indices
        """
        # Determine batch size and format
        if isinstance(transforms, list):
            B = transforms[0].shape[0]
            transforms_list = transforms
            per_body = True
        else:
            B = transforms.shape[0]
            transforms_list = transforms
            per_body = False

        W_list = bodies['W']
        device = (transforms_list[0].device if per_body else transforms_list.device)
        dtype = (transforms_list[0].dtype if per_body else transforms_list.dtype)

        if body_indices is None:
            body_indices = list(range(len(W_list)))

        W_transformed_list = []
        for i, W in enumerate(W_list):
            W = W.to(device=device, dtype=dtype)
            W_xyz = W[..., :3]  # (V, 3)
            W_ones = W[..., 3:4]  # (V, 1)

            if i in body_indices:
                if per_body:
                    # Get the specific transform for this body
                    transform = transforms_list[body_indices.index(i)]  # (B, 3, 3)
                    # (1, V, 3) @ (B, 3, 3) -> (B, V, 3)
                    W_xyz_transformed = torch.matmul(W_xyz.unsqueeze(0), transform.transpose(1, 2))
                else:
                    # Unified transform for all bodies
                    transform = transforms_list  # (B, 3, 3)
                    # (1, V, 3) @ (B, 3, 3) -> (B, V, 3)
                    W_xyz_transformed = torch.matmul(W_xyz.unsqueeze(0), transform.transpose(1, 2))
            else:
                # No transform: repeat original vertices B times
                W_xyz_transformed = W_xyz.unsqueeze(0).expand(B, -1, -1)

            # Add homogeneous coordinate (unchanged)
            W_ones_batch = W_ones.unsqueeze(0).expand(B, -1, -1)
            W_transformed = torch.cat([W_xyz_transformed, W_ones_batch], dim=-1)
            W_transformed_list.append(W_transformed)

        return W_transformed_list

    def eval_link_contact_energy_batch(self, system_def, transformed_bodies, qRFull):
        """Vectorized link contact energy."""
        if not hasattr(self, 'linkContactPairs') or self.linkContactPairs.shape[0] == 0:
            return torch.zeros(qRFull.shape[0], device=qRFull.device, dtype=qRFull.dtype)

        device = qRFull.device
        B = qRFull.shape[0]
        pairs = self.linkContactPairs
        b0_ids = pairs[:, 0].long()
        b1_ids = pairs[:, 1].long()
        measure_dont_sep = pairs[:, 2]

        le_base = system_def['link_le']
        r1_base = system_def['link_r1']
        r2_base = system_def['link_r2']
        stiffness = system_def['contact_stiffness']

        le = le_base
        r1 = r1_base
        r2 = r2_base

        pair_energies = []
        for pair_idx in range(pairs.shape[0]):
            b0 = b0_ids[pair_idx]
            b1 = b1_ids[pair_idx]

            q_b0 = qRFull[:, b0, :, :]
            q_b1 = qRFull[:, b1, :, :]

            W1 = transformed_bodies[b1]

            relT = q_b1[:, 3, :] - q_b0[:, 3, :]
            qRelT = torch.cat([q_b1[:, 0:3, :], relT.unsqueeze(1)], dim=1)
            qRel = torch.matmul(qRelT, q_b0[:, 0:3, :].transpose(1, 2))

            v10 = torch.matmul(W1, qRel)

            ly = torch.clamp(torch.abs(v10[..., 2]) - le, min=0.0)
            lxy = torch.sqrt(v10[..., 0] ** 2 + ly ** 2 + 1e-6) - r1
            l = torch.sqrt(v10[..., 1] ** 2 + lxy ** 2 + 1e-6) - r2
            c = torch.minimum(l, torch.zeros_like(l))
            sdf_term = torch.mean(c ** 2, dim=1)

            good_bbox = torch.tensor([r1 - 2 * r2, r2, le + r1 - 2 * r2],
                                     device=device, dtype=qRFull.dtype)
            good_bbox = good_bbox + r2 / 2
            dist_bbox = torch.sum(torch.clamp(torch.abs(v10) - good_bbox, min=0.0) ** 2, dim=-1)
            min_dist_bbox = torch.min(dist_bbox, dim=1).values
            min_dist_bbox = measure_dont_sep[pair_idx] * min_dist_bbox

            pair_penalty = sdf_term + 10 * min_dist_bbox
            pair_energies.append(pair_penalty)

        total_penalty = torch.stack(pair_energies, dim=1).sum(dim=1)
        return stiffness * total_penalty

    def potential_energy_batch(self, system_def, q_batch, shape, shape_body_indices=None):
        """Fully vectorized potential energy computation."""
        B, dtype, device = q_batch.shape[0], q_batch.dtype, q_batch.device
        num_bodies = system_def['mass'].numel() // (4 * 4)

        # --- reshape state ----------------------------------------------------
        fixed_pos = system_def['fixed_pos'].reshape(1, -1).to(dtype=dtype, device=device)
        q_full_batch = torch.cat([fixed_pos.expand(B, -1), q_batch], dim=1)
        qRFull = q_full_batch.reshape(B, -1, 4, 3)

        # --- shape transforms -------------------------------------------------
        if shape_body_indices is None:
            shape_body_indices = self.shape_transform_body_indices

        # Always compute base shape_transforms first
        shape_transforms = self.get_shape_transform_batch(shape.to(device=device, dtype=dtype))

        # Handle per-body scaling for bodies only
        if getattr(self, 'per_body_scaling', False):
            # Generate per-body transforms for body vertices
            transforms_list = []
            for i in range(len(self.per_body_shape_indices)):
                scale_factor = shape[:, i]  # (B,)
                transform = self.get_shape_transform_batch(scale_factor.unsqueeze(1).expand(-1, 3))
                transforms_list.append(transform)

            # For bodies: use the list of per-body transforms
            transformed_bodies = self.apply_shape_batch_shared_bodies(
                self.bodies, transforms_list, self.per_body_shape_indices)
        else:
            # Original unified scaling for bodies
            transformed_bodies = self.apply_shape_batch_shared_bodies(
                self.bodies, shape_transforms, shape_body_indices)

        # --- joint data (tensor-only, no dicts) -------------------------------
        nJ = len(self.original_joints)
        joint_energy = torch.zeros(B, dtype=dtype, device=device)

        if nJ > 0:
            # Body IDs
            b0_ids = torch.tensor([j['body_id0'] for j in self.original_joints], device=device)
            b1_ids = torch.tensor([j['body_id1'] for j in self.original_joints], device=device)

            # Original local positions
            local0 = torch.stack([j['pos_body0_original'] for j in self.original_joints]).to(dtype)
            local1 = torch.stack([j['pos_body1_original'] for j in self.original_joints]).to(dtype)

            # For per-body scaling, apply different transforms per joint
            if getattr(self, 'per_body_scaling', False):
                # Initialize output tensors
                new0 = torch.zeros(B, nJ, 3, device=device, dtype=dtype)
                new1 = torch.zeros(B, nJ, 3, device=device, dtype=dtype)

                # Apply per-body transforms
                for i, body_id in enumerate(self.per_body_shape_indices):
                    transform = transforms_list[i]  # (B, 3, 3)

                    # Find joints on this body
                    mask0 = (b0_ids == body_id).unsqueeze(0).unsqueeze(-1)  # (1, nJ, 1)
                    mask1 = (b1_ids == body_id).unsqueeze(0).unsqueeze(-1)

                    # Apply transform
                    T = transform.unsqueeze(1)  # (B, 1, 3, 3)
                    transformed0 = (local0.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
                    transformed1 = (local1.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)

                    # Update where applicable
                    new0 = torch.where(mask0, transformed0, new0)
                    new1 = torch.where(mask1, transformed1, new1)
            else:
                # Unified transform for all joints
                T = shape_transforms.unsqueeze(1)  # (B, 1, 3, 3)
                new0 = (local0.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
                new1 = (local1.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)

            # Vectors
            vec0 = torch.stack([j['vec_body0'] for j in self.original_joints]).to(dtype)
            vec1 = torch.stack([j['vec_body1'] for j in self.original_joints]).to(dtype)

            # Build 4-vectors
            one = torch.ones((B, nJ, 1), dtype=dtype, device=device)  # (B, nJ, 1)
            zero = torch.zeros((B, nJ, 1), dtype=dtype, device=device)

            pb0 = torch.cat([new0, one], dim=-1)  # (B, nJ, 3) + (B, nJ, 1) -> (B, nJ, 4)
            pb1 = torch.cat([new1, one], dim=-1)
            vb0 = torch.cat([vec0.expand(B, -1, -1), zero], dim=-1)
            vb1 = torch.cat([vec1.expand(B, -1, -1), zero], dim=-1)

            # Gather body transforms (one-shot advanced indexing)
            batch_idx = torch.arange(B, device=device).unsqueeze(1)
            qR_b0 = qRFull[batch_idx, b0_ids]  # (B, nJ, 4, 3)
            qR_b1 = qRFull[batch_idx, b1_ids]

            # Transform to world space
            mask0 = (b0_ids != -1).unsqueeze(0).unsqueeze(-1)
            mask1 = (b1_ids != -1).unsqueeze(0).unsqueeze(-1)

            pb0_w = torch.where(mask0, (pb0.unsqueeze(-2) @ qR_b0).squeeze(-2), new0)
            pb1_w = torch.where(mask1, (pb1.unsqueeze(-2) @ qR_b1).squeeze(-2), new1)
            vb0_w = torch.where(mask0, (vb0.unsqueeze(-2) @ qR_b0).squeeze(-2), vec0.expand(B, -1, -1))
            vb1_w = torch.where(mask1, (vb1.unsqueeze(-2) @ qR_b1).squeeze(-2), vec1.expand(B, -1, -1))

            # Compute joint energy (vectorized across all joints)
            d = pb1_w - pb0_w
            dist_sq = (d * d).sum(dim=-1)
            align = 1.0 - (vb0_w * vb1_w).sum(dim=-1)

            joint_energy = 0.5 * 300000.0 * dist_sq.sum(dim=1) + 0.5 * 500.0 * align.sum(dim=1)

        # --- remaining energy terms (unchanged) ----------------------------
        contact_energy = self.eval_link_contact_energy_batch(system_def, transformed_bodies, qRFull)

        ext_force_energy = torch.zeros(B, dtype=dtype, device=device)
        external_forces = system_def['external_forces']
        forcedBodyId = system_def.get('forcedBodyId', num_bodies // 2)
        for axis, key in enumerate(['force_strength_x', 'force_strength_y', 'force_strength_z']):
            if key in external_forces:
                ext_force_energy += qRFull[:, forcedBodyId, 3, axis] * float(external_forces[key])

        qR = q_batch.reshape(B, -1, 4, 3)
        massR = system_def['mass'].reshape(1, -1, 4, 4).expand(B, -1, -1, -1)
        gravity = system_def['gravity']
        c_weighted = massR[:, :, 3, 3].unsqueeze(2) * qR[:, :, 3, :]
        gravity_energy = -torch.sum(c_weighted * gravity.unsqueeze(0), dim=(1, 2))

        rotT = qR[:, :, 0:3, :]
        ide = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        const = torch.matmul(rotT, rotT.transpose(2, 3)) - ide
        rigid_energy = 5000.0 * torch.sum(const ** 2, dim=(1, 2, 3))

        return joint_energy + gravity_energy + ext_force_energy + rigid_energy + contact_energy

    def potential_energy(self, system_def, q, shape, shape_body_indices=None):
        """Non-batched version."""
        B = 1
        q_batch = q.unsqueeze(0)
        shape_batch = shape.unsqueeze(0) if shape.ndim == 1 else shape
        energy_batch = self.potential_energy_batch(system_def, q_batch, shape_batch, shape_body_indices)
        return energy_batch[0]

    def kinetic_energy_batch(self, system_def, qdot_batch, shape):
        B = qdot_batch.shape[0]
        num_bodies = system_def['mass'].numel() // (4 * 4)

        qdotR = qdot_batch.reshape(B, num_bodies, 4, 3)
        massR = system_def['mass'].reshape(1, num_bodies, 4, 4).expand(B, -1, -1, -1)

        A = torch.matmul(torch.matmul(qdotR.transpose(2, 3), massR), qdotR)
        tr = A.diagonal(dim1=-2, dim2=-1).sum(-1)

        energies = 0.5 * tr.sum(dim=1)
        return energies

    def kinetic_energy(self, system_def, q_dot):
        q_dotR = q_dot.reshape(-1, 4, 3)
        massR = system_def['mass'].reshape(-1, 4, 4)

        A = torch.matmul(torch.matmul(q_dotR.transpose(1, 2), massR), q_dotR)
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

    def _transform_joint_positions(self, shape_transforms, body_indices):
        """Handle both per-body (list) and unified (tensor) transforms."""

        if len(self.original_joints) == 0:
            return [[]]
        B = shape_transforms[0].shape[0] if isinstance(shape_transforms, list) else shape_transforms.shape[0]
        device = shape_transforms[0].device if isinstance(shape_transforms, list) else shape_transforms.device
        dtype = shape_transforms[0].dtype if isinstance(shape_transforms, list) else shape_transforms.dtype
        nJ = len(self.original_joints)

        b0_ids = torch.tensor([j['body_id0'] for j in self.original_joints], device=device)
        b1_ids = torch.tensor([j['body_id1'] for j in self.original_joints], device=device)
        local0 = torch.stack([j['pos_body0_original'] for j in self.original_joints]).to(dtype)
        local1 = torch.stack([j['pos_body1_original'] for j in self.original_joints]).to(dtype)

        # Initialize output tensors
        new0 = torch.zeros(B, nJ, 3, device=device, dtype=dtype)
        new1 = torch.zeros(B, nJ, 3, device=device, dtype=dtype)

        if isinstance(shape_transforms, list):
            # Per-body transforms
            for i, body_id in enumerate(body_indices):
                transform = shape_transforms[i]  # (B, 3, 3)
                mask0 = (b0_ids == body_id).unsqueeze(0).unsqueeze(-1)
                mask1 = (b1_ids == body_id).unsqueeze(0).unsqueeze(-1)
                T = transform.unsqueeze(1)
                transformed0 = (local0.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
                transformed1 = (local1.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
                new0 = torch.where(mask0, transformed0, new0)
                new1 = torch.where(mask1, transformed1, new1)
        else:
            # Unified transform
            T = shape_transforms.unsqueeze(1)
            transformed0 = (local0.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
            transformed1 = (local1.unsqueeze(0).unsqueeze(2) @ T.transpose(-1, -2)).squeeze(-2)
            if body_indices is not None:
                idx = torch.as_tensor(body_indices, device=device)
                mask0 = torch.isin(b0_ids, idx).unsqueeze(0).unsqueeze(-1)
                mask1 = torch.isin(b1_ids, idx).unsqueeze(0).unsqueeze(-1)
                new0 = torch.where(mask0, transformed0, local0.unsqueeze(0))
                new1 = torch.where(mask1, transformed1, local1.unsqueeze(0))
            else:
                new0, new1 = transformed0, transformed1

        # Return as list-of-dicts for backward compatibility
        return [
            [{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in joint.items() if
              not k.endswith('_original')}
             for joint in self.original_joints]
            for _ in range(B)
        ]

    def visualize(self, system_def, x, shape, shape_body_indices=None):
        """Visualize with per-body or unified scaling."""
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        xr = torch.cat((system_def['fixed_pos'].to(dtype), x.to(dtype)), dim=0).reshape(-1, 4, 3)
        if (shape_body_indices is None):
            shape_body_indices = self.shape_transform_body_indices
        device = xr.device
        shape_transforms = self.get_shape_transform_batch(shape.to(device=device, dtype=dtype))

        # Transform joints first (same logic as potential_energy_batch)
        if getattr(self, 'per_body_scaling', False):
            # Generate per-body transforms for joints
            transforms_list = []
            for i in range(len(self.per_body_shape_indices)):
                scale_factor = shape[i] if shape.ndim == 1 else shape[:, i]
                if scale_factor.ndim == 0:
                    scale_factor = scale_factor.unsqueeze(0)
                transform = self.get_shape_transform_batch(scale_factor.unsqueeze(0).expand(1, 3))
                transforms_list.append(transform)
            transformed_joints_batch = self._transform_joint_positions(transforms_list, self.per_body_shape_indices)
        else:
            # Unified transform
            transformed_joints_batch = self._transform_joint_positions(shape_transforms, shape_body_indices)

        # Transform bodies (same logic as potential_energy_batch)
        if getattr(self, 'per_body_scaling', False):
            # Generate per-body transforms for bodies
            transforms_bodies = []
            for i in range(len(self.per_body_shape_indices)):
                scale_factor = shape[i] if shape.ndim == 1 else shape[:, i]
                if scale_factor.ndim == 0:
                    scale_factor = scale_factor.unsqueeze(0)
                transform = self.get_shape_transform_batch(scale_factor.unsqueeze(0).expand(1, 3))[0]
                transforms_bodies.append(transform.cpu().numpy())

            transformed_bodies = self.apply_shape_batch_shared_bodies(
                self.bodies, transforms_list, self.per_body_shape_indices)
        else:
            # Unified transform
            transformed_bodies = self.apply_shape_batch_shared_bodies(
                self.bodies, shape_transforms, shape_body_indices)
            T_single = shape_transforms.mean(dim=0).cpu().numpy()

        # Visualize bodies
        for bid in range(self.n_bodies):
            body = self.bodiesRen[bid]
            W_np = body['W']
            f_np = body['f']

            if self.per_body_scaling and bid in self.per_body_shape_indices:
                idx = self.per_body_shape_indices.index(bid)
                W_xyz = W_np[:, :3] @ transforms_bodies[idx].T
            elif not self.per_body_scaling and (shape_body_indices is None or bid in shape_body_indices):
                W_xyz = W_np[:, :3] @ T_single.T
            else:
                W_xyz = W_np[:, :3]

            W_scaled = np.concatenate([W_xyz, W_np[:, 3:4]], axis=1)
            xr_bid = xr[bid].cpu().numpy()
            v_world = W_scaled @ xr_bid
            ps.register_surface_mesh(f"body{bid}", v_world, f_np)

        # Visualize joints
        joint_positions_body0 = []
        joint_positions_body1 = []
        joint_body0_ids = []
        joint_body1_ids = []
        joint_line_positions = []
        joint_line_edges = []

        for joint_idx, joint in enumerate(transformed_joints_batch[0]):
            b0id = joint['body_id0']
            b1id = joint['body_id1']

            pos0_local = joint['pos_body0'].to(device, dtype)
            pos1_local = joint['pos_body1'].to(device, dtype)

            # Transform to world space
            pos0_world = self._local_to_world(pos0_local, xr, b0id, device, dtype)
            pos1_world = self._local_to_world(pos1_local, xr, b1id, device, dtype)

            # Store points
            joint_positions_body0.append(pos0_world.cpu().numpy())
            joint_positions_body1.append(pos1_world.cpu().numpy())
            joint_body0_ids.append(b0id if b0id != -1 else -1)
            joint_body1_ids.append(b1id if b1id != -1 else -1)

            # Store line segment (2 points)
            joint_line_positions.append(pos0_world.cpu().numpy())
            joint_line_positions.append(pos1_world.cpu().numpy())
            joint_line_edges.append([2 * joint_idx, 2 * joint_idx + 1])

        # Visualize body0 points (red)
        if joint_positions_body0:
            positions0 = np.stack(joint_positions_body0)
            ps_joints0 = ps.register_point_cloud("joints_body0", positions0, radius=0.008)
            ps_joints0.add_color_quantity("joint_colors", np.array([[1.0, 0.0, 0.0]] * len(positions0)), enabled=True)
            ps_joints0.add_scalar_quantity("body_id", np.array(joint_body0_ids), enabled=True)

        # Visualize body1 points (blue)
        if joint_positions_body1:
            positions1 = np.stack(joint_positions_body1)
            ps_joints1 = ps.register_point_cloud("joints_body1", positions1, radius=0.008)
            ps_joints1.add_color_quantity("joint_colors", np.array([[0.0, 0.0, 1.0]] * len(positions1)), enabled=True)
            ps_joints1.add_scalar_quantity("body_id", np.array(joint_body1_ids), enabled=True)

        # Visualize connecting lines (gray)
        if joint_line_positions:
            line_pos = np.stack(joint_line_positions)
            line_edges = np.array(joint_line_edges)
            ps_lines = ps.register_curve_network("joint_lines", line_pos, line_edges)
            ps_lines.set_radius(0.003)
            ps_lines.add_color_quantity("line_colors", np.array([[0.5, 0.5, 0.5]] * len(line_pos)), enabled=True)

    def _local_to_world(self, pos_local, xr, body_id, device, dtype):
        """Helper: transform local position to world space."""
        if body_id != -1:
            vec4 = torch.cat([pos_local, torch.tensor([1.0], device=device, dtype=dtype)])
            return (vec4.unsqueeze(0) @ xr[body_id]).squeeze(0)
        else:
            return pos_local

    def export(self, system_def, x, prefix=""):
        pass

    def visualize_set_nice_view(self, system_def, x):
        ps.look_at((1.5, 1.5, 1.5), (0., -.2, 0.))