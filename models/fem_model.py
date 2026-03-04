import numpy as np
import potpourri3d as pp3d

import os
from io import StringIO

import polyscope as ps
import polyscope.imgui as psim
import torch

import system_utils

import igl


def linear_energy(system_def, FT, mesh):
    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))  # Plane strain

    E = 0.5 * (FT + FT.transpose(1, 2)) - torch.eye(dim, device=FT.device)[None, :, :]
    energies = mu * (E * E).sum(dim=(1, 2)) + 0.5 * lamb * torch.trace(E) ** 2

    return (A * energies).sum()


def StVK_energy(system_def, FT, mesh):
    dim = mesh["Vrest"].shape[1]
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))  # Plane strain

    E = 0.5 * (FT @ FT.transpose(1, 2) - torch.eye(dim, device=FT.device)[None, :, :])
    energies = mu * (E * E).sum(dim=(1, 2)) + 0.5 * lamb * torch.trace(E) ** 2

    return (A * energies).sum()

def neohook_energy(system_def, FT, mesh, A):
    poisson = system_def['poisson']
    Y = system_def['Y']
    #A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(dim=(1, 2))
    dec = torch.linalg.det(FT) - alpha

    energies0 = 0.5 * (mu * dim + lambRep * (1.0 - alpha) ** 2)
    energies = 0.5 * (mu * FTn + lambRep * dec ** 2)

    return (A * (energies - energies0)).sum()

def neohook_energy_batch(system_def, FT, mesh, A):
    """
    FT: [B, num_elements, D, D]
    Returns: [B]
    """
    B, num_elements, D, _ = FT.shape
    #A = mesh["A"]
    poisson = system_def['poisson']
    Y = system_def['Y']

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))  # Plane strain
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    # FTn: Frobenius norm squared per element
    FTn = (FT * FT).sum(dim=(-2, -1))  # [B, num_elements]

    # Determinant per element
    dec = torch.linalg.det(FT) - alpha  # [B, num_elements]

    # energies0: per element, broadcast across batch
    energies0 = 0.5 * (mu * D + lambRep * (1.0 - alpha) ** 2)  # scalar
    energies0 = energies0 * torch.ones_like(FTn)

    # energies per element
    energies = 0.5 * (mu * FTn + lambRep * dec ** 2)  # [B, num_elements]

    # Multiply by element areas (broadcast if needed)
    if A.ndim == 1:
        A = A.unsqueeze(0).expand(B, -1)  # [B, num_elements]

    return torch.sum(A * (energies - energies0), dim=1)  # [B]

def neohook_thin_energy(system_def, FT, mesh):
    poisson = system_def['poisson']
    Y = system_def['Y']
    A = mesh["A"]

    mu = 0.5 * Y / (1.0 + poisson)
    lamb = (Y * poisson) / (1.0 - poisson ** 2)  # Plane stress

    dim = mesh["Vrest"].shape[1]
    lambRep = lamb + mu
    alpha = 1.0 + (mu / lambRep)

    FTn = (FT * FT).sum(dim=(1, 2))
    dec = torch.linalg.det(FT) - alpha

    energies0 = 0.5 * (mu * dim + lambRep * (1.0 - alpha) ** 2)
    energies = 0.5 * (mu * FTn + lambRep * dec ** 2)

    return (A * (energies - energies0)).sum()

# def fem_energy(system_def, mesh, material_energy, V, s):
#     E = mesh["E"]
#     DTI = mesh["DTI"]
#
#     DT = V[E[:, 1:]] - V[E[:, 0:1]]  # shape (num_elements, dim, dim)
#     FT = torch.matmul(DTI, DT)
#
#     return material_energy(system_def, FT, mesh)

#@torch.compile()
def fem_energy_batch(system_def, mesh, material_energy, V_batch, s_batch):
    """
    Batched version of fem_energy.
    V_batch:  [B, N, D]
    s_batch:  [B] or [B,1] or [B,1,1]
    """
    E = mesh["E"]                 # [num_elements, dim]
    Vrest = mesh["Vrest"]         # [N, D]
    B, N, D = V_batch.shape
    num_elements = E.shape[0]

    # -------------------------------
    # 1. Scale the rest shape by s_batch
    # -------------------------------
    # Ensure s_batch is [B,1]
    if s_batch.ndim == 1:
        s_batch = s_batch[:, None]      # [B,1]

    # Broadcast to [B, N]
    s_for_rest = s_batch                 # [B,1]

    # Build Vrest_batch = [B, N, D]
    Vrest_batch = Vrest.unsqueeze(0).expand(B, -1, -1).clone()
    Vrest_batch[:, :, 1] = Vrest_batch[:, :, 1] * s_for_rest  # broadcast ok

    # -------------------------------
    # 2. Compute element edge matrices DT_R and DT
    # -------------------------------
    # Indices for vectorized gather:
    # E[:,0] is first vertex index,  E[:,1:] are other vertices
    # Shapes:
    #   V_batch[:, E[:,0], :] → [B, num_elements, D]
    #   but we want [B, num_elements, 1, D]
    first_rest = Vrest_batch[:, E[:, 0], :].unsqueeze(2)
    other_rest = Vrest_batch[:, E[:, 1:], :]

    DT_R = other_rest - first_rest       # [B, num_elements, D, D]

    # Same for current V
    first = V_batch[:, E[:, 0], :].unsqueeze(2)
    other = V_batch[:, E[:, 1:], :]

    DT = other - first                   # [B, num_elements, D, D]

    # -------------------------------
    # 3. Inverse of DT_R for each batch element
    # -------------------------------
    DTI = torch.linalg.inv(DT_R)         # [B, num_elements, D, D]

    # -------------------------------
    # 4. Element areas/volumes A
    # -------------------------------
    detR = torch.linalg.det(DT_R)        # [B, num_elements]
    A = torch.abs(detR) / (D * (D - 1))  # [B, num_elements]

    # -------------------------------
    # 5. Deformation gradients F = DTI @ DT
    # -------------------------------
    FT = torch.matmul(DTI, DT)           # [B, num_elements, D, D]

    # -------------------------------
    # 6. Material energy
    # -------------------------------
    return material_energy(system_def, FT, mesh, A)

def fem_energy(system_def, mesh, material_energy, V, s):
    dim = mesh["Vrest"].shape[1]

    # Apply shape scaling to rest positions
    Vrest = mesh["Vrest"].clone()
    Vrest[:, 1] *= s

    E = mesh["E"]

    # Compute element edge matrices: shape (num_elements, dim, dim)
    DT_R = Vrest[E[:, 1:]] - Vrest[E[:, 0:1]]

    # Inverse of element matrices
    DTI = torch.linalg.inv(DT_R)

    # Current element matrices
    DT = V[E[:, 1:]] - V[E[:, 0:1]]

    # Element areas / volumes
    A = torch.abs(torch.linalg.det(DT_R)) / (dim * (dim - 1))

    # mesh["DTI"] = DTI
    # mesh["A"] = A

    # Deformation gradients
    FT = torch.matmul(DTI, DT)  # shape: [num_elements, dim, dim]

    return material_energy(system_def, FT, mesh, A)

def mean_strain_metric(system_def, mesh, V):
    dim = mesh["Vrest"].shape[1]
    E_idx = mesh["E"]
    DTI = mesh["DTI"]
    A = mesh["A"]

    DT = V[E_idx[:, 1:]] - V[E_idx[:, 0:1]]
    FT = torch.matmul(DTI, DT)

    E = 0.5 * (FT @ FT.transpose(1, 2) - torch.eye(dim, device=FT.device)[None, :, :])
    rigidity_density = torch.sqrt((E * E).sum(dim=(1, 2)))

    return (A * rigidity_density).sum() / A.sum()


def tet_mesh_boundary_faces(tets):
    # numpy to numpy
    return igl.boundary_facets(tets)


def precompute_mesh(mesh):
    dim = mesh["Vrest"].shape[1]
    Vrest = mesh["Vrest"]
    E = mesh["E"]

    # Compute element edge matrices: shape (num_elements, dim, dim)
    DT = Vrest[E[:, 1:]] - Vrest[E[:, 0:1]]

    # Inverse of element matrices
    DTI = torch.linalg.inv(DT)

    # Element areas / volumes
    A = torch.abs(torch.linalg.det(DT)) / (dim * (dim - 1))

    mesh["DTI"] = DTI
    mesh["A"] = A

    # Lumped vertex areas/volumes
    VA = torch.zeros(Vrest.shape[0], dtype=Vrest.dtype, device=Vrest.device)

    Atile = A.repeat_interleave(dim + 1).reshape(E.shape[0], dim + 1) / (dim + 1)

    for i in range(E.shape[0]):
        VA[E[i]] += Atile[i]

    mesh["VA"] = VA

    if E.shape[-1] == 4:
        mesh['boundary_triangles'] = tet_mesh_boundary_faces(E)

    return mesh


def build_quad_mesh(device=None, dtype=torch.float32):
    mesh = {}
    mesh["Vrest"] = torch.tensor([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.]
    ], dtype=dtype, device=device)
    mesh["E"] = torch.tensor([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.long, device=device)
    return mesh


def build_tet_mesh(device=None, dtype=torch.float32):
    mesh = {}

    p = torch.tensor([1.0, 0., 0.], dtype=dtype, device=device)

    ang = torch.tensor(2.0 * torch.pi / 3.0, dtype=dtype, device=device)
    R = torch.tensor([
        [torch.cos(ang), 0.0, torch.sin(ang)],
        [0.0, 1.0, 0.0],
        [-torch.sin(ang), 0.0, torch.cos(ang)]
    ], dtype=dtype, device=device)

    Rp = R @ p
    RRp = R @ Rp

    mesh["Vrest"] = torch.stack([
        torch.tensor([0., 1.2, 0.], dtype=dtype, device=device),
        p,
        Rp,
        RRp
    ], dim=0)

    mesh["E"] = torch.tensor([
        [0, 1, 2, 3]
    ], dtype=torch.long, device=device)

    return mesh


def load_obj(filename):
    verts, faces = pp3d.read_mesh(filename)

    mesh = {}
    mesh["Vrest"] = verts[:, :2]
    mesh["E"] = faces

    return mesh



def load_tri_mesh(file_name_root, device=None, dtype=torch.float32):
    # --- Read element file ---
    with open(file_name_root + ".ele") as f:
        lines = [line for line in f.readlines() if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())

    # Load faces
    faces_array = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)
    faces = torch.tensor(faces_array[:, 1:4], dtype=torch.long, device=device)

    # Regions
    regions = 0
    if n_attribT > 0:
        regions_array = faces_array[:, 4]
        regions = torch.tensor(regions_array, dtype=torch.long, device=device)

    # --- Read node file ---
    with open(file_name_root + ".node") as f:
        lines = [line for line in f.readlines() if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())

    verts_array = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:, 1:3]
    verts = torch.tensor(verts_array, dtype=dtype, device=device)

    # Normalize vertices
    center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2
    scale = verts.max(dim=0).values - verts.min(dim=0).values
    verts = (verts - center) / scale.max()

    # Adjust faces to 0-based indexing
    first_node_index = int(np.loadtxt(StringIO(''.join(lines[1])), dtype=float)[0])
    faces -= first_node_index

    # Build mesh dict
    mesh = {
        "Vrest": verts,
        "E": faces
    }

    if n_attribT > 0:
        Y = torch.full((1, n_tri), 1.5e3, dtype=dtype, device=device)
        Y[:, regions == 2] = 1.5e2

        poisson = torch.full((1, n_tri), 0.4, dtype=dtype, device=device)
        poisson[:, regions == 2] = 0.4

        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh


def load_tet_mesh_igl(file_name_root, normalize=True):
    verts, tets, _ = igl.read_mesh(file_name_root)

    if normalize:
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        scale = verts.max(axis=0) - verts.min(axis=0)
        verts = (verts - center) / scale.max()

    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = tets

    return mesh


def load_tet_mesh(file_name_root):
    # ele file
    lines = open(file_name_root + ".ele").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_tri, n_nodesPerTriangle, n_attribT = map(int, lines[0].split())
    data = np.loadtxt(StringIO(''.join(lines[1:])), dtype=int)[:, 1:]
    tets = data[:, 0:4]
    regions = 0
    if n_attribT > 0:
        regions = data[:, -1]

    # node file
    lines = open(file_name_root + ".node").readlines()
    lines = [line for line in lines if line.strip() != '' and line[0] != '#']
    n_vert, n_dim, n_attrib, n_bmark = map(int, lines[0].split())
    verts = np.loadtxt(StringIO(''.join(lines[1:])), dtype=float)[:, 1:4]

    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    scale = verts.max(axis=0) - verts.min(axis=0)
    verts = (verts - center) / scale.max()

    first_node_index = np.loadtxt(StringIO(''.join(lines[1])), dtype=int)[0]
    tets = tets - first_node_index

    mesh = {}
    mesh["Vrest"] = verts
    mesh["E"] = tets
    if n_attribT > 0:
        Y = np.full((1, n_tri), 100e2)
        np.put(Y, np.argwhere(regions[:] == 0), 1e2)
        poisson = np.full((1, n_tri), 0.4)
        np.put(poisson, np.argwhere(regions[:] == 0), 0.4)
        mesh["Y"] = Y
        mesh["poisson"] = poisson
        mesh["regions"] = regions

    return mesh


###

class FEMSystem:

    def __init__(self):

        self.mesh = None

    @staticmethod
    def construct(problem_name, config):

        system_def = {}
        system = FEMSystem()

        system.system_name = "FEM"
        system.problem_name = str(problem_name)

        # set some defaults
        system_def['external_forces'] = {}
        system_def['cond_param'] = torch.zeros((0,))
        system.cond_dim = 0

        def update_conditional(self, system_def):
            return system_def  # default does nothing

        system.update_conditional = update_conditional

        system.material_energy = neohook_energy
        system.material_energy_batch = neohook_energy_batch

        if problem_name == 'bistable':

            mesh = load_tri_mesh(os.path.join("..", "data", "longerCantileverP2"))
            mesh["Vrest"][:, 1] *= 1  # scale y-coordinate
            # Precompute mesh quantities (assume precompute_mesh now uses torch)
            mesh = precompute_mesh(mesh)

            # System properties
            system_def["gravity"] = torch.tensor([0., 0.], dtype=torch.float32)
            system_def['poisson'] = torch.tensor(0.45, dtype=torch.float32)
            system_def['Y'] = torch.tensor(1e3, dtype=torch.float32)
            system_def['density'] = torch.tensor(10.0, dtype=torch.float32)

            verts = torch.tensor(mesh["Vrest"], dtype=torch.float32)

            # Compressed and initial vertex positions
            verts_compress = verts.clone()
            verts_compress[:, 0] *= 0.8
            verts_init = verts * 0.8

            # Identify pinned vertices (on x-min and x-max)
            xmin = torch.amin(verts[:, 0], dim=0)
            xmax = torch.amax(verts[:, 0], dim=0)
            pinned_verts_mask = (verts[:, 0] < (xmin + 1e-3)) | (verts[:, 0] > (xmax - 1e-3))

            # Flatten mask for 2D coordinates
            pinned_verts_mask_flat = pinned_verts_mask.repeat_interleave(2)

            # Use fixed/unfixed indices
            fixed_inds, unfixed_inds, fixed_values, unfixed_values = system_utils.generate_fixed_entry_data(
                pinned_verts_mask_flat,
                verts_compress.flatten()
            )

            system_def["fixed_inds"] = fixed_inds
            system_def["unfixed_inds"] = unfixed_inds
            system_def["fixed_values"] = fixed_values

            # Configure external forces
            xmid = 0.5 * (xmin + xmax)
            force_verts_mask = (verts[:, 0] > xmid - 1e-1) & (verts[:, 0] < xmid + 1e-1)

            system_def['external_forces'] = None
            # system_def['external_forces']['force_verts_mask'] = force_verts_mask
            # system_def['external_forces']['pull_X'] = torch.tensor(0., dtype=torch.float32)
            # system_def['external_forces']['pull_Y'] = torch.tensor(0., dtype=torch.float32)
            # pull_minmax = (-0.1, 0.1)
            # system_def['external_forces']['pull_strength_minmax'] = pull_minmax
            # system_def['external_forces']['pull_strength'] = 0.5 * (pull_minmax[0] + pull_minmax[1])

            # Store mesh and initial positions in the system
            system.mesh = mesh
            system_def['init_pos'] = unfixed_values
            system.pos_dim = verts.shape[1]
            system.dim = unfixed_values.numel()

        # elif problem_name == 'load3d':
        #
        #     mesh = load_tet_mesh_igl(os.path.join(".", "data", "beam365.mesh"))
        #     mesh = precompute_mesh(mesh)
        #
        #     system.material_energy = StVK_energy
        #
        #     system_def["gravity"] = np.array([0, -0.98, 0])
        #     system_def['poisson'] = np.array(0.45)
        #     system_def['Y'] = np.array(5e3)
        #     system_def['density'] = np.array(100.0)
        #
        #     verts = np.array(mesh["Vrest"])
        #
        #     # identify verts that are on the x min and pin them.
        #     xmin = np.amin(verts, axis=0)[2]
        #     pinned_verts_mask = verts[:, 2] < xmin + 1e-3
        #     pinned_verts_mask_flat = np.repeat(pinned_verts_mask, 3)
        #     fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
        #         system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
        #     system_def["fixed_inds"] = fixed_inds
        #     system_def["unfixed_inds"] = unfixed_inds
        #     system_def["fixed_values"] = fixed_values
        #
        #     xmax = np.amax(verts, axis=0)[2]
        #     system_def['force_verts_mask'] = verts[:, 2] > (xmax - 1e-3)
        #
        #     system.mesh = mesh
        #     system_def['init_pos'] = unfixed_values
        #     system.pos_dim = verts.shape[1]
        #     system.dim = system_def['init_pos'].size
        #
        #
        # elif problem_name.startswith('heterobeam'):
        #
        #     mesh = load_tet_mesh(os.path.join(".", "data", "heterobeam"))
        #     mesh = precompute_mesh(mesh)
        #
        #     if problem_name == 'heterobeam-gravity':
        #         system_def["gravity"] = np.array([0, -1.0, 0])
        #     else:
        #         system_def["gravity"] = np.array([0, -1, 0])
        #
        #     system_def['poisson'] = np.array(mesh["poisson"])
        #     system_def['Y'] = np.array(mesh["Y"])
        #     system_def['density'] = np.array(100.0)
        #
        #     verts = np.array(mesh["Vrest"])
        #
        #     # identify verts that are on the x min and pin them.
        #     xmin = np.amin(verts, axis=0)[2]
        #     pinned_verts_mask = verts[:, 2] < xmin + 1e-3
        #     pinned_verts_mask_flat = np.repeat(pinned_verts_mask, 3)
        #     fixed_inds, unfixed_inds, fixed_values, unfixed_values = \
        #         system_utils.generate_fixed_entry_data(pinned_verts_mask_flat, verts.flatten())
        #     system_def["fixed_inds"] = fixed_inds
        #     system_def["unfixed_inds"] = unfixed_inds
        #     system_def["fixed_values"] = fixed_values
        #
        #     xmax = np.amax(verts, axis=0)[2]
        #
        #     # configure external forces
        #     xmax = np.amax(verts, axis=0)[2]
        #     system_def['external_forces']['force_verts_mask'] = verts[:, 2] > xmax - 1e-3
        #     system_def['external_forces']['pull_X'] = np.array(0.)
        #     system_def['external_forces']['pull_Y'] = np.array(0.)
        #     system_def['external_forces']['pull_Z'] = np.array(0.)
        #     pull_minmax = (-0.005, 0.005)
        #     system_def['external_forces']['pull_strength_minmax'] = pull_minmax
        #     system_def['external_forces']['pull_strength'] = 0.5 * (pull_minmax[0] + pull_minmax[1])
        #
        #     system.mesh = mesh
        #     system_def['init_pos'] = unfixed_values
        #     system.pos_dim = verts.shape[1]
        #     system.dim = system_def['init_pos'].size

        else:
            raise ValueError("unrecognized system problem_name")

        system_def['interesting_states'] = system_def['init_pos'][None, :]

        return system, system_def

    # ===========================================
    # === Energy functions 
    # ===========================================
    def apply_shape(self, values, shape):
        """
        Scale the y-coordinate of vertices by shape.

        Args:
            values: flat [N] vector of DOFs (N even, interpreted as N/2 vertices)
            shape: scalar or [B] or [B,1] batch of shape parameters

        Returns:
            values_scaled: same shape as input (or [B, N] for batch)
        """
        N = values.numel()
        D = 2
        assert N % D == 0, "Values length must be divisible by 2"

        verts = values.view(-1, D).clone()  # [num_vertices, 2]

        if torch.is_tensor(shape):
            if shape.ndim == 2 and shape.shape[1] == 1:
                shape = shape.squeeze(1)  # [B, 1] -> [B]
            if shape.ndim == 1:
                # batch case
                B = shape.shape[0]
                verts = verts.unsqueeze(0).expand(B, -1, -1).clone()  # [B, num_vertices, 2]
                verts[:, :, 1] *= shape[:, None]  # scale y-coordinate
            else:
                # unexpected shape
                raise ValueError(f"Shape tensor must be scalar or 1D, got {shape.shape}")
        else:
            # scalar case
            verts[:, 1] *= shape

        return verts.view_as(values) if not (torch.is_tensor(shape) and shape.ndim == 1) else verts


    def get_full_position(self, system_def, q, shape):
        pos = system_utils.apply_fixed_entries(
            system_def['fixed_inds'], system_def['unfixed_inds'],
            self.apply_shape(system_def['fixed_values'], shape), q).reshape(-1, self.pos_dim)
        return pos

    def get_full_position_batch(self, system_def, q_batch, shape_batch):
        """
        q_batch: [B, Q] (latent/unfixed DOFs)
        Returns: pos_batch: [B, N, D]
        """
        B, Q = q_batch.shape
        N = len(system_def['fixed_inds']) + len(system_def['unfixed_inds'])
        D = self.pos_dim

        # Create tensor to hold full positions
        pos_batch = torch.empty((B, N), dtype=q_batch.dtype, device=q_batch.device)

        # Place unfixed entries
        pos_batch[:, system_def['unfixed_inds']] = q_batch

        # Place fixed entries (broadcast if necessary)
        fixed_values = self.apply_shape(system_def['fixed_values'], shape_batch)
        if fixed_values.ndim == 2:
            fixed_values = fixed_values.unsqueeze(0).expand(B, -1, -1)
        pos_batch[:, system_def['fixed_inds']] = fixed_values.reshape(B, -1)

        # Reshape to [B, N_nodes, D]
        pos_batch = pos_batch.reshape(B, -1, D)
        return pos_batch

    def mean_strain(self, system_def, q, shape):

        pos = self.get_full_position(system_def, q, shape)

        return mean_strain_metric(system_def, self.mesh, pos)


    def potential_energy(self, system_def, q, shape):
        system_def = self.update_conditional(self, system_def)
        pos = self.get_full_position(system_def, q, shape)
        mass_lumped = self.mesh["VA"] * system_def['density']

        # Gravity energy (vectorized)
        gravity = system_def["gravity"]
        gravity_energy = -torch.sum(pos * gravity[None, :] * mass_lumped[:, None])

        # Contact energy
        contact_energy = 0  # keep as-is if no logic

        # External forces
        ext_forces = system_def.get('external_forces', None)
        if ext_forces is not None:
            mask = ext_forces["force_verts_mask"][:, None]  # [N,1]
            pull_strength = ext_forces['pull_strength']

            # Pre-create direction matrix for 3D/2D
            if pos.shape[1] == 2:
                directions_mat = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=pos.dtype, device=pos.device)
            else:
                directions_mat = torch.eye(3, dtype=pos.dtype, device=pos.device)  # X,Y,Z

            # Collect all forces that exist in system_def
            applied_forces = [f for f in ['pull_X', 'pull_Y', 'pull_Z'] if f in ext_forces]

            if applied_forces:
                # Build direction tensor for all forces
                dir_tensor = directions_mat[[['pull_X', 'pull_Y', 'pull_Z'].index(f) for f in applied_forces]]  # [F,D]
                force_values = torch.stack([ext_forces[f] for f in applied_forces], dim=0)  # [F]
                # Broadcast: [F,D] * [F,1] * [N,1,D] -> [F,N,D]
                masked_forces = (
                            mask[None, :, :] * dir_tensor[:, None, :] * force_values[:, None, None] * pull_strength)
                # Sum over all forces and vertices
                ext_force_energy = torch.sum(masked_forces * pos[None, :, :])
            else:
                ext_force_energy = torch.tensor(0., dtype=pos.dtype, device=pos.device)
        else:
            ext_force_energy = torch.tensor(0., dtype=pos.dtype, device=pos.device)

        # FEM energy
        fem_e = fem_energy(system_def, self.mesh, self.material_energy, pos, shape)

        return fem_e + gravity_energy + contact_energy + ext_force_energy

    #@torch.compile()
    def potential_energy_batch(self, system_def, q_batch, shape_batch):
        """
        Vectorized potential energy for a batch of q's.
        q_batch: [B, Q]
        Returns: E_pots [B]
        """
        B = q_batch.size(0)
        system_def = self.update_conditional(self, system_def)

        # Get full positions for the batch
        pos_batch = self.get_full_position_batch(system_def, q_batch, shape_batch)  # [B, N, D]
        mass_lumped = self.mesh["VA"] * system_def['density']  # [N]

        # Gravity energy
        gravity = system_def["gravity"]
        gravity_energy = -torch.sum(pos_batch * gravity[None, None, :] * mass_lumped[None, :, None], dim=(1, 2))  # [B]

        # Contact energy placeholder
        contact_energy = torch.zeros(B, device=pos_batch.device, dtype=pos_batch.dtype)

        # External forces
        ext_forces = system_def.get('external_forces', None)
        if ext_forces is not None:
            mask = ext_forces["force_verts_mask"][None, :, None]  # [1, N, 1]
            pull_strength = ext_forces['pull_strength']

            # Direction matrix for 3D/2D
            D = pos_batch.size(2)
            if D == 2:
                directions_mat = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=pos_batch.dtype, device=pos_batch.device)
            else:
                directions_mat = torch.eye(3, dtype=pos_batch.dtype, device=pos_batch.device)

            applied_forces = [f for f in ['pull_X', 'pull_Y', 'pull_Z'] if f in ext_forces]
            if applied_forces:
                dir_tensor = directions_mat[[['pull_X', 'pull_Y', 'pull_Z'].index(f) for f in applied_forces]]  # [F,D]
                force_values = torch.stack([ext_forces[f] for f in applied_forces], dim=0)  # [F]

                # Broadcast: [F,1,1,D] * [1,N,D] -> [B,F,N,D]
                masked_forces = mask * dir_tensor[:, None, :] * force_values[:, None, None] * pull_strength
                # Sum over vertices and directions, then broadcast to batch
                ext_force_energy = torch.sum(masked_forces[None, :, :, :] * pos_batch[:, None, :, :],
                                             dim=(1, 2, 3))  # [B]
            else:
                ext_force_energy = torch.zeros(B, dtype=pos_batch.dtype, device=pos_batch.device)
        else:
            ext_force_energy = torch.zeros(B, dtype=pos_batch.dtype, device=pos_batch.device)

        # FEM energy: batch
        fem_e = fem_energy_batch(system_def, self.mesh, self.material_energy_batch, pos_batch, shape_batch)  # returns [B]
        fem_ee = fem_energy(system_def, self.mesh, self.material_energy, pos_batch[0, :], shape_batch[0])
        return fem_e + gravity_energy + contact_energy + ext_force_energy

    @torch.compile()
    def kinetic_energy(self, system_def, q, q_dot):
        system_def = self.update_conditional(self, system_def)

        pos_dot = system_utils.apply_fixed_entries(
            system_def['fixed_inds'], system_def['unfixed_inds'],
            torch.tensor(0., dtype=q_dot.dtype, device=q_dot.device), q_dot
        ).reshape(-1, self.pos_dim)

        mass_lumped = self.mesh["VA"] * system_def['density']
        ke = 0.5 * torch.sum(mass_lumped * torch.sum(pos_dot ** 2, dim=-1))
        return ke

    @torch.compile()
    def kinetic_energy_batch(self, system_def, q_dot_batch, shape_batch):
        system_def = self.update_conditional(self, system_def)
        batch_size = q_dot_batch.shape[0]

        pos_dot_batch = self.batched_apply_fixed_entries(
            system_def['fixed_inds'],
            system_def['unfixed_inds'],
            torch.tensor(0., dtype=q_dot_batch.dtype, device=q_dot_batch.device),
            q_dot_batch  # [batch_size, Q]
        )

        pos_dot_batch = pos_dot_batch.reshape(batch_size, -1, self.pos_dim)

        mass_lumped = self.mesh["VA"] * system_def['density']  # [num_vertices] or scalar

        velocity_squared = torch.sum(pos_dot_batch ** 2, dim=-1)  # [batch_size, num_vertices]
        ke_batch = 0.5 * torch.sum(mass_lumped * velocity_squared, dim=-1)  # [batch_size]

        return ke_batch

    #@torch.compile()
    def batched_apply_fixed_entries(self, fixed_inds, unfixed_inds, fixed_values, unfixed_values_batch):
        batch_size = unfixed_values_batch.shape[0]
        N = fixed_inds.numel() + unfixed_inds.numel()

        if not torch.is_tensor(fixed_values):
            fixed_values = torch.tensor(fixed_values, dtype=unfixed_values_batch.dtype,
                                        device=unfixed_values_batch.device)

        out_batch = torch.zeros(batch_size, N,
                                dtype=unfixed_values_batch.dtype,
                                device=unfixed_values_batch.device)

        out_batch[:, fixed_inds] = fixed_values

        out_batch[:, unfixed_inds] = unfixed_values_batch

        return out_batch

    # ===========================================
    # === Conditional systems
    # ===========================================

    @torch.compile()
    def sample_conditional_params(self, system_def, rngkey, rho=1.):
        return torch.zeros((0,), dtype=torch.float32)

    # ===========================================
    # === Visualization routines
    # ===========================================

    def build_system_ui(self, system_def):

        if psim.TreeNode("system UI"):

            psim.TextUnformatted("External forces:")

            if "pull_X" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_X']
                _, pulling = psim.Checkbox("pull_X", pulling)
                system_def['external_forces']['pull_X'] = np.where(pulling, 1., 0.)

            if "pull_Y" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Y']
                _, pulling = psim.Checkbox("pull_Y", pulling)
                system_def['external_forces']['pull_Y'] = np.where(pulling, 1., 0.)

            if "pull_Z" in system_def["external_forces"]:
                pulling = system_def['external_forces']['pull_Z']
                _, pulling = psim.Checkbox("pull_Z", pulling)
                system_def['external_forces']['pull_Z'] = np.where(pulling, 1., 0.)

            if "pull_strength" in system_def["external_forces"]:
                low, high = system_def['external_forces']['pull_strength_minmax']
                _, system_def['external_forces']['pull_strength'] = psim.SliderFloat("pull_strength",
                                                                                     system_def['external_forces'][
                                                                                         'pull_strength'], low, high)

            psim.TreePop()

    def visualize(self, system_def, q, shape, prefix="", transparency=1.0):
        system_def = self.update_conditional(self, system_def)

        name = self.problem_name + prefix

        pos = self.get_full_position(system_def, q, shape).cpu().detach().numpy()

        elem_list = self.mesh['E']

        if self.pos_dim == 2:
            ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(elem_list))
            if 'regions' in self.mesh.keys():
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='faces', enabled=True)
        else:
            if 'regions' in self.mesh.keys():
                ps_elems = ps.register_volume_mesh(name + " mesh", pos, np.array(elem_list))
                regions = self.mesh['regions']
                ps_elems.add_scalar_quantity("material colors r", regions, defined_on='cells', enabled=True)
            else:
                ps_elems = ps.register_surface_mesh(name + " mesh", pos, np.array(self.mesh['boundary_triangles']))

        if (transparency < 1.):
            ps_elems.set_transparency(transparency)

        if self.problem_name == "bistable":
            s = 0.4
            w = 0.07
            h = 0.1
            quad_block_verts = np.array([
                [-s - w, -h],
                [-s, -h],
                [-s, +h],
                [-s - w, +h],
                [+s, -h],
                [+s + w, -h],
                [+s + w, +h],
                [+s, +h],
            ])
            ps_endcaps = ps.register_surface_mesh("endcaps", quad_block_verts, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                                                  color=(0.7, 0.7, 0.7), edge_width=4.)

        return (ps_elems)

    # def export(self, system_def, x, prefix=""):
    #
    #     system_def = self.update_conditional(self, system_def)
    #
    #     pos = self.get_full_position(self, system_def, x)
    #     tri_list = self.mesh['boundary_triangles']
    #
    #     filename = prefix + f"{self.problem_name}_mesh.obj"
    #
    #     utils.write_obj(filename, pos, np.array(tri_list))

    def visualize_set_nice_view(self, system_def, q):

        if self.problem_name == 'spot':
            ps.look_at((-1.2, 0.8, -1.8), (0., 0., 0.))

        ps.look_at((2., 1., 2.), (0., 0., 0.))