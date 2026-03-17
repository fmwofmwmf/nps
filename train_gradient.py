import io
import json
import random
import sys, os

from networkx.algorithms.threshold import eigenvectors

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pstats
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from gradients import *
from torch import optim
from tqdm import tqdm

from config_utils import load_config, Config, system_to_name
from integrators import *
from layers import SubspaceMLP
from Args import Args
from loss import *

from gradient_helpers import *

def initialize_single_bar_buffer_arc(
        system,
        system_def,
        integrator,
        buffer_size=10000,
        n_steps_per_sample=20,
        space=None,
        pivot=torch.tensor([0.0, 2.0]),
):
    """
    Generate a buffer of configurations for a single bar of length 1,
    attached at a fixed pivot point, sampling theta uniformly in [-pi, pi].

    State: (x_center, y_center, theta)

    Args:
        buffer_size: number of configurations
        pivot: fixed point of attachment (2,)
        bar_length: length of the bar
        device: torch device
        cache_path: path to save/load buffer

    Returns:
        config_buffer: (buffer_size, 3)
    """

    # Sample theta uniformly
    theta = -torch.pi * torch.rand(buffer_size)  # [-pi, pi]

    # Center of bar = pivot + 0.5 * [cos(theta), sin(theta)]
    offset = 0.5 * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (B,2)
    x_center = pivot[0] + offset[:,0]
    y_center = pivot[1] + offset[:,1]

    config_buffer = torch.stack([x_center, y_center, theta], dim=1)

    return config_buffer


def initialize_configuration_buffer(
        system,
        system_def,
        integrator,
        buffer_size=10000,
        n_steps_per_sample=20,
        space=None
):
    """
    Pre-generate a buffer of configurations by running physics simulations.
    Do this once at the start of training.

    Args:
        system: Physics system
        system_def: System definition
        integrator: Your integrator function
        state_to_system: State conversion function
        buffer_size: Total number of configurations to generate
        n_steps_per_sample: Integration steps per trajectory
        space: Space parameter for integrator

    Returns:
        config_buffer: (buffer_size, dim) - pre-computed configurations
    """
    cache_path = "config_buffer_link.pt"
    if os.path.exists(cache_path):
        print(f"Loading cached configuration buffer from {cache_path}...")
        config_buffer = torch.load(cache_path)
        print(f"Loaded {config_buffer.shape[0]} configurations")
        return config_buffer

    dim = system.dim
    device = system_def['init_pos'].device

    config_list = [system_def['init_pos']]

    print(f"Generating {buffer_size} configurations...")
    pbar = tqdm(total=buffer_size, desc="Generating configs")
    i = 0
    while len(config_list) < buffer_size:
        # Random initial condition
        q_init = config_list[random.randint(0, len(config_list) - 1)]
        system_def["external_forces"]["torque_strength"] = random.uniform(-3, 3)
        system_def["external_forces"]["torque_joint_idx"] = random.randint(0, 1)
        i += 1
        qdot_init = torch.zeros(dim, device=device)

        int_state = {
            'q_t': q_init,
            'qdot_t': qdot_init
        }

        # Run integration and save configurations along the way
        for step in range(n_steps_per_sample):

            int_state['q_t'], int_state['qdot_t'], r = integrator(
                system,
                system_def,
                lambda sd, x, s: x,
                int_state['q_t'],
                int_state['qdot_t'],
                space
            )
            pbar.update(1)
            # Save this configuration
            config_list.append(int_state['q_t'].detach().clone())

            if len(config_list) >= buffer_size:
                break

    config_buffer = torch.stack(config_list[:buffer_size], dim=0)  # (buffer_size, dim)

    print(f"Caching configuration buffer to {cache_path}...")
    torch.save(config_buffer, cache_path)

    return config_buffer


def visualize_buffer_animation(
        system,
        system_def,
        config_buffer,
        n_frames=100,
        fps=30
):
    """
    Create an animation showing configurations from the buffer.

    Cycles through random configurations at a controlled FPS.
    """
    import polyscope as ps
    import time
    import torch

    # Sample configurations
    indices = torch.randint(0, config_buffer.shape[0], (n_frames,))

    ps.init()

    frame = 0
    frame_time = 1.0 / fps  # seconds per frame
    last_time = time.time()

    def callback():
        nonlocal frame, last_time

        current_time = time.time()
        if current_time - last_time < frame_time:
            return  # skip until next frame time

        q = config_buffer[indices[frame]]
        system.visualize(system_def, q)

        frame = (frame + 1) % n_frames
        last_time = current_time

    ps.set_user_callback(callback)
    ps.show()
def compute_pca_basis(gradients, k):
    """
    Compute top k principal components of gradient data.

    Args:
        gradients: (N, dim) - collection of gradients
        k: number of components

    Returns:
        V: (dim, k) - orthonormal basis (columns are eigenvectors)
    """
    # Center the gradients
    mean = gradients.mean(dim=0, keepdim=True)  # (1, dim)
    centered = gradients - mean  # (N, dim)

    # Compute covariance matrix
    # C = (1/N) * centered^T @ centered
    cov = (centered.T @ centered) / (gradients.shape[0] - 1)  # (dim, dim)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # eigh for symmetric

    # Sort by eigenvalue (descending)
    idx = torch.argsort(eigenvalues, descending=False)
    eigenvectors = eigenvectors[:, idx]  # (dim, dim)

    # Take top k
    V = eigenvectors[:, :k]  # (dim, k)

    return V


def reconstruction_loss(B_pred, gradients_batch):
    """
    Ensure basis can reconstruct gradients well.

    Args:
        B_pred: (B, dim, k) - predicted bases
        gradients_batch: (B, dim) - actual gradients

    Returns:
        loss: scalar
    """
    # Project gradients onto predicted subspace
    # coefficients: (B, k)
    coeffs = torch.bmm(
        gradients_batch.unsqueeze(1),  # (B, 1, dim)
        B_pred  # (B, dim, k)
    ).squeeze(1)  # (B, k)

    # Reconstruct: g_reconstructed = B @ coeffs
    g_recon = torch.bmm(
        B_pred,  # (B, dim, k)
        coeffs.unsqueeze(2)  # (B, k, 1)
    ).squeeze(2)  # (B, dim)

    # Reconstruction error
    error = gradients_batch - g_recon  # (B, dim)
    loss = (error ** 2).sum(dim=1).mean()

    return loss

from torch.func import vmap, hessian

def compute_local_pca_targets(q_batch, system, system_def, space, k=1):
    """
    Compute local PCA target basis for each configuration using
    Hessian soft modes + gradient. Fully batched and vectorized.
    """

    B, dim = q_batch.shape
    device, dtype = q_batch.device, q_batch.dtype

    # --- Expand space if needed ---
    if space.dim() == 1:
        space_batch = space.unsqueeze(0).expand(B, -1)
    else:
        space_batch = space

    # -------------------------------------------------------
    # 1. Energy function (single sample)
    # -------------------------------------------------------
    def energy_single(q, s):
        return system.potential_energy_batch(
            system_def,
            q.unsqueeze(0),
            s.unsqueeze(0)
        ).squeeze(0)

    # Vectorized energy
    energy_batch = lambda q: vmap(energy_single)(q, space_batch)

    q_batch = q_batch.requires_grad_(True)
    E = energy_batch(q_batch)
    grads = torch.autograd.grad(E.sum(), q_batch, create_graph=True)[0]

    hess_fn = hessian(energy_single)
    H_batch = vmap(hess_fn)(q_batch, space_batch)  # (B, dim, dim)

    M_test = system.physical_mass_matrix(system_def, q_batch[0])

    if M_test.dim() == 2:
        M_batch = M_test.unsqueeze(0).expand(B, dim, dim)
    else:
        M_batch = vmap(lambda q: system.physical_mass_matrix(system_def, q))(q_batch)

    L = torch.linalg.cholesky(M_batch)
    L_inv = torch.linalg.inv(L)

    H_t = torch.bmm(L_inv, torch.bmm(H_batch, L_inv.transpose(1, 2)))
    eigvals, eigvecs_w = torch.linalg.eigh(H_t)
    U = torch.bmm(L_inv.transpose(1, 2), eigvecs_w)

    # ---- select soft modes ----
    eigvals_k = eigvals[:, :k]
    U_k = U[:, :, :k]

    return U_k.detach(), eigvals_k.detach(), U.detach()
def diagnose_interpolation_quality(
    config_buffer,
    pca_targets_buffer,
    system,
    system_def,
    space,
    B=32,
    k_neighbors=16,
    noise_std=0.01,
    bandwidth=0.01,
):
    """
    Diagnostic: compare interpolated PCA targets (Grassmann-weighted kNN)
    with freshly recomputed targets at perturbed points.

    This tells you whether interpolation is faithful to the underlying map.
    """

    device = config_buffer.device
    N = config_buffer.shape[0]
    k = pca_targets_buffer.shape[2]

    print("\n=== Interpolation Diagnostic (Grassmann) ===")

    # -------------------------------------------------
    # 1️⃣ Sample random base configs
    # -------------------------------------------------
    idx = torch.randint(0, N, (B,), device=device)
    base = config_buffer[idx]

    # -------------------------------------------------
    # 2️⃣ Perturb
    # -------------------------------------------------
    q_new = base + noise_std * torch.randn_like(base)

    # -------------------------------------------------
    # 3️⃣ Interpolate using Grassmann projection averaging
    # -------------------------------------------------
    dists = torch.cdist(q_new, config_buffer)  # (B, N)
    knn_idx = dists.topk(k_neighbors, largest=False).indices  # (B, kN)

    neighbor_targets = pca_targets_buffer[knn_idx]  # (B, kN, dim, k)
    neighbor_dists = dists.gather(1, knn_idx)        # (B, kN)

    # projection matrices
    P_neighbors = neighbor_targets @ neighbor_targets.transpose(-1, -2)  # (B, kN, dim, dim)

    # distance weights
    weights = 1/(neighbor_dists + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, kN, 1, 1)

    # weighted average of projections
    P_avg = (P_neighbors * weights).sum(dim=1)  # (B, dim, dim)

    # extract top-k eigenvectors (subspace)
    eigvals, eigvecs = torch.linalg.eigh(P_avg)
    V_interp = eigvecs[:, :, -k:]  # (B, dim, k)

    # orthonormal (should already be)
    V_interp, _ = torch.linalg.qr(V_interp, mode="reduced")

    # -------------------------------------------------
    # 4️⃣ Recompute ground-truth PCA at perturbed points
    # -------------------------------------------------
    V_true = compute_local_pca_targets(
        q_new,
        system,
        system_def,
        space,
        k=k,
    )

    # -------------------------------------------------
    # 5️⃣ Subspace error (Grassmann-style)
    # -------------------------------------------------
    def subspace_error(Va, Vb):
        """
        Frobenius distance between projection matrices:
            || VaVa^T - VbVb^T ||_F
        """
        Pa = Va @ Va.transpose(-1, -2)
        Pb = Vb @ Vb.transpose(-1, -2)
        return ((Pa - Pb) ** 2).sum(dim=(1, 2))

    errors = subspace_error(V_true, V_interp)

    print(f"Mean interpolation error: {errors.mean().item():.6e}")
    print(f"Max interpolation error:  {errors.max().item():.6e}")
    print(f"Min interpolation error:  {errors.min().item():.6e}")

    # -------------------------------------------------
    # 6️⃣ Optional: report scale vs noise (Lipschitz hint)
    # -------------------------------------------------
    print(f"Noise scale (std): {noise_std:.4e}")
    print(f"Bandwidth:         {bandwidth:.4e}")

    return errors

def gradient_at_config(system, system_def, q, space=None):
    """
    Compute the gradient of the potential energy at a single configuration.

    Args:
        system: Physical system object (must implement potential_energy_batch)
        system_def: System definition
        q: (dim,) torch tensor - configuration
        space: optional extra space info

    Returns:
        grad: (dim,) torch tensor - dE/dq at this configuration
    """
    q = q.clone().detach().requires_grad_(True)  # track gradients

    # Compute potential energy (as batch of 1)
    E = system.potential_energy_batch(
        system_def,
        q.unsqueeze(0),
        None if space is None else space.unsqueeze(0)
    )[0]

    # Compute gradient
    grad = torch.autograd.grad(E, q)[0]  # (dim,)

    return grad

def compute_gradient_targets(config_buffer, system, system_def):
    """
    For 1D systems, the subspace is just the normalized gradient at each configuration.
    Returns: targets_buffer of shape (N, dim, 1)
    """
    targets_buffer = []
    for q in config_buffer:
        q_grad = q.clone().detach().requires_grad_(True)
        E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0))[0]
        grad = torch.autograd.grad(E, q_grad)[0]  # (dim,)
        grad = grad / (grad.norm() + 1e-8)        # normalize
        targets_buffer.append(grad.unsqueeze(-1)) # (dim, 1)

    return torch.stack(targets_buffer, dim=0)     # (N, dim, 1)

def sample_interpolated_batch(
    config_buffer,
    pca_targets_buffer,
    B,
    k_neighbors=16,
    noise_std=0.01,
):
    """
    Generate B new samples by:
    1) choosing random points from dataset
    2) perturbing them
    3) averaging kNN PCA targets
    """

    device = config_buffer.device
    N, dim = config_buffer.shape
    k = pca_targets_buffer.shape[2]

    idx = torch.randint(0, N, (B,), device=device)
    base_points = config_buffer[idx]  # (B, dim)

    q_new = base_points + noise_std * torch.randn_like(base_points)

    return q_new, grassmann_weighted_knn_interpolation(
                                                    q_new,
                                                    config_buffer,
                                                    pca_targets_buffer, k_neighbors=k_neighbors)


def sample_batch(
    config_buffer,
    pca_targets_buffer,
    B
):

    device = config_buffer.device
    N, dim = config_buffer.shape
    idx = torch.randint(0, N, (B,), device=device)

    return config_buffer[idx], pca_targets_buffer[idx]
def grassmann_weighted_knn_interpolation(
    q_new,
    config_buffer,
    pca_targets_buffer,
    k_neighbors=16,
):
    """
    Interpolate PCA targets using weighted projection averaging
    on the Grassmann manifold.

    Args:
        q_new: (B, dim)
        config_buffer: (N, dim)
        pca_targets_buffer: (N, dim, k)
    Returns:
        V_interp: (B, dim, k)
    """

    device = config_buffer.device
    B = q_new.shape[0]
    dim = config_buffer.shape[1]
    k = pca_targets_buffer.shape[2]

    # -------------------------------------------------
    # 1️⃣ kNN search
    # -------------------------------------------------
    dists = torch.cdist(q_new, config_buffer)  # (B, N)
    knn_idx = dists.topk(k_neighbors, largest=False).indices  # (B, kN)

    neighbor_targets = pca_targets_buffer[knn_idx]  # (B, kN, dim, k)
    neighbor_dists = dists.gather(1, knn_idx)       # (B, kN)

    # -------------------------------------------------
    # 2️⃣ Compute projection matrices
    # -------------------------------------------------
    # P = V V^T
    P_neighbors = torch.matmul(
        neighbor_targets,
        neighbor_targets.transpose(-1, -2)
    )  # (B, kN, dim, dim)

    # -------------------------------------------------
    # 3️⃣ Distance weights
    # -------------------------------------------------
    weights = 1/(neighbor_dists + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, kN, 1, 1)

    # -------------------------------------------------
    # 4️⃣ Weighted projection average
    # -------------------------------------------------
    P_avg = (P_neighbors * weights).sum(dim=1)  # (B, dim, dim)

    # -------------------------------------------------
    # 5️⃣ Extract top-k eigenvectors
    # -------------------------------------------------
    eigvals, eigvecs = torch.linalg.eigh(P_avg)

    # take largest k eigenvectors
    V_interp = eigvecs[:, :, -k:]  # (B, dim, k)

    return V_interp

def upsample_buffer_with_noise(
    config_buffer,
    system,
    system_def,
    space,
    k_pca=4,
    samples_per_point=2,
    noise_std=0.01,
    batch_size=512,
):
    """
    Upsample buffer by adding Gaussian noise to each config point
    and recomputing PCA targets.

    Returns:
        new_config_buffer
        new_target_buffer
    """

    device = config_buffer.device
    N, dim = config_buffer.shape

    print("\n=== Upsampling Buffer (Gaussian noise) ===")

    # -------------------------------------------------
    # 1️⃣ Generate noisy configs
    # -------------------------------------------------
    noise = noise_std * torch.randn(
        N, samples_per_point + 1, dim, device=device
    )
    noise[:, 0, :] = 0.0

    q_new = config_buffer.unsqueeze(1) + noise
    q_new = q_new.reshape(-1, dim)

    print(f"Generated {q_new.shape[0]} new config samples")

    # -------------------------------------------------
    # 2️⃣ Batched PCA recomputation (with progress bar)
    # -------------------------------------------------
    new_targets = []
    new_targets1 = []
    new_targets2 = []
    total = q_new.shape[0]
    num_batches = (total + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, total, batch_size),
        total=num_batches,
        desc="Recomputing PCA",
    ):
        end = min(start + batch_size, total)
        batch = q_new[start:end]

        V_batch, vals_batch, vecs_batch = compute_local_pca_targets(
            batch,
            system,
            system_def,
            space,
            k=k_pca,
        )

        new_targets.append(V_batch)
        new_targets1.append(vals_batch)
        new_targets2.append(vecs_batch)

    new_targets = torch.cat(new_targets, dim=0)

    # -------------------------------------------------
    # 3️⃣ Concatenate buffers
    # -------------------------------------------------
    new_config_buffer = q_new
    new_target_buffer = new_targets

    print(f"New buffer size: {new_config_buffer.shape[0]}")

    return new_config_buffer, new_target_buffer, torch.cat(new_targets1, dim=0), torch.cat(new_targets2, dim=0)

def compare_basis_schemes(
    system,
    system_def,
    config_buffer,
    space,
    basis_fn_a,
    basis_fn_b,
    num_vel_samples=32,
    vel_scale=1e-2,
):

    state_to_system = lambda system_def, z, space: z

    captures_a = []
    captures_b = []
    update_sizes = []

    for q0 in tqdm(config_buffer, desc="Comparing basis schemes"):

        U_a = basis_fn_a(system, system_def, q0, space)
        U_b = basis_fn_b(system, system_def, q0, space)

        z_dots = vel_scale * torch.randn(num_vel_samples, *q0.shape, device=q0.device)

        def single_step(z_dot):
            q_new, _, _ = latent_step_newton_batch(
                system,
                system_def,
                state_to_system,
                q0,
                z_dot,
                space,
            )
            return q_new - q0

        deltas = torch.vmap(single_step)(z_dots)

        delta_norms = deltas.norm(dim=1) + 1e-12

        proj_a = (U_a @ (U_a.T @ deltas.T)).T
        proj_b = (U_b @ (U_b.T @ deltas.T)).T

        capture_a = proj_a.norm(dim=1) / delta_norms
        capture_b = proj_b.norm(dim=1) / delta_norms

        captures_a.append(capture_a)
        captures_b.append(capture_b)
        update_sizes.append(delta_norms)

    captures_a = torch.cat(captures_a).cpu().numpy()
    captures_b = torch.cat(captures_b).cpu().numpy()
    update_sizes = torch.cat(update_sizes).cpu().numpy()

    # log size
    log_sizes = update_sizes

    # automatic bins
    bins = np.histogram_bin_edges(log_sizes, bins="auto")

    # bin indices
    inds = np.digitize(log_sizes, bins) - 1

    mean_a = []
    mean_b = []
    centers = []

    for i in range(len(bins) - 1):
        mask = inds == i
        if mask.sum() == 0:
            continue

        mean_a.append(captures_a[mask].mean())
        mean_b.append(captures_b[mask].mean())
        centers.append((bins[i] + bins[i + 1]) / 2)

    centers = np.array(centers)

    # plot
    plt.figure()
    plt.plot(centers, mean_a, label="Basis A")
    plt.plot(centers, mean_b, label="Basis B")

    plt.xlabel("update size")
    plt.ylabel("Average fraction captured")
    plt.title("Basis Capture vs Update Magnitude")
    plt.legend()

    plt.show()

    return captures_a.mean(), captures_b.mean()

def compute_pca_bases_from_buffer(
    system,
    system_def,
    config_buffer,
    space,
    num_samples=32,
    vel_scale=1e-2,
    k=2,
):
    bases = []

    for q0 in tqdm(config_buffer, desc="Computing local PCA bases"):
        # basis = local_pca_one_step_random(
        #     system=system,
        #     system_def=system_def,
        #     q0=q0,
        #     space=space,
        #     num_samples=num_samples,
        #     vel_scale=vel_scale,
        #     k=k,
        # )
        basis = gradient_pca(
            system=system,
            system_def=system_def,
            q=q0,
            space=space,
            add_gradient=False,
            k=k,
        )
        bases.append(basis)

    return torch.stack(bases)  # (N, dim, k)


def train_gradient_basis_field(
        model,
        system,
        system_def,
        k,
        n_iters_per_epoch=2000,
        n_epochs=10,
        batch_size=512,
        space=None
):
    """
    Main training loop with pre-computed configuration buffer.
    """
    target_path = os.path.join("precompute", f"{config.experiment_name}_{k}D_pca_targets.pt")
    config_path = os.path.join("precompute", f"{config.experiment_name}_dataset.pt")

    configs = torch.load(config_path)

    config_buffer = configs["q"].to(device="cuda")
    visualize_buffer_animation(
        system=system,
        system_def=system_def, config_buffer=config_buffer)
    # test = config_buffer[8]
    # basis = local_pca_one_step_random(
    #     system=system,
    #     system_def=system_def,
    #     q0=test,
    #     space=space,
    #     num_samples=32,
    #     vel_scale=1e-2,
    #     k=2,
    # )
    #
    # compare_integration_error(
    #     system,
    #     system_def,
    #     test,
    #     basis,
    #     space,
    #     n_samples=32,
    #     vel_scale=1e-2,
    # )

    basis_fn_a = lambda system, system_def, q0, space: local_pca_one_step_random(
            system=system,
            system_def=system_def,
            q0=q0,
            space=space,
            num_samples=128*2,
            vel_scale=.1,
            k=k,
        )
    basis_fn_b = lambda system, system_def, q0, space: gradient_pca(
        system=system,
        system_def=system_def,
        q=q0,
        space=space,
        add_gradient=False,
        k=k)
    # compare_basis_schemes(
    #     system,
    #     system_def,
    #     config_buffer[:150],
    #     space,
    #     basis_fn_a,
    #     basis_fn_b,
    #     num_vel_samples=32*32,
    #     vel_scale=1e-2,
    # )
    # basis_fn_a(system, system_def, q0, space)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    N = config_buffer.shape[0]

    print(config_buffer.shape)

    if os.path.exists(target_path):
        pca_targets_buffer_new = torch.load(target_path)
        print("loaded cached PCA targets")
    else:
        pca_targets_buffer_new = compute_pca_bases_from_buffer(
            system,
            system_def,
            config_buffer,
            space,
            num_samples=128*16,
            vel_scale=1e-2,
            k=k,
        )
        torch.save(pca_targets_buffer_new, target_path)
    print(pca_targets_buffer_new.shape, k)
    pbar = tqdm(total=n_iters_per_epoch * n_epochs, desc="Training", unit="iter")
    for epoch in range(n_epochs):
        # samples = 2
        # config_buffer_new, pca_targets_buffer_new, eigenvalues_buffer_new, eigenvectors_buffer_new = upsample_buffer_with_noise(
        #     config_buffer,
        #     system,
        #     system_def,
        #     space,
        #     k_pca=k,
        #     samples_per_point=samples,
        #     noise_std=0.1,
        #     batch_size=1024,
        # )
        # print(pca_targets_buffer_new.shape, eigenvectors_buffer_new.shape, eigenvalues_buffer_new.shape)
        # print(pca_targets_buffer_new[3].shape, model(config_buffer[3]).shape)
        # print(subspace_distance_loss(pca_targets_buffer_new[3].unsqueeze(0), model(config_buffer[3]).unsqueeze(0)).mean())
        for iteration in range(n_iters_per_epoch):
            q_batch, V_target = sample_batch(
                        config_buffer,
                        pca_targets_buffer_new,
                        batch_size,
                )

            # === 4. Predict basis from model ===
            B_pred = model(q_batch)  # (B, dim, k)

            # === 5. Compute losses ===
            loss_subspace = subspace_distance_loss(B_pred, V_target).mean()
            #loss_ortho = orthonormality_loss(B_pred)

            # === 6. Combine losses ===
            total_loss = (
                    1.0 * loss_subspace
            )

            # === 7. Optimize ===
            optimizer.zero_grad()
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix({
                "Epoch": epoch,
                'Loss': f"{total_loss.item():.6f}",
                'Subspace': f"{loss_subspace.item():.6f}",
            })

    model_path = os.path.join("experiments", f"{config.experiment_name}_{k}D_model_local.pt")
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    args = Args()

    config = load_config(args.config_file)

    system, system_def = system_to_name(config)
    system.training = True

    in_dim = system.dim + config.shape_space_dim
    k = config.subspace_dim

    model = SmoothGradientBasisField(in_dim, k, config["network"]["n_hidden_layers"], config["network"]["n_neurons"])

    train_gradient_basis_field(
        model,
        system,
        system_def,
        k,
        space=torch.empty(0))