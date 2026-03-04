
import io
import json
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pstats

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from config_utils import load_config, Config, system_to_name
from layers import SubspaceMLP
from Args import Args
from loss import *

def train_system(args: Args, config: Config):

    system, system_def = system_to_name(config)
    system.training = True
    base_state = system_def['interesting_states'][0, :]

    in_dim = config.subspace_dim + config.shape_space_dim
    model_spec = {
        "in_dim": in_dim,
        "out_dim": system.dim,
        "model_type": config["network"]["otype"],
        "activation": config["network"]["activation"],
        "MLP_hidden_layers": config["network"]["n_hidden_layers"],
        "MLP_hidden_layer_width": config["network"]["n_neurons"],
    }

    model = SubspaceMLP(model_spec, base_output=base_state).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["optimizer"]["lr_decay_every"], gamma=config["optimizer"]["lr_decay_frac"])

    def apply_subspace_batch(z_batch, shape, t_schedule):
        # Concatenate along feature dim
        z_cond = torch.cat([z_batch, shape], dim=-1)
        return model(z_cond, t_schedule)

    def orthogonality_loss_covariance(z):
        """
        Penalize correlation between latent dimensions in the batch.
        Encourages decorrelated/orthogonal latent directions.

        z: [B, D] latent codes
        Returns: scalar loss
        """
        B, D = z.size()

        # Center the latent codes
        z_centered = z - z.mean(dim=0, keepdim=True)  # [B, D]

        # Compute covariance matrix
        cov = (z_centered.T @ z_centered) / (B - 1)  # [D, D]

        # We want diagonal elements to be non-zero (variance)
        # and off-diagonal elements to be zero (no correlation)

        # Off-diagonal penalty
        off_diag_mask = 1.0 - torch.eye(D, device=z.device, dtype=z.dtype)
        off_diag_loss = (cov * off_diag_mask).pow(2).sum()

        # Variance regularization: penalize if variance is too small
        # This prevents collapse to a single point
        diag = torch.diagonal(cov)
        variance_loss = torch.sum(torch.relu(1.0 - diag))  # Penalize if variance < 1

        return off_diag_loss + 0.1 * variance_loss

    def orthogonality_loss_jacobian(z, q):
        """
        Ensure Jacobian columns (dq/dz_i) are orthogonal.
        This is stronger: it ensures the latent directions map to
        orthogonal directions in the output space.

        z: [B, D] latent codes (requires grad)
        q: [B, Q] decoded outputs
        Returns: scalar loss
        """
        B, D = z.shape
        Q = q.shape[1]

        # Compute Jacobian dq/dz for the batch
        # We want columns of J to be orthogonal

        # Efficient batch Jacobian computation
        jacobians = []
        for i in range(D):
            # Gradient of q w.r.t. z[:, i]
            grad_outputs = torch.zeros_like(q)
            grad_outputs[:, :] = 1.0

            grads = torch.autograd.grad(
                outputs=q,
                inputs=z,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]  # [B, D]

            jacobians.append(grads[:, i:i + 1])  # [B, 1]

        J = torch.cat(jacobians, dim=1)  # [B, D]

        # Compute Gram matrix: J^T J (averaged over batch)
        G = (J.T @ J) / B  # [D, D]

        # We want G ≈ I (orthonormal Jacobian columns)
        I = torch.eye(D, device=z.device, dtype=z.dtype)

        return ((G - I) ** 2).mean()

    def orthogonality_loss_jacobian_efficient(z, q):
        """
        More efficient version using per-dimension scalar outputs.

        z: [B, D] latent codes (requires grad)
        q: [B, Q] decoded outputs
        Returns: scalar loss
        """
        B, D = z.shape

        # For each output dimension, compute gradient w.r.t. z
        # Then ensure these gradients are orthogonal across latent dims

        # Sum over output dimensions to get a scalar per sample
        q_sum = q.sum(dim=1)  # [B]

        # Compute gradient
        grads = torch.autograd.grad(
            outputs=q_sum,
            inputs=z,
            grad_outputs=torch.ones_like(q_sum),
            create_graph=True,
            retain_graph=True
        )[0]  # [B, D]

        # Gram matrix (averaged over batch)
        G = (grads.T @ grads) / B  # [D, D]

        I = torch.eye(D, device=z.device, dtype=z.dtype)

        return ((G - I) ** 2).mean()

    def orthogonality_loss_variance_preserving(z, q):
        """
        Combined loss that:
        1. Keeps latent dimensions decorrelated
        2. Ensures Jacobian columns are orthogonal
        3. Prevents variance collapse

        z: [B, D] latent codes (requires grad)
        q: [B, Q] decoded outputs
        Returns: scalar loss
        """
        B, D = z.shape

        # 1. Latent covariance decorrelation
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_z = (z_centered.T @ z_centered) / (B - 1)

        off_diag_mask = 1.0 - torch.eye(D, device=z.device, dtype=z.dtype)
        latent_decorr_loss = (cov_z * off_diag_mask).pow(2).sum()

        # 2. Variance preservation (penalize if variance too small)
        diag_z = torch.diagonal(cov_z)
        variance_loss = torch.sum(torch.relu(0.5 - diag_z))

        # 3. Jacobian orthogonality
        q_sum = q.sum(dim=1)
        grads = torch.autograd.grad(
            outputs=q_sum,
            inputs=z,
            grad_outputs=torch.ones_like(q_sum),
            create_graph=True,
            retain_graph=True
        )[0]

        G = (grads.T @ grads) / B
        I = torch.eye(D, device=z.device, dtype=z.dtype)
        jacobian_ortho_loss = ((G - I) ** 2).mean()

        # Combine
        total_loss = latent_decorr_loss + variance_loss + jacobian_ortho_loss

        return total_loss

    def orthogonality_loss_output_spread(z, q):
        """
        Alternative: ensure outputs are spread out when latents are orthogonal.
        This prevents mode collapse by ensuring different latent samples
        produce different outputs.

        z: [B, D] latent codes
        q: [B, Q] decoded outputs
        Returns: scalar loss
        """
        B, D = z.shape

        # 1. Decorrelate latent dimensions
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_z = (z_centered.T @ z_centered) / (B - 1)
        off_diag_mask = 1.0 - torch.eye(D, device=z.device, dtype=z.dtype)
        decorr_loss = (cov_z * off_diag_mask).pow(2).sum()

        # 2. Ensure output variance is maintained
        q_centered = q - q.mean(dim=0, keepdim=True)
        output_variance = (q_centered ** 2).mean()

        # Penalize if output variance is too small (collapse)
        variance_preservation = torch.relu(1.0 - output_variance)

        # 3. Ensure different samples produce different outputs (anti-collapse)
        # Compute pairwise output distances
        q_dists = torch.cdist(q, q, p=2).pow(2)
        z_dists = torch.cdist(z, z, p=2).pow(2).clamp(min=1e-6)

        # Penalize if output distance is small when latent distance is large
        # This prevents the map from collapsing
        collapse_penalty = torch.relu(z_dists - q_dists).mean()

        return decorr_loss + variance_preservation + 0.1 * collapse_penalty

    # Recommended usage in your training loop:
    def get_orthogonality_loss(z, q, loss_type='variance_preserving'):
        """
        Wrapper to select which orthogonality loss to use.

        Args:
            z: [B, D] latent codes (requires grad)
            q: [B, Q] decoded outputs
            loss_type: 'covariance', 'jacobian', 'variance_preserving', 'output_spread'
        """
        if loss_type == 'covariance':
            return orthogonality_loss_covariance(z)
        elif loss_type == 'jacobian':
            return orthogonality_loss_jacobian_efficient(z, q)
        elif loss_type == 'variance_preserving':
            return orthogonality_loss_variance_preserving(z, q)
        elif loss_type == 'output_spread':
            return orthogonality_loss_output_spread(z, q)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def sample_shape(batch_size, mins, mids, maxs, device=None):
        """
        Sample B x D values from per-dimension [mid - t*(mid-min), mid + t*(max-mid)].
        """
        mins = torch.as_tensor(mins, device=device)
        mids = torch.as_tensor(mids, device=device)
        maxs = torch.as_tensor(maxs, device=device)

        # Uniform in [-1, 1]
        r = torch.rand(batch_size, mids.shape[0], device=device) * 2 - 1

        left = mids - mins
        right = maxs - mids

        return torch.where(
            r < 0,
            mids + r * left,  # negative side
            mids + r * right,  # positive side
        )

    def sample_system_and_Epot_batch(system_def, t_schedule, batch_size):
        shape_ranges = config["subspace"]["shape_space_range"]

        mins = [x[0] for x in shape_ranges]
        mids = [x[1] for x in shape_ranges]
        maxs = [x[2] for x in shape_ranges]

        # scale outward from middle
        mins = torch.tensor(mins, device=device)
        mids = torch.tensor(mids, device=device)
        maxs = torch.tensor(maxs, device=device)

        shape = sample_shape(
            batch_size,
            mins,
            mids,
            maxs,
            device=device,
        )

        # apply schedule
        shape = mids + t_schedule * (shape - mids)

        # Sample batch of latent vectors
        z_batch = torch.randn((batch_size, config.subspace_dim), device=device)
        z_batch.requires_grad_()
        # Apply subspace in batch
        q_batch = apply_subspace_batch(z_batch, shape, t_schedule)

        E_pots = system.potential_energy_batch(system_def, q_batch, shape)

        return z_batch, q_batch, E_pots, shape

    #@torch.compile()
    def batch_repulsion_neo(z_batch, q_batch, t_schedule, system_def, sigma_scale):
        DIST_EPS = 1e-8
        B = z_batch.shape[0]
        dim = q_batch.shape[1]
        # Z distances
        z_dists_sq = torch.cdist(z_batch, z_batch, p=2).square()

        q_start = q_batch.unsqueeze(1).expand(B, B, dim).reshape(B*B, dim)
        q_diffs = q_batch.unsqueeze(1) - q_batch.unsqueeze(0)  # (B, B, dim)
        q_diffs_flat = q_diffs.reshape(B * B, -1)

        all_q_dists = (system.kinetic_energy_batch(system_def, q_start, q_diffs_flat, None) + DIST_EPS).view(B, B)

        # Combined log computation
        factor = torch.log(t_schedule * sigma_scale * z_dists_sq + DIST_EPS) - torch.log(all_q_dists)
        repel_term = 0.25 * factor.square().sum(dim=-1)

        return repel_term, {'mean_scale_log': -factor.mean().detach()}

    def batch_repulsion(z_batch, q_batch, t_schedule):
        DIST_EPS = 1e-8
        stats = {}
        B = q_batch.shape[0]

        all_z_dists = torch.cdist(z_batch, z_batch, p=2).square()

        q_i = q_batch.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
        q_j = q_batch.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
        q_diffs = q_j - q_i
        all_q_dists_flat = system.batched_kinetic_energy(system_def, q_i, q_diffs)
        all_q_dists = all_q_dists_flat.reshape(B, B)

        all_q_dists += DIST_EPS

        factor = torch.log(t_schedule * config["training"]["sigma_scale"] * all_z_dists + DIST_EPS) - torch.log(all_q_dists)
        repel_term = torch.sum((0.5 * factor) ** 2, dim=-1)

        stats['mean_scale_log'] = torch.mean(-factor)

        return repel_term, stats

    pbar = tqdm(total=config.n_train_iters, desc="Training", unit="iter")

    # pr = cProfile.Profile()
    # pr.enable()

    for i_train_iter in range(config.n_train_iters):

        t_schedule = i_train_iter / float(config.n_train_iters)

        optimizer.zero_grad()

        z_batch, q_batch, E_pots, shape = sample_system_and_Epot_batch(system_def, t_schedule, config.batch_size)

        expand_loss, repel_stats = batch_repulsion_neo(z_batch, q_batch, t_schedule, system_def, config["training"]["sigma_scale"])

        E_pot = E_pots.mean()
        E_exp = expand_loss.mean() * config["training"]["weight_expand"]
        # lip

        weight_landscape_reg = config["training"]["weight_landscape"] * max(t_schedule * 2-1, 0)
        if weight_landscape_reg > 0:
            E_landscape = landscape_loss(model, system, system_def, z_batch, q_batch, shape, t_schedule, device) * weight_landscape_reg
        else:
            E_landscape = torch.tensor(0.0, device=device)

        orth_weight = 1e-1
        #E_ortho = get_orthogonality_loss(z_batch, q_batch, loss_type='output_spread') * orth_weight
        E_ortho = 0#torch.clamp(E_ortho, max=50 * t_schedule)

        loss_curvature = 0#loss_multiscale_curvature(model, system, system_def, z_batch, q_batch, shape, t_schedule, device) * 1e-3 * t_schedule

        total_loss = E_pot + E_exp + E_landscape + E_ortho + loss_curvature

        total_loss.backward()

        optimizer.step()
        scheduler.step()

        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{total_loss.item():.6f}",
            'E_pot': f"{E_pot.item():.6f}",
            'E_exp': f"{E_exp.item():.6f}",
            'E_lip': f"{E_landscape.item():.6f}",
            'E_orth': f"{E_ortho:.6f}",
            'E_hess': f"{loss_curvature:.6f}",
            'stretch': f"{torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item():.6f}",
            't_sched': f"{t_schedule:.3f}"
        })
        if i_train_iter % config["training"]["save_every"] == 0:
            pbar.write(
                f'\n== iter {i_train_iter}/{config["training"]["n_train_iters"]}  ({100. * i_train_iter / config["training"]["n_train_iters"]:.2f}%)')
            pbar.write(f"   loss: {total_loss.item():.6f}")
            pbar.write(f"   E_pots: {E_pot.item():.6f}")
            pbar.write(f"   E_exp: {E_exp.item():.6f}")
            pbar.write(f"   mean metric stretch: {torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item():.6f}")
            save_model(model, model_spec, args, config, i_train_iter, t_schedule)

            # pr.disable()
            # s = io.StringIO()
            # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            # ps.print_stats(20)  # Show top 20 functions
            # print(s.getvalue())
            # pr = cProfile.Profile()
            # pr.enable()

    save_model(model, model_spec, args, config, "_final", 1.0)


def save_model(model, model_spec, args: Args, config: Config, suffix, t_schedule):
    filename = os.path.join(config.output_dir, config.experiment_name, args.experiment_id, f"data_{suffix}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename + ".pt")
    with open(filename + ".json", "w") as f:
        json.dump(model_spec, f)
    np.save(filename + "_info.npy", {
        "system": config.system_name,
        "problem_name": config.problem_name,
        "subspace_dim": config.subspace_dim,
        "t_schedule_final": t_schedule,
    })
    print(f"Saved model to {filename}.pt")



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    args = Args()

    config = load_config(args.config_file)

    train_system(args, config)