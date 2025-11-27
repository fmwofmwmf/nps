# This is a sample Python script.
import cProfile
import io
import json
import os
import pstats

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from fem_model import FEMSystem

import layers
from layers import SubspaceMLP
from Args import Args
from rb_model import Rigid3DSystem


def train_system(args: Args, system, system_def, subspace_domain_dict, base_state, target_dim):

    in_dim = args.subspace_dim
    model_spec = {
        "in_dim": in_dim,
        "out_dim": target_dim,
        "model_type": args.model_type,
        "activation": args.activation,
        "MLP_hidden_layers": args.MLP_hidden_layers,
        "MLP_hidden_layer_width": args.MLP_hidden_layer_width,
    }

    model = SubspaceMLP(model_spec, base_output=base_state).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_frac)

    def apply_subspace(x, shape, t_schedule):
        z = torch.cat([x, shape], dim=-1)
        return model(z, t_schedule)

    def mollify_norm(x, eps=1e-20):
        return torch.sqrt(torch.sum(x**2) + eps)

    # def sample_system_and_Epot(system_def, t_schedule):
    #     z = torch.randn((args.subspace_dim,), device=device)
    #     q = apply_subspace(z, t_schedule)
    #     E_pot = system.potential_energy(system_def, q)
    #     return z, q, E_pot

    def apply_subspace_batch(z_batch, shape, t_schedule):
        # Concatenate along feature dim
        z_cond = torch.cat([z_batch, shape.unsqueeze(-1)], dim=-1)
        return model(z_cond, t_schedule)

    @torch.compile()
    def orthogonality_loss(z):
        """
        z: [batch_size, dim] (masked latent)
        Returns scalar loss penalizing correlation between dims
        """
        B, D = z.size()
        # center along batch
        z_centered = z - z.mean(dim=0, keepdim=True)
        # covariance matrix
        cov = (z_centered.T @ z_centered) / B  # [D, D]
        # zero out diagonal
        diag_mask = torch.eye(D, device=z.device)
        off_diag = cov * (1 - diag_mask)
        loss = (off_diag ** 2).sum()
        return loss

    def jacobian_orthogonality_loss(z, out):
        B, D = z.shape

        # per-sample scalar (do not collapse batch!)
        scalar = out.sum(dim=1)  # [B]

        grad = torch.autograd.grad(
            scalar,
            z,
            grad_outputs=torch.ones_like(scalar),
            create_graph=True,
            retain_graph=True
        )[0]  # [B, D]

        # Gram matrix (averaged over batch)
        G = (grad.transpose(0, 1) @ grad) / B  # [D, D]

        I = torch.eye(D, device=z.device)
        return ((G - I) ** 2).mean()

    def sample_system_and_Epot_batch(system_def, t_schedule, batch_size):
        shape_min = 0.1
        shape_max = 1.5

        shape = torch.rand(batch_size, device=device) * (shape_max - shape_min) + shape_min

        # Sample batch of latent vectors
        z_batch = torch.randn((batch_size, args.subspace_dim), device=device)
        z_batch.requires_grad_()
        # Apply subspace in batch
        q_batch = apply_subspace_batch(z_batch, shape, t_schedule)

        E_pots = system.potential_energy_batch(system_def, q_batch, shape) / shape

        return z_batch, q_batch, E_pots, shape

    def batch_repulsion_neo(z_batch, q_batch, t_schedule, system_def):
        DIST_EPS = 1e-8
        B = z_batch.shape[0]

        # Compute z distances more efficiently
        z_dists_sq = torch.cdist(z_batch, z_batch, p=2).square()

        # Compute q distances in batch without reshaping
        q_i = q_batch.unsqueeze(1).expand(B, B, -1)
        q_j = q_batch.unsqueeze(0).expand(B, B, -1)
        q_diffs = q_j - q_i
        q_diffs_flat = q_diffs.reshape(B * B, -1)

        all_q_dists_flat = system.kinetic_energy_batch(system_def, q_diffs_flat, None) + DIST_EPS
        all_q_dists = all_q_dists_flat.view(B, B)

        # Compute factor efficiently
        factor = torch.log(t_schedule * args.sigma_scale * z_dists_sq + DIST_EPS) - torch.log(all_q_dists)
        repel_term = torch.sum(0.25 * factor.square(), dim=-1)

        return repel_term, {'mean_scale_log': -factor.mean()}

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

        factor = torch.log(t_schedule * args.sigma_scale * all_z_dists + DIST_EPS) - torch.log(all_q_dists)
        repel_term = torch.sum((0.5 * factor) ** 2, dim=-1)

        stats['mean_scale_log'] = torch.mean(-factor)

        return repel_term, stats

    pbar = tqdm(total=args.n_train_iters, desc="Training", unit="iter")

    pr = cProfile.Profile()
    pr.enable()

    for i_train_iter in range(args.n_train_iters):

        t_schedule = i_train_iter / float(args.n_train_iters)

        optimizer.zero_grad()

        z_batch, q_batch, E_pots, shape = sample_system_and_Epot_batch(system_def, t_schedule, args.batch_size)

        expand_loss, repel_stats = batch_repulsion_neo(z_batch, q_batch, t_schedule, system_def)

        E_pot = E_pots.mean()
        E_exp = expand_loss.mean() * args.weight_expand
        total_loss = E_pot + E_exp
        total_loss.backward()

        optimizer.step()
        scheduler.step()

        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{total_loss.item():.6f}",
            'E_pot': f"{E_pot.item():.6f}",
            'E_exp': f"{E_exp.item():.6f}",
            'stretch': f"{torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item():.6f}",
            't_sched': f"{t_schedule:.3f}"
        })
        if i_train_iter % args.report_every == 0:
            pbar.write(
                f"\n== iter {i_train_iter}/{args.n_train_iters}  ({100. * i_train_iter / args.n_train_iters:.2f}%)")
            pbar.write(f"   loss: {total_loss.item():.6f}")
            pbar.write(f"   E_pots: {E_pot.item():.6f}")
            pbar.write(f"   E_exp: {E_exp.item():.6f}")
            pbar.write(f"   mean metric stretch: {torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item():.6f}")
            save_model(model, model_spec, args, i_train_iter, t_schedule)

            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Show top 20 functions
            print(s.getvalue())
            pr = cProfile.Profile()
            pr.enable()

    save_model(model, model_spec, args, "_final", 1.0)


def save_model(model, model_spec, args: Args, suffix, t_schedule):
    filename = os.path.join(args.output_dir, f"{args.model_type}_{suffix}")
    torch.save(model.state_dict(), filename + ".pt")
    torch.set_default_dtype(torch.float32)
    with open(filename + ".json", "w") as f:
        json.dump(model_spec, f)
    np.save(filename + "_info.npy", {
        "system": args.system_name,
        "problem_name": args.problem_name,
        "subspace_domain_type": args.subspace_domain_type,
        "subspace_dim": args.subspace_dim,
        "t_schedule_final": t_schedule,
    })
    print(f"Saved model to {filename}.pt")



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    args = Args()

    system, system_def = Rigid3DSystem.construct("links")

    target_dim = system.dim
    base_state = system_def['interesting_states'][0, :]

    # Construct the learned subspace operator
    in_dim = args.subspace_dim + system.cond_dim
    model_spec = layers.model_spec_from_args(args, in_dim, target_dim)

    train_system(args, system, system_def, None, base_state, target_dim)