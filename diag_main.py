import time
import torch
from torch import optim
from tqdm import tqdm
import layers
from layers import SubspaceMLP
from Args import Args
from rb_model import Rigid3DSystem


def profile_train_system(args: Args, system, system_def, base_state, target_dim):
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=torch.cuda.is_available())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_frac)

    def apply_subspace_batch(z_batch, shape, t_schedule):
        z_cond = torch.cat([z_batch, shape.unsqueeze(-1)], dim=-1)
        return model(z_cond, t_schedule)

    def sample_system_and_Epot_batch(system_def, t_schedule, batch_size):
        shape_min = 0.1
        shape_max = 1.5
        shape = torch.rand(batch_size, device=device) * (shape_max - shape_min) + shape_min
        z_batch = torch.randn((batch_size, args.subspace_dim), device=device)
        z_batch.requires_grad_()
        q_batch = apply_subspace_batch(z_batch, shape, t_schedule)
        E_pots = system.potential_energy_batch(system_def, q_batch, shape) / shape
        return z_batch, q_batch, E_pots, shape

    def batch_repulsion_neo(z_batch, q_batch, t_schedule, system_def, sigma_scale):
        DIST_EPS = 1e-8
        B = z_batch.shape[0]
        z_dists_sq = torch.cdist(z_batch, z_batch, p=2).square()
        q_diffs = q_batch.unsqueeze(1) - q_batch.unsqueeze(0)
        q_diffs_flat = q_diffs.reshape(B * B, -1)
        all_q_dists = (system.kinetic_energy_batch(system_def, q_diffs_flat, None) + DIST_EPS).view(B, B)
        factor = torch.log(t_schedule * sigma_scale * z_dists_sq + DIST_EPS) - torch.log(all_q_dists)
        repel_term = 0.25 * factor.square().sum(dim=-1)
        return repel_term, {'mean_scale_log': -factor.mean().detach()}

    # Warm-up
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        z_batch, q_batch, E_pots, shape = sample_system_and_Epot_batch(system_def, 0.5, args.batch_size)
        expand_loss, repel_stats = batch_repulsion_neo(z_batch, q_batch, 0.5, system_def, args.sigma_scale)
        E_pot = E_pots.mean()
        E_exp = expand_loss.mean() * args.weight_expand
        total_loss = E_pot + E_exp
        total_loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Detailed timing for 10 iterations
    print("\n" + "=" * 80)
    print("DETAILED TIMING BREAKDOWN (10 iterations)")
    print("=" * 80)

    timings = {
        'forward_sample': [],
        'forward_repulsion': [],
        'backward': [],
        'optimizer_step': [],
        'total_iter': []
    }

    for i in range(10):
        iter_start = time.perf_counter()

        # Zero grad
        optimizer.zero_grad()

        # Forward - sampling
        t1 = time.perf_counter()
        z_batch, q_batch, E_pots, shape = sample_system_and_Epot_batch(system_def, 0.5, args.batch_size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        timings['forward_sample'].append(t2 - t1)

        # Forward - repulsion
        t1 = time.perf_counter()
        expand_loss, repel_stats = batch_repulsion_neo(z_batch, q_batch, 0.5, system_def, args.sigma_scale)
        E_pot = E_pots.mean()
        E_exp = expand_loss.mean() * args.weight_expand
        total_loss = E_pot + E_exp
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        timings['forward_repulsion'].append(t2 - t1)

        # Backward
        t1 = time.perf_counter()
        total_loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        timings['backward'].append(t2 - t1)

        # Optimizer step
        t1 = time.perf_counter()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        timings['optimizer_step'].append(t2 - t1)

        iter_end = time.perf_counter()
        timings['total_iter'].append(iter_end - iter_start)

    # Print results
    print(f"\nBatch size: {args.batch_size}")
    print(f"Subspace dim: {args.subspace_dim}")
    print(f"Target dim: {target_dim}")
    print(f"Number of bodies: {system.n_bodies}")
    print(f"Device: {device}\n")

    for key, times in timings.items():
        avg_time = sum(times) / len(times)
        pct = (avg_time / timings['total_iter'][0]) * 100 if key != 'total_iter' else 100
        print(f"{key:20s}: {avg_time * 1000:7.2f} ms  ({pct:5.1f}%)")

    print(f"\n{'=' * 80}")
    avg_iter_time = sum(timings['total_iter']) / len(timings['total_iter'])
    print(f"Average iteration time: {avg_iter_time * 1000:.2f} ms")
    print(f"Throughput: {1.0 / avg_iter_time:.2f} iter/s")
    print(f"{'=' * 80}\n")

    # Now profile individual components
    print("\n" + "=" * 80)
    print("COMPONENT-LEVEL PROFILING")
    print("=" * 80)

    # Profile potential energy
    z_batch, q_batch, E_pots, shape = sample_system_and_Epot_batch(system_def, 0.5, args.batch_size)

    print("\nPotential energy breakdown:")
    t1 = time.perf_counter()
    E_pots = system.potential_energy_batch(system_def, q_batch, shape)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.perf_counter()
    print(f"  Total potential energy: {(t2 - t1) * 1000:.2f} ms")

    # Profile kinetic energy
    print("\nKinetic energy (for repulsion):")
    B = args.batch_size
    q_diffs = q_batch.unsqueeze(1) - q_batch.unsqueeze(0)
    q_diffs_flat = q_diffs.reshape(B * B, -1)

    t1 = time.perf_counter()
    all_q_dists = system.kinetic_energy_batch(system_def, q_diffs_flat, None)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.perf_counter()
    print(f"  Kinetic energy ({B * B} evaluations): {(t2 - t1) * 1000:.2f} ms")

    # Profile cdist
    print("\nDistance computations:")
    z_batch_test = torch.randn((args.batch_size, args.subspace_dim), device=device)
    t1 = time.perf_counter()
    z_dists_sq = torch.cdist(z_batch_test, z_batch_test, p=2).square()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.perf_counter()
    print(f"  Z-space cdist: {(t2 - t1) * 1000:.2f} ms")

    print(f"\n{'=' * 80}\n")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    args = Args()

    # Test with different batch sizes
    for batch_size in [128]:
        args.batch_size = batch_size
        print(f"\n{'#' * 80}")
        print(f"# TESTING WITH BATCH SIZE: {batch_size}")
        print(f"{'#' * 80}\n")

        system, system_def = Rigid3DSystem.construct("links")
        target_dim = system.dim
        base_state = system_def['interesting_states'][0, :]

        profile_train_system(args, system, system_def, base_state, target_dim)