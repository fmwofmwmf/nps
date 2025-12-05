#!/usr/bin/env python3
"""
Modern training script for neural subspace learning.
Uses config files and tensorboard logging.
"""
import argparse
import torch
from torch import optim
from tqdm import tqdm

from config_utils import load_config
from logger import TrainingLogger
from network import make_network
from rb_model import Rigid3DSystem
from fem_model import FEMSystem
import subspace


def batch_repulsion_neo(z_batch, q_batch, t_schedule, system, system_def, sigma_scale):
    """
    Compute repulsion energy using kinetic energy metric.
    Matches the original main.py implementation.
    """
    DIST_EPS = 1e-8
    B = z_batch.shape[0]
    
    # Z distances in latent space (Euclidean)
    z_dists_sq = torch.cdist(z_batch, z_batch, p=2).square()
    
    # Q distances in configuration space (using kinetic energy metric)
    q_diffs = q_batch.unsqueeze(1) - q_batch.unsqueeze(0)  # (B, B, dim)
    q_diffs_flat = q_diffs.reshape(B * B, -1)
    
    all_q_dists = (system.kinetic_energy_batch(system_def, q_diffs_flat, None) + DIST_EPS).view(B, B)
    
    # Log ratio of distances
    factor = torch.log(t_schedule * sigma_scale * z_dists_sq + DIST_EPS) - torch.log(all_q_dists)
    repel_term = 0.25 * factor.square().sum(dim=-1)
    
    stats = {
        'mean_scale_log': -factor.mean().detach().item(),
    }
    
    return repel_term, stats


def train(config):
    """Main training loop."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)  # Set default device for all tensor creation
    print(f"Using device: {device}")
    
    # Setup logging
    logger = TrainingLogger(
        log_dir=config.output_dir,
        experiment_name=config.experiment_name,
        config=config.to_dict()
    )
    
    # Construct system
    system_name = config.system_name
    problem_name = config.problem_name
    
    if system_name == "Rigid3D":
        system, system_def = Rigid3DSystem.construct(problem_name)
    elif system_name == "FEM":
        system, system_def = FEMSystem.construct(problem_name)
    else:
        raise ValueError(f"Unknown system: {system_name}")
    
    # Move system and system_def tensors to device
    system.to(device)
    for key in system_def:
        if isinstance(system_def[key], torch.Tensor):
            system_def[key] = system_def[key].to(device)
    
    # Override contact stiffness if specified in config
    if 'contact_stiffness' in config['system']:
        system_def['contact_stiffness'] = config['system']['contact_stiffness']
        logger.log_message(f"Contact stiffness: {system_def['contact_stiffness']}")
    
    logger.log_message(f"System: {system_name} - {problem_name}")
    logger.log_message(f"System dimension: {system_def['init_pos'].shape[0]}")
    
    # Setup subspace parameters
    subspace_dim = config.subspace_dim
    shape_space_dim = config.shape_space_dim
    subspace_domain_type = config['subspace']['domain_type']
    subspace_domain_dict = subspace.get_subspace_domain_dict(subspace_domain_type)
    
    logger.log_message(f"Subspace dimension: {subspace_dim}")
    logger.log_message(f"Shape space dimension: {shape_space_dim}")
    logger.log_message(f"Domain type: {subspace_domain_type}")
    
    # Create network
    base_state = system_def['interesting_states'][0, :]
    target_dim = system_def['init_pos'].shape[0]
    
    # Build config dict for network
    network_config = {
        'in_dim': subspace_dim + shape_space_dim,
        'out_dim': target_dim,
        'network': config['network']
    }
    
    model = make_network(network_config, base_output=base_state, device=str(device))
    
    # Setup optimizer
    optimizer_type = config['optimizer']['otype']
    if optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
            eps=config['optimizer']['epsilon'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
            eps=config['optimizer']['epsilon'],
            weight_decay=config['optimizer']['weight_decay'],
            fused=config['optimizer'].get('fused', False) if torch.cuda.is_available() else False
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['optimizer']['lr_decay_every'],
        gamma=config['optimizer']['lr_decay_frac']
    )
    
    # Training parameters
    n_train_iters = config.n_train_iters
    batch_size = config.batch_size
    report_every = config['training']['report_every']
    save_every = config['training']['save_every']
    sigma_scale = config['training']['sigma_scale']
    weight_expand = config['training']['weight_expand']
    expand_type = config['training']['expand_type']
    
    logger.log_message(f"Training for {n_train_iters} iterations")
    logger.log_message(f"Batch size: {batch_size}")
    
    # Curriculum schedule for t_schedule
    t_schedule = 0.0
    t_schedule_final = 1.0
    
    # Training loop
    pbar = tqdm(range(n_train_iters), desc="Training")
    
    for i_iter in pbar:
        # Update t_schedule (curriculum learning)
        progress = i_iter / n_train_iters
        t_schedule = progress * t_schedule_final
        
        # Sample latent codes from subspace domain (normal distribution)
        z_batch = torch.randn(batch_size, subspace_dim, device=device)
        z_batch.requires_grad_()
        
        # Sample shape space with curriculum: starts at [1,1,1], expands to [1,1,0.1-3.0]
        shape_min = 0.1
        shape_max = 3.0
        mn = t_schedule * shape_min + (1 - t_schedule)
        mx = t_schedule * shape_max + (1 - t_schedule)
        
        # First two dimensions stay at 1.0, third dimension varies
        shape_batch = torch.ones(batch_size, shape_space_dim, device=device)
        shape_batch[:, 2] = torch.rand(batch_size, device=device) * (mx - mn) + mn
        
        # Concatenate latent and shape
        input_batch = torch.cat([z_batch, shape_batch], dim=-1)
        
        # Forward pass through model
        q_batch = model(input_batch, t_schedule=t_schedule)
        
        # Compute potential energy for the batch (vectorized)
        E_pot_batch = system.potential_energy_batch(system_def, q_batch, shape_batch)
        E_pot_mean = torch.mean(E_pot_batch)
        
        # Compute repulsion loss (returns per-sample repulsion)
        expand_loss, repel_stats = batch_repulsion_neo(
            z_batch,
            q_batch,
            t_schedule,
            system,
            system_def,
            sigma_scale
        )
        
        E_exp = expand_loss.mean() * weight_expand
        
        # Total loss
        loss = E_pot_mean + E_exp
        
        # Check for NaN
        if torch.isnan(loss):
            logger.log_message(f"\n!!! NaN detected at iteration {i_iter} !!!")
            logger.log_message(f"E_pot_mean: {E_pot_mean.item()}")
            logger.log_message(f"E_exp: {E_exp.item()}")
            logger.log_message(f"t_schedule: {t_schedule}")
            logger.log_message(f"z_batch stats - mean: {z_batch.mean().item()}, std: {z_batch.std().item()}")
            logger.log_message(f"q_batch stats - mean: {q_batch.mean().item()}, std: {q_batch.std().item()}")
            logger.log_message("Stopping training due to NaN")
            break
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if i_iter % report_every == 0:
            # Log to tensorboard
            logger.log_metrics({
                'loss/total': loss.item(),
                'loss/potential_energy': E_pot_mean.item(),
                'loss/expansion': E_exp.item(),
                'training/t_schedule': t_schedule,
                'training/learning_rate': optimizer.param_groups[0]['lr'],
                'stats/mean_scale_log': repel_stats['mean_scale_log'],
                'stats/metric_stretch': torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item(),
            }, i_iter)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'E_pot': f"{E_pot_mean.item():.6f}",
                'E_exp': f"{E_exp.item():.6f}",
                'stretch': f"{torch.exp(torch.tensor(repel_stats['mean_scale_log'])).item():.6f}",
                't_sched': f"{t_schedule:.3f}"
            })
            
            logger.flush()
        
        # Save checkpoint
        if i_iter % save_every == 0 or i_iter == n_train_iters - 1:
            checkpoint_dir = logger.get_checkpoint_dir()
            checkpoint_path = checkpoint_dir / f"model_{i_iter}"
            
            # Save model weights
            torch.save(model.state_dict(), str(checkpoint_path) + ".pt")
            
            # Save model config with training info
            import json
            checkpoint_config = {
                'system': config['system'],
                'subspace': config['subspace'],
                'training': config['training'],
                'optimizer': config['optimizer'],
                'encoding': config['encoding'],
                'network': config['network'],
                'logging': config['logging'],
                'in_dim': subspace_dim + shape_space_dim,
                'out_dim': target_dim,
                't_schedule_final': t_schedule
            }
            
            with open(str(checkpoint_path) + ".json", 'w') as f:
                json.dump(checkpoint_config, f, indent=2)
            
            logger.log_message(f"Saved checkpoint: {checkpoint_path}")
            # for i, p in enumerate(model.parameters()):
            #     if p.grad is None:
            #         print(f"param[{i}] grad is None")
            #     else:
            #         print(f"param[{i}] grad norm:", p.grad.norm().item())
    
    # Final save and cleanup
    logger.log_message("\nTraining completed!")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description='Train neural subspace model')
    parser.add_argument('--config', type=str, default='configs/links.json',
                        help='Path to config JSON file (default: configs/links.json)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config)


if __name__ == '__main__':
    # # Disable compilation via TorchInductor
    # torch._dynamo.reset()
    # torch._dynamo.config.verbose = True
    # torch._dynamo.config.suppress_errors = True
    #
    # # Force AOTAutograd or eager mode instead of Inductor
    # torch._dynamo.optimize("eager")(lambda x: x)
    main()
