import sys, os
from functools import partial
import argparse, json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

import integrators

# Imports from this project
import subspace
from network import SubspaceMLP

from config_utils import load_config
from fem_model import FEMSystem
from rb_model import Rigid3DSystem

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def find_latest_checkpoint(experiment_dir):
    """
    Find the latest checkpoint in an experiment directory.
    Looks for the newest timestamped run folder, then the highest numbered checkpoint.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/links_subspace)
    
    Returns:
        Path to checkpoint without extension, or None if not found
    """
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"Warning: Experiment directory {experiment_dir} does not exist")
        return None
    
    # Find all timestamped run directories (YYYYMMDD_HHMMSS format)
    run_dirs = []
    for item in exp_path.iterdir():
        if item.is_dir():
            # Try to parse as timestamp
            try:
                # Check if it matches YYYYMMDD_HHMMSS pattern
                datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                run_dirs.append(item)
            except ValueError:
                continue
    
    if not run_dirs:
        print(f"Warning: No timestamped run directories found in {experiment_dir}")
        return None
    
    # Sort by directory name (timestamp) to get latest
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    print(f"Found latest run: {latest_run.name}")
    
    # Look for checkpoints directory
    checkpoint_dir = latest_run / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Warning: No checkpoints directory in {latest_run}")
        return None
    
    # Find all checkpoint files (.pt files in checkpoints directory)
    # Supports both model_*.pt and <problem_name>_*.pt patterns
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        print(f"Warning: No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Extract iteration numbers and find the highest
    # Handle patterns like: model_50000.pt, links_50000.pt, SubspaceMLP_50000.pt
    checkpoints = []
    for ckpt_file in checkpoint_files:
        try:
            stem = ckpt_file.stem  # e.g., "model_50000" or "links_50000"
            parts = stem.split('_')
            if len(parts) >= 2:
                iter_num = int(parts[-1])  # Get last part as iteration number
                checkpoints.append((iter_num, ckpt_file))
        except (IndexError, ValueError):
            continue
    
    if not checkpoints:
        print(f"Warning: Could not parse checkpoint iteration numbers")
        return None
    
    # Get checkpoint with highest iteration
    latest_iter, latest_file = max(checkpoints, key=lambda x: x[0])
    checkpoint_path = str(latest_file.with_suffix(''))  # Remove .pt extension
    
    print(f"Found latest checkpoint: iteration {latest_iter}")
    print(f"Loading from: {checkpoint_path}")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained neural subspace model')
    parser.add_argument('--config', type=str, default='configs/links.json',
                        help='Path to config JSON file (default: configs/links.json)')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to model checkpoint OR experiment directory. If directory provided, loads latest checkpoint.')
    parser.add_argument('--no-auto-load', action='store_true',
                        help='Disable automatic checkpoint loading from experiment directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Auto-detect checkpoint from config if not explicitly provided
    checkpoint_path = args.checkpoint
    if not checkpoint_path and not args.no_auto_load:
        # Try to infer experiment directory from config
        if hasattr(config, 'output_dir'):
            # Get experiment name from config (default to problem name)
            exp_name = getattr(config, 'experiment_name', f"{config.problem_name}_subspace")
            experiment_dir = os.path.join(config.output_dir, exp_name)
            print(f"Auto-detecting checkpoint from: {experiment_dir}")
            checkpoint_path = find_latest_checkpoint(experiment_dir)
    elif checkpoint_path:
        # Check if provided path is a directory
        if os.path.isdir(checkpoint_path):
            print(f"Checkpoint path is a directory, finding latest checkpoint...")
            checkpoint_path = find_latest_checkpoint(checkpoint_path)
    
    # Construct system from config
    system, system_def = Rigid3DSystem.construct(config.problem_name)

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')

    #########################################################################
    ### Load subspace map (if given)
    #########################################################################

    # If we're running on a use_subspace system, load it
    subspace_model_params = None
    subspace_dim = config.subspace_dim
    subspace_domain_dict = None
    use_subspace = False
    apply_subspace = None
    
    if checkpoint_path:
        print(f"Loading subspace from {checkpoint_path}")

        # load subspace weights
        with open(checkpoint_path + '.json', 'r') as json_file:
            subspace_model_spec = json.loads(json_file.read())
        subspace_model = SubspaceMLP(subspace_model_spec, base_output=None)
        subspace_model.load_state_dict(torch.load(checkpoint_path + ".pt"))
        subspace_model.eval()  # Set to eval mode
        
        # Optimize for inference - disable gradient computation
        for param in subspace_model.parameters():
            param.requires_grad = False
        
        # Load info from checkpoint JSON
        subspace_dim = subspace_model_spec['subspace']['dim']
        subspace_domain_type = subspace_model_spec['subspace']['domain_type']
        t_schedule_final = subspace_model_spec.get('t_schedule_final', 1.0)
        system_name = subspace_model_spec['system']['name']
        problem_name = subspace_model_spec['system']['problem_name']

        subspace_domain_dict = subspace.get_subspace_domain_dict(subspace_domain_type)
        latent_comb_dim = system_def['interesting_states'].shape[0]

        @torch.inference_mode()
        def apply_subspace(x, space):
            return subspace_model(torch.concat((x, space,), dim=-1), t_schedule=t_schedule_final)

        if config.system_name != system_name:
            raise ValueError(f"system name does not match loaded weights: {config.system_name} != {system_name}")
        if config.problem_name != problem_name:
            raise ValueError(f"problem name does not match loaded weights: {config.problem_name} != {problem_name}")
        
        use_subspace = True
    else:
        # No checkpoint loaded - use full system state directly
        print("No checkpoint loaded - running in full state mode")
        def apply_subspace(x, space):
            # In full mode, just return the state as-is (it's already full DOFs)
            return system_def['interesting_states'][0, :]

    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    if use_subspace:
        print("Subspace dimension: " + str(subspace_dim))

    #########################################################################
    ### Set up state & UI params
    #########################################################################

    ## Integrator setup
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, "implicit-proximal")  # Default integrator

    ## State of the system

    # UI state
    run_sim = False
    optimize = False
    eval_energy_every = True
    update_viz_every = True
    space = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32)

    # Set up state parameters
    subspace_domain_dict = subspace.get_subspace_domain_dict(config['subspace']['domain_type'])
    base_latent = torch.zeros(subspace_dim) + subspace_domain_dict['initial_val']

    def reset_state():
        if use_subspace:
            int_state['q_t'] = base_latent
        else:
            # In full state mode, use the reference state
            int_state['q_t'] = system_def['interesting_states'][0, :]
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = torch.zeros_like(int_state['q_t'])

        system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

    def state_to_system(system_def, state, space):
        return apply_subspace(state, space)

    baseState = state_to_system(system_def, base_latent, space)

    subspace_fn = state_to_system

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz

    print(f"state_to_system dtype: {state_to_system(system_def, int_state['q_t'], space).dtype}")

    def eval_potential_energy(system_def, q, compare = False):
        pot = system.potential_energy(system_def, state_to_system(system_def, q, space), space)
        if compare:
            bpot = system.potential_energy_batch(system_def, state_to_system(system_def, q, space).unsqueeze(0), space.view(1, 3))
            return pot, bpot
        return pot

    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################

    def central_diff_grad(energy_fn, system_def, x, eps=1e-6):
        """
        Finite-difference gradient for torch tensors.
        x: 1D torch tensor, dtype float32 or float64
        """
        n = x.numel()
        grad = torch.zeros_like(x)

        for i in range(n):
            # unit vector ei
            e = torch.zeros_like(x)
            e[i] = eps

            f_plus = energy_fn(system_def, x + e)
            f_minus = energy_fn(system_def, x - e)

            grad[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def finite_diff_gd(energy_fn, system_def, x0, lr=1e-2, eps=1e-6,
                       steps=100, tol=1e-8, verbose=False):

        x = x0.clone().detach().float()

        history = {'loss': [], 'grad_norm': []}

        for t in range(steps):
            loss = float(energy_fn(system_def, x))

            grad = central_diff_grad(energy_fn, system_def, x, eps)
            grad_norm = grad.norm().item()

            history['loss'].append(loss)
            history['grad_norm'].append(grad_norm)

            if verbose and (t % max(1, steps // 10) == 0):
                print(f"iter {t:4d} loss={loss:.6e} ||g||={grad_norm:.3e}")

            if grad_norm < tol:
                if verbose:
                    print("converged (grad norm < tol)")
                break

            # gradient descent step
            x = x - lr * grad

        return x, history

    def main_loop():

        nonlocal run_sim, base_latent, update_viz_every, eval_energy_every, optimize, space

        # ============================================================
        # SHAPE SPACE PARAMETERS
        # ============================================================
        psim.PushItemWidth(200)
        
        # ============================================================
        # SHAPE SPACE PARAMETERS (system-specific)
        # ============================================================
        psim.TextUnformatted("=== SHAPE PARAMETERS ===")
        
        # Get shape parameter names from system definition
        if hasattr(system, 'shape_param_names') and system.shape_param_names:
            shape_names = system.shape_param_names
        else:
            # Default names based on shape_space_dim
            shape_space_dim = space.shape[0]
            shape_names = [f"Shape_{i}" for i in range(shape_space_dim)]
        
        # Create sliders for each shape parameter
        new_space = []
        for i, name in enumerate(shape_names):
            if i < space.shape[0]:
                _, val = psim.SliderFloat(name, space[i].item(), 0.1, 2.0)
                new_space.append(val)
        space = torch.tensor(new_space, dtype=torch.float32)
        
        psim.Separator()

        # Latent exploration (only when using subspace)
        if use_subspace:
            psim.TextUnformatted("=== LATENT SPACE EXPLORATION ===")
            if psim.TreeNode("Latent Coordinates"):
                psim.TextUnformatted(f"Domain: {subspace_domain_dict['domain_name']} | Latent Dim: {subspace_dim}")
                psim.Spacing()

                any_changed = False

                # Make a detached clone to avoid modifying int_state in-place until confirmed
                tmp_state_q = int_state['q_t'].clone() if torch.is_tensor(int_state['q_t']) else int_state['q_t'].copy()
                
                psim.TextUnformatted("Gradient Descent:")
                _, optimize = psim.SliderFloat("Optimize Power", optimize, 0, 4)
                if optimize > 0:
                    tmp_state_q, _ = finite_diff_gd(eval_potential_energy, system_def, tmp_state_q, lr=(10 ** (-(4-optimize))), eps=1e-6, steps=10, tol=1e-8, verbose=False)
                    any_changed = True

                psim.Spacing()
                psim.TextUnformatted("Latent Coordinates:")
                low = subspace_domain_dict['viz_entry_bound_low']
                high = subspace_domain_dict['viz_entry_bound_high']

                for i in range(subspace_dim):
                    s = f"z[{i}]"
                    val = tmp_state_q[i].item() if torch.is_tensor(tmp_state_q[i]) else float(tmp_state_q[i])
                    changed, new_val = psim.SliderFloat(s, val, low, high)
                    if changed:
                        any_changed = True
                        tmp_state_q[i] = new_val  # direct tensor or array assignment

                if any_changed:
                    # Ensure we’re working with the same type
                    if torch.is_tensor(tmp_state_q):
                        tmp_state_q = tmp_state_q.clone().detach()
                    else:
                        tmp_state_q = np.array(tmp_state_q, dtype=np.float32)

                    integrators.update_state(int_opts, int_state, tmp_state_q, with_velocity=True)
                    integrators.apply_domain_projection(int_state, subspace_domain_dict)
                    system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

                psim.TreePop()
            
            psim.Separator()

        # ============================================================
        # SYSTEM CONTROLS
        # ============================================================
        psim.TextUnformatted("=== SYSTEM CONTROLS ===")
        if psim.TreeNode("System Options"):
            system.build_system_ui(system_def)
            psim.TreePop()
        
        psim.Separator()
        psim.PopItemWidth()

        # ============================================================
        # VISUALIZATION & ENERGY
        # ============================================================
        if update_viz_every or run_sim:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

        if eval_energy_every:

            E, BE = eval_potential_energy(system_def, int_state['q_t'], True)
            E_str = f"Potential energy: {E}, {BE.item()}"
            #print(E, int_state['q_t'])
            psim.TextUnformatted(E_str)

        _, eval_energy_every = psim.Checkbox("Eval Energy Every Frame", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("Update Viz Every Frame", update_viz_every)
        
        # ============================================================
        # SIMULATION CONTROLS
        # ============================================================
        if psim.Button("Reset State"):
            reset_state()

        psim.SameLine()

        if psim.Button("Stop Velocity"):
            integrators.update_state(int_opts, int_state, int_state['q_t'], with_velocity=False)

        psim.SameLine()

        _, run_sim = psim.Checkbox("Run Simulation", run_sim)
        psim.SameLine()
        # if run_sim or psim.Button("single step"):
        #     # all-important timestep happens here
        #     int_state = integrators.timestep(system,
        #                                      system_def,
        #                                      int_state,
        #                                      int_opts,
        #                                      subspace_fn=subspace_fn,
        #                                      subspace_domain_dict=subspace_domain_dict)

    ps.set_user_callback(main_loop)
    ps.show()


if __name__ == '__main__':
    device = "cpu"
    torch.set_default_device(device)
    
    with torch.no_grad():
        main()
