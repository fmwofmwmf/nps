import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from functools import partial
import argparse, json
import copy
import hashlib

import numpy as np
import torch
from loss import *
import polyscope as ps
import polyscope.imgui as psim
from gradients import *
from integrators import *
import layers
import subspace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation
from Args import Args
from config_utils import *
from integrators import choose_integrator
from energy_diagnostics import diagnose_energy_landscape
from log_utils import save_frames_universal

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():
    args = Args()
    config = load_config(args.config_file)
    system, system_def = system_to_name(config)
    system.training = False

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')

    #########################################################################
    ### Load subspace map (if given)
    #########################################################################
    k = config.subspace_dim

    # load subspace weights
    network = SmoothGradientBasisField(system.dim, k, config["network"]["n_hidden_layers"], config["network"]["n_neurons"])
    network.load_state_dict(torch.load("experiments/model_local.pt"))

    network.eval()
    def model_eval(q, log=False):
        if not log:
            return network.inference_recompute(system, system_def, q, space) if args.use_subspace else (
                fake_gradient_pca(system, system_def, q, k, space))
        bench = fake_gradient_pca(system, system_def, q, k, space)
        predict = network.inference_recompute(system, system_def, q, space)
        # print(bench, predict)
        if (log): print(f"Basis Error: {subspace_distance_loss(predict.unsqueeze(0), bench.unsqueeze(0)).mean() }")
        #print((bench - predict).norm().item(), bench.norm().item())
        return predict if args.use_subspace else bench

    model = model_eval


    #########################################################################
    ### Set up state & UI params
    #########################################################################

    integrator = choose_integrator(args.integrator_name)
    int_state = {}
    ## State of the system

    # UI state
    run_sim = False
    reduced = False
    optimize = False
    show_gradient, show_pca, show_model = False, False, False
    eval_energy_every = True
    update_viz_every = True
    space = torch.ones(config.shape_space_dim)
    record_count = args.record_frames
    # Set up state parameters

    def reset_state():
        int_state['q_t'] = system_def['init_pos']
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = torch.zeros_like(int_state['q_t'])

        system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

    def state_to_system(system_def, state, space):
        return state

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz
    system.visualize_set_nice_view(system_def, space)
    print(f"state_to_system dtype: {state_to_system(system_def, int_state['q_t'], space).dtype}")

    def eval_potential_energy(system_def, q, compare = False):

        pot = system.potential_energy(system_def, state_to_system(system_def, q, space), space)
        if compare:
            #print(q.shape, system.dim, q.dtype)
            bpot = system.potential_energy_batch(system_def, state_to_system(system_def, q, space).unsqueeze(0), space.view(1, space.shape[0]))
            return pot, bpot
        return pot

    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################
    energy_history = []
    max_energy_history = 500

    def latent_kinetic_energy(system_def, z, z_dot, space):
        z = z.clone().detach().requires_grad_(True)

        def decode(zz):
            return state_to_system(system_def, zz, space)

        # Jacobian dq/dz
        J = torch.autograd.functional.jacobian(decode, z)
        # J shape: (q_dim, z_dim)

        # Physical velocity
        q_dot = J @ z_dot

        # Use your existing KE
        return system.kinetic_energy(system_def, decode(z), q_dot)

    def analyze_gradient_basis(system, system_def, model, q, space, k=10):
        """
        Analyze and print gradient, model prediction, and PCA basis at current configuration.

        Args:
            system: Physics system
            system_def: System definition
            model: Trained gradient basis field model
            q: Current configuration
            space: Space parameter
            k: Number of basis vectors to analyze
        """
        dim = q.shape[0]

        # 1. Compute actual gradient
        q_grad = q.clone().requires_grad_(True)
        E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0), space.unsqueeze(0))[0]
        grad_full = torch.autograd.grad(E, q_grad)[0]  # (dim,)

        print("\n" + "=" * 60)
        print("GRADIENT BASIS ANALYSIS")
        print("=" * 60)

        # 2. Get model's predicted basis
        B_pred = model(q.unsqueeze(0)).squeeze(0)  # (dim, k)

        print(f"\nActual gradient norm: {grad_full.norm().item():.6f}")
        print(f"Gradient: {grad_full.detach().cpu().numpy()}")
        print(f"Basis: {B_pred.detach().cpu().numpy()}")

        cos_sim = (grad_full @ B_pred) / (grad_full.norm() * B_pred.norm())

        print(f"Cos Sim: {cos_sim}")
        print(f"Norm: {B_pred.norm(dim=0, keepdim=True)}")

        # 3. Project gradient onto predicted basis
        grad_coeffs_pred = B_pred.T @ grad_full  # (k,)
        grad_reconstructed_pred = B_pred @ grad_coeffs_pred  # (dim,)
        recon_error_pred = (grad_full - grad_reconstructed_pred).norm().item()

        print(f"\n--- MODEL PREDICTION ---")
        print(f"Predicted basis shape: {B_pred.shape}")
        print(f"Gradient coefficients in predicted basis: {grad_coeffs_pred.detach().cpu().numpy()}")
        print(f"Reconstruction error: {recon_error_pred:.6f}")
        print(f"Explained variance: {1 - (recon_error_pred / grad_full.norm().item()) ** 2:.4f}")

        # Check orthonormality
        G = B_pred.T @ B_pred
        ortho_error = (G - torch.eye(G.shape[0])).norm().item()
        print(f"Orthonormality error (||B^T B - I||): {ortho_error:.6e}")

        # 4. Compute PCA basis from local gradient samples
        # Sample gradients near current configuration
        n_samples = 1000
        perturbations = torch.randn(n_samples, dim) * 0.01
        q_samples = q.unsqueeze(0) + perturbations  # (n_samples, dim)

        # Compute gradients at samples
        gradients_list = []
        for i in range(n_samples):
            q_sample = q_samples[i].requires_grad_(True)
            E_sample = system.potential_energy_batch(system_def, q_sample.unsqueeze(0), space.unsqueeze(0))[0]
            grad_sample = torch.autograd.grad(E_sample, q_sample)[0]
            gradients_list.append(grad_sample)

        gradients_samples = torch.stack(gradients_list, dim=0)  # (n_samples, dim)

        # Compute PCA
        mean_grad = gradients_samples.mean(dim=0)
        centered = gradients_samples - mean_grad
        cov = (centered.T @ centered) / (n_samples - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        V_pca = eigenvectors[:, :k]  # (dim, k)

        print(f"\n--- PCA BASIS (from {n_samples} local samples) ---")
        print(f"Top {k} eigenvalues: {eigenvalues[:k].detach().cpu().numpy()}")
        print(f"Explained variance ratio: {(eigenvalues[:k].sum() / eigenvalues.sum()).item():.4f}")

        # Project gradient onto PCA basis
        grad_coeffs_pca = V_pca.T @ grad_full
        grad_reconstructed_pca = V_pca @ grad_coeffs_pca
        recon_error_pca = (grad_full - grad_reconstructed_pca).norm().item()

        print(f"Gradient coefficients in PCA basis: {grad_coeffs_pca.detach().cpu().numpy()}")
        print(f"Reconstruction error: {recon_error_pca:.6f}")

        # 5. Compare predicted basis to PCA basis
        # Subspace distance (order-invariant)
        P_pred = B_pred @ B_pred.T
        P_pca = V_pca @ V_pca.T
        subspace_distance = (P_pred - P_pca).norm().item()

        print(f"\n--- COMPARISON ---")
        print(f"Subspace distance (||P_pred - P_pca||): {subspace_distance:.6f}")

        # Grassmann distance
        M = B_pred.T @ V_pca
        grassmann_dist = torch.sqrt(k - (M ** 2).sum() + 1e-8).item()
        print(f"Grassmann distance: {grassmann_dist:.6f}")

        print("=" * 60 + "\n")

    data = torch.load("dataset.pt")
    targets_buffer = data["q"].to(device="cuda")
    pca_targets_buffer = data["y"].to(device="cuda")
    frames = []

    def main_loop():

        nonlocal run_sim, update_viz_every, eval_energy_every, optimize, space, record_count, reduced
        nonlocal show_gradient, show_pca, show_model

        new_space = []
        for i, (low, _, high) in enumerate(config["subspace"]["shape_space_range"]):
            _, val = psim.SliderFloat(f"space_{i}", space[i].item(), low, high)
            new_space.append(val)

        space = torch.tensor(new_space)

        # Helpers to build other parts of the UI
        system.build_system_ui(system_def)

        if update_viz_every or run_sim:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

        if eval_energy_every:

            E, PE = eval_potential_energy(system_def, int_state['q_t'], True)
            E_str = f"Potential energy: {E}, {PE.item()}"
            #print(E, int_state['q_t'])
            psim.TextUnformatted(E_str)
            KE = latent_kinetic_energy(system_def, int_state['q_t'], int_state['qdot_t'], space)
            E_str = f"Kinetic energy: {KE}"
            psim.TextUnformatted(E_str)

            total_E = (PE + KE).item()

            energy_history.append(total_E)
            if len(energy_history) > max_energy_history:
                energy_history.pop(0)

        psim.Separator()
        psim.TextUnformatted("=== TOTAL ENERGY ===")

        if len(energy_history) > 1:
            psim.PlotLines(
                "E(t)",
                energy_history,  # MUST be a Python list
                scale_min=min(energy_history),
                scale_max=max(energy_history),
                graph_size=(300, 80)
            )
        else:
            psim.TextUnformatted("Not enough samples yet")

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()
        if psim.Button("reset KE"):
            int_state['qdot_t'] = torch.zeros_like(int_state['q_t'])
        if psim.Button("reset force"):
            ext_forces = system_def['external_forces']
            if 'torque_strength' in ext_forces:
                ext_forces['torque_strength'] = 0
            if 'wind_strength' in ext_forces:
                ext_forces['wind_strength'] = 0

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        _, reduced_n = psim.Checkbox("reduced basis mode", reduced)
        if reduced_n and not reduced: print("Switched to REDUCED")
        if not reduced_n and reduced: print("Switched to FULL")
        reduced = reduced_n
        psim.SameLine()
        # if run_sim or psim.Button("single step"):
        # Number of steps per frame


        if psim.Button("Analyze Gradient Basis"):
            analyze_gradient_basis(system, system_def, model, int_state['q_t'], space, k=5)

        if psim.TreeNode("Gradient Visualization"):
            _, show_gradient = psim.Checkbox("Show Gradient", show_gradient)
            _, show_pca = psim.Checkbox("Show PCA Basis", show_pca)
            _, show_model = psim.Checkbox("Show Model Basis", show_model)

            system.visualize_with_gradients(
                system,
                system_def,
                int_state['q_t'],
                model,
                space,
                show_gradient=show_gradient,
                show_pca=show_pca,
                show_model=show_model
            )

            psim.TreePop()

        if psim.Button("generateHash"):
            print(run_sim_and_hash(
                    50,
                    system,
                    system_def,
                    model,
                    state_to_system,
                    int_state,
                    space,
                    reduced,
                    integrator
            ))

        steps_per_frame = 1
        if run_sim or psim.Button("single step"):
            for _ in range(steps_per_frame):
                dists = torch.norm(targets_buffer.cpu() - int_state['q_t'], dim=1)
                min_dist, idx = dists.min(dim=0)
                targ = fake_gradient_pca(system, system_def, int_state['q_t'], k, space)
                basis_diff = subspace_distance_loss(model(int_state['q_t']).unsqueeze(0), targ.unsqueeze(0).cpu())[0]
                if len(frames) % 15 == 0:
                    print("Distance to nearest reference:", min_dist.item())
                    print("Basis error wrt nearest reference:", basis_diff.item())
                    model(int_state['q_t'], True)
                frames.append((int_state['q_t'].detach().cpu(), fake_gradient_pca(system, system_def, int_state['q_t'], k, space).detach().cpu()))
                int_state['q_t'], int_state['qdot_t'], n = run_sim_step(system, system_def, model, state_to_system, int_state, space, reduced, integrator)

            psim.LabelText("Newton Residual", f"{n:.3e}")
            q_vis = state_to_system(system_def, int_state['q_t'], space)
            pos = system.visualize(system_def, q_vis, space, return_transforms=True)

    ps.set_user_callback(main_loop)
    ps.show()

    # print("\n=== Subspace distances between subsequent frames ===")
    # for i in range(1, len(frames)):
    #     V_prev = frames[i - 1][1]
    #     V_curr = frames[i][1]
    #
    #     dist = subspace_distance_loss(V_prev.unsqueeze(0), V_curr.unsqueeze(0))[0]
    #
    #     print(f"{i - 1} -> {i}: {dist:.6e}")

    input("press enter to save")
    qs, ys = zip(*frames)
    qs, ys = torch.stack(qs), torch.stack(ys)
    torch.save({"q": qs, "y": ys}, "dataset.pt")
    print(ys.shape)
    return





def run_sim_and_hash(
    n_frames,
    system,
    system_def,
    model,
    state_to_system,
    int_state,
    space,
    reduced,
    integrator,
    steps_per_frame=1,
):
    # Deep copy so we don't mutate the original state
    sim_state = copy.deepcopy(int_state)

    q_history = []

    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            sim_state['q_t'], sim_state['qdot_t'], _ = run_sim_step(
                system,
                system_def,
                model,
                state_to_system,
                sim_state,
                space,
                reduced,
                integrator,
            )

        # Store detached CPU tensor
        q_history.append(sim_state['q_t'].detach().cpu())

    # Stack and convert to bytes
    q_tensor = torch.stack(q_history)  # (n_frames, dim)
    q_bytes = q_tensor.numpy().tobytes()

    # Create stable SHA256 hash
    sim_hash = hashlib.sha256(q_bytes).hexdigest()

    return sim_hash
def run_sim_step(system, system_def, model, state_to_system, int_state, space, reduced, integrator):
    if reduced:
        return gradient_basis_step_newton_new(system,
                                              system_def,
                                              model,
                                              int_state['q_t'],
                                              int_state['qdot_t'], space)
    else:
        return integrator(system,
                          system_def,
                          state_to_system,
                          int_state['q_t'],
                          int_state['qdot_t'], space)

if __name__ == '__main__':
    device = "cpu"
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)

    main()
