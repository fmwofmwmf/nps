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
from integrator_helpers import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation
from Args import Args
from config_utils import *
from integrators import choose_integrator
from energy_diagnostics import diagnose_energy_landscape
from log_utils import save_frames_universal
from gradient_helpers import *

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

    if args.use_subspace:
        network = SmoothGradientBasisField(system.dim, k, config["network"]["n_hidden_layers"], config["network"]["n_neurons"])
        model_path = os.path.join("experiments", f"{config.experiment_name}_{k}D_model_local.pt")
        network.load_state_dict(torch.load(model_path))

        network.eval()
    def model_eval(q, log=False):
        if not log:
            return network.inference_recompute(system, system_def, q, space) if args.use_subspace else (
                gradient_pca(system, system_def, q, k, space))
        bench = gradient_pca(system, system_def, q, k, space)
        predict = network.inference_recompute(system, system_def, q, space)
        # print(bench, predict)
        if log: print(f"Basis Error: {subspace_distance_loss(predict.unsqueeze(0), bench.unsqueeze(0)).mean() }")
        #print((bench - predict).norm().item(), bench.norm().item())
        return predict if args.use_subspace else bench

    model = model_eval


    #########################################################################
    ### Set up state & UI params
    #########################################################################

    integrator = choose_integrator(args.integrator_name)
    int_state = {}
    int_state_gt = {}
    ## State of the system

    # UI state
    run_sim = False
    run_alt_sim = False
    reduced = False
    optimize = False
    show_gradient, show_pca, show_model = False, False, False
    eval_energy_every = True
    update_viz_every = True
    space = torch.ones(config.shape_space_dim)
    record_count = args.record_frames
    # Set up state parameters

    def reset_state():
        nonlocal int_state_gt
        int_state['q_t'] = system_def['init_pos']
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = torch.zeros_like(int_state['q_t'])
        int_state_gt['q_t'] = system_def['init_pos']
        int_state_gt['q_tm1'] = int_state_gt['q_t']
        int_state_gt['qdot_t'] = torch.zeros_like(int_state['q_t'])

        system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)
        system.visualize(system_def, state_to_system(system_def, int_state_gt['q_t'], space), space,
                         name_prefix="GT",
                         offset=np.array(args.ground_truth_offset),
                         main_color=(0.8, 0.5, 0.2))

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

    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################
    energy_history = []
    max_energy_history = 500

    frames = []
    data = []

    def main_loop():

        nonlocal run_sim, update_viz_every, eval_energy_every, optimize, space, record_count, reduced
        nonlocal show_gradient, show_pca, show_model, run_alt_sim

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
            int_state_gt['qdot_t'] = torch.zeros_like(int_state['q_t'])
        if psim.Button("reset force"):
            ext_forces = system_def['external_forces']
            if 'torque_strength' in ext_forces:
                ext_forces['torque_strength'] = 0
            if 'wind_strength' in ext_forces:
                ext_forces['wind_strength'] = 0

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        _, reduced_n = psim.Checkbox("reduced basis mode", reduced)
        psim.Separator()
        psim.TextUnformatted(f"Configuration difference: {torch.norm(int_state['q_t'] - int_state_gt['q_t'])}")
        _, run_alt_sim = psim.Checkbox("Run GT Sim", run_alt_sim)
        if reduced_n and not reduced: print("Switched to REDUCED")
        if not reduced_n and reduced: print("Switched to FULL")
        reduced = reduced_n
        psim.SameLine()

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
                # if len(frames) % 1500 == 0:
                #     dists = torch.norm(targets_buffer.cpu() - int_state['q_t'], dim=1)
                #     targ = fake_gradient_pca(system, system_def, int_state['q_t'], k, space)
                #     min_dist, idx = dists.min(dim=0)
                #     basis_diff = subspace_distance_loss(model(int_state['q_t']).unsqueeze(0), targ.unsqueeze(0).cpu())[0]
                #     print("Distance to nearest reference:", min_dist.item())
                #     print("Basis error wrt nearest reference:", basis_diff.item())
                #     model(int_state['q_t'], True)
                # if len(frames) % 5 == 1:
                #     print("Integration analysis=====")
                #     targ = local_pca_one_step_random(
                #         system,
                #         system_def,
                #         int_state['q_t'],
                #         space,
                #         num_samples=128*128,
                #         vel_scale=1e-2,
                #         k=k,
                #     )
                #     targ = add_basis(targ, get_gradient(system, system_def, int_state['q_t'], space))
                #     error1 = compare_integration_error_single(
                #             system,
                #             system_def,
                #             int_state['q_t'],
                #             targ,
                #             space,
                #             int_state['qdot_t'],
                #     )
                #     targ = gradient_pca(system, system_def, int_state['q_t'], k, space)
                #     error2 = compare_integration_error_single(
                #         system,
                #         system_def,
                #         int_state['q_t'],
                #         targ,
                #         space,
                #         int_state['qdot_t'],
                #     )
                #     data.append([error1, error2])
                frames.append(int_state['q_t'].detach().cpu())
                # frames.append(1)
                ppos, pvel = int_state['q_t'], int_state['qdot_t']
                int_state['q_t'], int_state['qdot_t'], n = run_sim_step(system, system_def, model, state_to_system, int_state, space, reduced, integrator)
                if run_alt_sim:
                    pos_err, vel_err, pos_pct, vel_pct, bpos_pct, bvel_pct = compare_integration_error_basic(system,
                                                                                                             system_def, ppos, pvel, int_state['q_t'], int_state['qdot_t'],
                                                                                                             model(int_state['q_t']), space)
                    vel_err_percent = vel_err / (int_state['qdot_t'].norm() + 1e-12) * 100

                    # Display
                    psim.TextUnformatted(f"Error: {pos_err:.5e} q, {vel_err:.5e} ({vel_err_percent:.3f}%) q_dot")
                    psim.TextUnformatted(f"Error as percent of Delta: {pos_pct:.3f}% q, {vel_pct:.3f}% q_dot")
                    psim.TextUnformatted(f"Basis span: {bpos_pct:.3f}% q, {bvel_pct:.3f}% q_dot")

                    int_state_gt['q_t'], int_state_gt['qdot_t'], _ = run_sim_step(system, system_def, model, state_to_system,
                                                                            int_state_gt, space, False, integrator)

            psim.LabelText("Newton Residual", f"{n:.3e}")

            if run_alt_sim:
                system.visualize(system_def, state_to_system(system_def, int_state_gt['q_t'], space), space, return_transforms=True,
                                name_prefix="GT",
                                offset=np.array([0.5, 2, 3]),
                                main_color=(0.8, 0.5, 0.2),)
                system.visualize(system_def, state_to_system(system_def, int_state_gt['q_t'], space), space,
                                 name_prefix="GT",
                                 offset=np.array(args.ground_truth_offset),
                                 main_color=(0.8, 0.5, 0.2))
            pos = system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space, return_transforms=True)

    ps.set_user_callback(main_loop)
    ps.show()

    a = [p[0] for p in data]
    b = [p[1] for p in data]

    plt.figure()
    plt.plot(a, label="Perturb")
    plt.plot(b, label="Hessian")
    plt.xlabel("t")
    plt.ylabel("% error")
    plt.title("error")
    plt.legend()
    plt.show()

    # print("\n=== Subspace distances between subsequent frames ===")
    # for i in range(1, len(frames)):
    #     V_prev = frames[i - 1][1]
    #     V_curr = frames[i][1]
    #
    #     dist = subspace_distance_loss(V_prev.unsqueeze(0), V_curr.unsqueeze(0))[0]
    #
    #     print(f"{i - 1} -> {i}: {dist:.6e}")
    path = os.path.join("precompute", f"{config.experiment_name}_dataset.pt")
    input(f"press enter to save ({path}) {len(frames)} frames")
    qs = torch.stack(frames)
    torch.save({"q": qs}, path)
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
        return gradient_basis_step_newton_pos_only(system,
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
