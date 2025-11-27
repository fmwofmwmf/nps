import sys, os
from functools import partial
import argparse, json

import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

import integrators
# import igl

# Imports from this project
import layers
import subspace

from Args import Args
from fem_model import FEMSystem
from rb_model import Rigid3DSystem

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():
    args = Args()
    system, system_def = Rigid3DSystem.construct("links")

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')

    #########################################################################
    ### Load subspace map (if given)
    #########################################################################

    # If we're running on a use_subspace system, load it
    subspace_model_params = None
    subspace_dim = -1
    subspace_domain_dict = None
    if args.subspace_model:
        print(f"Loading subspace from {args.subspace_model}")

        # load subspace weights
        with open(args.subspace_model + '.json', 'r') as json_file:
            subspace_model_spec = json.loads(json_file.read())
        subspace_model = layers.create_model(subspace_model_spec)
        subspace_model.load_state_dict(torch.load(args.subspace_model + ".pt"))
        # _, subspace_model_static = eqx.partition(subspace_model, eqx.is_array)
        #
        # subspace_model_params = eqx.tree_deserialise_leaves(args.subspace_model + ".eqx",
        #                                                     subspace_model)
        subspace_model.eval()
        # load other info
        d = np.load(args.subspace_model + "_info.npy", allow_pickle=True).item()

        subspace_dim = d['subspace_dim']
        subspace_domain_dict = subspace.get_subspace_domain_dict(d['subspace_domain_type'])
        latent_comb_dim = system_def['interesting_states'].shape[0]
        t_schedule_final = d['t_schedule_final']

        def apply_subspace(x, space):
            return subspace_model(torch.concat((x, torch.tensor((space,))), dim=-1), t_schedule=t_schedule_final)

        if args.system_name != d['system']:
            raise ValueError("system name does not match loaded weights")
        if args.problem_name != d['problem_name']:
            raise ValueError("problem name does not match loaded weights")
    use_subspace = True

    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    if use_subspace:
        print("Subspace dimension: " + str(subspace_dim))

    #########################################################################
    ### Set up state & UI params
    #########################################################################

    ## Integrator setup
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, args.integrator)

    ## State of the system

    # UI state
    run_sim = False
    optimize = False
    eval_energy_every = True
    update_viz_every = True
    space = 0.5

    # Set up state parameters

    base_latent = torch.zeros(subspace_dim) + subspace_domain_dict['initial_val']

    def reset_state():
        pass
        if use_subspace:
            int_state['q_t'] = base_latent
        # else:
        #     int_state['q_t'] = system_def['init_pos']
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
            bpot = system.potential_energy_batch(system_def, state_to_system(system_def, q, space).unsqueeze(0), torch.tensor((space,)))
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

        # Define the GUI

        # some latent sliders
        if use_subspace:
            changed, space = psim.SliderFloat("space", space, .1, 2)

            psim.TextUnformatted(f"Subspace domain type: {subspace_domain_dict['domain_name']}")

            if psim.TreeNode("explore current latent"):

                psim.TextUnformatted("This is the current state of the system.")

                any_changed = False

                # Make a detached clone to avoid modifying int_state in-place until confirmed
                tmp_state_q = int_state['q_t'].clone() if torch.is_tensor(int_state['q_t']) else int_state['q_t'].copy()
                _, optimize = psim.SliderFloat("optimize", optimize, 0, 4)
                if optimize > 0:
                    tmp_state_q, _ = finite_diff_gd(eval_potential_energy, system_def, tmp_state_q, lr=(10 ** (-(4-optimize))), eps=1e-6, steps=10, tol=1e-8, verbose=False)
                    any_changed = True

                low = subspace_domain_dict['viz_entry_bound_low']
                high = subspace_domain_dict['viz_entry_bound_high']

                for i in range(subspace_dim):
                    s = f"latent_{i}"
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

        # Helpers to build other parts of the UI
        # integrators.build_ui(int_opts, int_state)
        system.build_system_ui(system_def)

        if update_viz_every or run_sim:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

        if eval_energy_every:

            E, BE = eval_potential_energy(system_def, int_state['q_t'], True)
            E_str = f"Potential energy: {E}, {BE.item()}"
            #print(E, int_state['q_t'])
            psim.TextUnformatted(E_str)

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()

        psim.SameLine()

        if psim.Button("stop velocity"):
            integrators.update_state(int_opts, int_state, int_state['q_t'], with_velocity=False)

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
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
    with torch.no_grad():
        main()
