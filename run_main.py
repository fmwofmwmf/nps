import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from functools import partial
import argparse, json

import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

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

    # If we're running on a use_subspace system, load it
    subspace_model_params = None
    subspace_dim = -1

    use_subspace = args.use_subspace
    if use_subspace:
        model_name = os.path.join(config.output_dir, config.experiment_name, args.experiment_id, f"data_{args.subspace_model}")
        print(f"Loading subspace from {model_name}")

        # load subspace weights
        with open(model_name + '.json', 'r') as json_file:
            subspace_model_spec = json.loads(json_file.read())
        subspace_model = layers.create_model(subspace_model_spec, device=device)
        subspace_model.load_state_dict(torch.load(model_name + ".pt"))

        subspace_model.eval()
        # load other info
        d = np.load(model_name + "_info.npy", allow_pickle=True).item()

        subspace_dim = d['subspace_dim']

        latent_comb_dim = system_def['interesting_states'].shape[0]
        t_schedule_final = d['t_schedule_final']

        def apply_subspace(x, space):
            return subspace_model(torch.concat((x, space,), dim=-1), t_schedule=t_schedule_final)

        if config.system_name != d['system']:
            raise ValueError("system name does not match loaded weights")
        if config.problem_name != d['problem_name']:
            raise ValueError("problem name does not match loaded weights")

    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    if use_subspace:
        print("Subspace dimension: " + str(subspace_dim))
        base_latent = torch.zeros(subspace_dim)

    #########################################################################
    ### Set up state & UI params
    #########################################################################

    integrator = choose_integrator(args.integrator_name)
    int_state = {}
    ## State of the system

    # UI state
    run_sim = False
    optimize = False
    eval_energy_every = True
    update_viz_every = True
    space = torch.ones(config.shape_space_dim)
    record_count = args.record_frames
    # Set up state parameters

    def reset_state():
        pass
        if use_subspace:
            int_state['q_t'] = base_latent

        else:
            int_state['q_t'] = system_def['init_pos']

        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = torch.zeros_like(int_state['q_t'])

        system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

    def state_to_system(system_def, state, space):
        return apply_subspace(state, space) if use_subspace else state


    subspace_fn = state_to_system

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz

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

        x = x0.clone().detach()

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

    def compute_energy_slice_2d_batch(system_def, q_current, space,
                                      i=0, j=1, n=30):
        low = -5
        high = 5

        xs = torch.linspace(low, high, n)
        ys = torch.linspace(low, high, n)

        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        Xf = X.flatten()
        Yf = Y.flatten()

        B = Xf.shape[0]

        q_batch = q_current.repeat(B, 1)
        q_batch[:, i] = Xf
        q_batch[:, j] = Yf

        space_batch = space.view(1, space.shape[0]).repeat(B, 1)

        states = state_to_system(system_def, q_batch, space_batch)
        E = system.potential_energy_batch(system_def, states, space_batch)

        return X, Y, E.view(n, n)



    record_frames = []
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

    def animate_encoder_warp(
            encoder_fn,
            grid_lim=1.0,
            grid_res=20,
            frames=120,
            interval=50,
            device="cpu"
    ):
        """
        Visualize how a 2D latent grid is warped by an encoder.

        encoder_fn: function (N,2) -> (N,D)
        """

        # --- Build grid lines ---
        xs = np.linspace(-grid_lim, grid_lim, grid_res)
        ys = np.linspace(-grid_lim, grid_lim, grid_res)

        lines = []
        for x in xs:
            lines.append(np.stack([np.full_like(ys, x), ys], axis=1))
        for y in ys:
            lines.append(np.stack([xs, np.full_like(xs, y)], axis=1))

        lines = np.stack(lines)  # (L, P, 2)
        L, P, _ = lines.shape

        z = torch.tensor(lines.reshape(-1, 2), dtype=torch.float32, device=device)
        z_flat = z.reshape(L * P, 2)  # (B·P, 2)

        with torch.no_grad():
            z_enc_flat = encoder_fn(z_flat)

        z_enc = z_enc_flat.reshape(L, P, 2).cpu().numpy()

        # --- Project to 2D if needed ---
        if z_enc.shape[2] > 2:
            z_mean = z_enc.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(z_enc - z_mean)
            z_enc = (z_enc - z_mean) @ Vh[:2].T

        z_enc = z_enc.reshape(L, P, 2)
        z = z.cpu().reshape(L, P, 2)

        # --- Plot setup ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-grid_lim * 2, grid_lim * 2)
        ax.set_ylim(-grid_lim * 2, grid_lim * 2)
        ax.set_aspect("equal")
        ax.set_title("Grid warp")

        orig_lines = []
        for i in range(L):
            ln, = ax.plot(z[i, :, 0], z[i, :, 1], lw=1, color="lightgray", alpha=0.5)
            orig_lines.append(ln)

        line_artists = [
            ax.plot([], [], lw=1)[0] for _ in range(L)
        ]

        # --- Animation ---
        def update(frame):
            t = frame / (frames - 1)

            interp = (1 - t) * z + t * z_enc
            interp = interp.detach().cpu().numpy()  # 🔑 IMPORTANT

            for i, ln in enumerate(line_artists):
                ln.set_data(interp[i, :, 0], interp[i, :, 1])

            ax.set_title(f"Encoder warp  t={t:.2f}")
            return line_artists

        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=False
        )

        plt.show(block=True)
        return anim



    def main_loop():

        nonlocal run_sim, base_latent, update_viz_every, eval_energy_every, optimize, space, record_count

        new_space = []
        for i, (low, _, high) in enumerate(config["subspace"]["shape_space_range"]):
            _, val = psim.SliderFloat(f"space_{i}", space[i].item(), low, high)
            new_space.append(val)

        space = torch.tensor(new_space)

        if use_subspace:
            psim.TextUnformatted(f"Subspace type: SUBSPACE")
        else:
            psim.TextUnformatted(f"Subspace type: FULL SPACE")
        if psim.TreeNode("explore current latent"):

            psim.TextUnformatted("This is the current state of the system.")

            any_changed = False

            # Make a detached clone to avoid modifying int_state in-place until confirmed
            tmp_state_q = int_state['q_t'].clone() if torch.is_tensor(int_state['q_t']) else int_state['q_t'].copy()
            _, optimize = psim.SliderFloat("optimize", optimize, 0, 4)
            if optimize > 0:
                tmp_state_q, _ = finite_diff_gd(eval_potential_energy, system_def, tmp_state_q, lr=(10 ** (-(4-optimize))), eps=1e-6, steps=10, tol=1e-8, verbose=False)
                any_changed = True

            low = -3
            high = 3
            if use_subspace:
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
                int_state['q_t'] = tmp_state_q
                system.visualize(system_def, state_to_system(system_def, int_state['q_t'], space), space)

            psim.TreePop()

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

        psim.Separator()
        psim.TextUnformatted("=== ENERGY VISUALIZATION ===")

        if psim.Button("Open Energy Surface (3D plt)"):
            with torch.no_grad():
                X, Y, Z = compute_energy_slice_2d_batch(
                    system_def,
                    int_state['q_t'],
                    space,
                    i=0,
                    j=1,
                    n=50
                )

            Xn = X.numpy()
            Yn = Y.numpy()
            Zn = Z.numpy()

            plt.figure("Energy Surface 3D")
            plt.clf()
            ax = plt.axes(projection="3d")

            surf = ax.plot_surface(
                Xn, Yn, Zn,
                cmap="viridis",
                linewidth=0
            )

            ax.set_xlabel("latent[0]")
            ax.set_ylabel("latent[1]")
            ax.set_zlabel("Energy")
            ax.set_title("Energy landscape slice")

            plt.colorbar(surf, shrink=0.6, aspect=10)
            plt.tight_layout()
            plt.show(block=False)

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()
        if psim.Button("reset force"):
            ext_forces = system_def['external_forces']
            if 'torque_strength' in ext_forces:
                ext_forces['torque_strength'] = 0
            if 'wind_strength' in ext_forces:
                ext_forces['wind_strength'] = 0

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        psim.SameLine()
        # if run_sim or psim.Button("single step"):
        # Number of steps per frame
        steps_per_frame = 2

        if psim.Button("cool button"):
            def decode(zz):
                return state_to_system(system_def, zz, space.unsqueeze(0).expand(zz.shape[0], -1))
            animate_encoder_warp(
                encoder_fn=decode,
                grid_lim=50.0,
                grid_res=25
            )

        if psim.Button("graph autograd"):
            results = diagnose_energy_landscape(
                system_def,
                int_state['q_t'],
                space,
                eval_potential_energy,
                subspace_fn=state_to_system
            )

        if psim.Button("graph fin diff"):
            results = diagnose_energy_landscape(
                system_def,
                int_state['q_t'],
                space,
                eval_potential_energy,
                subspace_fn=state_to_system,
                use_finite_diff=True
            )

        if run_sim or psim.Button("single step"):

            for _ in range(steps_per_frame):
                int_state['q_t'], int_state['qdot_t'], n = integrator(system,
                                                                      system_def,
                                                                      state_to_system,
                                                                      int_state['q_t'],
                                                                      int_state['qdot_t'], space)
            psim.LabelText("Newton Residual", f"{n:.3e}")
            q_vis = state_to_system(system_def, int_state['q_t'], space)
            pos = system.visualize(system_def, q_vis, space, return_transforms=True)

            # Record frame for later
            if record_count != 0:
                record_frames.append(pos)
                print(len(record_frames))
                if len(record_frames) >= record_count:
                    if use_subspace:
                        save_path = f"recordings/[{config.system_name}]{config.problem_name}_{model_name.split('/')[-1]}[{args.experiment_id}].usda"
                    else:
                        save_path = f"recordings/[{config.system_name}]_{config.problem_name}_full[{args.experiment_id}].usda"
                    save_frames_universal(save_path, record_frames, system, fps=24)
                    record_count = 0

    ps.set_user_callback(main_loop)
    ps.show()
    return



if __name__ == '__main__':
    device = "cpu"
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)

    main()
