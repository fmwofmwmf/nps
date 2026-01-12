import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch

import polyscope as ps
import polyscope.imgui as psim

import integrators
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from Args import Args
from fem_model import FEMSystem
from rb_model import Rigid3DSystem
from rb_2d_model import Pendulum2DSystem

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():
    args = Args()

    # ------------------------------------------------------------
    # System
    # ------------------------------------------------------------
    system, system_def = Pendulum2DSystem.construct("double")
    dim = system_def["init_pos"].shape[0]

    print("System dimension:", dim)

    # ------------------------------------------------------------
    # Polyscope init
    # ------------------------------------------------------------
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_automatically_compute_scene_extents(False)

    # ------------------------------------------------------------
    # Integrator state
    # ------------------------------------------------------------
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, args.integrator)

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    run_sim = False
    update_viz_every = True
    eval_energy_every = True

    # Space is still a parameter of the system
    space = torch.tensor((1.0, 1.0, 1.0))

    # ------------------------------------------------------------
    # State initialization (FULL SPACE)
    # ------------------------------------------------------------
    def reset_state():
        int_state["q_t"] = system_def["init_pos"].clone()
        int_state["q_tm1"] = int_state["q_t"].clone()
        int_state["qdot_t"] = torch.zeros_like(int_state["q_t"])

        system.visualize(system_def, int_state["q_t"], space)

    reset_state()

    # ------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------
    def eval_potential_energy(system_def, q, compare=False):
        E = system.potential_energy(system_def, q, space)
        if compare:
            Eb = system.potential_energy_batch(
                system_def,
                q.unsqueeze(0),
                space.unsqueeze(0),
            )[0]
            return E, Eb
        return E

    # ------------------------------------------------------------
    # Finite-difference tools (still useful for debugging)
    # ------------------------------------------------------------
    def central_diff_grad(energy_fn, system_def, x, eps=1e-6):
        grad = torch.zeros_like(x)
        for i in range(x.numel()):
            e = torch.zeros_like(x)
            e[i] = eps
            grad[i] = (
                energy_fn(system_def, x + e)
                - energy_fn(system_def, x - e)
            ) / (2 * eps)
        return grad

    # ------------------------------------------------------------
    # Full-space dynamics (autograd force)
    # ------------------------------------------------------------
    def full_step(q, q_dot, dt=0.01, gamma=0.05, mass=1.0):
        q = q.clone().detach().requires_grad_(True)
        print(system_def['external_forces'])
        E = system.potential_energy_batch(system_def, q.unsqueeze(0), space.unsqueeze(0))[0]
        #E = system.potential_energy(system_def, q, space)
        F = -torch.autograd.grad(E, q, create_graph=False)[0]
        F_max = 1
        F = torch.clamp(F, -F_max, F_max)
        q_dot_new = q_dot + dt * F / mass - dt * gamma * q_dot
        q_new = q + dt * q_dot_new

        return q_new.detach(), q_dot_new.detach()

    def implicit_euler_step_linesearch(
            energy_fn,
            q, qdot,
            dt=1e-2,
            gamma=0.05,
            mass=1.0,
            newton_iters=1,
            alpha_init=1.0,
            beta=0.5,
            c=1e-4
    ):
        qn = q.detach()
        v = qdot.detach()

        qk = (qn + dt * v).clone().detach().requires_grad_(True)

        for _ in range(newton_iters):

            # --- Current residual ---
            E = energy_fn(qk)
            gradE = torch.autograd.grad(E, qk, create_graph=True)[0]

            G = qk - qn - dt * v + (dt * dt / mass) * gradE
            Phi = 0.5 * (G ** 2).sum()

            gradPhi = torch.autograd.grad(Phi, qk)[0]

            # --- Backtracking line search ---
            alpha = alpha_init

            while alpha > 1e-6:
                q_trial = (qk - alpha * gradPhi).detach().requires_grad_(True)

                E_t = energy_fn(q_trial)
                print(E_t)
                gradE_t = torch.autograd.grad(E_t, q_trial)[0]

                G_t = q_trial - qn - dt * v + (dt * dt / mass) * gradE_t
                Phi_t = 0.5 * (G_t ** 2).sum()

                if Phi_t <= Phi - c * alpha * (gradPhi ** 2).sum():
                    break

                alpha *= beta

            # Accept step
            qk = (qk - alpha * gradPhi).detach().requires_grad_(True)

        v_next = (qk - qn) / dt
        v_next *= (1.0 - gamma * dt)

        return qk.detach(), v_next.detach()

    # ------------------------------------------------------------
    # Energy slice visualization (full space)
    # ------------------------------------------------------------
    def compute_energy_slice_2d_batch(
        system_def, q_current, space, i=0, j=1, n=40, radius=0.5
    ):
        xs = torch.linspace(-radius, radius, n)
        ys = torch.linspace(-radius, radius, n)

        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        Xf = X.flatten()
        Yf = Y.flatten()

        B = Xf.shape[0]
        q_batch = q_current.repeat(B, 1)
        q_batch[:, i] += Xf
        q_batch[:, j] += Yf

        space_batch = space.view(1, 3).repeat(B, 1)

        E = system.potential_energy_batch(
            system_def, q_batch, space_batch
        )

        return X, Y, E.view(n, n)

    def latent_energy(z):
        return system.potential_energy_batch(
            system_def,
            z.unsqueeze(0),
            space.unsqueeze(0)
        )[0]
    # ------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------
    record_frames = []

    # ------------------------------------------------------------
    # Main UI loop
    # ------------------------------------------------------------
    def main_loop():
        nonlocal run_sim, update_viz_every, eval_energy_every, space

        # ---------------- UI: space ----------------
        psim.TextUnformatted("=== SPACE ===")
        _, sx = psim.SliderFloat("space x", space[0].item(), 0.1, 2.0)
        _, sy = psim.SliderFloat("space y", space[1].item(), 0.1, 2.0)
        _, sz = psim.SliderFloat("space z", space[2].item(), 0.1, 2.0)
        space = torch.tensor((sx, sy, sz))

        psim.Separator()

        # ---------------- UI: system ----------------
        system.build_system_ui(system_def)

        # ---------------- Energy ----------------
        if eval_energy_every:
            E, Eb = eval_potential_energy(system_def, int_state["q_t"], True)
            psim.TextUnformatted(
                f"Energy: {E:.6e}  | batch: {Eb:.6e}"
            )

        psim.Separator()

        # ---------------- Visualization ----------------
        if update_viz_every or run_sim:
            system.visualize(system_def, int_state["q_t"], space)

        # ---------------- Energy surface ----------------
        if psim.Button("Energy slice (3D)"):
            with torch.no_grad():
                X, Y, Z = compute_energy_slice_2d_batch(
                    system_def,
                    int_state["q_t"],
                    space,
                    i=0,
                    j=1,
                    n=80,
                )

            fig = plt.figure("Energy slice")
            plt.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(
                X.numpy(), Y.numpy(), Z.numpy(),
                cmap="viridis",
                linewidth=0
            )
            ax.set_xlabel("q[0]")
            ax.set_ylabel("q[1]")
            ax.set_zlabel("Energy")
            plt.tight_layout()
            plt.show(block=False)

        psim.Separator()

        # ---------------- Controls ----------------
        _, eval_energy_every = psim.Checkbox("eval energy", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("update viz", update_viz_every)

        if psim.Button("reset"):
            reset_state()

        psim.SameLine()
        if psim.Button("stop velocity"):
            int_state["qdot_t"].zero_()

        psim.SameLine()
        _, run_sim = psim.Checkbox("run", run_sim)

        # ---------------- Simulation ----------------
        if run_sim or psim.Button("step"):
            for _ in range(1):
                int_state['q_t'], int_state['qdot_t'] = implicit_euler_step_linesearch(
                    latent_energy,
                    int_state['q_t'],
                    int_state['qdot_t'],
                    dt=0.01
                )

            pos = system.visualize(
                system_def,
                int_state["q_t"],
                space,
                return_transforms=True
            )
            record_frames.append(pos)

    ps.set_user_callback(main_loop)
    ps.show()

    # ------------------------------------------------------------
    # USD export
    # ------------------------------------------------------------
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew("simulation2.usda")
    world = UsdGeom.Xform.Define(stage, "/World")

    num_frames = len(record_frames)
    num_bodies = len(record_frames[0])

    body_meshes = []
    for bid in range(num_bodies):
        body_path = f"/World/Body{bid}"
        body_xf = UsdGeom.Xform.Define(stage, body_path)
        mesh = UsdGeom.Mesh.Define(stage, body_path + "/Mesh")
        mesh.CreateSubdivisionSchemeAttr().Set("none")

        faces = record_frames[0][bid]["faces"]
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr(faces.flatten())
        body_meshes.append(mesh)

    for t, frame in enumerate(record_frames):
        for bid, data in enumerate(frame):
            body_meshes[bid].CreatePointsAttr().Set(
                data["vertices"], time=t
            )

    stage.GetRootLayer().Save()
    print("Saved", num_frames, "frames")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    main()
