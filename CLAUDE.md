# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **neural reduced-order modeling (ROM) framework** for mechanical systems (bar-joint mechanisms, pendulums, rigid bodies, FEM). It trains an MLP (`SmoothGradientBasisField`) to predict low-dimensional orthonormal subspaces that capture system dynamics, enabling faster simulation via subspace integration.

## Running the Code

All scripts read configuration from `Args.py` (edit that file to change what runs):

```bash
# Train a subspace model — reads Args.py for config_file, experiment_id, etc.
python train_gradient.py

# Interactive simulation with Polyscope visualization (compare full vs. reduced dynamics)
python run_main_gradient.py

# GUI tool for designing bar-joint mechanisms, export to objects/*.txt
python mechanism_maker.py
```

There are no build steps, linting configs, or test suites.

## Key Configuration

`Args.py` controls all runtime behavior:
- `config_file` — path to JSON in `configs/` (e.g., `configs/bar_walker_lots.json`)
- `integrator_name` — `"imp"` (implicit Euler, default), `"exp"`, `"imp_torch"`, `"imp_semi"`
- `use_subspace` — toggle between reduced and full-space dynamics
- `experiment_id` — subfolder in `experiments/` for checkpoints
- `subspace_model` — checkpoint step number to load (e.g., `"180000"`)

Config JSONs specify system geometry, physics params, network architecture, and training hyperparameters.

## Architecture

### Physics Layer
All physics systems live in `models/` and expose:
- `.potential_energy_batch(system_def, q, space)` → scalar energy
- `.physical_mass_matrix(system_def, q)` → mass matrix
- `.visualize(system_def, q, space, ...)` → Polyscope rendering

`config_utils.py` acts as a factory (`system_to_name()`) that instantiates the right system from a JSON config.

The primary system is **BarJoint2D** (`models/barjoint2d_model.py`): 2D bars with 3 DOF each (x, y, θ). Mechanism geometry is defined in `objects/*.txt` files, which `mechanism_maker.py` can generate.

### Neural Basis Field (`gradients.py`)
`SmoothGradientBasisField`: MLP mapping configuration q → orthonormal k-dimensional subspace B(q) ∈ ℝ^{dim×k}. Uses Gram-Schmidt (`gradient_helpers.py`) for orthonormalization. The basis includes k−1 smallest Hessian eigenvectors plus the gradient direction. Periodic DOFs (angles) are handled via a periodic mask applied every 3rd coordinate.

### Reduced Dynamics (`integrators.py`)
Implicit Euler in latent space z:
- Position: `z_new = z + dt * z_dot_new`
- Velocity: `M(z) * (z_dot_new - z_dot) = -dt * ∇_z E(z)`

Where M(z) = J^T M_phys J (latent mass matrix via Jacobian of the decoder). Newton-Raphson solves each step. `choose_integrator(name)` dispatches to the right stepper.

### Training (`train_gradient.py`)
1. Fill a configuration buffer by simulating the system
2. Sample batches of configurations z from the buffer
3. Compute ground-truth PCA bases from mass-whitened Hessian eigenvectors (`loss.py:fake_gradient_pca_batched_vmap`)
4. Predict bases via `SmoothGradientBasisField`
5. Optimize with Adam using `subspace_distance_loss` + `orthonormality_loss` + `landscape_loss`
6. Save checkpoints to `experiments/<experiment_id>/`

### Loss Functions (`loss.py`)
- `subspace_distance_loss` — Frobenius norm of projection matrix difference (primary loss)
- `orthonormality_loss` — penalizes deviation from B^T B = I
- `landscape_loss` / `curvature_loss` — Lipschitz-like regularization for smooth energy landscape
- `fake_gradient_pca_batched_vmap` — computes ground-truth Hessian eigenvectors via vmap

### Analytical Gradients (`analytical_gradients.py`)
Closed-form Hessian/gradient for BarJoint2D (used instead of autodiff for speed during training).

## Dependencies

PyTorch (with `torch.func`: vmap, jacrev, grad, hessian), Polyscope, NumPy, DearPyGui (mechanism editor), libigl, tqdm, matplotlib.