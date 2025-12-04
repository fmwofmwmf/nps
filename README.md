# Neural Parametric Subspaces for Rigid Body Physics

Modern training system with JSON configuration, tensorboard logging, and CUDA-accelerated PyTorch networks.

## Quick Start

### 1. Install Dependencies

```bash
conda create -n nps python=3.10
conda activate nps
pip install -r requirements.txt
```

Required packages:
- PyTorch with CUDA 12.1+ support
- Geometry visualization (polyscope, potpourri3d)
- Logging tools (tensorboard, tqdm)

### 2. Configure Training

Edit `configs/links.json` to customize training:

```json
{
    "system": {
        "name": "Rigid3D",
        "problem_name": "links",
        "contact_stiffness": 1e7
    },
    "subspace": {
        "dim": 8,
        "shape_space_dim": 3,
        "domain_type": "normal"
    },
    "training": {
        "n_train_iters": 500000,
        "batch_size": 64,
        "report_every": 100,
        "save_every": 10000,
        "weight_expand": 1.0,
        "sigma_scale": 1.0,
        "expand_type": "iso"
    },
    "optimizer": {
        "otype": "AdamW",
        "learning_rate": 5e-5,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "fused": true,
        "lr_decay_every": 100000,
        "lr_decay_frac": 0.5
    },
    "encoding": {
        "otype": "Identity"
    },
    "network": {
        "otype": "SubspaceMLP",
        "activation": "ELU",
        "output_activation": "None",
        "n_neurons": 128,
        "n_hidden_layers": 5
    },
    "logging": {
        "output_dir": "./experiments",
        "experiment_name": "links"
    }
}
```

### 3. Train

```bash
python train.py --config configs/links.json
```

Training outputs:
- Checkpoints: `experiments/links/YYYYMMDD_HHMMSS/checkpoints/model_*.{pt,json}`
- Logs: `experiments/links/YYYYMMDD_HHMMSS/logs/training.log`
- Tensorboard: `experiments/links/YYYYMMDD_HHMMSS/tensorboard/`

### 4. Visualize Results

**Auto-load latest checkpoint:**
```bash
python inference.py --config configs/links.json
```

**Specify checkpoint:**
```bash
python inference.py --config configs/links.json --checkpoint experiments/links
```

The inference tool will:
- Auto-detect the latest checkpoint from experiment directory
- Load model and configuration
- Launch interactive 3D visualization with polyscope

## Network Architecture

### SubspaceMLP (Pure PyTorch)
Standard MLP with proper weight initialization:
```json
"network": {
    "otype": "SubspaceMLP",
    "activation": "ELU",
    "n_neurons": 128,
    "n_hidden_layers": 5
}
```

**Supported activations:**
- `"ELU"` - Exponential Linear Unit (recommended, uses custom initialization)
- `"ReLU"` - Rectified Linear Unit (Kaiming initialization)
- `"Sine"` - Sine activation (SIREN initialization for coordinate networks)

**Key features:**
- Curriculum learning via `t_schedule` parameter
- Base output initialized to system rest state
- Proper weight initialization per activation type
- Residual output: `output = base_state + t_schedule * network(z)`

## System Configuration

### Contact Stiffness
Controls penetration penalty between rigid bodies:
```json
"system": {
    "contact_stiffness": 1e8
}
```

### Shape Space
Dynamic shape parameters per problem:
- **Links**: `["Link Width", "Link Thickness", "Link Length"]`
- **Klann/Stewart**: No shape space (fixed geometry)

Curriculum: starts at `[1,1,1]`, expands 3rd dim to `[1,1,0.1-3.0]`

## Training Parameters

### Repulsion Loss
Uses kinetic energy metric for proper Riemannian distance:
```json
"training": {
    "weight_expand": 1.0,    // Repulsion weight (balance with potential energy)
    "sigma_scale": 1.0,      // Distance scaling (affects repulsion strength)
    "expand_type": "iso"     // Isotropic distance metric
}
```

Formula: `loss = E_potential + weight_expand * E_repulsion`

Where: `E_repulsion = 0.25 * (log(t*sigma*z_dist²) - log(q_dist))²`

### Learning Rate Schedule
```json
"optimizer": {
    "learning_rate": 5e-5,      // Start conservative for stability
    "lr_decay_every": 100000,   // Decay every 100k iterations
    "lr_decay_frac": 0.5        // Multiply by 0.5 each decay
}
```

Schedule: 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 3.125e-6

## Logging & Monitoring

### Tensorboard
View training progress:
```bash
tensorboard --logdir=./experiments/links
```

**Tracked metrics:**
- `loss/total` - Combined loss (can be negative!)
- `loss/potential_energy` - Physical energy (negative = stable)
- `loss/expansion` - Repulsion loss (should decrease then stabilize)
- `stats/metric_stretch` - Latent space stretch (aim for ~0.8-1.0)
- `stats/mean_scale_log` - Log of distance ratio
- `training/t_schedule` - Curriculum progress (0→1)
- `training/learning_rate` - Current LR

### Console Output
Real-time progress bar with:
```
loss: -45.234 | E_pot: -52.118 | E_exp: 6.884 | stretch: 0.856 | t_sched: 0.247
```

## Interactive Visualization

The inference tool provides:

**Shape Space Controls:**
- Dynamic sliders based on `system.shape_param_names`
- Real-time geometry updates

**Latent Space Exploration:**
- 8D latent coordinate sliders
- Gradient descent optimizer (adjust power 0-4)
- Domain bounds from subspace configuration

**System Controls:**
- Energy evaluation (potential + batch comparison)
- Visualization updates
- State reset

## Problem Types

### Links (24-body chain)
```json
"problem_name": "links"
```
- 24 interlocking rigid bodies
- Shape space: width, thickness, length
- Contact-rich dynamics

### Klann Linkage
```json
"problem_name": "klann"
```
- 5-body walking mechanism
- Fixed geometry (no shape space)
- Complex joint constraints

### Stewart Platform
```json
"problem_name": "stewart"  
```
- Hexapod parallel manipulator
- Fixed geometry
- 6-DOF motion

## File Structure

```
configs/
  links.json           # Training configuration
  
train.py               # Modern training script
inference.py           # Interactive visualization (auto-loads latest)
run_main.py            # Legacy visualization (manual checkpoint)

config_utils.py        # Config loading with convenience properties
logger.py              # TrainingLogger with tensorboard
network.py             # SubspaceMLP with proper initialization
rb_model.py            # Rigid body physics (Rigid3DSystem)
fem_model.py           # FEM physics (FEMSystem)
subspace.py            # Subspace domain utilities

experiments/           # Training outputs (gitignored)
  links/
    YYYYMMDD_HHMMSS/
      checkpoints/     # model_*.{pt,json}
      logs/            # training.log
      tensorboard/     # tensorboard events
```

## Device Handling

All scripts use `torch.set_default_device(device)` for automatic GPU placement:
- Tensors created without `device=` arg automatically use CUDA
- System tensors moved to device during initialization
- No manual `.to(device)` calls needed in training loop

## Checkpoint Format

**Modern format** (JSON + PyTorch state dict):
```
model_50000.pt       # PyTorch state dict
model_50000.json     # Full config + metadata
```

JSON contains:
- System configuration
- Subspace settings
- Network architecture
- Training state (`t_schedule_final`)