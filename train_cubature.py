"""
train_cubature.py

Learns cubature weights for the regular joint springs in a BarJoint2D system.

Method: ECSW with gradient + optional Hessian matching (NNLS + OMP).
  For each sampled pose q_s with basis B(q_s):
    gradient row:   A[:, j]  = B^T ∇_q E_j(q_s)           (k,) per joint
    hessian rows:   A[:, j] += B^T H_j(q_s) B[:,i]         (k*k,) flattened per joint
  Solve:  min ||Aw - b||_2  s.t.  w >= 0  (OMP + NNLS)

Usage:
    python train_cubature.py          # reads Args.py for config + experiment_id
    python train_cubature.py --config_file configs/bar_walker_lots.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from scipy.optimize import nnls
from tqdm import tqdm

from Args import Args
from config_utils import load_config, system_to_name


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Max number of poses to sample from the dataset for NNLS.
MAX_SAMPLES = 500

# Whether to include Hessian-vector products in the NNLS system.
# More accurate but k× slower to build.  Strongly recommended when gradient-only
# cubature gives poor integration results.
MATCH_HESSIAN = True

# OMP: stop when relative residual drops below this threshold.
OMP_TOL = 1e-3   # tighter than default 1e-2

# Cubature fractions to solve and save simultaneously.
# Each entry produces a separate file: {name}_cubature_{pct}pct.pt
# The first entry is also saved as {name}_cubature.pt (the default loaded at runtime).
CUBATURE_FRACS = [0.05, 0.1, 0.20, 0.30, 0.67]

# Samples processed per vmap call for the Hessian rows.
# Each sample needs ~n_joints × dim × 8 bytes of GPU memory during the nested
# jacrev; lower this if you hit OOM.
HESS_BATCH = 16


def build_nnls_matrix(system, system_def, model, q_samples, match_hessian=MATCH_HESSIAN):
    """Build the NNLS matrix A and vector b for cubature weight solving.

    Gradient rows (always):
      A_block[:, j] = B^T ∇_q E_j(q)          (k,) per joint per sample
      b_block       = B^T ∇_q Σ_j E_j(q)       (k,)

    Hessian rows (when match_hessian=True):
      A_block[:, j] += flatten(B^T H_j(q) B)   (k*k,) per joint per sample
      b_block       += flatten(B^T H_full(q) B)  (k*k,)

    Each sample's block is L2-normalised so all poses contribute equally
    regardless of energy magnitude.

    Uses vmap(jacrev) to compute all sample Jacobians in one batched call
    instead of a Python loop.
    """
    n_joints = system.num_joints
    N = len(q_samples)

    # Stack all samples into a single batch tensor
    q_batch = torch.stack([q.to(device=DEVICE, dtype=torch.float64) for q in q_samples])  # (N, dim)

    # Batch model inference — one forward pass per sample but no jacrev overhead
    print("Running model inference...")
    with torch.no_grad():
        B_batch = torch.stack([model(q).squeeze(0) for q in q_batch])  # (N, dim, k)
    k = B_batch.shape[2]

    def per_joint_fn(q):
        return system.energy_per_joint_batch(system_def, q.unsqueeze(0)).squeeze(0)

    # ── All Jacobians in one vmap call: (N, n_joints, dim) ───────────────────
    print("Computing Jacobians (batched)...")
    J_batch = torch.func.vmap(torch.func.jacrev(per_joint_fn))(q_batch)

    # Gradient rows for all samples: A_g (N, k, n_joints), b_g (N, k)
    # A_g[n] = B[n]^T @ J[n]^T
    A_g_batch = torch.bmm(B_batch.permute(0, 2, 1), J_batch.permute(0, 2, 1))   # (N, k, n_joints)
    # b_g[n] = B[n]^T @ J[n].sum(joints)
    b_g_batch = torch.einsum('ndk,nd->nk', B_batch, J_batch.sum(dim=1))          # (N, k)

    # ── Hessian rows via mini-batched vmap HVPs ──────────────────────────────
    # Each sample in the nested jacrev needs ~n_joints × dim × 8 bytes of
    # scratch memory, so we vmap over HESS_BATCH samples at a time.
    A_h_list = []  # one (N, k, n_joints) per basis direction
    b_h_list = []  # one (N, k) per basis direction
    if match_hessian:
        def directional_jac_fn(q_, v_):
            # d/dq [J(q) @ v] using reverse-over-forward (jvp for inner product).
            # jvp computes J(q)v in one forward pass — O(dim) memory vs O(n_joints²×dim)
            # for reverse-over-reverse.
            def fn(q):
                _, Jv = torch.func.jvp(per_joint_fn, (q,), (v_,))
                return Jv   # (n_joints,)
            return torch.func.jacrev(fn)(q_)   # (n_joints, dim)

        for i in range(k):
            print(f"Computing HVPs for basis direction {i+1}/{k} "
                  f"(mini-batches of {HESS_BATCH})...")
            v_batch = B_batch[:, :, i].detach()  # (N, dim)

            chunks = []
            for start in range(0, N, HESS_BATCH):
                end = min(start + HESS_BATCH, N)
                chunks.append(torch.func.vmap(directional_jac_fn)(
                    q_batch[start:end], v_batch[start:end]))
            dJv_batch = torch.cat(chunks, dim=0)   # (N, n_joints, dim)

            A_h_list.append(torch.bmm(B_batch.permute(0, 2, 1), dJv_batch.permute(0, 2, 1)))  # (N, k, n_joints)
            b_h_list.append(torch.einsum('ndk,nd->nk', B_batch, dJv_batch.sum(dim=1)))         # (N, k)

    # ── Assemble A and b with per-sample L2 normalisation ────────────────────
    rows_per_sample = k * (1 + k) if match_hessian else k
    A_np = np.empty((N * rows_per_sample, n_joints), dtype=np.float64)
    b_np = np.empty(N * rows_per_sample, dtype=np.float64)

    A_g_np = A_g_batch.cpu().numpy()   # (N, k, n_joints)
    b_g_np = b_g_batch.cpu().numpy()   # (N, k)
    A_h_np = [x.cpu().numpy() for x in A_h_list]  # k × (N, k, n_joints)
    b_h_np = [x.cpu().numpy() for x in b_h_list]  # k × (N, k)

    for n in range(N):
        if match_hessian:
            A_block = np.vstack([A_g_np[n]] + [A_h_np[i][n] for i in range(k)])
            b_block = np.concatenate([b_g_np[n]] + [b_h_np[i][n] for i in range(k)])
        else:
            A_block = A_g_np[n]
            b_block = b_g_np[n]
        scale = np.linalg.norm(b_block) + 1e-12
        start = n * rows_per_sample
        A_np[start:start + rows_per_sample] = A_block / scale
        b_np[start:start + rows_per_sample] = b_block / scale

    print(f"NNLS system: {A_np.shape[0]} rows ({N} samples × "
          f"{rows_per_sample} rows/sample), {n_joints} joints")
    return A_np, b_np


def _nnls_l1(A, b, lam, threshold=1e-3):
    """NNLS with L1 regularisation via augmented system.

    Minimises  ||Aw - b||² + λ ||w||₁  s.t.  w ≥ 0.
    (With w ≥ 0, ||w||₁ = 1ᵀw, so augmenting A with √λ·I achieves this.)
    """
    n = A.shape[1]
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    w, _ = nnls(A_aug, b_aug)
    if w.max() > 0:
        w[w < threshold * w.max()] = 0.0
    return w


def solve_cubature(A, b, max_frac=CUBATURE_FRACS[0], tol=OMP_TOL):
    """L1-regularised NNLS with binary search on λ.

    Finds the globally optimal sparse non-negative weights (vs. greedy OMP).
    Binary-searches λ to get the sparsest solution within `tol` relative error
    and at most `max_points` non-zero weights.

    Returns (indices, weights) as numpy arrays.
    """
    n_joints = A.shape[1]
    max_points = max(1, int(round(n_joints * max_frac)))
    b_norm = np.linalg.norm(b) + 1e-12
    print(f"L1-NNLS cubature: A={A.shape}, {n_joints} joints, "
          f"tol={tol:.1e}, max_pts={max_points} ({max_frac*100:.0f}%)")

    def eval_lam(lam):
        w = _nnls_l1(A, b, lam)
        r = b - A @ w
        return w, np.linalg.norm(r) / b_norm, int((w > 0).sum())

    # ── Find lambda bounds ────────────────────────────────────────────────────
    # λ=0 → trivial w≈1 (accurate but dense); λ large → w≈0 (sparse but wrong)
    lam_lo, lam_hi = 1e-10, 1e2

    w_lo, err_lo, n_lo = eval_lam(lam_lo)
    w_hi, err_hi, n_hi = eval_lam(lam_hi)
    print(f"  bounds: λ={lam_lo:.1e} → err={err_lo:.4f} n={n_lo} | "
          f"λ={lam_hi:.1e} → err={err_hi:.4f} n={n_hi}")

    if err_lo > tol:
        print(f"  ⚠ Cannot reach tol={tol:.1e} even at λ_min — "
              f"using λ_min solution (err={err_lo:.4f})")
        w_best = w_lo
    else:
        # Binary search: largest λ (sparsest) that still meets tol AND max_points
        w_best = w_lo
        for step in range(25):
            lam_mid = np.sqrt(lam_lo * lam_hi)   # geometric bisection
            w_mid, err_mid, n_mid = eval_lam(lam_mid)

            feasible = err_mid <= tol and n_mid <= max_points
            bar = '█' * int((1 - min(err_mid, 1)) * 20) + '░' * int(min(err_mid, 1) * 20)
            print(f"  [{step+1:2d}] λ={lam_mid:.2e}  err={err_mid:.4f}  "
                  f"n={n_mid:3d}  |{bar}|  {'✓' if feasible else '✗'}")

            if feasible:
                w_best = w_mid
                lam_lo = lam_mid   # can go sparser
            else:
                lam_hi = lam_mid   # need to be more accurate

            if lam_hi / lam_lo < 1.05:
                break

    indices = np.where(w_best > 0)[0]
    weights = w_best[indices]

    # Hard-enforce max_points only when the binary search found a genuinely
    # sparse solution (weights are non-uniform).  If all weights are nearly
    # equal the search fell back to the trivial all-joints solution — keep it
    # as-is so accuracy is preserved; pruning to max_points here would just
    # pick an arbitrary bad subset.
    if len(indices) > max_points:
        cv = weights.std() / (weights.mean() + 1e-12)
        if cv > 0.2:
            # Non-uniform weights: a real sparse solution that happened to be
            # slightly over budget.  Prune to the top-weight joints.
            top = np.argsort(weights)[::-1][:max_points]
            indices = indices[top]
            weights, _ = nnls(A[:, indices], b)
        else:
            print(f"  ⚠ No sparse solution found (weights near-uniform, cv={cv:.3f}); "
                  f"keeping all {len(indices)} joints (trivial fallback)")

    r_final = b - A[:, indices] @ weights
    final_err = np.linalg.norm(r_final) / b_norm
    print(f"\n  {'─'*60}")
    print(f"  Selected {len(indices)} / {n_joints} joints   rel_err={final_err:.6f}")
    print(f"  joint_indices: {indices.tolist()}")
    print(f"  joint_weights: {[round(float(v), 4) for v in weights]}")
    print(f"  {'─'*60}")
    return indices, weights


def main(args=None):
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(DEVICE)

    if args is None:
        from get_args import get_args
        args = get_args()

    config = load_config(args.config_file)
    system, system_def = system_to_name(config)
    system.training = False

    # --- Load trained basis model ---
    k         = config.subspace_dim
    bonus_dim = config["subspace"].get("bonus_dim", 0)
    name      = config.experiment_name
    model_file = (f"{name}_{k}D_{bonus_dim}P_model_local.pt" if bonus_dim > 0
                  else f"{name}_{k}D_model_local.pt")
    model_path = os.path.join("experiments", model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model checkpoint at {model_path}")

    from gradients import SmoothGradientBasisField
    net_cfg = config["network"]
    model = SmoothGradientBasisField(
        dim=system.dim,
        k=k + bonus_dim,
        hidden_layers=net_cfg["n_hidden_layers"],
        hidden_width=net_cfg["n_neurons"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {model_path}")

    # --- Load precomputed dataset ---
    dataset_path = os.path.join("precompute", f"{name}_dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No dataset at {dataset_path}. Run bench_playback first.")

    data = torch.load(dataset_path, map_location="cpu")
    q_all = data["q"]  # (N_total, dim)
    print(f"Dataset: {q_all.shape[0]} frames, dim={q_all.shape[1]}")

    # Subsample
    n = min(MAX_SAMPLES, len(q_all))
    idx = torch.randperm(len(q_all))[:n]
    q_samples = [q_all[i] for i in idx]
    print(f"Using {n} samples for NNLS")

    # --- Build NNLS system ---
    A, b = build_nnls_matrix(system, system_def, model, q_samples,
                             match_hessian=MATCH_HESSIAN)

    # --- Solve at each requested sparsity level ---
    for i, frac in enumerate(CUBATURE_FRACS):
        pct = int(round(frac * 100))
        print(f"\n{'═'*60}")
        print(f"  Solving cubature at {pct}% ({int(round(system.num_joints * frac))}/{system.num_joints} joints max)")
        print(f"{'═'*60}")
        indices, weights = solve_cubature(A, b, max_frac=frac)

        # Save per-fraction file: {name}_cubature_{pct}pct.pt
        pct_path = os.path.join("experiments", f"{name}_cubature_{pct}pct.pt")
        payload = {
            "joint_indices": torch.tensor(indices, dtype=torch.long),
            "joint_weights": torch.tensor(weights, dtype=torch.float64),
        }
        torch.save(payload, pct_path)
        print(f"Saved → {pct_path}")

        # Also save the first fraction as the default {name}_cubature.pt
        if i == 0:
            default_path = os.path.join("experiments", f"{name}_cubature.pt")
            torch.save(payload, default_path)
            print(f"Saved → {default_path}  (default)")


if __name__ == "__main__":
    main()
