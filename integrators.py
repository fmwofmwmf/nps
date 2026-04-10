import time
import torch
from torch.func import jacrev
from torch.func import grad, jacfwd, vmap


def _t():
    """Wall-clock timer, GPU-safe: syncs CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _print_profile(timings, n_iters):
    total = sum(timings.values())
    print(f"\n{'─'*55}")
    print(f"  Integrator profile  ({n_iters} Newton iter{'s' if n_iters != 1 else ''})")
    print(f"{'─'*55}")
    for label, t in timings.items():
        pct = 100 * t / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:<28s} {t*1e3:7.2f} ms  {pct:5.1f}%  {bar}")
    print(f"  {'TOTAL':<28s} {total*1e3:7.2f} ms")
    print(f"{'─'*55}\n")

def latent_step(system, system_def, state_to_system, z, z_dot, space, dt=0.01, gamma=0.05, mass=1.0):
    """
    One step of latent-space dynamics.
    z, z_dot: latent state and velocity (torch tensors)
    """

    z = z.clone().detach().requires_grad_(True)  # important!

    # Full system state
    q = state_to_system(system_def, z, space)

    # Compute energy
    E = system.potential_energy_batch(system_def, q.unsqueeze(0), space.unsqueeze(0))[0]

    # Actually simpler: use torch.autograd.grad with chain rule
    F_z = -torch.autograd.grad(E, z, create_graph=False)[0]  # direct latent force

    # Integrate (symplectic Euler)
    z_dot_new = z_dot + dt * F_z / mass - dt * gamma * z_dot
    z_new = z + dt * z_dot_new

    return z_new.detach(), z_dot_new.detach(), 0

def apply_A(v):
    # M v
    Mv = M @ v

    # K v = -H(E) v
    grad = torch.autograd.grad(E, z_new, create_graph=True)[0]
    Hv = torch.autograd.grad(grad, z_new, v, retain_graph=True)[0]

    return Mv - dt * dt * (-Hv)

def cg_solve(A, b, x0=None, tol=1e-6, max_iters=50):
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A(x)
    p = r.clone()
    rs_old = r @ r

    for _ in range(max_iters):
        Ap = A(p)
        alpha = rs_old / (p @ Ap + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

def latent_step_newton_cg(
    system, system_def, state_to_system,
    z, z_dot, space,
    dt=0.01, max_iters=10, tol=1e-6
):
    n = z.shape[0]

    z_new = z.clone().detach()
    z_dot_new = z_dot.clone().detach()

    # --- Compute mass matrix once ---
    z_req = z.detach().requires_grad_(True)
    q = state_to_system(system_def, z_req, space)

    def jvp_fn(v):
        _, jvp = torch.autograd.functional.jvp(
            lambda z_: state_to_system(system_def, z_, space),
            z_req, v
        )
        return jvp

    M_phys = system.physical_mass_matrix(system_def, q)

    n_phys = M_phys.shape[0]
    J = torch.zeros(n_phys, n, device=z.device, dtype=z.dtype)

    for i in range(n):
        e = torch.zeros_like(z)
        e[i] = 1.0
        J[:, i] = jvp_fn(e)

    M = J.T @ M_phys @ J

    for _ in range(max_iters):
        z_req = z_new.detach().requires_grad_(True)
        q = state_to_system(system_def, z_req, space)

        E = system.potential_energy_batch(
            system_def, q.unsqueeze(0), space.unsqueeze(0)
        )[0]

        F = -torch.autograd.grad(E, z_req, create_graph=True)[0]

        # Residual
        R = M @ (z_dot_new - z_dot) - dt * F
        if R.norm() < tol:
            break

        # Matrix-free operator
        def A(v):
            grad = torch.autograd.grad(E, z_req, create_graph=True)[0]
            Hv = torch.autograd.grad(grad, z_req, v, retain_graph=True)[0]
            return M @ v - dt * dt * (-Hv)

        # Solve (M - dt^2 K) dz = -R
        delta_z = cg_solve(A, -R, tol=tol)

        delta_v = delta_z / dt

        z_new = (z_new + delta_z).detach()
        z_dot_new = (z_dot_new + delta_v).detach()

    return z_new, z_dot_new, R.norm().item()


from torch.func import grad, jacfwd, vmap

def latent_step_newton_batch(system, system_def, state_to_system,
                                   z, z_dot, space, dt=0.01, max_iters=10, tol=1e-6):
    n_latent = z.shape[0]
    I = torch.eye(n_latent, device=z.device, dtype=z.dtype)

    # --- Mass matrix (computed once) ---
    def q_of_z(z_):
        return state_to_system(system_def, z_, space)

    J = _jacrev(q_of_z)(z)                          # (n_phys, n_latent)
    q = q_of_z(z)
    M_phys = system.physical_mass_matrix(system_def, q)
    M = J.T @ M_phys @ J                            # (n_latent, n_latent)
    M_z_dot = M @ z_dot

    # --- Energy / force / stiffness via torch.func ---
    def energy_fn(z_):
        q_ = state_to_system(system_def, z_, space)
        return system.potential_energy_batch(
            system_def, q_.unsqueeze(0), space.unsqueeze(0)
        )[0]

    neg_grad_energy = grad(lambda z_: -energy_fn(z_))  # F_z = -dE/dz
    hess_neg_energy = jacfwd(neg_grad_energy)           # K  = d(F_z)/dz = -d²E/dz²

    z_new = z.clone()
    z_dot_new = z_dot.clone()
    residual_norm = torch.tensor(float('inf'), device=z.device)

    for _ in range(max_iters):
        F_z = neg_grad_energy(z_new)          # (n_latent,)   — one reverse pass
        K   = hess_neg_energy(z_new)          # (n_latent, n_latent) — fwd-over-rev
        E   = energy_fn(z_new)

        M_z_dot_new = M @ z_dot_new
        R_v = M_z_dot_new - M_z_dot - dt * F_z
        R_z = z_new - z - dt * z_dot_new

        residual_norm = torch.cat([R_z, R_v]).norm()

        lhs = torch.cat([
            torch.cat([I,        -dt * I], dim=1),
            torch.cat([-dt * K,  M      ], dim=1),
        ], dim=0)

        try:
            delta = torch.linalg.solve(lhs, -torch.cat([R_z, R_v]))
        except RuntimeError:
            delta = torch.cat([-R_z, -R_v / M.diagonal().mean()])

        delta_z, delta_v = delta[:n_latent], delta[n_latent:]

        KE = 0.5 * z_dot_new @ M_z_dot_new
        E0 = KE + E

        alpha = line_search(E0, z_new, z_dot_new, delta_z, delta_v, M, F_z, system, system_def, space)

        z_new = z_new + alpha * delta_z
        z_dot_new = z_dot_new + alpha * delta_v

    return z_new, z_dot_new, residual_norm


def latent_step_newton(system, system_def, state_to_system, z, z_dot, space,
                             dt=0.01, max_iters=10, tol=1e-6, pca_k=0, profile=False):
    n_latent = z.shape[0]
    device, dtype = z.device, z.dtype

    z_new = z.clone().detach()
    z_dot_new = z_dot.clone().detach()

    timings = {"mass_jacobian": 0.0, "energy_force": 0.0,
               "stiffness": 0.0, "newton_solve": 0.0}
    n_iters = 0

    # Compute mass matrix
    t0 = _t()
    q = state_to_system(system_def, z, space)
    J = jacrev(lambda z_: state_to_system(system_def, z_, space))(z)
    M_phys = system.physical_mass_matrix(system_def, q)
    M = J.T @ M_phys @ J
    timings["mass_jacobian"] += _t() - t0

    def energy_fn(q_):
        return system.potential_energy_batch(system_def, q_.unsqueeze(0),
                                             space.unsqueeze(0))[0]

    for iteration in range(max_iters):
        n_iters += 1

        # Current q
        q_new = state_to_system(system_def, z_new, space)

        # Force via single backward pass
        t0 = _t()
        z_new_grad = z_new.clone().requires_grad_(True)
        q_grad = state_to_system(system_def, z_new_grad, space)
        E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0),
                                          space.unsqueeze(0))[0]
        F_z = -torch.autograd.grad(E, z_new_grad)[0]
        timings["energy_force"] += _t() - t0

        # Stiffness via jacrev
        t0 = _t()
        def force_fn(z_):
            q_ = state_to_system(system_def, z_, space)
            E_ = system.potential_energy_batch(system_def, q_.unsqueeze(0),
                                               space.unsqueeze(0))[0]
            return -torch.autograd.grad(E_, z_, create_graph=True)[0]

        K = jacrev(force_fn)(z_new)
        timings["stiffness"] += _t() - t0

        # Implicit Euler residuals
        R_v = M @ (z_dot_new - z_dot) - dt * F_z
        R_z = z_new - z - dt * z_dot_new

        residual_norm = torch.cat([R_z, R_v]).norm()
        if residual_norm < tol:
            break

        # Newton system
        t0 = _t()
        I = torch.eye(n_latent, device=device, dtype=dtype)
        top = torch.cat([I, -dt * I], dim=1)
        bottom = torch.cat([-dt * K, M], dim=1)
        lhs = torch.cat([top, bottom], dim=0)
        rhs = -torch.cat([R_z, R_v], dim=0)

        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            delta = torch.cat([-R_z, -R_v / M.diagonal().mean()])
        timings["newton_solve"] += _t() - t0

        delta_z = delta[:n_latent]
        delta_v = delta[n_latent:]

        # =========================================================
        # DEBUG: Analyze delta_q alignment with PCA basis
        # =========================================================

        if pca_k > 0:
            # Map delta_z to full space delta_q
            J_new = jacrev(lambda z_: state_to_system(system_def, z_, space))(z_new)
            delta_q = J_new @ delta_z

            # Get gradient in full space
            with torch.no_grad():
                g_full = grad(energy_fn)(q_new)

            g_norm = g_full.norm()
            g_dir = g_full / (g_norm + 1e-10)

            # Project out gradient from delta_q
            delta_q_grad_component = (delta_q @ g_dir) * g_dir
            delta_q_no_grad = delta_q - delta_q_grad_component

            # Get PCA basis (no gradient added)
            with torch.no_grad():
                H_pca = jacrev(grad(energy_fn))(q_new)
                eigenvalues, eigenvectors = torch.linalg.eigh(H_pca)
                U_pca = eigenvectors[:, :pca_k]

            # Compute alignment of delta_q with orthogonalized basis
            delta_q_norm = delta_q.norm()
            if delta_q_norm > 1e-10:
                delta_q_dir = delta_q / delta_q_norm

                # Orthogonalize: gradient first, then PCA modes
                U_full = torch.cat([g_dir.unsqueeze(1), U_pca], dim=1)
                U_orth, _ = torch.linalg.qr(U_full)  # (dim, 1 + pca_k)

                alignments_sq = (U_orth.T @ delta_q_dir) ** 2
            else:
                alignments_sq = torch.zeros(1 + pca_k, device=device, dtype=dtype)

            total_captured = alignments_sq.sum().item()
            print(f"Norm: {delta_q_norm}")
            print(f"\nPCA eigenvalues (lowest {pca_k}): {[f'{e:.3e}' for e in eigenvalues[:pca_k].tolist()]}")
            print(f"\nAlignment² with orthogonalized basis:")
            print(f"  grad dir: {alignments_sq[0].item():.4f} {'█' * int(alignments_sq[0].item() * 50)}")
            for i in range(pca_k):
                a = alignments_sq[i + 1].item()
                bar = '█' * int(a * 50)
                print(f"  mode {i:2d} (λ={eigenvalues[i].item():+.3e}): {a:.4f} {bar}")
            print(f"\n  Total captured: {total_captured:.4f} ({100 * total_captured:.1f}%)")
            print(f"  Residual: {1 - total_captured:.4f} ({100 * (1 - total_captured):.1f}%)")

        # =========================================================

        z_new = (z_new + delta_z).detach()
        z_dot_new = (z_dot_new + delta_v).detach()

    if profile:
        _print_profile(timings, n_iters)
    return z_new, z_dot_new, residual_norm.item()


from torch.func import grad, jvp

def _basis_and_mass(model, system, system_def, q, g):
    """Compute basis (with gradient direction) and reduced mass matrix.
    g: full-space energy gradient (dim,), pre-computed by the caller."""
    with torch.no_grad():
        U = model(q, g)   # model_eval(q, g) → network.inference(q, g) → (dim, k+1)
        M_phys = system.physical_mass_matrix(system_def, q)
        M_reduced = U.T @ M_phys @ U
    return U, M_reduced

@torch.inference_mode()
def _newton_linalg(M_reduced, H_reduced, g_reduced, disp_reduced, dt2):
    """Reduced-space linear solve — pure linear algebra, no grad."""
    R = M_reduced @ disp_reduced + dt2 * g_reduced
    lhs = M_reduced + dt2 * H_reduced
    delta_reduced = torch.linalg.solve(lhs, -R)
    return R, delta_reduced

@torch.inference_mode()
def _line_search_residual(U_model, M_reduced, q_trial, q, q_dot, g_r_trial, dt, dt2):
    """Evaluate Newton residual at trial point — g_r_trial is already reduced (k,)."""
    disp_trial = U_model.T @ (q_trial - q - dt * q_dot)
    return M_reduced @ disp_trial + dt2 * g_r_trial


def _energy_for_newton(system, system_def, q_grad):
    """Non-gravity energy for Newton solve — uses cubature if present in system_def."""
    cubature = system_def.get('cubature')
    if cubature is not None:
        return system.potential_energy_cubature_batch(
            system_def, q_grad.unsqueeze(0), cubature
        )[0]
    return system.energy_springs_batch(system_def, q_grad.unsqueeze(0))[0]


def gradient_basis_step_newton_pos_only(
        system, system_def, model, q, q_dot, space,
        dt=0.01, max_iters=20, tol=1e-6, profile=False
):
    """Implicit Euler in the reduced basis (rolling-frame Newton).

    The basis U is recomputed at the CURRENT Newton iterate q_new each
    iteration, so the projected equation is self-consistent:
        M_r(q*) U(q*)^T (q* - q - dt qdot) + dt^2 U(q*)^T g(q*) = 0
    """
    q_new = q.clone().detach()
    dt2 = dt * dt

    residual_norm = float("inf")

    timings = {"basis_model": 0.0, "energy_grad": 0.0,
               "hessian_HU": 0.0, "newton_solve": 0.0, "line_search": 0.0}
    n_iters = 0

    # Gravity gradient is constant — precompute once.
    g_grav = system.gravity_gradient(system_def, q)

    sparse_info = system_def.get('cubature_sparse')
    if sparse_info is not None:
        dof_idx = sparse_info['dof_indices'].to(q_new.device)

    for iteration in range(max_iters):
        n_iters += 1

        # --- Full-space gradient at q_new (for basis construction) ---
        t0 = _t()
        if sparse_info is not None:
            #print(sparse_info)
            # Sparse: differentiate only active DOFs, scatter back to full space.
            with torch.enable_grad():
                q_active_g = q_new[dof_idx].clone().requires_grad_(True)
                E_g = system.energy_cubature_sparse(system_def, q_active_g, sparse_info)
                g_sparse = torch.autograd.grad(E_g, q_active_g)[0].detach()
            g_spr = torch.zeros_like(q_new)
            g_spr[dof_idx] = g_sparse
        else:
            with torch.enable_grad():
                q_g = q_new.clone().requires_grad_(True)
                E_g = _energy_for_newton(system, system_def, q_g)
                g_spr = torch.autograd.grad(E_g, q_g)[0].detach()
        g_total = g_spr + g_grav
        timings["energy_grad"] += _t() - t0

        # --- Recompute basis at CURRENT iterate (rolling frame) ---
        t0 = _t()
        U_model, M_reduced = _basis_and_mass(model, system, system_def, q_new, g_total)
        k = U_model.shape[1]
        if sparse_info is not None:
            U_cub = U_model[dof_idx, :]
        timings["basis_model"] += _t() - t0

        # --- Reduced gradient + Hessian via s-parameterisation ---
        t0 = _t()
        s = torch.zeros(k, requires_grad=True)
        if sparse_info is not None:
            q_act = q_new[dof_idx] + U_cub @ s
            E = system.energy_cubature_sparse(system_def, q_act, sparse_info)
        else:
            E = _energy_for_newton(system, system_def, q_new + U_model @ s)
        E_current = E.detach() + (g_grav * q_new).sum()
        g_r_sp = torch.autograd.grad(E, s, create_graph=True)[0]
        g_reduced = g_r_sp.detach() + U_model.T @ g_grav
        H_reduced = torch.stack([
            torch.autograd.grad(g_r_sp[i], s,
                                retain_graph=(i < k - 1), create_graph=False)[0]
            for i in range(k)
        ], dim=0).detach()
        #print(H_reduced.shape)
        timings["hessian_HU"] += _t() - t0

        disp_reduced = U_model.T @ (q_new - q - dt * q_dot)

        t0 = _t()
        R, delta_reduced = _newton_linalg(M_reduced, H_reduced, g_reduced, disp_reduced, dt2)
        residual_norm = R.norm().item()
        timings["newton_solve"] += _t() - t0

        if residual_norm < tol:
            break

        delta_q = (U_model @ delta_reduced).detach()

        # --- line search (energy decrease) ---
        t0 = _t()
        # alpha = 1.0
        # for _ in range(5):
        #     q_trial = (q_new + alpha * delta_q).detach()
        #     with torch.no_grad():
        #         if sparse_info is not None:
        #             E_trial = system.energy_cubature_sparse(system_def, q_trial[dof_idx], sparse_info)
        #         else:
        #             E_trial = _energy_for_newton(system, system_def, q_trial)
        #         E_trial = E_trial + (g_grav * q_trial).sum()
        #     if E_trial.item() < E_current.item():
        #         break
        #     alpha *= 0.5
        # timings["line_search"] += _t() - t0

        # --- line search (residual decrease) --- [old version]
        alpha = 1.0
        for _ in range(5):
            q_trial = (q_new + alpha * delta_q).detach()
            s_t = torch.zeros(k, requires_grad=True)
            if sparse_info is not None:
                q_act_t = q_trial[dof_idx] + U_cub @ s_t
                E_t = system.energy_cubature_sparse(system_def, q_act_t, sparse_info)
            else:
                E_t = _energy_for_newton(system, system_def, q_trial + U_model @ s_t)
            g_r_trial = torch.autograd.grad(E_t, s_t)[0].detach() + U_model.T @ g_grav
            R_trial = _line_search_residual(U_model, M_reduced, q_trial, q, q_dot, g_r_trial, dt, dt2)
            if R_trial.norm().item() < residual_norm:
                break
            alpha *= 0.5
        timings["line_search"] += _t() - t0

        q_new = (q_new + alpha * delta_q).detach()

    if profile:
        _print_profile(timings, n_iters)

    with torch.inference_mode():
        q_dot_new = (q_new - q) / dt

    return q_new, q_dot_new, residual_norm

def gradient_basis_step_newton_new(system, system_def, model, q, q_dot, space,
                                    dt=0.01, max_iters=10, tol=1e-6):
    dim = q.shape[0]
    q_new = q.clone().detach()
    q_dot_new = q_dot.clone().detach()

    residual_norm = torch.tensor(float('inf'))

    for iteration in range(max_iters):
        # Recompute basis and mass at current iterate
        U = model(q_new).squeeze(0)                        # (dim, k)
        # print(U.shape)
        k = U.shape[1]
        M_phys = system.physical_mass_matrix(system_def, q_new)
        M_reduced = U.T @ M_phys @ U                      # (k, k)

        # Energy, forces in full space
        q_new_grad = q_new.requires_grad_(True)
        E = system.potential_energy_batch(
            system_def, q_new_grad.unsqueeze(0), space.unsqueeze(0)
        )[0]
        grad_full = torch.autograd.grad(E, q_new_grad, create_graph=True)[0]
        F_full = -grad_full.detach()

        # Reduced Hessian without forming full (dim, dim) matrix
        HU = torch.stack([
            torch.autograd.grad(grad_full @ U[:, i], q_new_grad,
                                retain_graph=(i < k - 1))[0]
            for i in range(k)
        ], dim=1)                                           # (dim, k)
        K_reduced = -U.T @ HU                              # (k, k)

        # Residuals in full space, projected to reduced
        R_q_full = q_new - q - dt * q_dot_new
        R_v_full = M_phys @ (q_dot_new - q_dot) - dt * F_full
        R_q_reduced = U.T @ R_q_full
        R_v_reduced = U.T @ R_v_full

        residual_norm = torch.cat([R_q_full, R_v_full]).norm()  # track full residual
        if residual_norm < tol:
            break

        # Newton solve in reduced space
        I_k = torch.eye(k, device=q.device, dtype=q.dtype)
        lhs = torch.cat([
            torch.cat([I_k,        -dt * I_k  ], dim=1),
            torch.cat([-dt * K_reduced, M_reduced], dim=1),
        ], dim=0)
        rhs = -torch.cat([R_q_reduced, R_v_reduced])

        delta_reduced = torch.linalg.solve(lhs, rhs)
        delta_q_full = U @ delta_reduced[:k]
        delta_v_full = U @ delta_reduced[k:]

        KE = 0.5 * q_dot_new @ M_phys @ q_dot_new
        E0 = KE + E.detach()

        alpha = 1 # line_search(E0, q_new, q_dot_new, delta_q_full, delta_v_full, M_phys, F_full, system, system_def, space)

        q_new = (q_new + alpha * delta_q_full).detach()
        q_dot_new = (q_dot_new + alpha * delta_v_full).detach()

    return q_new, q_dot_new, residual_norm.item()

def line_search(E0, q, q_dot, delta_q, delta_v, M_phys, F, system, system_def, space, max_ls_iters = 10):
    alpha = 1.0
    c = 1e-1

    for ls_iter in range(max_ls_iters):

        q_trial = q + alpha * delta_q
        q_dot_trial = q_dot + alpha * delta_v

        KE_trial = 0.5 * q_dot_trial @ M_phys @ q_dot_trial
        PE_trial = system.potential_energy_batch(system_def, q_trial.unsqueeze(0),
                                                 space.unsqueeze(0))[0]
        E_trial = KE_trial + PE_trial

        if E_trial <= E0 + c:
            break
        alpha *= 0.5

    return alpha

def gradient_basis_step_newton(system, system_def, model, q, q_dot, space,
                                   dt=0.01, max_iters=10, tol=1e-6):
    """
    Implicit Euler using bottom k eigenvectors of Hessian as basis.

    The bottom k eigenvectors correspond to the "softest" directions
    where the energy changes most slowly - these are the natural
    coordinates for low-frequency dynamics.
    """
    dim = q.shape[0]
    k = 1  # Number of soft modes to keep

    # Compute Hessian at initial configuration
    q_grad = q.clone().requires_grad_(True)
    E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0),
                                      space.unsqueeze(0))[0]

    # Compute gradient and Hessian
    grad = torch.autograd.grad(E, q_grad, create_graph=True)[0]

    # Hessian of potential energy
    H = torch.zeros(dim, dim, device=q.device, dtype=q.dtype)
    for i in range(dim):
        H[i] = torch.autograd.grad(grad[i], q_grad, retain_graph=True)[0]

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(H)

    # Take BOTTOM k eigenvectors (softest modes)
    U = eigenvectors[:, :k]  # (dim, k)

    print(f"\n=== Soft Mode Basis (k={k}) ===")
    print(f"Bottom {k} eigenvalues: {eigenvalues[:k].detach().cpu().numpy()}")
    print(f"Top eigenvalue (stiffest): {eigenvalues[-1].item():.2f}")

    # Project state into soft-mode coordinates
    q_reduced = U.T @ q  # (k,)
    q_dot_reduced = U.T @ q_dot  # (k,)

    # Check how much velocity is lost
    q_dot_projected = U @ q_dot_reduced  # Project back to full space
    q_dot_lost = q_dot - q_dot_projected

    velocity_norm = q_dot.norm().item()
    velocity_lost_norm = q_dot_lost.norm().item()

    if velocity_norm > 1e-10:
        fraction_lost = velocity_lost_norm / velocity_norm
        print(f"Velocity norm: {velocity_norm:.6f}")
        print(f"Velocity lost in projection: {velocity_lost_norm:.6f}")
        print(f"Fraction of velocity lost: {fraction_lost:.2%}")
    else:
        print("Velocity is near zero")
    print("=" * 40)

    # Initial guess
    q_reduced_new = q_reduced.clone().detach()
    q_dot_reduced_new = q_dot_reduced.clone().detach()

    # Mass matrix in reduced coordinates
    M_phys = system.physical_mass_matrix(system_def, q)
    M_reduced = U.T @ M_phys @ U  # (k, k)

    for iteration in range(max_iters):
        # Lift to full space
        q_new = U @ q_reduced_new  # (dim,)
        q_dot_new = U @ q_dot_reduced_new  # (dim,)

        # Compute energy and gradient at new position
        q_new_grad = q_new.clone().requires_grad_(True)
        E_new = system.potential_energy_batch(system_def, q_new_grad.unsqueeze(0),
                                              space.unsqueeze(0))[0]
        grad_new = torch.autograd.grad(E_new, q_new_grad, create_graph=True)[0]

        # Project gradient to reduced space
        g_reduced = U.T @ grad_new  # (k,)
        F_reduced = -g_reduced

        # Compute reduced Hessian H_reduced = U^T @ H @ U
        H_new = torch.zeros(dim, dim, device=q.device, dtype=q.dtype)
        for i in range(dim):
            H_new[i] = torch.autograd.grad(grad_new[i], q_new_grad,
                                           retain_graph=True)[0]

        H_reduced = U.T @ H_new @ U  # (k, k)
        K_reduced = -H_reduced  # Stiffness matrix (negative Hessian)

        # Implicit Euler residuals
        R_v = M_reduced @ (q_dot_reduced_new - q_dot_reduced) - dt * F_reduced  # (k,)
        R_q = q_reduced_new - q_reduced - dt * q_dot_reduced_new  # (k,)

        residual_norm = torch.cat([R_q, R_v]).norm()
        if residual_norm < tol:
            break

        # Build Newton system
        # [ I              -dt*I        ] [delta_q] = [-R_q]
        # [-dt*K_reduced   M_reduced    ] [delta_v] = [-R_v]
        I_k = torch.eye(k, device=q.device, dtype=q.dtype)
        top = torch.cat([I_k, -dt * I_k], dim=1)
        bottom = torch.cat([-dt * K_reduced, M_reduced], dim=1)
        lhs = torch.cat([top, bottom], dim=0)  # (2k, 2k)

        rhs = -torch.cat([R_q, R_v], dim=0)

        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            delta = torch.cat([
                -R_q,
                -R_v / M_reduced.diagonal().mean()
            ])

        delta_q_reduced = delta[:k]
        delta_v_reduced = delta[k:]

        # Line search
        alpha = 1.0
        max_ls_iters = 10

        KE = 0.5 * q_dot_new @ M_phys @ q_dot_new
        PE = E_new
        E0 = KE + PE

        c = 1e-4
        dir_deriv = torch.dot(-F_reduced, delta_q_reduced)

        for ls_iter in range(max_ls_iters):
            q_reduced_trial = q_reduced_new + alpha * delta_q_reduced
            q_dot_reduced_trial = q_dot_reduced_new + alpha * delta_v_reduced

            q_trial = U @ q_reduced_trial
            q_dot_trial = U @ q_dot_reduced_trial

            KE_trial = 0.5 * q_dot_trial @ M_phys @ q_dot_trial
            PE_trial = system.potential_energy_batch(system_def, q_trial.unsqueeze(0),
                                                     space.unsqueeze(0))[0]
            E_trial = KE_trial + PE_trial

            if E_trial <= E0 + c * alpha * dir_deriv:
                break
            alpha *= 0.5

        # Accept step
        q_reduced_new = (q_reduced_new + alpha * delta_q_reduced).detach()
        q_dot_reduced_new = (q_dot_reduced_new + alpha * delta_v_reduced).detach()

    # Lift final state to full space
    q_new = (U @ q_reduced_new).detach()
    q_dot_new = (U @ q_dot_reduced_new).detach()

    # Print final velocity loss
    q_dot_new_projected = U @ q_dot_reduced_new
    q_dot_new_lost = q_dot_new - q_dot_new_projected

    final_velocity_norm = q_dot_new.norm().item()
    final_velocity_lost_norm = q_dot_new_lost.norm().item()

    print(f"After step: velocity norm = {final_velocity_norm:.6f}, lost = {final_velocity_lost_norm:.6f}")

    return q_new, q_dot_new, residual_norm.item()

def reduced_symplectic_step(system, system_def, model, q, q_dot, space, dt=0.01, gamma=0.05):
    """
    Symplectic Euler step in a reduced subspace.

    Args:
        system: physical system object
        system_def: system definition
        model: gradient basis field (maps q -> B of shape dim x k)
        q: full-space positions (dim,)
        q_dot: full-space velocities (dim,)
        space: optional extra info
        dt: timestep
        gamma: damping coefficient

    Returns:
        q_new: updated full-space positions (dim,)
        q_dot_new: updated full-space velocities (dim,)
        z_dot_new: latent-space velocity (k,) for diagnostics
    """

    q_grad = q.clone().detach().requires_grad_(True)
    E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0), space.unsqueeze(0))[0]
    F_q = -torch.autograd.grad(E, q_grad, create_graph=False)[0]  # (dim,)

    B = F_q / F_q.norm()#model(q_grad.unsqueeze(0))[0]  # (dim, k)
    B = B.unsqueeze(1)  # (dim, 1) → proper basis

    F_z = B.T @ F_q  # (k,)

    z_dot = B.T @ q_dot  # project full-space velocity (dim -> k)

    z_dot_new = (z_dot + dt * F_z) / (1.0 + dt * gamma)
    z_new = dt * z_dot_new  # step in latent coordinates


    q_new = q + B @ z_new
    q_dot_new = q_dot + B @ z_dot_new

    return q_new.detach(), q_dot_new.detach(), z_dot_new.detach().norm()


# ALTERNATIVE: Simpler semi-implicit Euler (often more stable)
def latent_step_semiimplicit(system, system_def, state_to_system, z, z_dot, space, dt=0.01, gamma=0.05):
    """
    Semi-implicit (symplectic) Euler - simpler and often more stable.

    v_{n+1} = v_n + dt * (F/M - gamma * v_{n+1})
    z_{n+1} = z_n + dt * v_{n+1}

    Rearranging:
    v_{n+1} = (v_n + dt * F/M) / (1 + dt * gamma)
    """

    z_grad = z.clone().requires_grad_(True)
    q = state_to_system(system_def, z_grad, space)
    E = system.potential_energy_batch(system_def, q.unsqueeze(0), space.unsqueeze(0))[0]
    F_z = -torch.autograd.grad(E, z_grad, create_graph=False)[0]

    # Approximate mass as identity (or compute proper mass)
    # For better results, compute M as in Newton version
    M_inv_approx = 1.0  # Simplification

    # Semi-implicit update
    z_dot_new = (z_dot + dt * F_z * M_inv_approx) / (1.0 + dt * gamma)
    z_new = z + dt * z_dot_new

    return z_new.detach(), z_dot_new.detach(), 0.0

def latent_step_implicit_v2(system, system_def, state_to_system, z, z_dot, space, dt=0.01, gamma=0.05, max_iters=50, tol=1e-6):
    """
    More robust implicit integration using optimization approach.
    """
    # Compute mass matrix at current position
    M = compute_mass_matrix_safe(system, system_def, state_to_system, z, space)

    # Variables to optimize
    z_new = z.clone().detach().requires_grad_(True)
    z_dot_new = z_dot.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [z_new, z_dot_new],
        lr=1.0,
        max_iter=max_iters,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn='strong_wolfe'
    )

    # Store final residual for return value
    final_residual = [0.0]

    def closure():
        optimizer.zero_grad()

        # Compute force
        q = state_to_system(system_def, z_new, space)
        E = system.potential_energy_batch(
            system_def,
            q.unsqueeze(0),
            space.unsqueeze(0)
        )[0]
        F = -torch.autograd.grad(E, z_new, create_graph=True)[0]

        # Implicit Euler residual
        R_v = M @ (z_dot_new - z_dot) - dt * F + dt * gamma * M @ z_dot_new
        R_z = z_new - z - dt * z_dot_new

        loss = 0.5 * (R_z.norm() ** 2 + R_v.norm() ** 2)

        # Store residual norm (before backward, while graph exists)
        with torch.no_grad():
            final_residual[0] = torch.cat([R_z.detach(), R_v.detach()]).norm().item()

        loss.backward()

        return loss

    optimizer.step(closure)

    return z_new.detach(), z_dot_new.detach(), final_residual[0]


def compute_mass_matrix_safe(system, system_def, state_to_system, z, space):
    """Safe mass matrix computation with error handling"""
    try:
        z_pos = z.detach()
        z_vel_dummy = torch.zeros_like(z_pos).requires_grad_(True)

        def kinetic_energy_latent(z_vel):
            z_for_jac = z_pos.clone().requires_grad_(True)
            q_decoded = state_to_system(system_def, z_for_jac, space)
            J = torch.autograd.functional.jacobian(
                lambda zp: state_to_system(system_def, zp, space),
                z_for_jac
            )
            q_dot = J @ z_vel
            return system.kinetic_energy(
                system_def,
                state_to_system(system_def, z_pos, space),
                q_dot,
                space
            )

        M = torch.autograd.functional.hessian(kinetic_energy_latent, z_vel_dummy)

        # Add small regularization for numerical stability
        M = M + 1e-6 * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        return M
    except:
        # Fallback to identity
        print("Warning: Mass matrix computation failed, using identity")
        return torch.eye(z.shape[0], device=z.device, dtype=z.dtype)

def choose_integrator(integrator_name):
    #exp, imp, imp_torch, imp_semi
    if integrator_name == "imp_torch":
        return latent_step_implicit_v2
    elif integrator_name == "exp":
        return latent_step
    elif integrator_name == "imp":
        return latent_step_newton
    elif integrator_name == "imp_semi":
        return latent_step_semiimplicit
    else:
        raise ValueError("Integrator name not recognized")