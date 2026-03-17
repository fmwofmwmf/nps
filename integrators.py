import torch
from torch.func import jacrev as _jacrev
from torch.func import grad, jacfwd, vmap

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

        alpha = 1 # line_search(E0, z_new, z_dot_new, delta_z, delta_v, M, F_z, system, system_def, space)

        z_new = z_new + alpha * delta_z
        z_dot_new = z_dot_new + alpha * delta_v

    return z_new, z_dot_new, residual_norm

def latent_step_newton(system, system_def, state_to_system, z, z_dot, space,
                       dt=0.01, max_iters=10, tol=1e-6):
    """
    Fully physically accurate latent-space implicit Euler step.

    Args:
        system: Physical system object (needs kinetic_energy, mass_matrix, potential_energy, etc.)
        system_def: System definition
        state_to_system: Function mapping latent z -> physical q
        z: latent positions (n_latent,)
        z_dot: latent velocities (n_latent,)
        space: optional extra space info for system
        dt: timestep
        max_iters: max Newton iterations
        tol: convergence tolerance (residual norm)

    Returns:
        z_new: updated latent positions
        z_dot_new: updated latent velocities
        residual_norm: final residual norm
    """
    n_latent = z.shape[0]

    # Initial guess: explicit Euler
    z_new = z.clone().detach()
    z_dot_new = z_dot.clone().detach()

    # Compute mass matrix: M = J^T M_phys J
    q = state_to_system(system_def, z, space)  # physical config
    J = torch.autograd.functional.jacobian(
        lambda z_: state_to_system(system_def, z_, space), z
    )  # shape (n_phys, n_latent)
    # print(J.shape)
    M_phys = system.physical_mass_matrix(system_def, q)  # shape (n_phys, n_phys)
    M = J.T @ M_phys @ J  # shape (n_latent, n_latent)

    for iteration in range(max_iters):
        # Compute forces at current Newton iterate
        z_new_grad = z_new.clone().requires_grad_(True)
        q_new = state_to_system(system_def, z_new_grad, space)
        E = system.potential_energy_batch(system_def, q_new.unsqueeze(0),
                                          space.unsqueeze(0))[0]
        F_z = -torch.autograd.grad(E, z_new_grad)[0]  # latent force

        # Compute stiffness matrix K = dF/dz
        def force_fn(z_):
            q_ = state_to_system(system_def, z_, space)
            E_ = system.potential_energy_batch(system_def, q_.unsqueeze(0),
                                               space.unsqueeze(0))[0]
            return -torch.autograd.grad(E_, z_, create_graph=True)[0]

        K = torch.autograd.functional.jacobian(force_fn, z_new)  # (n_latent, n_latent)

        # Implicit Euler residuals
        R_v = M @ (z_dot_new - z_dot) - dt * F_z
        R_z = z_new - z - dt * z_dot_new

        residual_norm = torch.cat([R_z, R_v]).norm()
        if residual_norm < tol:
            break

        # Build full Newton system
        # [ I         -dt*I ] [dz]   = [-R_z]
        # [-dt*K   M + dt*γ*M] [dv]   = [-R_v]
        I = torch.eye(n_latent, device=z.device, dtype=z.dtype)
        top = torch.cat([I, -dt * I], dim=1)
        bottom = torch.cat([-dt * K, M], dim=1)
        lhs = torch.cat([top, bottom], dim=0)  # (2n_latent, 2n_latent)

        rhs = -torch.cat([R_z, R_v], dim=0)

        # Solve Newton system
        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            print("something really bad")
            # fallback: damped gradient
            delta = torch.cat([
                -R_z,
                -R_v / M.diagonal().mean()
            ])

        delta_z = delta[:n_latent]
        delta_v = delta[n_latent:]

        KE = 0.5 * z_dot_new @ M @ z_dot_new
        PE = system.potential_energy_batch(system_def,
                                           state_to_system(system_def, z_new, space).unsqueeze(0),
                                           space.unsqueeze(0))[0]
        E0 = KE + PE


        alpha = 1 #line_search(E0, z_new, z_dot_new, delta_z, delta_v, M, F_z, system, system_def, space)

        # Accept step
        z_new = (z_new + alpha * delta_z).detach()
        z_dot_new = (z_dot_new + alpha * delta_v).detach()

    return z_new, z_dot_new, residual_norm.item()

def gradient_basis_step_newton_pos_only(
    system, system_def, model, q, q_dot, space,
    dt=0.01, max_iters=10, tol=1e-6
):
    q_new = q.clone().detach()
    residual_norm = torch.tensor(float("inf"))

    for iteration in range(max_iters):

        U = model(q_new).squeeze(0)
        k = U.shape[1]

        M_phys = system.physical_mass_matrix(system_def, q_new)
        M_reduced = U.T @ M_phys @ U

        q_new_grad = q_new.requires_grad_(True)

        E = system.potential_energy_batch(
            system_def, q_new_grad.unsqueeze(0), space.unsqueeze(0)
        )[0]

        grad_full = torch.autograd.grad(E, q_new_grad, create_graph=True)[0]
        g_full = grad_full.detach()

        # reduced Hessian
        HU = torch.stack([
            torch.autograd.grad(grad_full @ U[:, i], q_new_grad,
                                retain_graph=(i < k - 1))[0]
            for i in range(k)
        ], dim=1)

        H_reduced = U.T @ HU

        g_reduced = U.T @ g_full

        # residual
        R = M_reduced @ (U.T @ (q_new - q - dt * q_dot)) + dt**2 * g_reduced

        residual_norm = R.norm()
        if residual_norm < tol:
            break

        lhs = M_reduced + dt**2 * H_reduced
        delta_reduced = torch.linalg.solve(lhs, -R)

        delta_q = U @ delta_reduced

        q_new = (q_new + delta_q).detach()

    # velocity recovered from position change
    q_dot_new = (q_new - q) / dt

    return q_new, q_dot_new, residual_norm.item()

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