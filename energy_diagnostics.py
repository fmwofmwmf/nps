import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class EnergyDiagnostics:
    """
    Tools for analyzing smoothness and high-frequency noise in energy landscapes.
    """

    @staticmethod
    def compute_derivatives_1d(energy_fn, system_def, z_center, direction,
                               n_samples=100, sample_range=2.0, use_finite_diff=False):
        """
        Sample energy and derivatives along a 1D line.

        Args:
            energy_fn: function(system_def, z) -> scalar
            z_center: center point in latent space
            direction: direction vector (will be normalized)
            n_samples: number of sample points
            sample_range: range in each direction (total span = 2 * sample_range)
            use_finite_diff: if True, use finite differences instead of autograd

        Returns:
            dict with 't', 'E', 'dE', 'd2E', 'd3E', 'd4E'
        """
        direction = direction / torch.norm(direction)

        # Sample points along line
        t_vals = torch.linspace(-sample_range, sample_range, n_samples)
        z_samples = z_center.unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)

        # Storage
        E_vals = torch.zeros(n_samples)
        dE_vals = torch.zeros(n_samples)
        d2E_vals = torch.zeros(n_samples)
        d3E_vals = torch.zeros(n_samples)
        d4E_vals = torch.zeros(n_samples)

        if use_finite_diff:
            # Finite difference approach
            dt = t_vals[1] - t_vals[0]

            # Compute energies
            for i in range(n_samples):
                z = z_samples[i]
                E_vals[i] = energy_fn(system_def, z).item()

            # First derivative (central differences)
            for i in range(1, n_samples - 1):
                dE_vals[i] = (E_vals[i + 1] - E_vals[i - 1]) / (2 * dt)
            dE_vals[0] = (E_vals[1] - E_vals[0]) / dt  # forward
            dE_vals[-1] = (E_vals[-1] - E_vals[-2]) / dt  # backward

            # Second derivative
            for i in range(1, n_samples - 1):
                d2E_vals[i] = (E_vals[i + 1] - 2 * E_vals[i] + E_vals[i - 1]) / (dt ** 2)
            d2E_vals[0] = d2E_vals[1]
            d2E_vals[-1] = d2E_vals[-2]

            # Third derivative (from dE)
            for i in range(1, n_samples - 1):
                d3E_vals[i] = (dE_vals[i + 1] - 2 * dE_vals[i] + dE_vals[i - 1]) / (dt ** 2)
            d3E_vals[0] = d3E_vals[1]
            d3E_vals[-1] = d3E_vals[-2]

            # Fourth derivative (from d2E)
            for i in range(1, n_samples - 1):
                d4E_vals[i] = (d2E_vals[i + 1] - 2 * d2E_vals[i] + d2E_vals[i - 1]) / (dt ** 2)
            d4E_vals[0] = d4E_vals[1]
            d4E_vals[-1] = d4E_vals[-2]

        else:
            # Autograd approach
            for i in range(n_samples):
                z = z_samples[i].requires_grad_(True)

                # Energy
                E = energy_fn(system_def, z)
                E_vals[i] = E.item()

                # First derivative (gradient in direction)
                dE_dz = torch.autograd.grad(E, z, create_graph=True)[0]
                dE = torch.dot(dE_dz, direction)
                dE_vals[i] = dE.item()

                # Second derivative (Hessian in direction)
                d2E_dz2 = torch.autograd.grad(dE, z, create_graph=True)[0]
                d2E = torch.dot(d2E_dz2, direction)
                d2E_vals[i] = d2E.item()

                # Third derivative
                d3E_dz3 = torch.autograd.grad(d2E, z, create_graph=True)[0]
                d3E = torch.dot(d3E_dz3, direction)
                d3E_vals[i] = d3E.item()

                # Fourth derivative
                d4E_dz4 = torch.autograd.grad(d3E, z, create_graph=True)[0]
                d4E = torch.dot(d4E_dz4, direction)
                d4E_vals[i] = d4E.item()

        return {
            't': t_vals.numpy(),
            'E': E_vals.numpy(),
            'dE': dE_vals.numpy(),
            'd2E': d2E_vals.numpy(),
            'd3E': d3E_vals.numpy(),
            'd4E': d4E_vals.numpy()
        }

    @staticmethod
    def compute_hessian_eigenvalues(energy_fn, system_def, z, space=None):
        """
        Compute eigenvalues of Hessian at a point.

        Returns:
            eigenvalues: sorted eigenvalues
            condition_number: ratio of max/min eigenvalue
        """
        z = z.clone().requires_grad_(True)

        # Compute Hessian
        E = energy_fn(system_def, z)
        grad = torch.autograd.grad(E, z, create_graph=True)[0]

        n = z.shape[0]
        H = torch.zeros(n, n)

        for i in range(n):
            grad_i = torch.autograd.grad(grad[i], z, retain_graph=True)[0]
            H[i] = grad_i

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(H)
        eigenvalues_sorted = torch.sort(eigenvalues)[0]

        # Condition number (watch for negative eigenvalues!)
        min_eig = eigenvalues_sorted[0].item()
        max_eig = eigenvalues_sorted[-1].item()

        if abs(min_eig) > 1e-10:
            cond = abs(max_eig / min_eig)
        else:
            cond = float('inf')

        return eigenvalues_sorted.numpy(), cond

    @staticmethod
    def frequency_analysis_1d(signal, dt=1.0):
        """
        Perform FFT to identify high-frequency components.

        Args:
            signal: 1D array
            dt: spacing between samples

        Returns:
            freqs: frequency bins
            power: power spectrum
        """
        n = len(signal)

        # FFT
        fft_vals = np.fft.fft(signal)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(n, dt)

        # Only positive frequencies
        mask = freqs >= 0
        freqs = freqs[mask]
        power = power[mask]

        return freqs, power

    @staticmethod
    def gradient_smoothness_metric(energy_fn, system_def, z_samples):
        """
        Compute how much the gradient varies across samples.
        High variation = rough landscape.

        Args:
            z_samples: (N, dim) tensor of sample points

        Returns:
            smoothness: lower is smoother
        """
        N = z_samples.shape[0]
        gradients = []

        for i in range(N):
            z = z_samples[i].requires_grad_(True)
            E = energy_fn(system_def, z)
            grad = torch.autograd.grad(E, z)[0]
            gradients.append(grad)

        gradients = torch.stack(gradients)

        # Compute variance in gradient direction
        grad_diffs = gradients[1:] - gradients[:-1]
        smoothness = torch.norm(grad_diffs, dim=1).mean().item()

        return smoothness, gradients

    @staticmethod
    def plot_diagnostic_suite(energy_fn, system_def, z_center, directions=None,
                              n_samples=100, sample_range=2.0, use_finite_diff=False):
        """
        Generate comprehensive diagnostic plots.

        Args:
            directions: list of direction vectors (if None, uses random directions)
            use_finite_diff: if True, use finite differences instead of autograd
        """
        if directions is None:
            dim = z_center.shape[0]
            n = 2  # or however many you want

            # First n standard basis vectors e_0, e_1, ..., e_{n-1}
            directions = [
                torch.eye(dim, device=z_center.device, dtype=z_center.dtype)[i]
                for i in range(n)
            ]

        n_dirs = len(directions)

        fig = plt.figure(figsize=(16, 4 * n_dirs))
        gs = GridSpec(n_dirs, 3, figure=fig, hspace=0.4)

        method_str = "Finite Diff" if use_finite_diff else "Autograd"

        for dir_idx, direction in enumerate(directions):
            # Compute derivatives along this direction
            data = EnergyDiagnostics.compute_derivatives_1d(
                energy_fn, system_def, z_center, direction,
                n_samples, sample_range, use_finite_diff=use_finite_diff
            )

            t = data['t']
            dt = t[1] - t[0]

            # Plot 1: Energy
            ax1 = fig.add_subplot(gs[dir_idx, 0])
            ax1.plot(t, data['E'], 'b-', linewidth=2, label='Energy')
            ax1.set_xlabel('t', fontsize=10)
            ax1.set_ylabel('E(t)', fontsize=10)
            ax1.set_title(f'Direction {dir_idx + 1}: Energy', fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=8)

            # Plot 2: First & Second derivatives
            ax2 = fig.add_subplot(gs[dir_idx, 1])
            line1 = ax2.plot(t, data['dE'], 'g-', label="dE/dt (gradient)", linewidth=2)
            ax2.set_xlabel('t', fontsize=10)
            ax2.set_ylabel('dE/dt', color='g', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='g')

            ax2_twin = ax2.twinx()
            line2 = ax2_twin.plot(t, data['d2E'], 'r-', label="d²E/dt² (curvature)", linewidth=2)
            ax2_twin.set_ylabel('d²E/dt²', color='r', fontsize=10)
            ax2_twin.tick_params(axis='y', labelcolor='r')

            ax2.set_title('Gradient & Curvature', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best', fontsize=8)

            # Plot 3: Higher derivatives (noise indicators)
            ax3 = fig.add_subplot(gs[dir_idx, 2])
            ax3.plot(t, data['d3E'], 'orange', label='d³E/dt³', linewidth=1.5)
            ax3.plot(t, data['d4E'], 'purple', label='d⁴E/dt⁴', linewidth=1.5)
            ax3.set_xlabel('t', fontsize=10)
            ax3.set_ylabel('Higher derivatives', fontsize=10)
            ax3.set_title('High-order derivatives', fontsize=11, fontweight='bold')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Add interpretation text
            d3_std = np.std(data['d3E'])
            d4_std = np.std(data['d4E'])

            # # Plot 4: Frequency analysis of gradient
            # ax4 = fig.add_subplot(gs[dir_idx, 3])
            # freqs_dE, power_dE = EnergyDiagnostics.frequency_analysis_1d(data['dE'], dt)
            # ax4.semilogy(freqs_dE, power_dE, 'g-', linewidth=2, label='Gradient power')
            # ax4.set_xlabel('Frequency (1/unit)', fontsize=10)
            # ax4.set_ylabel('Power (log scale)', fontsize=10)
            # ax4.set_title('Gradient frequency spectrum', fontsize=11, fontweight='bold')
            # ax4.grid(True, alpha=0.3, which='both')
            # ax4.legend(loc='best', fontsize=8)
            #
            # # Add high-frequency indicator
            # hf_power_ratio = power_dE[len(power_dE) // 2:].sum() / power_dE.sum()
            # color = 'red' if hf_power_ratio > 0.3 else 'green'
            # ax4.text(0.98, 0.98, f'HF ratio: {hf_power_ratio:.1%}',
            #          transform=ax4.transAxes, fontsize=9, color=color,
            #          verticalalignment='top', horizontalalignment='right',
            #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            #
            # # Plot 5: Frequency analysis of 4th derivative (noise!)
            # ax5 = fig.add_subplot(gs[dir_idx, 4])
            # freqs_d4E, power_d4E = EnergyDiagnostics.frequency_analysis_1d(data['d4E'], dt)
            # ax5.semilogy(freqs_d4E, power_d4E, 'purple', linewidth=2, label='d⁴E/dt⁴ power')
            # ax5.set_xlabel('Frequency (1/unit)', fontsize=10)
            # ax5.set_ylabel('Power (log scale)', fontsize=10)
            # ax5.set_title('4th derivative spectrum (HF noise)', fontsize=11, fontweight='bold')
            # ax5.grid(True, alpha=0.3, which='both')
            # ax5.legend(loc='best', fontsize=8)

            # Print statistics
            print(f"\n=== Direction {dir_idx + 1} Statistics ===")
            print(f"Energy range: [{data['E'].min():.3e}, {data['E'].max():.3e}]")
            print(f"Gradient std: {np.std(data['dE']):.3e}")
            print(f"d²E/dt² std: {np.std(data['d2E']):.3e}")
            print(f"d³E/dt³ std: {d3_std:.3e}")
            print(f"d⁴E/dt⁴ std: {d4_std:.3e}")
            # print(f"High-freq power ratio (dE): {hf_power_ratio:.3f}")

        # Add overall title with method indicator
        fig.suptitle(f'Energy Landscape Diagnostics ({method_str})',
                     fontsize=16, fontweight='bold', y=0.998)

        return fig

    @staticmethod
    def check_lipschitz_continuity(energy_fn, system_def, z_center,
                                   n_samples=50, radius=1.0):
        """
        Estimate Lipschitz constant of gradient (smoothness measure).

        ||∇E(z1) - ∇E(z2)|| ≤ L ||z1 - z2||

        Large L means gradient changes rapidly (rough landscape).
        """
        dim = z_center.shape[0]

        # Generate random samples around center
        z_samples = z_center.unsqueeze(0) + radius * torch.randn(n_samples, dim)

        gradients = []
        for i in range(n_samples):
            z = z_samples[i].requires_grad_(True)
            E = energy_fn(system_def, z)
            grad = torch.autograd.grad(E, z)[0]
            gradients.append(grad)

        gradients = torch.stack(gradients)

        # Compute pairwise Lipschitz estimates
        L_estimates = []
        for i in range(min(n_samples, 20)):  # Don't compute all pairs
            for j in range(i + 1, min(n_samples, 20)):
                grad_diff = torch.norm(gradients[i] - gradients[j])
                z_diff = torch.norm(z_samples[i] - z_samples[j])
                if z_diff > 1e-8:
                    L_estimates.append((grad_diff / z_diff).item())

        L_estimates = np.array(L_estimates)

        return {
            'mean': np.mean(L_estimates),
            'max': np.max(L_estimates),
            'std': np.std(L_estimates),
            'median': np.median(L_estimates)
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def diagnose_energy_landscape(system_def, z_current, space, energy_fn,
                              subspace_fn=None, use_finite_diff=False):
    """
    Complete diagnostic workflow.

    Call this from your main code when things are exploding!

    Args:
        use_finite_diff: if True, use finite differences instead of autograd for derivatives
    """
    print("\n" + "=" * 60)
    print("ENERGY LANDSCAPE DIAGNOSTICS")
    print("=" * 60)

    method_str = "Finite Differences" if use_finite_diff else "Automatic Differentiation"
    print(f"Using: {method_str}")

    # 1. Check Hessian condition number
    print("\n1. Hessian Analysis at current point:")
    eigenvalues, cond_num = EnergyDiagnostics.compute_hessian_eigenvalues(
        energy_fn, system_def, z_current
    )
    print(f"   Eigenvalue range: [{eigenvalues[0]:.3e}, {eigenvalues[-1]:.3e}]")
    print(f"   Condition number: {cond_num:.3e}")
    print(f"   Negative eigenvalues: {np.sum(eigenvalues < 0)}")

    if cond_num > 1e6:
        print("   ⚠️  WARNING: Very ill-conditioned Hessian!")
    if np.any(eigenvalues < -1e-6):
        print("   ⚠️  WARNING: Negative eigenvalues detected!")

    # 2. Lipschitz continuity
    print("\n2. Lipschitz constant estimates (gradient smoothness):")
    lipschitz = EnergyDiagnostics.check_lipschitz_continuity(
        energy_fn, system_def, z_current
    )
    print(f"   Mean L: {lipschitz['mean']:.3e}")
    print(f"   Max L:  {lipschitz['max']:.3e}")
    print(f"   Median L: {lipschitz['median']:.3e}")

    if lipschitz['max'] > 1e3:
        print("   ⚠️  WARNING: Very large Lipschitz constant - rough landscape!")

    # 3. Random sample smoothness
    print("\n3. Gradient variation over random samples:")
    z_samples = z_current.unsqueeze(0) + torch.randn(20, z_current.shape[0]) * 0.5
    smoothness, _ = EnergyDiagnostics.gradient_smoothness_metric(
        energy_fn, system_def, z_samples
    )
    print(f"   Gradient variation metric: {smoothness:.3e}")

    # 4. Generate full diagnostic plots
    print("\n4. Generating diagnostic plots...")
    dim = z_current.shape[0]

    # Use random directions for high-dim, or coordinate axes for low-dim
    if True: #dim <= 10:
        directions = [torch.zeros(dim) for _ in range(min(dim, 2))]
        for i, d in enumerate(directions):
            d[i] = 1.0
    else:
        directions = [torch.randn(dim) for _ in range(2)]

    fig = EnergyDiagnostics.plot_diagnostic_suite(
        energy_fn, system_def, z_current,
        directions=directions,
        n_samples=300,
        sample_range=2.0,
        use_finite_diff=use_finite_diff
    )

    plt.show(block=False)

    # 5. Summary recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    issues = []
    if cond_num > 1e6:
        issues.append("- Use smaller timestep (try dt < 0.001)")
        issues.append("- Add mass matrix regularization")
    if lipschitz['max'] > 1e3:
        issues.append("- Increase damping coefficient")
        issues.append("- Consider semi-implicit Euler instead of Newton")
    if np.any(eigenvalues < -1e-6):
        issues.append("- Energy is not convex - expect instability")
        issues.append("- Use explicit method or trust-region approach")

    if len(issues) == 0:
        print("✓ Energy landscape looks reasonable!")
    else:
        for issue in issues:
            print(issue)

    print("=" * 60 + "\n")

    return {
        'eigenvalues': eigenvalues,
        'condition_number': cond_num,
        'lipschitz': lipschitz,
        'smoothness': smoothness
    }


# ============================================================================
# Integration with your code - add this to main_loop():
# ============================================================================