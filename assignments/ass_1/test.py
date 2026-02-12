import numpy as np
import matplotlib.pyplot as plt

def solve_contraction_streamfunction(phi=5.0, refine=1, omega=1.7, tol=1e-8, max_iter=20000):
    """
    Solves Laplace(psi)=0 for the 2D contraction in the assignment figure,
    using Gauss-Seidel SOR on a masked Cartesian grid.

    Geometry (base grid):
      Nx = 10 nodes in x (i = 0..9)
      Ny =  6 nodes in y (j = 0..5)

      Step is at i = 5, and step height is j = 2 (bottom after step is y = 2*Δ).
      Solid region: i > 5 and j < 2
    Refinement:
      refine=2 or 3 subdivides each Δ into refine parts (still uniform spacing).
    """

    # ---- base-grid "node counts" inferred from the figure ----
    Nx0, Ny0 = 10, 6
    i_step0 = 5          # x-location of step (node index)
    j_step0 = 2          # y-index of step height (node index)

    # ---- refined grid sizes (nodes) ----
    Nx = (Nx0 - 1) * refine + 1
    Ny = (Ny0 - 1) * refine + 1

    # step indices on refined grid
    i_step = i_step0 * refine
    j_step = j_step0 * refine

    # coordinate arrays (Δ=1 in index-space; physical scaling not needed for streamlines)
    x = np.arange(Nx, dtype=float)
    y = np.arange(Ny, dtype=float)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ---- active mask: True = fluid/boundary, False = solid ----
    active = np.ones((Nx, Ny), dtype=bool)
    # Solid block is bottom-right corner: i > i_step and j < j_step
    active[(np.arange(Nx)[:, None] > i_step) & (np.arange(Ny)[None, :] < j_step)] = False

    # Initialize psi, fixed mask
    psi = np.zeros((Nx, Ny), dtype=float)
    fixed = np.zeros((Nx, Ny), dtype=bool)

    # Helper to set Dirichlet conditions
    def set_dirichlet(mask, values):
        nonlocal psi, fixed
        psi[mask] = values[mask] if isinstance(values, np.ndarray) else values
        fixed[mask] = True

    # ---- Boundary conditions ----
    # Top wall: psi = phi (only where active)
    top = (np.arange(Ny)[None, :] == (Ny - 1)) & active
    set_dirichlet(top, phi)

    # Bottom wall (upstream part): j=0, i <= i_step : psi = 0
    bottom_up = (np.arange(Ny)[None, :] == 0) & (np.arange(Nx)[:, None] <= i_step) & active
    set_dirichlet(bottom_up, 0.0)

    # Vertical step wall: i=i_step, j=0..j_step : psi = 0
    step_wall = (np.arange(Nx)[:, None] == i_step) & (np.arange(Ny)[None, :] <= j_step) & active
    set_dirichlet(step_wall, 0.0)

    # Horizontal wall after step (the ledge): j=j_step, i=i_step..Nx-1 : psi = 0
    ledge = (np.arange(Ny)[None, :] == j_step) & (np.arange(Nx)[:, None] >= i_step) & active
    set_dirichlet(ledge, 0.0)

    # Left inlet boundary: i=0, j=0..Ny-1 (full height), linear from 0 to phi
    inlet = (np.arange(Nx)[:, None] == 0) & active
    inlet_profile = phi * (Y / (Ny - 1))
    set_dirichlet(inlet, inlet_profile)

    # Right outlet boundary: i=Nx-1, j=j_step..Ny-1, linear from 0 at ledge to phi at top
    outlet = (np.arange(Nx)[:, None] == (Nx - 1)) & active
    outlet_profile = np.zeros_like(psi)
    H_out = (Ny - 1) - j_step
    outlet_profile[:, :] = np.nan
    # only define where active outlet exists:
    outlet_profile[outlet] = phi * ((Y[outlet] - j_step) / H_out)
    set_dirichlet(outlet, outlet_profile)

    # Any inactive cells -> NaN for plotting
    psi[~active] = np.nan

    # ---- Iterative solver (Gauss-Seidel + SOR) ----
    # Update only nodes that are active and not fixed
    update_mask = active & (~fixed)

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if not update_mask[i, j]:
                    continue

                # All neighbors should be active in this stair-stepped geometry
                # (nodes adjacent to the solid block are on/far from fixed walls).
                p_new = 0.25 * (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1])

                # SOR relaxation
                p_old = psi[i, j]
                psi[i, j] = (1 - omega) * p_old + omega * p_new
                max_delta = max(max_delta, abs(psi[i, j] - p_old))

        if max_delta < tol:
            # converged
            break

    return X, Y, psi, active, it + 1

def plot_streamlines(X, Y, psi, active, phi=5.0, n_lines=15, title="Streamlines"):
    # Mask solid as NaN for contour
    psi_plot = psi.copy()
    psi_plot[~active] = np.nan

    plt.figure(figsize=(8, 4))
    levels = np.linspace(0.0, phi, n_lines)
    cs = plt.contour(X, Y, psi_plot, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # draw domain outline roughly by showing where solid is
    plt.imshow((~active).T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
               alpha=0.15, aspect='auto')

    plt.title(title)
    plt.xlabel("x (grid units)")
    plt.ylabel("y (grid units)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    phi = 5.0

    # Coarse grid (as in the figure)
    X, Y, psi, active, niter = solve_contraction_streamfunction(phi=phi, refine=1)
    print(f"Coarse grid converged in {niter} iterations")
    plot_streamlines(X, Y, psi, active, phi=phi, n_lines=15, title="Streamlines (coarse grid)")

    # Refined grid (factor 2)
    X2, Y2, psi2, active2, niter2 = solve_contraction_streamfunction(phi=phi, refine=2)
    print(f"Refine=2 grid converged in {niter2} iterations")
    plot_streamlines(X2, Y2, psi2, active2, phi=phi, n_lines=20, title="Streamlines (refine=2)")