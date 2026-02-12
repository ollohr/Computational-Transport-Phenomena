import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg


def poisson(Nx: int, Ny: int, i_step=5, j_step=2) -> scipy.sparse.lil_matrix:
    Ntot = Nx * Ny
    A = scipy.sparse.lil_matrix((Ntot, Ntot))

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j * Nx

            # Dirichlet BC at Inlet
            if i == 0:
                A[idx, idx] = 1

            # Dirichlet BC at Top wall
            elif j == Ny - 1:
                A[idx, idx] = 1

            # Bottom wall before the step
            elif (j == 0) and (i < i_step):
                A[idx, idx] = 1

            # Vertical step face (at i_step) from bottom up to j_step
            elif (i == i_step) and (j <= j_step):
                A[idx, idx] = 1

            # Horizontal top of step (at j_step) for i > i_step
            elif (j == j_step) and (i > i_step):
                A[idx, idx] = 1

            # Outlet (only open part)
            elif (i == Nx - 1) and (j >= j_step):
                A[idx, idx] = 1

            # Solid/slab INTERIOR (consistent definition)
            elif (i > i_step) and (j < j_step):
                A[idx, idx] = 1

            # Interior fluid nodes: Laplace stencil
            else:
                A[idx, idx] = -4
                A[idx, idx + 1] = 1
                A[idx, idx - 1] = 1
                A[idx, idx + Nx] = 1
                A[idx, idx - Nx] = 1

    return A


def rhs_vector(Nx: int, Ny: int, Q: float, i_step=5, j_step=2) -> np.ndarray:
    Ntot = Nx * Ny
    b = np.zeros(Ntot)

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j * Nx

            # Inlet
            if i == 0:
                b[idx] = Q * j / (Ny - 1)

            # Top wall
            elif j == Ny - 1:
                b[idx] = Q

            # Outlet (open part only)
            elif (i == Nx - 1) and (j >= j_step):
                b[idx] = Q * (j - j_step) / (Ny - 1 - j_step)

            # Bottom wall before step
            elif (j == 0) and (i < i_step):
                b[idx] = 0

            # Step face
            elif (i == i_step) and (j <= j_step):
                b[idx] = 0

            # Top of step
            elif (j == j_step) and (i > i_step):
                b[idx] = 0

            # Solid/slab interior (DO NOT USE NaN — use Dirichlet value)
            elif (i > i_step) and (j < j_step):
                b[idx] = 0

            else:
                b[idx] = 0

    return b


# Parameters
Nx = 10
Ny = 6
Q = 5
i_step = 5
j_step = 2

# Build system
A = poisson(Nx, Ny, i_step=i_step, j_step=j_step)
b = rhs_vector(Nx, Ny, Q, i_step=i_step, j_step=j_step)

# Solve
psi = scipy.sparse.linalg.spsolve(A.tocsr(), b)
psi_grid = psi.reshape((Ny, Nx))

# Grid coordinates
x = np.linspace(0, Nx - 1, Nx)
y = np.linspace(0, Ny - 1, Ny)
X, Y = np.meshgrid(x, y)

# Solid mask (MUST match the slab interior definition exactly)
solid_mask = np.zeros((Ny, Nx), dtype=bool)
for j in range(Ny):
    for i in range(Nx):
        if (i > i_step) and (j < j_step):
            solid_mask[j, i] = True

psi_plot = psi_grid.copy()
psi_plot[solid_mask] = np.nan

# 1) Heat map
plt.figure(figsize=(8, 5))
cp = plt.contourf(X, Y, psi_plot, levels=50, cmap="viridis")
plt.colorbar(cp)
plt.title("ψ(x, y) Heat Map (solid masked correctly)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# 2) Streamlines (psi = const)
plt.figure(figsize=(8, 5))
levels = np.linspace(0, Q, 15)
cs = plt.contour(X, Y, psi_plot, levels=levels)
plt.clabel(cs, inline=True, fontsize=8)
plt.title("Streamlines (ψ = const)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()