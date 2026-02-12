import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg

def is_solid(i, j, i_step=5, j_step=2):
    # solid interior only (same as "better code")
    return (i > i_step) and (j < j_step)

def poisson(Nx: int, Ny: int, i_step=5, j_step=2) -> scipy.sparse.lil_matrix:
    Ntot = Nx * Ny
    A = scipy.sparse.lil_matrix((Ntot, Ntot))

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j * Nx

            inlet = (i == 0)
            top   = (j == Ny - 1)

            bottom_upstream = (j == 0 and i <= i_step)
            step_wall = (i == i_step and j <= j_step)
            ledge = (j == j_step and i >= i_step)

            outlet = (i == Nx - 1 and j >= j_step)

            solid = is_solid(i, j, i_step, j_step)

            # Dirichlet on boundaries + solid interior (kept in system but pinned)
            if inlet or top or bottom_upstream or step_wall or ledge or outlet or solid:
                A[idx, idx] = 1.0
            else:
                A[idx, idx] = -4.0
                A[idx, idx + 1] = 1.0
                A[idx, idx - 1] = 1.0
                A[idx, idx + Nx] = 1.0
                A[idx, idx - Nx] = 1.0

    return A

def rhs_vector(Nx: int, Ny: int, Q: float, i_step=5, j_step=2) -> np.ndarray:
    Ntot = Nx * Ny
    b = np.zeros(Ntot)

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j * Nx

            inlet = (i == 0)
            top   = (j == Ny - 1)

            bottom_upstream = (j == 0 and i <= i_step)
            step_wall = (i == i_step and j <= j_step)
            ledge = (j == j_step and i >= i_step)

            outlet = (i == Nx - 1 and j >= j_step)

            solid = is_solid(i, j, i_step, j_step)

            if bottom_upstream or step_wall or ledge or solid:
                b[idx] = 0.0
            elif top:
                b[idx] = Q
            elif inlet:
                b[idx] = Q * j / (Ny - 1)
            elif outlet:
                b[idx] = Q * (j - j_step) / ((Ny - 1) - j_step)
            else:
                b[idx] = 0.0

    return b

# Parameters
Nx = 10
Ny = 6
Q = 5.0
i_step = 5
j_step = 2

A = poisson(Nx, Ny, i_step, j_step)
b = rhs_vector(Nx, Ny, Q, i_step, j_step)

psi = scipy.sparse.linalg.spsolve(A.tocsr(), b)
psi_grid = psi.reshape((Ny, Nx))

# --- mask solid for plotting ONLY ---
for j in range(Ny):
    for i in range(Nx):
        if is_solid(i, j, i_step, j_step):
            psi_grid[j, i] = np.nan

# Plot
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 5))
levels = np.linspace(0, Q, 15)
cs = plt.contour(X, Y, psi_grid, levels=levels)
plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
plt.title("Streamlines (coarse grid)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()