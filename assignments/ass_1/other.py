import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

# ---- Parameters ----
Nx = 10
Ny = 6
h = 1.0
Q = 5

# ---- Solid mask (curb) ----
solid = np.zeros((Nx, Ny), dtype=bool)
for i in range(Nx):
    for j in range(Ny):
        if i >= 5 and j <= 3:
            solid[i,j] = True

# ---- Build Poisson matrix ----
def poisson_matrix(Nx, Ny, solid):
    Ntot = Nx*Ny
    A = scipy.sparse.lil_matrix((Ntot,Ntot))
    
    for i in range(Nx):
        for j in range(Ny):
            idx = i + j*Nx
            
            # Solid
            if solid[i,j]:
                A[idx, idx] = 1
                continue
            
            # Boundaries
            if i == 0:           # inlet
                A[idx, idx] = 1
            elif i == Nx-1:      # outlet (Neumann will be applied later in b)
                A[idx, idx] = 1
            elif j == 0:         # bottom wall
                A[idx, idx] = 1
            elif j == Ny-1:      # top wall
                A[idx, idx] = 1
            else:                # interior
                A[idx, idx] = -4
                A[idx, idx+1] = 1
                A[idx, idx-1] = 1
                A[idx, idx+Nx] = 1
                A[idx, idx-Nx] = 1
    return A.tocsr()

# ---- RHS vector ----
def rhs_vector(Nx, Ny, h, Q, solid):
    Ntot = Nx*Ny
    b = np.zeros(Ntot)
    
    for i in range(Nx):
        for j in range(Ny):
            idx = i + j*Nx
            y = j*h
            
            if solid[i,j]:
                b[idx] = 0
            elif i == 0:           # inlet
                b[idx] = Q * y / (h*(Ny-1))
            elif i == Nx-1:        # outlet (Neumann)
                b[idx] = 0
            elif j == 0:           # bottom
                b[idx] = 0
            elif j == Ny-1:        # top
                b[idx] = Q
            else:
                b[idx] = 0
    return b

# ---- Solve ----
A = poisson_matrix(Nx, Ny, solid)
b = rhs_vector(Nx, Ny, h, Q, solid)
psi = scipy.sparse.linalg.spsolve(A, b)
psi_grid = psi.reshape((Ny, Nx))  # [y, x] orientation

####### Plotting 
# Grid coordinates
x = np.linspace(0, (Nx - 1) , Nx)
y = np.linspace(0, (Ny - 1) , Ny)  
X, Y = np.meshgrid(x, y)  # 2D grid for plotting

# Contour plot of the solution
plt.figure(figsize=(8, 5))
cp = plt.contourf(X, Y, psi_grid, levels=50, cmap="viridis")
plt.colorbar(cp)
plt.title("Poisson Solution ψ(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

# Plot profiles along x (rows) at different y positions
plt.figure()
for j in range(0, Ny, max(1, Ny // 5)):
    plt.plot(x, psi_grid[j, :], label=f"y = {y[j]:.2f}")
plt.xlabel("x")
plt.ylabel("ψ")
plt.title("Profiles at Different y Positions")
plt.legend()
plt.show()

# Plot profiles along y (columns) at different x positions
plt.figure()
for i in range(0, Nx, max(1, Nx // 5)):
    plt.plot(y, psi_grid[:, i], label=f"x = {x[i]:.2f}")
plt.xlabel("y")
plt.ylabel("ψ")
plt.title("Profiles at Different x Positions")
plt.legend()
plt.show()
