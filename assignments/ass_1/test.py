import numpy as np
import scipy.sparse 
import matplotlib.pyplot as plt


def poisson_matrix(Nx, Ny):
    Ntot = Nx * Ny
    A = scipy.sparse.lil_matrix((Ntot, Ntot))

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j * Nx
            is_boundary = False

            if i == 0:  # inlet
                A[idx, idx] = 1
                is_boundary = True
            elif i == Nx - 1 and j > 3:  # outlet
                A[idx, idx] = 1
                is_boundary = True
            elif j == 0 and i < 5:  # bottom wall
                A[idx, idx] = 1
                is_boundary = True
            elif j == Ny - 1:  # top wall
                A[idx, idx] = 1
                is_boundary = True
            elif i > 5 and j < 3:  # curb
                A[idx, idx] = 1
                is_boundary = True

            if not is_boundary:  # interior
                A[idx, idx] = -4
                if i > 0 and not (i-1 > 5 and j < 3):  # skip blocked neighbor
                    A[idx, idx - 1] = 1
                if i < Nx - 1 and not (i+1 > 5 and j < 3):  # skip blocked neighbor
                    A[idx, idx + 1] = 1
                if j > 0 and not (i > 5 and j-1 < 3):  # skip blocked neighbor
                    A[idx, idx - Nx] = 1
                if j < Ny - 1 and not (i > 5 and j+1 < 3):  # skip blocked neighbor
                    A[idx, idx + Nx] = 1
            
    return A.tocsr()

def rhs_vector(Nx, Ny, h, Q):
    """
    The solution vector
    ....
    """

    Ntot = Nx*Ny
    b = np.zeros(Ntot)

    H = (Ny - 1) * h                # idk what this line does maybe remove
    for j in range(Ny):
        for i in range(Nx):
            idx = i + j*Nx
            y = j*h

            # bottom wall 
            if j ==0 and i<5:
                b[idx] = 0
            # top wall:
            elif j ==Ny-1:
                b[idx] = Q
            # inlet
            elif i ==0:
                b[idx] = Q*y/H
            # outlet
            elif i == Nx -1 and j>3:
                b[idx] = Q*y/H

            # # curb
            elif i > 5 and j < 3:  # curb
                b[idx] = 0
            
            # interior
            else:
                b[idx] = 0
            
    return b



# Parameters
Nx = 10
Ny = 6
h = 1/5
Q = 5

# Build system
A = poisson_matrix(Nx, Ny)
b = rhs_vector(Nx, Ny, h, Q)

# Solve
psi = scipy.sparse.linalg.spsolve(A, b)
psi_grid = psi.reshape((Ny, Nx))

print(psi_grid)
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
