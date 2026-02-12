import numpy as np
import matplotlib.pyplot as plt
import scipy 
    
def poisson(
        Nx      : int, 
        Ny      : int,
        i_step=5,
        j_step=2  
)->scipy.sparse.lil_matrix:

    """This function creates the poisson matrix ...
    
    Params:
    ...
    
    Return:
    ...
    """
    Ntot = Nx * Ny
    A = scipy.sparse.lil_matrix((Ntot, Ntot))

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j*Nx

            # define the laplacian operators
            P = -4
            W = 1
            E = 1
            S = 1
            N = 1

            # Dirichlet BC at Inlet 
            if i == 0:
                A[idx, idx] = 1
            
            # Dirichlet BC at Top wall
            elif j == Ny -1:
                A[idx, idx] = 1
            
            # Dirichlet BC at bottom wall
            elif i<i_step and j==0:
                A[idx, idx] = 1
            
            elif i==i_step and j<=j_step:
                A[idx, idx] = 1
            
            elif i>i_step and j==j_step:
                A[idx,idx] = 1

            # Dirichelt BC at Outlet 
            elif i==Nx-1 and j>=j_step:
                A[idx, idx] = 1
            
            # Dirichlet in slab region
            elif i>i_step and j<j_step:
                A[idx, idx] = 1

            # set the interior points 
            else: 
                A[idx, idx] = P
                A[idx, idx+1] = E
                A[idx, idx-1] = W
                A[idx, idx + Nx] = N
                A[idx, idx - Nx] = S
    return A

def rhs_vector(
        Nx      : int,
        Ny      : int,
        Q       : int,
        i_step=5,
        j_step=2 
)-> np.ndarray:
    """This function generates the solution vector...
    
    Params: 
    ...
        j_step          (int): the height of the slab
    Returns:
    ...
    """ 
    Ntot = Nx*Ny
    b = np.zeros(Ntot)

    for j in range(Ny):
        for i in range(Nx):
            idx = i + j*Nx

            # Inlet
            if i == 0:
                b[idx] = Q*j/(Ny-1)
            
            # Top wall
            elif j ==Ny-1:
                b[idx] = Q

            # Outlet
            elif i==Nx-1 and j>=j_step: 
                b[idx] = Q*(j-j_step)/(Ny-1-j_step)
            
            # bottom wall
            elif i<= i_step and j==0:
                b[idx] = 0
            elif i==i_step and j<=j_step:
                b[idx] = 0
            elif i >= i_step and j==j_step:
                b[idx ]= 0

            # slab 
            elif i>i_step and j<j_step:
                b[idx] = np.nan

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