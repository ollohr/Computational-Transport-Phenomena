import numpy as np
import matplotlib.pyplot as plt
import scipy

def poisson_matrix(
        Nx: int, 
        Ny: int, 
        h : float,
): 
    """
    This function initializes the poisson matrix. 

    Parameters:
        Nx      (int): Number of points in the x direction 
        Ny      (int): Number of points in the y direction
        h      (float): this si the 
    """
    Ntot = Nx * Ny

    P = np.full(Ntot,-1)   # point
    E =  np.full(Ntot-1,1/4)   # east
    W =  np.full(Ntot-1,1/4)   # west
    N =  np.full(Ntot-(Nx-1),1/4)   # north 
    S =  np.full(Ntot-(Nx-1),1/4)   # south

    diag = [
        P,
        E,
        W,
        N,
        S
    ]
    offsets = [0,1,-1,(Nx-1),-(Nx-1)]       # for N and S points this should maybe be (Nx-2) not sure, this depends on the BC 
    A = scipy.sparse.diags(diag, offsets, shape=(Ntot,Ntot), format = 'csr')
    return  A 


# conditios
u = 5           # m2/s
Nx = 10
Ny = 5
h = 1/5

matrix = poisson_matrix(Nx,Ny, h).toarray()
print(matrix)

