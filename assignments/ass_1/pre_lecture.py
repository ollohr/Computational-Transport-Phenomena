import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse  

def poisson_matrix(
        Nx: int, 
        Ny: int, 
        h : float,
)-> sparse:
    """
    This function initializes the poisson matrix. 

    Parameters:
        Nx      (int): Number of points in the x direction 
        Ny      (int): Number of points in the y direction
        h      (float): this si the 
    """
    N = Nx * Ny

    P = np.full(N,-1)   # point
    E =  np.full(N-1,1/4)   # east
    W =  np.full(N-1,1/4)   # west
    N =  np.full(N-1,1/4)   # north 
    S =  np.full(N-1,1/4)   # south

    diag = [
        P,
        E,
        W,
        N,
        S
    ]
    offsets = [0,1,-1,(Nx-1),-(Nx-1)]       # for N and S points this should maybe be (Nx-2) not sure, this depends on the BC 
    A = sparse.diag(diag, offsets, shape=(N,N), format = 'csr')
    return  A 


# conditios
u = 5           # m2/s
Nx = 10
Ny = 5
h = 1/5

matrix = poisson_matrix(Nx,Ny, h)
print(matrix)

