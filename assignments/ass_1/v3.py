import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def channel_solve(phi, n_streamlines, tol = 1e-10, maxit = 1000):
    x = np.arange(0, 10, 1)
    y = np.arange(0, 6, 1)
    nx, ny = len(x), len(y)

    #accounting for step in channel
    i_step5 = np.where(x >=5)[0][0]
    j_y2 = np.where(np.isclose(y, 2))[0][0]

    ### starting psi and BSc
    psi = np.zeros((ny,nx), dtype=float)

    psi[-1,:] = phi #top wall
    psi[0, :i_step5] = 0 #bottom wall
    psi[j_y2, i_step5:] = 0 #new bottol wall at y=2
    psi[:j_y2 + 1, i_step5] = 0 #wall at x=5

    psi[:, 0] = phi * (y / 5) #inlet for each streamline

    psi[:j_y2, -1] = 0 #outlet for blocked step
    psi[j_y2:, -1] = phi * ((y[j_y2:]- 2)/3) #outlet for actual channel

    step = np.zeros((ny,nx), dtype=bool)

    #step
    for i in range(nx):
        for j in range(ny):

            #step at x=5 and y=2, all coordinates inside it
            if j< j_y2 and i >= i_step5:
                step[j, i] = True
            
            #top wall
            if j == ny -1:
                step[j,i ] = True
            
            #bottom wall till x=5
            if j == 0 and i < i_step5:
                step[j, i] = True
            
            #new bottom wall at x=5
            if j == j_y2 and i >= i_step5:
                step[j, i] = True
            
            #inlet
            if i == 0:
                step[j, i] = True

            #outlet
            if i == nx -1:
                step[j, i] = True

            #wall at x=5 going up
            if j <= j_y2 and i == i_step5:
                step[j, i] = True

    ###solve Laplace
    for it in range(maxit):
        delta = 0

        for j in range(1, ny-1):
            for i in range(1,nx -1):
                
                if step[j, i]:
                    continue

                #west
                if j < j_y2 and (i - 1 >= i_step5):
                    W = 0
                else:
                    W = psi[j, i -1 ]
                
                #east
                if j < j_y2 and (i +1 >= i_step5):
                    E = 0
                else:
                    E = psi[j, i+1]

                #north
                if (j + 1 < j_y2) and i >= i_step5:
                    N = 0
                else:
                    N = psi[j +1, i]

                #south
                if (j - 1 < j_y2) and i >=i_step5:
                    S = 0
                else:
                    S = psi[j -1, i]

                psi_new = (W + E + S + N) / 4

                diff = abs(psi_new - psi[j,i])
                if diff > delta:
                    delta = diff

                psi[j,i] = psi_new
        
        if delta < tol:
            break

    ###plotting
    X, Y = np.meshgrid(x,y)
    psi_plot = psi.copy()

    #heatmap
    plt.figure(figsize=(10,5))
    plt.contourf(X, Y, psi_plot, levels = 50)
    plt.colorbar(label = 'Streamline')
    plt.plot([0,5,5,9], [0,0,2,2])
    plt.plot([0,9], [5,5])
    plt.xlim(0,9)
    plt.ylim(-0.1,5.1)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    step_patch = Polygon([(5,0), (9,0), (9,2), (5,2)],
                         closed=True, facecolor='white', edgecolor='white', linewidth=2)
    plt.gca().add_patch(step_patch)
    plt.show()

    # =======================
    # CHANGE #1: mask the step region instead of using NaN
    mask = (X >= x[i_step5]) & (Y < y[j_y2])
    psi_masked = np.ma.array(psi_plot, mask=mask)

    # CHANGE #2: exclude wall levels 0 and phi (more robust near boundaries)
    levels = np.linspace(0, phi, n_streamlines + 2)[1:-1]
    # =======================

    #streamlines
    plt.figure(figsize=(10,5))

    # CHANGE #3: corner_mask=False avoids dropped segments near masked corners
    psi_on_line = plt.contour(X, Y, psi_masked, levels=levels, corner_mask=False)
    plt.clabel(psi_on_line, inline=True)

    plt.plot([0,5,5,9], [0,0,2,2])
    plt.plot([0,9], [5,5])

    plt.xlim(0,9)
    plt.ylim(-0.1,5.1)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

    return psi

channel_solve(5, 20)