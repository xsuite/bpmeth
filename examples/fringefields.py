import bpmeth
import numpy as np
import matplotlib.pyplot as plt
import mpmath

"""
Plot the numerical solution for the Hamiltonian for a fringe field 
"""

def K0ggtanh(b1, a, L):
    """    
    K0 fringe field integral for a tanh times the gap height squared.
    
    :param b1: fringe field coefficient.
    :param a: fringe field range.
    :param L: integral around the fringe field ranges from -L to L.
        (In theory the value of K0gg should not depend on L, otherwise L is chosen too small.)
    """

    return b1* ( L**2/2 - L*a/2*np.log((1+np.exp(-2*L/a))*(1+np.exp(2*L/a))) \
        + a**2/4*(mpmath.polylog(2, -np.exp(-2*L/a))-mpmath.polylog(2, -np.exp(2*L/a))) )


# Properties
h = 0
lenght = 5
b1 = 0.1
aa = 1
b1shape = f"(tanh(s/{aa})+1)/2"
b1fringe = f"{b1}*{b1shape}" # Fringe field integral K = aa/(2g)
Kg = aa/2
K0gg = K0ggtanh(b1, aa, lenght/2)


#################################
# Compare the trajectories      #
#################################

qp0 = [0,1,0,0,0,0]  # x, y, tau, px, py, ptau
coord = np.array([[qp0[0]], [qp0[3]], [qp0[1]], [qp0[4]]])  # format that allows many particles

fringe_thin = bpmeth.ThinNumericalFringe(b1, b1shape, length=lenght, nphi=5)
trajectories_thin = fringe_thin.track(coord.copy())
trajectory_thin = trajectories_thin.get_trajectory(0)

fringe_thick = bpmeth.ThickNumericalFringe(b1, b1shape, lenght=lenght, nphi=5)
trajectories_thick = fringe_thick.track(coord.copy())
trajectory_thick = trajectories_thick.get_trajectory(0)

fringe_forest = bpmeth.ForestFringe(b1, Kg, K0gg, closedorbit=False)
trajectories_forest = fringe_forest.track(coord.copy())
trajectory_forest = trajectories_forest.get_trajectory(0)
fringe_forest_co = bpmeth.ForestFringe(b1, Kg, K0gg, closedorbit=True)
trajectories_forest_co = fringe_forest.track(coord.copy())
trajectory_forest_co = trajectories_forest_co.get_trajectory(0)

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
trajectory_thin.plot_3d(ax=ax3d, label='thin')
trajectory_thick.plot_3d(ax=ax3d, label='thick')
trajectory_forest.plot_3d(ax=ax3d, label='forest')
trajectory_forest_co.plot_3d(ax=ax3d, label='forest with closed orbit effect')

figx, axx = plt.subplots()
trajectory_thin.plot_x(ax=axx, label='thin')
trajectory_thick.plot_x(ax=axx, label='thick')
trajectory_forest.plot_x(ax=axx, label='forest')
trajectory_forest_co.plot_x(ax=axx, label='forest with closed orbit effect')

figy, axy = plt.subplots()
trajectory_thin.plot_y(ax=axy, label='thin')
trajectory_thick.plot_y(ax=axy, label='thick')
trajectory_forest.plot_y(ax=axy, label='forest')
trajectory_forest_co.plot_y(ax=axy, label='forest with closed orbit effect')

figpx, axpx = plt.subplots()
trajectory_thin.plot_px(ax=axpx, label='thin')
trajectory_thick.plot_px(ax=axpx, label='thick')
trajectory_forest.plot_px(ax=axpx, label='forest')
trajectory_forest_co.plot_px(ax=axpx, label='forest with closed orbit effect')

figpy, axpy = plt.subplots()
trajectory_thin.plot_py(ax=axpy, label='thin')
trajectory_thick.plot_py(ax=axpy, label='thick')
trajectory_forest.plot_py(ax=axpy, label='forest')
trajectory_forest_co.plot_py(ax=axpy, label='forest with closed orbit effect')

# => Thick fringe is not the real physical effect! It assumes that the maps commute which is not true


#################################
# Make plots to see trends in y #
#################################

npart = 10
part = np.zeros((4, npart))
part[2] = np.linspace(-1, 1, npart)
part[3] = 0.2

fringe_thin = bpmeth.ThinNumericalFringe(b1, b1shape, lenght=lenght, nphi=5)
trajectories_thin = fringe_thin.track(part.copy())

fringe_thick = bpmeth.ThickNumericalFringe(b1, b1shape, lenght=lenght, nphi=5)
trajectories_thick = fringe_thick.track(part.copy())

fringe_forest = bpmeth.ForestFringe(b1, Kg, K0gg)
trajectories_forest = fringe_forest.track(part.copy(), closedorbit=False)
trajectories_forest_co = fringe_forest.track(part.copy(), closedorbit=True)

ax = trajectories_thin.plot_final(label="thin map")
trajectories_thick.plot_final(ax=ax, label="thick map")
trajectories_forest.plot_final(ax=ax, label="forest map")
trajectories_forest_co.plot_final(ax=ax, label="forest map with closed orbit effect")


index = 4

ax1 = trajectories_thin.get_trajectory(index).plot_x(label='thin')
trajectories_thick.get_trajectory(index).plot_x(ax=ax1, label='thick')
trajectories_forest.get_trajectory(index).plot_x(ax=ax1, label='forest')

ax2 = trajectories_thin.get_trajectory(index).plot_y(label='thin')
trajectories_thick.get_trajectory(index).plot_y(ax=ax2, label='thick')
trajectories_forest.get_trajectory(index).plot_y(ax=ax2, label='forest')

# => The "focussing term" in forest is actually physical, with a third order correction (the sadistic term)