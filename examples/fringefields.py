import bpmeth
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the numerical solution for the Hamiltonian for a fringe field 
"""


# Properties
h = 0
len = 5
b1 = 0.1
aa = 1
b1shape = f"(tanh(s/{aa})+1)/2"
b1fringe = f"{b1}*{b1shape}" # Fringe field integral K = aa/(2g)
Kg = aa/2


qp0 = [0,1,0,0,0,0]  # x, y, tau, px, py, ptau
coord = np.array([[qp0[0]], [qp0[3]], [qp0[1]], [qp0[4]]])  # format that allows many particles

fringe_thin = bpmeth.ThinNumericalFringe(b1, b1shape, len=len, nphi=5)
trajectories_thin = fringe_thin.track(coord.copy())
trajectory_thin = trajectories_thin.get_trajectory(0)

fringe_thick = bpmeth.ThickNumericalFringe(b1, b1shape, len=len, nphi=5)
trajectories_thick = fringe_thick.track(coord.copy())
trajectory_thick = trajectories_thick.get_trajectory(0)

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
trajectory_thin.plot_3d(ax=ax3d)
trajectory_thick.plot_3d(ax=ax3d)

figx, axx = plt.subplots()
trajectory_thin.plot_x(ax=axx)
trajectory_thick.plot_x(ax=axx)

figx, axy = plt.subplots()
trajectory_thin.plot_y(ax=axy)
trajectory_thick.plot_y(ax=axy)

figpx, axpx = plt.subplots()
trajectory_thin.plot_px(ax=axpx)
trajectory_thick.plot_px(ax=axpx)

figpy, axpy = plt.subplots()
trajectory_thin.plot_py(ax=axpy)
trajectory_thick.plot_py(ax=axpy)



#################################
# Make plots to see trends in y #
#################################

npart = 10
part = np.zeros((4, npart))
part[2] = np.linspace(-1, 1, npart)

fringe_thin = bpmeth.ThinNumericalFringe(b1, b1shape, len=len, nphi=5)
trajectories_thin = fringe_thin.track(part.copy())
ax = trajectories_thin.plot_final(label="thin map")

fringe_thick = bpmeth.ThickNumericalFringe(b1, b1shape, len=len, nphi=5)
trajectories_thick = fringe_thick.track(part.copy())
trajectories_thick.plot_final(ax=ax, label="thick map")

fringe_forest = bpmeth.ForestFringe(b1, Kg)
trajectories_forest = fringe_forest.track(part.copy())
trajectories_forest.plot_final(ax=ax, label="forest map")

index = 4

ax1 = trajectories_thin.get_trajectory(index).plot_x(label='thin')
trajectories_thick.get_trajectory(index).plot_x(ax=ax1, label='thick')
trajectories_forest.get_trajectory(index).plot_x(ax=ax1, label='forest')

ax2 = trajectories_thin.get_trajectory(index).plot_y(label='thin')
trajectories_thick.get_trajectory(index).plot_y(ax=ax2, label='thick')
trajectories_forest.get_trajectory(index).plot_y(ax=ax2, label='forest')

# => The "focussing term" in forest is actually physical, with a third order correction (the sadistic term)