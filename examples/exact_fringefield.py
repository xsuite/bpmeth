import bpmeth
import numpy as np
import matplotlib.pyplot as plt
import mpmath

#####################################################################
# Plot the numerical solution for the Hamiltonian for a fringe field #
#####################################################################

# Properties
h = 0
length = 5
b1 = 0.1
aa = 1
b1shape = f"(tanh(s/{aa})+1)/2"
b1fringe = f"{b1}*{b1shape}" # Fringe field integral K = aa/(2g)
Kg = aa/2


# ----- Track -----

qp0 = [0,0.1,0,0,0,0]  # x, y, tau, px, py, ptau
part = bpmeth.MultiParticle(1, x=qp0[0], y=qp0[1], tau=qp0[2], px=qp0[3], py=qp0[4], ptau=qp0[5])

fringe_thin = bpmeth.ThinDipoleFringe(b1, b1shape, length=length, nphi=5)
trajectories_thin = fringe_thin.track(part.copy(), return_trajectories=True)
trajectory_thin = trajectories_thin.get_trajectory(0)

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
trajectory_thin.plot_3d(ax=ax3d, label='thin')

figx, axx = plt.subplots()
trajectory_thin.plot_x(ax=axx, label='thin')

figy, axy = plt.subplots()
trajectory_thin.plot_y(ax=axy, label='thin')

figpx, axpx = plt.subplots()
trajectory_thin.plot_px(ax=axpx, label='thin')

figpy, axpy = plt.subplots()
trajectory_thin.plot_py(ax=axpy, label='thin')


# ----- Make plots to see trends in y -----

npart = 10
yvals = np.linspace(-1, 1, npart)
part = bpmeth.MultiParticle(npart, y=yvals)

fringe_thin = bpmeth.ThinDipoleFringe(b1, b1shape, length=length, nphi=5)
trajectories_thin = fringe_thin.track(part.copy(), return_trajectories=True)


ax = trajectories_thin.plot_final(label="thin map")
# => The "focussing term" in forest is actually physical, with a third order correction (the sadistic term)