import bpmeth
import numpy as np
import matplotlib.pyplot as plt


# Parameters
length = 10
angle = 0

BB = 0.05 # Only q By / P0 plays a role so lets call this quantity BB
x0 = 0.5
y0 = 0.1
px0 = 0.2
py0 = 0.3
ptau0 = 0
beta0 = 1

# Frame in the origin
fr = bpmeth.Frame()

# Curvilinear coordinates in global frame
h = angle/length
fb = bpmeth.BendFrame(fr,length,angle)

# Plot a trajectory with dipole in y direction
delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
theta0 = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

smax = 0.99 * ( px0 + np.sqrt( (1+delta)**2 - py0**2) ) / BB
s = np.linspace(0,smax,100)
x = 1/BB * ( np.sqrt((1+delta)**2-(-BB*s+px0)**2-py0**2) - np.sqrt((1+delta)**2-px0**2-py0**2) ) + x0
y = -py0/BB * ( np.arcsin( (-BB*s+px0) / np.sqrt((1+delta)**2-py0**2) ) - np.arcsin( px0 / np.sqrt((1+delta)**2-py0**2) ) )
fb.plot_trajectory_zx(s,x,y, figname="figures/straight_dipole.png")
fb.plot_trajectory_zxy(s,x,y, figname="figures/straight_dipole_3D.png")

plt.show()
