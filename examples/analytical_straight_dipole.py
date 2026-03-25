import bpmeth
import numpy as np
import matplotlib.pyplot as plt

###################################################################
# Plot the analytical solution for a straight dipole using bpmeth #
###################################################################

# Parameters
length = 10
angle = 0

b1 = 0.05 
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

smax = 0.99 * ( px0 + np.sqrt( (1+delta)**2 - py0**2) ) / b1
s = np.linspace(0,smax,100)
x = 1/b1 * ( np.sqrt((1+delta)**2-(-b1*s+px0)**2-py0**2) - np.sqrt((1+delta)**2-px0**2-py0**2) ) + x0
y = -py0/b1 * ( np.arcsin( (-b1*s+px0) / np.sqrt((1+delta)**2-py0**2) ) - np.arcsin( px0 / np.sqrt((1+delta)**2-py0**2) ) )
fb.plot_trajectory_zx(s,x,y)
fb.plot_trajectory_zxy(s,x,y)

plt.show()
