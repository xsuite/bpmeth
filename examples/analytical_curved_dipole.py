import bpmeth
import numpy as np
import matplotlib.pyplot as plt

####################################################################
# Plot the analytical solution for a curved dipole in curved frame #
####################################################################

# Parameters
length = 1
angle = 1

b1 = 1
x0 = 0
y0 = 0
px0 = 0
py0 = 0
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

s = np.linspace(0,1,100)
px = px0*np.cos(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*np.sin(s*h)
pxder = -px0*h*np.sin(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*h*np.cos(s*h)
x = 1/h * 1/b1 * ( h*np.sqrt((1+delta)**2-px**2-py0**2) - pxder - b1)
y = h*py0/b1*s - py0/b1 * ( np.arcsin(px/np.sqrt((1+delta)**2)) - np.arcsin(px0/np.sqrt((1+delta)**2)) ) + y0

fb.plot_trajectory_zx(s,x,y)
fb.plot_trajectory_zxy(s,x,y)

plt.show()
