import bpmeth
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the analytical solution for a solenoid using bpmeth
"""

# Parameters
length = 5
angle = 0

BBs = 1 # Only q Bs / P0 plays a role so lets call this quantity BBs
x0 = 0
y0 = 0
px0 = 0.2
py0 = 0.2
ptau0 = 0
beta0 = 1

# Frame in the origin
fr = bpmeth.Frame()

# Curvilinear coordinates in global frame
h = angle/length
fb = bpmeth.BendFrame(fr,length,angle)

# Plot a trajectory with solenoid
delta0 = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
# Square root in initial coordinates since conserved
omega = BBs / np.sqrt((1+delta0)**2 - (px0 + BBs/2*y0)**2 * (py0 - BBs/2*x0)**2) 

s = np.linspace(0,length,500)
cc = np.cos(omega*s)
ss = np.sin(omega*s)

x = 1/2*(cc+1)*x0 + 1/BBs*ss*px0 + 1/2*ss*y0 - 1/BBs*(cc-1)*py0
y = -1/2*ss*x0 + 1/BBs*(cc-1)*px0 + 1/2*(cc+1)*y0 + 1/BBs*ss*py0

fb.plot_trajectory_zx(s,x,y, figname="figures/solenoid.png")
fb.plot_trajectory_zxy(s,x,y, figname="figures/solenoid_3D.png")

plt.show()
