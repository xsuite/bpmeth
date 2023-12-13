import bpmeth
import numpy as np
import matplotlib.pyplot as plt


# Parameters
length = 10
angle = 0.5

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

s = np.linspace(0,10,100)
px = px0*np.cos(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*BB )*np.sin(s*h)
pxder = -px0*h*np.sin(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*BB )*h*np.cos(s*h)
x = 1/h * 1/BB * ( h*np.sqrt((1+delta)**2-px**2-py0**2) - pxder - BB)
y = h*py0/BB*s - py0/BB * ( np.arcsin(px/np.sqrt((1+delta)**2)) - np.arcsin(px0/np.sqrt((1+delta)**2)) ) + y0

fb.plot_trajectory_zx(s,x,y, figname="figures/curved_dipole.png")
fb.plot_trajectory_zxy(s,x,y, figname="figures/curved_dipole_3D.png")

plt.show()
