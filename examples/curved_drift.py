import bpmeth
import numpy as np
import matplotlib.pyplot as plt


# Parameters
length = 10
angle = 0.5

x0 = 0.5
y0 = 0.1
px0 = -0.2
py0 = 0.3
ptau0 = 0
beta0 = 1

# Frame in the origin
fr = bpmeth.Frame()

# Curvilinear coordinates in global frame
h = angle/length
fb = bpmeth.BendFrame(fr,length,angle)

# Plot a drift trajectory
delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
theta0 = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

s = np.linspace(0,10,100)
x = (1/h + x0) * (np.cos(theta0) / np.cos(theta0+s*h)) - 1/h 
y = py0 / (h * np.sqrt((1+delta)**2 - py0**2)) * (1+h*x0) * np.cos(theta0) * np.tan(theta0+h*s) + y0
fb.plot_trajectory_zx(s,x,y, figname="figures/curved_drift.png")
fb.plot_trajectory_zxy(s,x,y, figname="figures/curved_drift_3D.png")

plt.show()

