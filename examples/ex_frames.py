import bpmeth
import numpy as np


# Frame in the origin
fr = bpmeth.Frame().plot_zx()

# Translated frame
offset = [0.4,0,0.3]
fr2 = fr.copy().move_by(offset).plot_zx()

# Curvilinear coordinates in global frame
length = 10
angle = 0.5
h = angle/length
fb = bpmeth.BendFrame(fr2,length,angle).plot_zx()

# Plot a trajectory with constant x
s = np.linspace(0,10,100)
x = 0.1 + 0*s
y = 0*s
fb.plot_trajectory_zx(s,x,y)

# Plot a drift trajectory
x0 = 0.5
y0 = 0.1
px0 = -0.2
py0 = 0.3
ptau0 = 0
beta0 = 1
delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
theta0 = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

s = np.linspace(0,10,100)
x = (1/h + x0) * (np.cos(theta0) / np.cos(theta0+s*h)) - 1/h 
y = py0 / (h * np.sqrt((1+delta)**2 - py0**2)) * (1+h*x0) * np.cos(theta0) * np.tan(theta0+h*s) + y0
fb.plot_trajectory_zx(s,x,y)
fb.plot_trajectory_zxy(s,x,y, figname="drift.png")

# Plot a trajectory with dipole in y direction for h=0
length = 10
angle = 0
h = angle/length
fb0 = bpmeth.BendFrame(fr2,length,angle)

BB = 0.05 # Only q By / P0 plays a role so lets call this quantity BB
x0 = 0.5
y0 = 0.1
px0 = 0.2
py0 = 0.3
ptau0 = 0
beta0 = 1
delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
theta0 = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

smax = 0.99 * ( px0 + np.sqrt( (1+delta)**2 - py0**2) ) / BB
s = np.linspace(0,smax,100)
x = 1/BB * ( np.sqrt((1+delta)**2-(-BB*s+px0)**2-py0**2) - np.sqrt((1+delta)**2-px0**2-py0**2) ) + x0
y = -py0/BB * ( np.arcsin( (-BB*s+px0) / np.sqrt((1+delta)**2-py0**2) ) - np.arcsin( px0 / np.sqrt((1+delta)**2-py0**2) ) )
fb0.plot_trajectory_zx(s,x,y)
fb0.plot_trajectory_zxy(s,x,y, figname="dipoleh0.png")


