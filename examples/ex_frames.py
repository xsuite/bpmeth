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
px0 = -0.2
py0 = 0.1
ptau0 = 0
beta0 = 1
delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
b = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

s = np.linspace(0,10,100)
x = (1/h + x0) * (np.cos(b) / np.cos(b+s*h)) - 1/h 
y = py0 / (h * np.sqrt((1+delta)**2 - py0**2)) * (1+h*x0) * np.cos(b) * np.tan(b+h*s)
fb.plot_trajectory_zx(s,x,y)
fb.plot_trajectory_zxy(s,x,y)

