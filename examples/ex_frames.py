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

