import bpmeth
import numpy as np

fr = bpmeth.Frame()

fr.plot_zx()


fr2=fr.copy().move_by([0.4,0,0.3]).plot_zx()


fb=bpmeth.BendFrame(fr2,10,0.5).plot_zx()

s=np.linspace(0,10,100)
x=0.1-0.03*s
y=0*s
fb.plot_trajectory_zx(s,x,y)

