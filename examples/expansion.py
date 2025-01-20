import bpmeth
import numpy as np
import sympy as sp

fringe = bpmeth.FieldExpansion(b=("(tanh(s)+1)/2",))
fringe.plotfield()

xmin, xmax, xstep = -2, 2, 0.2
ymin, ymax, ystep = -2, 2, 0.2
zmin, zmax, zstep = -3, 3, 0.2
X = np.arange(xmin, xmax, xstep)
Y = np.arange(ymin, ymax, ystep)
Z = np.arange(zmin, zmax, zstep)
fringe.writefile(X, Y, Z, "tanh")

Ax, Ay, As = fringe.get_A()

Bxvec = As.diff(fringe.y)
Byvec = 1/(1+fringe.hs*fringe.x) * (Ax.diff(fringe.s) - ((1+fringe.hs*fringe.x)*As).diff(fringe.x))
Bsvec = - Ax.diff(fringe.y)


