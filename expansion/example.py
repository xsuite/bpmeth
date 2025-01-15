import generate_expansion
import numpy as np
import sympy as sp

fringe = generate_expansion.FieldExpansion(b=("(tanh(s)+1)/2",))
fringe.plotfield()

xmin, xmax, xstep = -2, 2, 0.2
ymin, ymax, ystep = -2, 2, 0.2
zmin, zmax, zstep = -3, 3, 0.2
X = np.arange(xmin, xmax, xstep)
Y = np.arange(ymin, ymax, ystep)
Z = np.arange(zmin, zmax, zstep)
fringe.writefile(X, Y, Z, "tanh")