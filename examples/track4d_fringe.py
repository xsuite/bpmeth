import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp

"""
Analysis of a ring with a single fringe field
"""

npart = 10
part = np.zeros((4, npart))
part[0] = np.linspace(0, 0.8, npart)
Qx = 0.112
Qy = 0.221
nturns = 100
xlims=[-1,1] 
ylims=[-1,1]
len = 1

b1 = 1
aa = 0.1
Kg = aa/2
b1shape = f"(tanh(s/{aa})+1)/2"
s = sp.symbols("s")
b1sym = eval(b1shape, sp.__dict__, {"s": s})

line_fringe = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ThinNumericalFringe(b1, b1shape, len=len, nphi=5)])
o_fringe = line_fringe.track(part, num_turns=nturns)
o_fringe.plot_xpx(xlims=[-2, 2], ylims=ylims)
o_fringe.plot_ypy(xlims=[-1, 1], ylims=ylims)

line_forest = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ForestFringe(b1, Kg)])
o_forest = line_forest.track(part, num_turns=nturns)
o_forest.plot_xpx(xlims=[-2, 2], ylims=ylims)
o_forest.plot_ypy(xlims=[-2, 2], ylims=ylims)


"""
Problem: 
RDTs for fringe fields are s-dependent. What happens for the phase advance inside the fringe field?
Should we approximate the fringe field with a single RDT based on solid approximations?
"""

# Solution with normal forms
h = np.zeros((4,4,4,4))

# First term Hamiltonian: closed orbit distortion
h[1,0,0,0] = 1/2 * b1sym
h[0,1,0,0] = 1/2 * b1sym

# Second term Hamiltonian: "sextupole-like"
h[1,0,2,0] = -1/16 * 1j * b1sym.diff(s)
h[0,1,2,0] =  1/16 * 1j * b1sym.diff(s)
h[1,0,0,2] = -1/16 * 1j * b1sym.diff(s)
h[0,1,0,2] =  1/16 * 1j * b1sym.diff(s)
h[1,0,1,1] = -1/16 * 2 * 1j * b1sym.diff(s)
h[0,1,1,1] =  1/16 * 2 * 1j * b1sym.diff(s)

# Third term Hamiltonian


frin4d = bpmeth.NormalForms4d(h, 2*np.pi*Qx, 2*np.pi*Qy, Qx, Qy, nturns)
o_norm = sext4d.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)