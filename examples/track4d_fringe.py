import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import bpmeth
import mpmath

"""
Analysis of a ring with a single fringe field
"""

def K0ggtanh(b1, a, L):
    """    
    K0 fringe field integral for a tanh times the gap height squared.

    :param b1: fringe field coefficient.
    :param a: fringe field range.
    :param L: integral around the fringe field ranges from -L to L.
        (In theory the value of K0gg should not depend on L, otherwise L is chosen too small.)
    """

    return b1* ( L**2/2 - L*a/2*np.log((1+np.exp(-2*L/a))*(1+np.exp(2*L/a))) \
        + a**2/4*(mpmath.polylog(2, -np.exp(-2*L/a))-mpmath.polylog(2, -np.exp(2*L/a))) )


npart = 10
part = np.zeros((4, npart))
part[0] = np.linspace(0, 0.5, npart)
part[2] = np.linspace(0, 0.5, npart)
Qx = 0.112
Qy = 0.252
nturns = 2000
xlims=[-1,1] 
ylims=[-1,1]

b1 = 0.1
aa = 1
Kg = aa/2
b1shape = f"(tanh(s/{aa})+1)/2"
s = sp.symbols("s")
b1sym = eval(b1shape, sp.__dict__, {"s": s})
len = 5
K0gg = K0ggtanh(b1, aa, len/2)


line_thinfringe = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ThinNumericalFringe(b1, b1shape, len=len, nphi=5)])
o_thinfringe = line_thinfringe.track(part, num_turns=nturns)
# o_thinfringe.plot_xpx(xlims=xlims, ylims=ylims)
# o_thinfringe.plot_ypy(xlims=xlims, ylims=ylims)

line_forest = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ForestFringe(b1, Kg)])
o_forest = line_forest.track(part, num_turns=nturns)
# o_forest.plot_xpx(xlims=xlims, ylims=ylims)
# o_forest.plot_ypy(xlims=xlims, ylims=ylims)

line_forest_co = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ForestFringe(b1, Kg, K0gg, closedorbit=True)])
o_forest_co = line_forest_co.track(part, num_turns=nturns)
# o_forest_co.plot_xpx(xlims=xlims, ylims=ylims)
# o_forest_co.plot_ypy(xlims=xlims, ylims=ylims)


index=1
padding=2*nturns
log=True
unwrap=True


figx, axx = plt.subplots(2)
plt.title('Spectrum x')

o_thinfringe.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                             label="Thin", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                             color="orange")
o_forest.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                         label="Forest", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                         color="green")
o_forest_co.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                            label="Forest with closed orbit", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                            color="blue")

sl_x = [(0,0), (0, -2), (0, 2), (0, -4), (0, 4)]
for xx, yy in sl_x:
    val = xx*Qx + yy*Qy
    if -0.6 < val < 0.6:
        axx[0].vlines(val, -9, -7, color="red")

axx[0].set_xlim(-0.6, 0.6)
axx[1].set_xlim(-0.6, 0.6)
axx[0].set_ylabel("Amplitude")
axx[1].set_ylabel("Phase")


figy, axy = plt.subplots(2)
plt.title('Spectrum y')

o_thinfringe.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                             label="Thin", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                             color="orange")
o_forest.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                         label="Forest", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                         color="green")
o_forest_co.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                            label="Forest with closed orbit", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                            color="blue")

sl_y = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (0, 3), (0, 1), (0, -1), (0, -3)]
for xx, yy in sl_y:
    val = xx*Qx + yy*Qy
    if -0.6 < val < 0.6:
        axy[0].vlines(val, -7, -5, color="red")

axy[0].set_xlim(-0.6, 0.6)
axy[1].set_xlim(-0.6, 0.6)
axy[0].set_ylabel("Amplitude")
axy[1].set_ylabel("Phase")




nsvals = 10

# Solution with normal forms
hsym = list(np.zeros((4,4,4,4)))
hsym = np.zeros((4,4,4,4,nsvals))


# First term Hamiltonian: closed orbit distortion
hsym[1,0,0,0] += 1/2 * b1sym
hsym[0,1,0,0] += 1/2 * b1sym

# Second term Hamiltonian: "sextupole-like"
hsym[1,0,2,0] += -1/16 * 1j * b1sym.diff(s)
hsym[0,1,2,0] +=  1/16 * 1j * b1sym.diff(s)
hsym[1,0,0,2] += -1/16 * 1j * b1sym.diff(s)
hsym[0,1,0,2] +=  1/16 * 1j * b1sym.diff(s)
hsym[1,0,1,1] += -1/16 * 2 * 1j * b1sym.diff(s)
hsym[0,1,1,1] +=  1/16 * 2 * 1j * b1sym.diff(s)

# Third term Hamiltonian

for p in range(len(h)):
    for q in range(len(h[p])):
        for r in range(len(h[p, q])):
            for t in range(len(h[p, q, r])):
                



frin4d = bpmeth.NormalForms4d(h, 2*np.pi*Qx, 2*np.pi*Qy, Qx, Qy, nturns)
o_norm = sext4d.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)