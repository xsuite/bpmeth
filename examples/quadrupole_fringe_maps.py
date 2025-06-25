import numpy as np
import matplotlib.pyplot as plt
import bpmeth
from numpy.lib.stride_tricks import sliding_window_view
import xtrack as xt
import sympy as sp

def mavg(x, window_shape=4):
    return np.average(sliding_window_view(x, window_shape = window_shape), axis=1)


data_cutquad=np.loadtxt("../fieldmaps/quadrupole_toymodels/field_map_quad_edge_30deg_resol_2mm.txt",skiprows=9)
cutquad = bpmeth.Fieldmap(data_cutquad)

#fig, ax = plt.subplots()
bparam, bcov = cutquad.fit_multipoles(bpmeth.spEnge, components=[2,3], design=2, zmin=0.4, zmax=0.7, zedge=0.55, nparams=6, guess=[-10, 0,0,0,0,0])
#plt.show()

cutquad_b2enge = [-9.94420385e+00, -3.69102167e+03, 1.27929108e+03, -1.16519236e+02, 3.68663359e+01, 9.98984576e-02]
cutquad_b3enge_der = [-6.36160140e+00, 1.16143036e+04, 3.76870754e+03, -2.87748724e+02, 5.71152200e+01, 1.44880010e-01] 
b2amp = -9.94420385e+00

# -s since fit is for an exit fringe
s = sp.symbols("s")
b2 = bpmeth.get_derivative(0, 6, bpmeth.spEnge, lambdify=False)(-s, *cutquad_b2enge)
b3 = bpmeth.get_derivative(1, 6, bpmeth.spEnge, lambdify=False)(-s, *cutquad_b3enge_der)

# Particles
xvals = np.linspace(-0.5, 0.5, 11)
yvals = np.linspace(-0.5, 0.5, 11)
pxvals = np.linspace(-0.05, 0.05, 11)
pyvals = np.linspace(-0.05, 0.05, 11)
p = xt.Particles(x=xvals)
p2 = p.copy()

# Entrance fringe!
quadfringe = bpmeth.NumericalQuadFringe(b2, f"{b2amp}", "0", length=0.3, nphi=5)
ForestFringe = bpmeth.ForestQuadFringe(b2amp)

fig, ax = plt.subplots()
traj = quadfringe.track(p)
traj.plot_px(ax=ax)

traj_forest = ForestFringe.track(p2)
traj_forest.plot_px(ax=ax, marker='o', markeredgecolor='black')

plt.show()


