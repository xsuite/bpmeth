import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd

############################
# Magnet design parameters #
############################

phi = 60/180*np.pi
rho = 0.927  # https://edms.cern.ch/ui/file/1311860/2.0/LNA-MBHEK-ER-0001-20-00.pdf
dipole_h = 1/rho
l_magn = rho*phi
gap=0.076
theta_E = 17/180*np.pi
fint = 0.424

############################################# 
# bpmeth dipole fieldmap model with splines #
#############################################

# My dipole with splines, matched such that the closed orbit remains zero
data = np.loadtxt("../dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

kf_splines=1.000447  # No closed orbit
# nphi=3 includes up to second order derivatives of b.
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=3, hgap=gap/2, nphi=3, plot=False, symmetric=True, step=50, radius=0.0025)

######################################### 
# Edge model: what to expect for dipole #
######################################### 

# Edge of magnet fieldmap
xmin, xmax, n_x = -0.4*gap, 0.4*gap, 51
zmin, zmax, n_z = -0.2*l_magn, 0.2*l_magn, 201
xarr = np.linspace(xmin, xmax, n_x)
yarr = [0]
zarr = np.linspace(zmin, zmax, n_z)

data_edge = np.loadtxt("../dipole/ELENA_edge.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]
dipole_edge = bpmeth.Fieldmap(data_edge)
central_field_from_enge_fit = 4.28304565e-01  # Determined from directly fitting fieldmap without rescaling

dipole_edge.rescale(dipole_h /central_field_from_enge_fit)
dipole_edge = dipole_edge.calc_edge_frame(xarr, yarr, zarr, theta_E, rho, phi, radius=0.005)
fig, ax = plt.subplots()
guess = [dipole_h,  5.45050869e+02, -1.34905910e+02, 2.84624272e+01,  5.49796478e-01]
# guess = [5.375e-7, -3.85190565e+03, 1.29508736e+03, -1.29954076e+02, 2.50076967e+01, 5.22942000e-01]
engeparams, engecov = dipole_edge.fit_multipoles(bpmeth.spEnge, components=[1], design=1, nparams=5, ax=ax,
                                                 guess = guess, padding=True, entrance=False, design_field=dipole_h)

# Compare fieldmap with model based on edge enge and rotation
s = sp.symbols('s')
b1 = sp.Function('b1')(s)
b1model = bpmeth.spEnge(-s, *engeparams[0])

dipole_model = bpmeth.GeneralVectorPotential(b=(b1model,))
dipole_model_FS = dipole_model.transform(theta_E=theta_E, maxpow=3)

fig, ax = plt.subplots()
dipole_splines.fieldmap.z_multipoles(2, ax=ax)
dipole_model_FS.plot_components(smin=-0.5, smax=0.5, s_shift=-l_magn/2, ax=ax)
plt.axvline(x=-l_magn/2, color='black', ls='--', label='Edge of magnet')
s_min=0.3
s_max=0.2
plt.axvline(x=-s_min-l_magn/2, color='gray', ls='--')
plt.axvline(x=s_max-l_magn/2, color='gray', ls='--')
plt.savefig("figures/ELENA_edge_model.png", dpi=500)
plt.close()

print("Integrated components splines edge: ", dipole_splines.fieldmap.integratedfield(3, zmin=-s_min-l_magn/2, zmax=s_max-l_magn/2))
print("Integrated components model edge: ", dipole_model_FS.integrate_components(smin=-s_min, smax=s_max))
print("Integrated component splines body: ", dipole_splines.fieldmap.integratedfield(3, zmin=s_max-l_magn/2, zmax=l_magn/2-s_max))