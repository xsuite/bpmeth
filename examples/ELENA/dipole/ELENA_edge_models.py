import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp

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

###########################
# Edge of magnet fieldmap #
###########################

xmin, xmax, n_x = -0.4*gap, 0.4*gap, 51
zmin, zmax, n_z = -0.2*l_magn, 0.2*l_magn, 201
xarr = np.linspace(xmin, xmax, n_x)
yarr = [0]
zarr = np.linspace(zmin, zmax, n_z)

data_edge = np.loadtxt("ELENA_edge.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]
dipole_edge = bpmeth.Fieldmap(data_edge)
central_field_from_enge_fit = 4.28304565e-01  # Determined from directly fitting fieldmap without rescaling

dipole_edge.rescale(dipole_h /central_field_from_enge_fit)
dipole_edge = dipole_edge.calc_edge_frame(xarr, yarr, zarr, theta_E, rho, phi, radius=0.005)
fig, ax = plt.subplots()
guess = [dipole_h,  5.45050869e+02, -1.34905910e+02, 2.84624272e+01,  5.49796478e-01]
# guess = [5.375e-7, -3.85190565e+03, 1.29508736e+03, -1.29954076e+02, 2.50076967e+01, 5.22942000e-01]
engeparams, engecov = dipole_edge.fit_multipoles(bpmeth.spEnge, components=[1], design=1, nparams=5, ax=ax,
                                                 guess = guess, padding=True, entrance=False, design_field=dipole_h)
plt.savefig("figures/ELENA_edge_enge_fit.png", dpi=500)
plt.close()

# If we would plot the higher order components of the edge fieldmap, we can see that there is a quadrupole 
# and sextupole component as well. These are from the design.

######################################################################
# Model based on real b1 component only, but as seen from a FS frame #
######################################################################

s = sp.symbols('s')
b1 = sp.Function('b1')(s)
b1model = bpmeth.spEnge(-s, *engeparams[0])
edge_model = bpmeth.GeneralVectorPotential(b=(b1model,))

# Taking into account curvature symbolically is heavy, but it can be used for understanding
edge_model_without_curv = edge_model.transform(theta_E=theta_E, maxpow=3)

# Extract points from a fieldmap, computationally favoured
Bx, By, Bs = edge_model.get_Bfield()
xx, yy, zz = np.meshgrid(np.linspace(-0.2, 0.2, 500), [0], np.linspace(-0.5, 0.5, 200))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

Bxvals = Bx(xx, yy, zz)
Byvals = By(xx, yy, zz)
Bsvals = Bs(xx, yy, zz)

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bsvals))

# Create fieldmap object based on this data, and rotate and translate it to the position of the edge
model_edge_fieldmap = bpmeth.Fieldmap(data)
model_edge_fieldmap = model_edge_fieldmap.rotate(phi/2-theta_E)
model_edge_fieldmap = model_edge_fieldmap.translate(dx=rho*np.cos(-phi/2), dy=0, dz=rho*np.sin(-phi/2))  # Global frame of edge fieldmap

xFS = np.linspace(-0.05, 0.05, 201)
yFS = [0]
sFS = np.arange(-l_magn, 0, 0.001)

# Edge as seen in FS coordinates, so what we would expect from the ELENA fieldmap if there were no design components
FS_model_edge_fieldmap = model_edge_fieldmap.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.01)

############################################# 
# bpmeth dipole fieldmap model with splines #
#############################################

# My dipole with splines, matched such that the closed orbit remains zero
data = np.loadtxt("ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

kf_splines=1.000447  # No closed orbit
# nphi=3 includes up to second order derivatives of b.
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=3, hgap=gap/2, nphi=3, plot=False, symmetric=True, step=50, radius=0.0025)

############
# Plotting #
############

fig, ax = plt.subplots()
FS_model_edge_fieldmap.z_multipoles(2, ax=ax)
edge_model_without_curv.plot_components(smin=-0.5, smax=0.5, ns=300, s_shift=-l_magn/2, ax=ax)
plt.axvline(x=-l_magn/2, color='black', ls='--', label='Edge of magnet')
dipole_splines.fieldmap.z_multipoles(2, ax=ax)
plt.close()

