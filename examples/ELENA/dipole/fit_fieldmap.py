# ELENA ring:
#     - 6 dipoles
#     - magnetic length              0.9708 m
#     - aperture                     0.076 m
#     - bending angles               60°
#     - bending radii                0.927 m
#     - edge angles (beam focusing)  17°
#     - field at (0,0,0)             5.3810e-07
#     - Brho                         4.9749e-07

import bpmeth
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

rho = 0.927
phi = 60/180*np.pi
l_magn = rho*phi
apt = 0.076
hgap = apt/2
theta_E = 17/180*np.pi
B = 5.3810e-07
Brho = B*rho

# x,y,z,phi_m,Ax,Ay,Az,Bx,By,Bz
data = np.loadtxt("ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

ELENA_dipole = bpmeth.Fieldmap(data)

xFS = np.linspace(-apt/2, apt/2, 31)
yFS = np.linspace(-apt/2, apt/2, 11)
sFS = np.linspace(-0.75*l_magn, 0.75*l_magn, 201)
ELENA_dipole_FS = ELENA_dipole.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.05)
# ELENA_dipole_FS.plot()


###############################################
# FIT DIPOLE ENGE IN STRAIGHT/BENT FRAME      #
###############################################

# Multipole components fitted over the whole s-range at once
fig, ax = plt.subplots()
params, cov = ELENA_dipole_FS.fit_multipoles(bpmeth.spEnge, components=[1,2,3], design=1, zmin=0, zmax=l_magn/2+3.8*hgap, zedge=l_magn/2, ax=ax)


fig, ax = plt.subplots()
params, cov = ELENA_dipole_FS.fit_multipoles(bpmeth.spTanh, components=[1,2,3], design=1, zmin=0, zmax=l_magn/2+3.8*hgap, zedge=l_magn/2, ax=ax, nparams=2, guess=[5e-7, -0.1])

# These multipole coefficient expressions can now be given to a field expansion, to be used in tracking code.



###############################################
# DETERMINE MULTIPOLES IN STRAIGHT/BENT FRAME #
###############################################

# Determine multipoles over the whole s-range
zvals, coeffs, coeffsstd = ELENA_dipole_FS.z_multipoles(4)

fig, ax = plt.subplots()
ax.errorbar(zvals, coeffs[:,0], label="b1", yerr=coeffsstd[:,0], marker=".", ls="", capsize=3, color="blue")
ax.errorbar(zvals, coeffs[:,1], label="b2", yerr=coeffsstd[:,1], marker=".", ls="", capsize=3, color="orange")
ax.errorbar(zvals, coeffs[:,2], label="b3", yerr=coeffsstd[:,2], marker=".", ls="", capsize=3, color="green")
plt.vlines(-l_magn/2, -5e-6, 8e-6, color="black", ls="--")
plt.vlines(l_magn/2, -5e-6, 8e-6, color="black", ls="--")
ax.set_ylim(-5e-6, 8e-6)


###############################################
# MODEL BASED ON FIT AT THE HARD EDGE         #
###############################################

# Parameters at the hard edge
engeparams = [5.375e-7, -3.85190565e+03, 1.29508736e+03, -1.29954076e+02, 2.50076967e+01, 5.22942000e-01]
s = sp.symbols("s")
b1model = bpmeth.spEnge(-s, *engeparams)

# Model with edge angle in the straight part
ss = np.linspace(-0.2-l_magn/2, -l_magn/2, 50)
ax.plot(ss, [b1model.subs({s:(sval+l_magn/2)*np.cos(theta_E)}).evalf() for sval in ss], label="model b1", color="lightskyblue")
ax.plot(ss, [-b1model.diff(s).subs({s:(sval+l_magn/2)*np.cos(theta_E)}).evalf() * np.sin(theta_E) for sval in ss], label="model b2", color="moccasin")
ax.plot(ss, [b1model.diff(s, 2).subs({s:(sval+l_magn/2)*np.cos(theta_E)}).evalf() * np.sin(theta_E)**2 for sval in ss], label="model b3", color="limegreen")

# Model with curvature in curved part ASSUMES RECTANCULAR BEND, BUT FOR PURE DIPOLE THIS IS OK
ss = np.linspace(-l_magn/2, 0, 250)
ax.plot(ss, [b1model.subs({s:rho*np.sin(theta_E) - rho*np.sin(theta_E - (sval+l_magn/2)/rho)}).evalf() for sval in ss], color="lightskyblue")
ax.plot(ss, [-b1model.diff(s).subs({s:rho*np.sin(theta_E) - rho*np.sin(theta_E - (sval+l_magn/2)/rho)}).evalf() * np.sin(theta_E - (sval+l_magn/2)/rho) for sval in ss], color="moccasin")
ax.plot(ss, [b1model.diff(s, 2).subs({s:rho*np.sin(theta_E) - rho*np.sin(theta_E - (sval+l_magn/2)/rho)}).evalf() * np.sin(theta_E - (sval+l_magn/2)/rho)**2 for sval in ss], color="limegreen")

plt.legend()


###############################################
# XSUITE TRACKING                             #
###############################################

# Design field: 0.42 T, there is a mu0 factor missing in the fieldmap, so divide by 4 pi 1e-7
# but immediately corrects the length as well!
# We want the integrated field to be k0l, so design field = k0 = 1/rho?
dipole = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn, design_field=1/rho, shape="enge", hgap=apt/2, apt=apt, radius=0.05, order=1, nphi=2, plot=True)

import xtrack as xt
p = xt.Particles(x=np.linspace(-2e-3, 2e-3, 11), y=np.linspace(-2e-3, 2e-3, 11), p0c=0.1, mass0=0.938272, q0=1)
dipole.track(p, plot=True)