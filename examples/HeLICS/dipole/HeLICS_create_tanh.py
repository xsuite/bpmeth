import numpy as np
import bpmeth
import matplotlib.pyplot as plt
import sympy as sp

############################
# Magnet design parameters #
############################

rho = 2.737
phi = 60/180*np.pi
theta_E = 30/180*np.pi
l_magn = rho*phi
gap = 0.07

dipole_k0 = 1/rho
dipole_k1 = -0.11

############################################################
# Estimate parameters for HeLICS based on ELENA            #
# See ELENA_create_tanh.py for the fitting to the fieldmap #
############################################################

b1_tanhparams = [dipole_k0, -0.9*gap]
b2_tanhparams = [dipole_k1, -0.9*gap]

####################################
# Create half a magnet tanh dipole # 
# Inspired by my model for ELENA   #
####################################

# Created orthogonal to the edge
s = sp.symbols('s')
b1model_tanh = bpmeth.spTanh(-s, *b1_tanhparams)
b1_edge_tanh = bpmeth.GeneralVectorPotential(b=(b1model_tanh,))

Bx, By, Bs = b1_edge_tanh.get_Bfield()

xx, yy, zz = np.meshgrid(np.linspace(-0.6, 0.6, 500), [0], np.linspace(-0.5, 2.5, 500))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

Bxvals = Bx(xx, yy, zz)
Byvals = By(xx, yy, zz)
Bsvals = Bs(xx, yy, zz)

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bsvals))

# Translate and rotate to the position of the edge
b1_edge_fieldmap = bpmeth.Fieldmap(data)
b1_edge_fieldmap = b1_edge_fieldmap.rotate(phi/2-theta_E)
b1_edge_fieldmap = b1_edge_fieldmap.translate(dx=rho*np.cos(-phi/2), dy=0, dz=rho*np.sin(-phi/2))  # Global frame of edge fieldmap

xFS = np.linspace(-0.05, 0.05, 201)
yFS = [0]
sFS = np.arange(-l_magn, 0, 0.001)

# Read in FS coordinates
FS_b1_edge_fieldmap = b1_edge_fieldmap.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.01)

fig, ax = plt.subplots()
FS_b1_edge_fieldmap.z_multipoles(2, ax=ax)
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_tanh_dipole.png", dpi=500)
plt.close()

#################################################
# Create half a magnet tanh quadrupole with cut #
#################################################

# Create immediately in FS coordinates: half straight, half curved frame
s = sp.symbols('s')
b2model_tanh = bpmeth.spTanh(-s, *b2_tanhparams)
b2_edge_tanh = bpmeth.GeneralVectorPotential(b=("0", b2model_tanh,))
b2_edge_tanh = b2_edge_tanh.cut_at_angle(theta_E)

b2_enge_tanh_curved = bpmeth.GeneralVectorPotential(b=b2_edge_tanh.b, hs=f"{1/rho}")

b2_edge_tanh.translate(ds=-l_magn/2)
b2_enge_tanh_curved.translate(ds=-l_magn/2)

Bx_neg, By_neg, Bs_neg = b2_edge_tanh.get_Bfield()
Bx_pos, By_pos, Bs_pos = b2_enge_tanh_curved.get_Bfield()

xx = FS_b1_edge_fieldmap.src['x']
yy = FS_b1_edge_fieldmap.src['y']
zz = FS_b1_edge_fieldmap.src['z']

mask_neg = zz<=-l_magn/2
xx_neg = xx[mask_neg]
yy_neg = yy[mask_neg]
zz_neg = zz[mask_neg]
mask_pos = zz>-l_magn/2
xx_pos = xx[mask_pos]
yy_pos = yy[mask_pos]
zz_pos = zz[mask_pos]

Bxvals_neg = Bx_neg(xx_neg, yy_neg, zz_neg)
Byvals_neg = By_neg(xx_neg, yy_neg, zz_neg)
Bsvals_neg = Bs_neg(xx_neg, yy_neg, zz_neg)
Bxvals_pos = Bx_pos(xx_pos, yy_pos, zz_pos)
Byvals_pos = By_pos(xx_pos, yy_pos, zz_pos)
Bsvals_pos = Bs_pos(xx_pos, yy_pos, zz_pos)

xx = np.concatenate((xx_neg, xx_pos))
yy = np.concatenate((yy_neg, yy_pos))
zz = np.concatenate((zz_neg, zz_pos))
Bxvals = np.concatenate((Bxvals_neg, Bxvals_pos))
Byvals = np.concatenate((Byvals_neg, Byvals_pos))
Bsvals = np.concatenate((Bsvals_neg, Bsvals_pos))

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bsvals))

FS_b2_edge_fieldmap = bpmeth.Fieldmap(data)

fig, ax = plt.subplots()
FS_b2_edge_fieldmap.z_multipoles(2, ax=ax)
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_tanh_quadrupole.png", dpi=500)
plt.close()

#######################################
# Add dipole and quadrupole component #
#######################################

FS_edge_fieldmap = FS_b1_edge_fieldmap + FS_b2_edge_fieldmap

fig, ax = plt.subplots()
FS_edge_fieldmap.z_multipoles(2, ax=ax)
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_tanh.png", dpi=500)
plt.close()

fieldmap = FS_edge_fieldmap.mirror()

fig, ax = plt.subplots()
fieldmap.z_multipoles(2, ax=ax)
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_tanh.png", dpi=500)
plt.close()

data = fieldmap.get_data()
np.savetxt("HeLICS_fieldmap_tanh.csv", data, delimiter=",", header="x,y,z,Bx,By,Bs", comments='')

