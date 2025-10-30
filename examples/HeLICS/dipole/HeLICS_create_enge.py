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

###################################################################
# Estimate parameters for HeLICS based on ELENA                   #
# See ELENA_edge_models.py for fit of enge orthogonal to the edge #
###################################################################

# elena_engeparams = [1.07874854, 545.05297078, -134.90628896, 28.46243753, 0.54979632]  # Melvin
elena_engeparams = [0.44378056, 66.41477208, -58.12780624, 30.6935168, 0.44352517]  # Vittorio: more clean!
nparams_elena = len(elena_engeparams)
elena_gap = 0.076

# Rescale s with gap difference to keep same value of fringe field integral
b1_engeparams = [dipole_k0, *[val*(elena_gap/gap)**(nparams_elena-2-i) for i, val in enumerate(elena_engeparams[1:])]]
b2_engeparams = [dipole_k1, *b1_engeparams[1:]]

####################################
# Create half a magnet enge dipole #
# Inspired by my model for ELENA   #
####################################

# Created orthogonal to the edge
s = sp.symbols('s')
b1_enge = bpmeth.spEnge(-(s+0.00707465219), *b1_engeparams)  # Shift edge to match integrated field 
b1_edge = bpmeth.GeneralVectorPotential(b=(b1_enge,))

Bx, By, Bs = b1_edge.get_Bfield()

xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, 500), [0], np.linspace(-0.5, 2.5, 500))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

Bxvals = Bx(xx, yy, zz)
Byvals = By(xx, yy, zz)
Bsvals = Bs(xx, yy, zz)

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bsvals))

# Translate and rotate to the position of the edge
b1_edge_fieldmap = bpmeth.Fieldmap(data)

print("Fringe field integral: ", b1_edge_fieldmap.calc_Fint(gap, dipole_k0))

b1_edge_fieldmap = b1_edge_fieldmap.rotate(phi/2-theta_E)
b1_edge_fieldmap = b1_edge_fieldmap.translate(dx=rho*np.cos(-phi/2), dy=0, dz=rho*np.sin(-phi/2))  # Global frame of edge fieldmap

xFS = np.linspace(-0.05, 0.05, 201)
yFS = [0]
sFS = np.arange(-l_magn, 0, 0.001)

# Read in FS coordinates
FS_b1_edge_fieldmap = b1_edge_fieldmap.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.01)
print(FS_b1_edge_fieldmap.integratedfield(1)[0]*2)
print(l_magn * dipole_k0)

fig, ax = plt.subplots()
FS_b1_edge_fieldmap.z_multipoles(2, ax=ax, marker='.')
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_enge_dipole.png", dpi=500)
plt.close()

#################################################
# Create half a magnet enge quadrupole with cut #
#################################################

# Create immediately in FS coordinates: half straight, half curved frame
s = sp.symbols('s')
b2_enge = bpmeth.spEnge(-(s+0.00725905843), *b2_engeparams)  # Shift edge to match integrated field 
b2_edge = bpmeth.GeneralVectorPotential(b=("0", b2_enge,), nphi=3)
b2_edge = b2_edge.cut_at_angle(theta_E, maxpow=3)

b2_edge_curved = bpmeth.GeneralVectorPotential(b=b2_edge.b, hs=f"{1/rho}", nphi=3)

b2_edge.translate(ds=-l_magn/2)
b2_edge_curved.translate(ds=-l_magn/2)

Bx_neg, By_neg, Bs_neg = b2_edge.get_Bfield()
Bx_pos, By_pos, Bs_pos = b2_edge_curved.get_Bfield()

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
print(FS_b2_edge_fieldmap.integratedfield(2)[1]*2)
print(l_magn * dipole_k1)

fig, ax = plt.subplots()
FS_b2_edge_fieldmap.z_multipoles(2, ax=ax, marker='.')
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_enge_quadrupole.png", dpi=500)
plt.close()

#######################################
# Add dipole and quadrupole component #
#######################################

FS_edge_fieldmap = FS_b1_edge_fieldmap + FS_b2_edge_fieldmap

fieldmap = FS_edge_fieldmap.mirror()

fig, ax = plt.subplots()
fieldmap.z_multipoles(2, ax=ax, marker='.')
ax.set_xlabel("s [m]")
plt.legend()
plt.savefig("figures/HeLICS_fieldmap_enge.png", dpi=500)
plt.close()

data = fieldmap.get_data()
np.savetxt("HeLICS_fieldmap_enge.csv", data, delimiter=",", header="x,y,z,Bx,By,Bs", comments='')

