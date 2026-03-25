import numpy as np
import sympy as sp
import bpmeth
import xtrack as xt
import matplotlib.pyplot as plt

####################################################
# Create an RBend fieldmap with tanh fringe fields #
####################################################

# ----- Dipole parameters -----

l_bend = 0.425
l_bend_mech = 0.3275
fint_h_var = 0.5
h_gap = 0.02
angle_mbh = -15.0*np.pi/180.0

bb = xt.RBend(length_straight=l_bend, k0_from_h=True, angle=-angle_mbh, edge_entry_fint=fint_h_var, edge_exit_fint=fint_h_var, edge_entry_hgap=h_gap, edge_exit_hgap=h_gap)
b1 = bb.h

xapt = 0.1  # Horizontal aperture, can be changed, only for fieldmap points 


# ----- Make fieldmap -----

# The first parameter is the amplitude, the second paramter specifies the range a, b1 = tanh(z/a)
# For this model we have fint = a / (2 * gap)
b1_tanhparams = [b1, fint_h_var*h_gap*4]  

s = sp.symbols('s')
b1_entry = bpmeth.spTanh(s+l_bend/2, *b1_tanhparams)
b1_exit = bpmeth.spTanh(-(s-l_bend/2), *b1_tanhparams)
magnet = bpmeth.GeneralVectorPotential(b=(b1_entry + b1_exit - b1,))


# ----- RBend fieldmap in global coordinates -----

Bx, By, Bz = magnet.get_Bfield()

saggita = 1/bb.h*(1-np.cos(angle_mbh/2))
print("Saggita of RBend: ", saggita)

xarr = np.linspace(-xapt - saggita + 1/bb.h, xapt+1/bb.h, 100)  # Make sure to have enough margin
yarr = np.array([0]) 
zarr = np.arange(-l_bend/2-10*h_gap, l_bend/2+10*h_gap, 0.001)

xx, yy, zz = np.meshgrid(xarr, yarr, zarr)
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

Bxvals = Bx(xx, yy, zz)
Byvals = By(xx, yy, zz)
Bzvals = Bz(xx, yy, zz)

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bzvals))

np.savetxt('RBend_fieldmap.txt', data, delimiter=",", header='x [m], y [m], z [m], Bx [T], By [T], Bz [T]')


# ----- Fieldmap in coordinates along particle trajectory -----

fieldmap = bpmeth.Fieldmap(data)

xFS = np.linspace(-xapt/2, xapt/2, 51)  # Best to have point in center if fitting multipoles (for my tracking) is desired
yFS = [0] 
smin=-l_bend/2-10*h_gap
smax=l_bend/2+10*h_gap
sFS = np.arange(smin, smax, 0.001)
fieldmap_FS = fieldmap.calc_FS_coords(xFS, yFS, sFS, rho=1/bb.h, phi=-angle_mbh, radius=0.002)

data_FS = fieldmap_FS.get_data()

np.savetxt('RBend_fieldmap_FS.txt', data_FS, delimiter=",", header='x [m], y [m], s [m], Bx [T], By [T], Bs [T]')


# ----- Tracking -----

order = 2  # Order of multipole expansion used in tracking, can be changed
dipole = bpmeth.Splines_Magnet(data_FS, bb.h, l_bend, order=order, radius=0.002, in_FS_coord=True, smin=smin, smax=smax)

p0 = xt.Particles()
dipole.track(p0, plot=True)
plt.show()
