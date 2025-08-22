import numpy as np
import bpmeth
import matplotlib.pyplot as plt
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

########################################
# Edge of magnet fieldmap and tanh fit #
########################################

xmin, xmax, n_x = -0.4*gap, 0.4*gap, 51
zmin, zmax, n_z = -0.2*l_magn, 0.2*l_magn, 201
xarr = np.linspace(xmin, xmax, n_x)
yarr = [0]
zarr = np.linspace(zmin, zmax, n_z)

data_edge = np.loadtxt("ELENA_edge.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]
dipole_edge = bpmeth.Fieldmap(data_edge)
central_field_from_enge_fit = 4.28304565e-01  # Determined from directly fitting fieldmap without rescaling

dipole_edge.rescale(dipole_h /central_field_from_enge_fit)
dipole_edge = dipole_edge.calc_edge_frame(xarr, yarr, zarr, theta_E, rho, phi, radius=0.005)# Fit tanh function

tanhparams, tanhcov = dipole_edge.fit_multipoles(bpmeth.spTanh, components=[1], design=1, nparams=2, 
                                                 guess=[dipole_h, -0.07], padding=True, entrance=False, design_field=dipole_h)

############################################################
# Model based on b1 tanh only, but as seen from a FS frame #
############################################################

s = sp.symbols('s')
b1model_tanh = bpmeth.spTanh(-s, *tanhparams[0])
edge_model_tanh = bpmeth.GeneralVectorPotential(b=(b1model_tanh,))

Bx, By, Bs = edge_model_tanh.get_Bfield()

xx, yy, zz = np.meshgrid(np.linspace(-0.2, 0.2, 500), [0], np.linspace(-0.5, 0.5, 200))
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

Bxvals = Bx(xx, yy, zz)
Byvals = By(xx, yy, zz)
Bsvals = Bs(xx, yy, zz)

data = np.column_stack((xx, yy, zz, Bxvals, Byvals, Bsvals))

# Create fieldmap object based on this data, and rotate and translate it to the position of the edge
model_edge_fieldmap_tanh = bpmeth.Fieldmap(data)
model_edge_fieldmap_tanh = model_edge_fieldmap_tanh.rotate(phi/2-theta_E)
model_edge_fieldmap_tanh = model_edge_fieldmap_tanh.translate(dx=rho*np.cos(-phi/2), dy=0, dz=rho*np.sin(-phi/2))  # Global frame of edge fieldmap

xFS = np.linspace(-0.05, 0.05, 201)
yFS = [0]
sFS = np.arange(-l_magn, 0, 0.001)

FS_model_edge_fieldmap_tanh = model_edge_fieldmap_tanh.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.01)
fieldmap_tanh_complete = FS_model_edge_fieldmap_tanh.mirror()

data = fieldmap_tanh_complete.get_data()
np.savetxt("ELENA_fieldmap_tanh.csv", data, delimiter=",", header="x,y,z,Bx,By,Bs", comments='')
