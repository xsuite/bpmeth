import numpy as np
import matplotlib.pyplot as plt
import bpmeth
import xtrack as xt


data = np.loadtxt("ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

rho = 0.927
phi = 60/180*np.pi
l_magn = rho*phi
apt = 0.076
hgap = apt/2
theta_E = 17/180*np.pi
B = 5.3810e-07
Brho = B*rho

guess = [ 4.27855927e-01, 7.63365939e+02, -1.44485719e+02, 2.59387643e+01, 5.81379154e-01]
dipole_enge = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn, design_field=1/rho, shape="enge", hgap=apt/2, apt=apt, radius=0.05, order=1, nphi=2, plot=True, guess=guess)

xFS = np.linspace(-apt/2, apt/2, 51)
yFS = [0]
sFS = np.arange(-l_magn/2-7.5*apt, l_magn/2+7.5*apt, 0.001)

x, y, s = np.meshgrid(xFS, yFS, sFS)
Bx, By, Bs = dipole_enge.get_Bfield(x, y, s)

np.savetxt("enge_fieldmap.csv", np.column_stack((x.flatten(), y.flatten(), s.flatten(), Bx.flatten(), By.flatten(), Bs.flatten())), delimiter=",", header="x,y,s,Bx,By,Bs", comments='')
