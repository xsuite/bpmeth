import numpy as np
import matplotlib.pyplot as plt
import bpmeth
import xtrack as xt

data = np.loadtxt("dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]
data_enge = np.loadtxt("dipole/enge_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,3,4,5]]

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

dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho, order=3, hgap=apt/2, nphi=2, plot=True, step=25)

# For checking
# dipole_splines_from_enge = bpmeth.MagnetFromFieldmap(data_enge, 1/rho, l_magn, design_field=1/rho, order=1, hgap=apt/2, nphi=2, plot=True, step=50, in_FS_coord=True)


p = xt.Particles(y=np.linspace(-2e-3, 2e-3, 11), p0c=0.1, mass0=0.938272, q0=1)
p_enge = p.copy()
p_splines = p.copy()
# p_splines_enge = p.copy()

dipole_enge.track(p_enge, plot=True)
dipole_splines.track(p_splines, plot=True)
# dipole_splines_from_enge.track(p_splines_enge, plot=True)

print(p_enge.s[0], p_enge.x[0], p_enge.y[0], p_enge.px[0], p_enge.py[0])
print(p_splines.s[0], p_splines.x[0], p_splines.y[0], p_splines.px[0], p_splines.py[0])
# print(p_splines_enge.s[0], p_splines_enge.x[0], p_splines_enge.y[0], p_splines_enge.px[0], p_splines_enge.py[0])