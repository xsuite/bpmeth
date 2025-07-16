import numpy as np
import matplotlib.pyplot as plt
import bpmeth
import xtrack as xt


data = np.loadtxt("dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

rho = 0.927
phi = 60/180*np.pi
l_magn = rho*phi
apt = 0.076
hgap = apt/2
theta_E = 17/180*np.pi
B = 5.3810e-07
Brho = B*rho

xFS = np.linspace(-apt/2, apt/2, 51)
yFS = [0]
sFS = np.arange(-l_magn/2-7.5*apt, l_magn/2+7.5*apt, 0.001)

guess = [ 4.27855927e-01, 7.63365939e+02, -1.44485719e+02, 2.59387643e+01, 5.81379154e-01]
dipole_enge = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn, design_field=1/rho, shape="enge", hgap=apt/2, apt=apt, radius=0.05, order=1, nphi=2, plot=True, guess=guess)

dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho, order=1, hgap=apt/2, nphi=2, plot=True, step=100)

p_enge = xt.Particles(x=0, y=0, p0c=0.1, mass0=0.938272, q0=1)
p_splines = xt.Particles(x=0, y=0, p0c=0.1, mass0=0.938272, q0=1)

dipole_enge.track(p_enge, plot=True)
dipole_splines.track(p_splines, plot=True)

print(p_enge.s, p_enge.x, p_enge.y, p_enge.px, p_enge.py)
print(p_splines.s, p_splines.x, p_splines.y, p_splines.px, p_splines.py)