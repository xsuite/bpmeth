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


kf_enge=1.0008781
guess = [ 1.09058298e+00, 3.19608461e+02, -1.03315497e+02, 2.80836370e+01, 4.23617617e-01]
dipole_enge = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_enge, shape="enge", hgap=apt/2, apt=apt, radius=0.0025, order=2, nphi=2, plot=False, guess=guess)

kf_splines=1.000447
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=2, hgap=apt/2, nphi=2, plot=False, symmetric=True, step=50, radius=0.0025)



p = xt.Particles(x=np.linspace(-2e-3, 2e-3, 11), y=np.linspace(-2e-3, 2e-3, 11), p0c=0.1, mass0=0.938272, q0=1)
p_splines = p.copy()
p_enge = p.copy()

dipole_splines.track(p_splines, plot=False)
dipole_enge.track(p_enge, plot=False)

assert np.isclose(p_splines.s[5], dipole_splines.length)
print(p_splines.s[0], p_splines.x[0], p_splines.px[0], p_splines.y[0], p_splines.py[0])
print(p_enge.s[0], p_enge.x[0], p_enge.px[0], p_enge.y[0], p_enge.py[0])
print(p_splines.s[5], p_splines.x[5], p_splines.px[5], p_splines.y[5], p_splines.py[5])
print(p_enge.s[5], p_enge.x[5], p_enge.px[5], p_enge.y[5], p_enge.py[5])

# For checking
# dipole_splines_from_enge = bpmeth.MagnetFromFieldmap(data_enge, 1/rho, l_magn, design_field=1/rho, order=1, hgap=apt/2, nphi=2, plot=True, step=50, in_FS_coord=True)
# p_splines_enge = p.copy()
# dipole_splines_from_enge.track(p_splines_enge, plot=True)