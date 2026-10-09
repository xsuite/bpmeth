import numpy as np
import matplotlib.pyplot as plt
import gzip
import bpmeth
import math

phi = 60/180*np.pi
rho = 0.927  # https://edms.cern.ch/ui/file/1311860/2.0/LNA-MBHEK-ER-0001-20-00.pdf
dipole_h = 1/rho
l_magn = rho*phi
gap=0.076
theta_E = 17/180*np.pi
fint = 0.424
B_dip_T = 0.42881  # Design dipole field in Tesla

data = np.loadtxt(gzip.open("../../fieldmaps/ELENA/ELENA_fieldmap.csv.gz"), skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]  # Fieldmap in Tesla
Brho = B_dip_T * rho  # T.m
magnet = bpmeth.Fieldmap(data)
magnet.rescale(1/Brho)

radius=0.0025
magnet_FS = magnet.calc_FS_coords(xFS=np.linspace(-0.025, 0.025, 51), yFS=[0], sFS=np.arange(-l_magn, l_magn, 0.001), rho=rho, phi=phi, radius=radius)
magnet_FS = magnet_FS.symmetrize(radius=radius)

for method in ["polynomial", "finite_difference"]:
    fig, ax = plt.subplots(figsize=(6,4))
    magnet_FS.s_multipoles(3, ax=ax, method=method)
    ax.set_yscale('symlog')
    ax.set_xlabel("s [m]")
    ax.set_ylabel(r"multipole strength $[m^{-n}]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ELENA_multipoles_{method}.png", dpi=300)
    plt.close()


data = np.loadtxt("../../fieldmaps/ELENA/ELENA_fieldmap_cylindrical_morepoints.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]  # Fieldmap in Tesla
Brho = B_dip_T * rho  # T.m
magnet = bpmeth.Fieldmap(data)
magnet.rescale(1/Brho)

rmin, rmax, nr = 0.001, 0.03, 21
ntheta = 64
rFS = np.linspace(rmin, rmax, nr)
thetaFS = np.arange(ntheta)/ntheta*2*np.pi

radius = 0.00000001
vapt = 0.076
sFS = np.concatenate([np.arange(-l_magn/2 - 5*vapt, -l_magn/2, 0.001), np.arange(-l_magn/2, l_magn/2, 0.001), np.arange(l_magn/2, l_magn/2 + 5*vapt, 0.001)])
ns = len(sFS)
magnet_FS_c = magnet.calc_FS_coords_cylindrical(rFS, thetaFS, sFS, rho, phi, radius=radius)

fig, ax = plt.subplots()
rr = np.linspace(rmin, rmax, nr)
magnet_FS_c.s_harmonics(3, rr, ntheta=ntheta, ns=ns, ax=ax)
ax.set_yscale('symlog')
ax.set_xlabel("s [m]")
ax.set_ylabel(r"multipole strength $[m^{-n}]$")
plt.legend()
plt.tight_layout()
plt.savefig(f"ELENA_multipoles_harmonic_analysis.png", dpi=300)