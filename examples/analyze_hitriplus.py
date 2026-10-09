import numpy as np
import matplotlib.pyplot as plt
import gzip
import bpmeth
import math


rho = 1.65
phi = 30/180*np.pi
l_magn = phi*rho
apt = 0.08  # aperture of the magnet

data = np.loadtxt(gzip.open('../fieldmaps/hitriplus_cct/fieldmap-cct.txt.gz'), usecols=[0,2,1,3,5,4], skiprows=9)
cctmagnet = bpmeth.Fieldmap(data)
# cctmagnet.FS_frame_plot(rho, phi, "By", xmax=apt, ymax=0, nx=51, ny=1, ns=201, smax=0.9, radius=0.005)

xFS = np.linspace(-apt, apt, 101)
yFS = [0]
sFS = np.linspace(-0.9*l_magn, 0.9*l_magn, 101)
cctmagnet_FS = cctmagnet.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.005)

for method in ["polynomial", "finite_difference"]:
    fig, ax = plt.subplots(figsize=(6,4))
    cctmagnet_FS.s_multipoles(3, ax=ax, xmax=apt/2, method=method)
    ax.set_yscale('symlog')
    ax.set_xlabel("s [m]")
    ax.set_ylabel(r"multipole strength $[m^{-n}]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cct_multipoles_{method}.png", dpi=300)
    plt.close()


rmin, rmax, nr = 0.005, apt/2, 51
ntheta = 128
ns=101
rFS = np.linspace(rmin, rmax, nr)
thetaFS = np.arange(ntheta)/ntheta*2*np.pi
sFS = np.linspace(-0.9*l_magn, 0.9*l_magn, ns)

cctmagnet_FS_c = cctmagnet.calc_FS_coords_cylindrical(rFS, thetaFS, sFS, rho, phi, radius=0.005)
# cctmagnet_FS_c.harmonic_analysis_at_s(0, rr=rFS, ntheta=ntheta, ns=ns, order=3)

fig, ax = plt.subplots(figsize=(6,4))
cctmagnet_FS_c.s_harmonics(3, rFS, ntheta, ns, ax=ax)
ax.set_yscale('symlog')
ax.set_xlabel("s [m]")
ax.set_ylabel(r"multipole strength $[m^{-n}]$")
plt.legend()
plt.tight_layout()
plt.savefig(f"cct_multipoles_harmonic_analysis.png", dpi=300)
plt.close()