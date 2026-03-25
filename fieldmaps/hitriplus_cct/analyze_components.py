import numpy as np
import matplotlib.pyplot as plt
import gzip
import bpmeth


rho = 1.65
phi = 30/180*np.pi
l_magn = phi*rho
apt = 0.08  # aperture of the magnet

data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'), usecols=[0,2,1,3,5,4], skiprows=9)
cctmagnet = bpmeth.Fieldmap(data)
# cctmagnet.FS_frame_plot(rho, phi, "By", xmax=apt, ymax=0, nx=51, ny=1, ns=201, smax=0.9, radius=0.005)

xFS = np.linspace(-apt, apt, 101)
yFS = [0]
sFS = np.linspace(-0.9*l_magn, 0.9*l_magn, 501)
cctmagnet_FS = cctmagnet.calc_FS_coords(xFS, yFS, sFS, rho, phi, radius=0.005)

# fig, ax = plt.subplots()
# cctmagnet_FS.s_multipoles(3, ax=ax, elinewidth=0.2)
# ax.set_yscale('symlog')
# plt.legend()

fig, ax = plt.subplots()
coeffs, coeffsstd = cctmagnet_FS.fit_xprofile(0, 0.1, "By", 3, ax=ax, xmax=apt/2, radius=0.005)
x1, y1 = 0, 2
dx, dy = 0.04, 0
ax.arrow(x1, y1, dx, dy, color="gray", head_width=0.04, length_includes_head=True, head_length=0.004)
ax.arrow(x1, y1, -dx, dy, color="gray", head_width=0.04, length_includes_head=True, head_length=0.004)
ax.text(x1, y1, "aperture", ha="center", va="bottom", color="gray")