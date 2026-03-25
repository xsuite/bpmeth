import numpy as np
import matplotlib.pyplot as plt
import gzip
import bpmeth


rho = 1.65
phi = 30/180*np.pi
l_magn = phi*rho

data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'), usecols=[0,2,1,3,5,4], skiprows=9)
cctmagnet = bpmeth.Fieldmap(data)
cctmagnet.FS_frame_plot(rho, phi, "By", xmax=0.1, ymax=0, nx=51, ny=1, ns=201, smax=0.9)