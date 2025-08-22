import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp

import sys
import os
sys.path.append(os.path.abspath("../../ELENA/ring"))
from save_read_twiss import save_twiss

############################
# Magnet design parameters #
############################

rho = 2.737
phi = 60/180*np.pi
theta_E = 30/180*np.pi
l_magn = rho*phi
gap = 0.07

dipole_k0 = 1/rho
dipole_k1 = -0.11

########################################
# Fieldmap created from tanh function, #
# see ../dipole/ELENA_create_tanh.py   #
########################################

data = np.loadtxt("../dipole/HeLICS_fieldmap_tanh.csv", skiprows=1, delimiter=",")[:, [0,1,2,3,4,5]]

##############################################
# Rescale fieldmap to have zero closed orbit #
##############################################

# Need to cut in s because there is just not enough space!
smin = -l_magn/2 - 0.199
smax = l_magn/2 + 0.199

scalefactor = 0.9999001  # <1 because of quadrupole component
dipole_tanh = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=dipole_k0*scalefactor, order=3, 
                                        hgap=gap/2, nphi=3, plot=True, step=50, radius=0.0025, 
                                        in_FS_coord=True)
plt.savefig("figures/HeLICS_dipole_tanh_splines.png", dpi=500)
plt.close()

p0 = xt.Particles()
dipole_tanh.track(p0, plot=True)
assert p0.s == dipole_tanh.length
plt.savefig("figures/HeLICS_dipole_tanh_track.png", dpi=500)
plt.close()

##################
# HeLICS lattice #
##################

mad=Madx()
mad.call("/home/silke/Documents/HeLICS/HeLICS.str")
mad.call("/home/silke/Documents/HeLICS/HeLICS.seq")
mad.beam()
mad.use("he_ring")

line=xt.Line.from_madx_sequence(mad.sequence.he_ring)
line.configure_bend_model(edge='full', core='adaptive', num_multipole_kicks=30)
line.particle_ref=xt.Particles(mass0=0.931494 * 4.001506179127e9, q0=2., energy0=(0.931494 * 4.001506179127 + 0.02)*1e9)  # Energy in eV
start_elem = "qb1"
line.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice

tw = line.twiss4d()

###################################
# Line with replaced dipoles tanh #
###################################

line_tanh = line.copy()
for dipole_number in ["1a", "1b", "2a", "2b", "3a", "3b"]:
    dipole_s = line_tanh.get_s_position(at_elements=f"mb{dipole_number}") + l_magn/2  # Center
    line_tanh.remove(f"mb{dipole_number}")
    line_tanh.insert(f"tanh_{dipole_number}", dipole_tanh, at=dipole_s)
tw_tanh = line_tanh.twiss4d(include_collective=True, compute_chromatic_properties=True)

print("Horizontal tune fieldmap: ", tw_tanh.qx)
print("Vertical tune fieldmap: ", tw_tanh.qy)
print("Horizontal chromaticity fieldmap: ", tw_tanh.dqx)
print("Vertical chromaticity fieldmap: ", tw_tanh.dqy)

print("Horizontal tune design: ", tw.qx)
print("Vertical tune design: ", tw.qy)
print("Horizontal chromaticity design: ", tw.dqx)
print("Vertical chromaticity design: ", tw.dqy)

# Save to not rerun it every time
save_twiss(tw_tanh, "twissresults/twiss_tanh")
