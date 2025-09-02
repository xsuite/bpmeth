import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
from save_read_twiss import save_twiss

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
# Fieldmap created from tanh function, #
# see ../dipole/ELENA_create_tanh.py   #
########################################

data = np.loadtxt("../dipole/ELENA_fieldmap_tanh.csv", skiprows=1, delimiter=",")[:, [0,1,2,3,4,5]]

##############################################
# Rescale fieldmap to have zero closed orbit #
##############################################

scalefactor = 1.001785
dipole_tanh = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=dipole_h*scalefactor, order=3, 
                                        hgap=gap/2, nphi=3, plot=True, step=50, radius=0.0025, 
                                        in_FS_coord=True)
plt.savefig("figures/ELENA_dipole_tanh_splines.png", dpi=500)
plt.close()

p0 = xt.Particles()
dipole_tanh.track(p0, plot=True)
assert p0.s == dipole_tanh.length
plt.savefig("figures/ELENA_dipole_tanh_track.png", dpi=500)
plt.close()

##########################
# ELENA lattice in MAD-X #
##########################

madx = Madx()
madx.call("acc-models-elena/elena.seq")
madx.call("acc-models-elena/elena.dbx")
madx.call("acc-models-elena/tools/splitEle_installBPM.madx")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.str")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.beam")
madx.use("elena")

#########################
# ELENA model in XSuite #
#########################

line_mad = xt.Line.from_madx_sequence(madx.sequence.elena)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line_mad.particle_ref = xt.Particles(p0c=0.1e6, mass0=0.938272e6, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')
tw = line_mad.twiss4d()

###################################
# Line with replaced dipoles tanh #
###################################

line_tanh = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_tanh.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line_tanh.remove(f"lnr.mbhek.{dipole_number}.h1")
    line_tanh.remove(f"lnr.mbhek.{dipole_number}.m")
    line_tanh.remove(f"lnr.mbhek.{dipole_number}.h2")
    line_tanh.insert(f"spline_{dipole_number}", dipole_tanh, at=dipole_s)
tw_tanh = line_tanh.twiss4d(include_collective=True, compute_chromatic_properties=True)

print("Horizontal tune fieldmap with sextupole: ", tw_tanh.qx)
print("Vertical tune fieldmap with sextupole: ", tw_tanh.qy)
print("Horizontal chromaticity fieldmap with sextupole: ", tw_tanh.dqx)
print("Vertical chromaticity fieldmap with sextupole: ", tw_tanh.dqy)

# Save to not rerun it every time
save_twiss(tw_tanh, "twissresults/twiss_tanh")
