import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
from save_read_twiss import save_twiss

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
tw.plot()

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

############################################# 
# bpmeth dipole fieldmap model with splines #
#############################################

# My dipole with splines, matched such that the closed orbit remains zero
data = np.loadtxt("../dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

kf_splines=1.000447  # No closed orbit
# nphi=3 includes up to second order derivatives of b.
dipole_splines_without_sext = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=2, hgap=gap/2, nphi=3, plot=False, symmetric=True, step=50, radius=0.0025)

##########################################################
# Line with replaced dipoles without sextupole component #
##########################################################

line_splines_without_sext = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_mad.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.h1")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.m")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.h2")
    line_splines_without_sext.insert(f"spline_{dipole_number}", dipole_splines_without_sext, at=dipole_s)
tw_splines_without_sext = line_splines_without_sext.twiss4d(include_collective=True)

print("Horizontal fieldmap tune without sextupole: ", tw_splines_without_sext.qx)
print("Vertical fieldmap tune without sextupole: ", tw_splines_without_sext.qy)
print("Horizontal chromaticity fieldmap without sextupole: ", tw_splines_without_sext.dqx)
print("Vertical chromaticity fieldmap without sextupole: ", tw_splines_without_sext.dqy)

# Save to not rerun it every time
save_twiss(tw_splines_without_sext, "twissresults/twiss_splines_without_sext")
