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

line_mad = xt.Line.from_madx_sequence(madx.sequence.elena, deferred_expressions=True)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line_mad.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')
tw = line_mad.twiss4d()
# tw.plot()

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
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=3, hgap=gap/2, nphi=3, plot=True, symmetric=True, step=50, radius=0.0025)
plt.savefig("figures/ELENA_dipole_splines.png", dpi=500)
plt.close()

# Plot trajectory of particle on the closed orbit
p = xt.Particles()
dipole_splines.track(p, plot=True)
plt.savefig("figures/ELENA_orbit.png", dpi=500)
plt.close()

#######################################################
# Line with replaced dipoles with sextupole component #
#######################################################

line_splines = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_mad.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.h1")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.m")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.h2")
    line_splines.insert(f"spline_{dipole_number}", dipole_splines, at=dipole_s)
    
##################################
# Vary k1, k2, k3 of quadrupoles #
##################################    
    
k1val, k2val = 0, 0
for k3val in np.linspace(-0.25, 0.25, 5):
    try:
        print(k1val, k2val, k3val)
        line_splines.vars["lnr_kq1"] = k1val
        line_splines.vars["lnr_kq2"] = k2val
        line_splines.vars["lnr_kq3"] = k3val

        tw_splines = line_splines.twiss4d(include_collective=True, compute_chromatic_properties=True)

        # Save to not rerun it every time
        save_twiss(tw_splines, f"twissresults/MD/twiss_num_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
    except ValueError:
        print(f"Computation failed for k1={k1val}, k2={k2val}, k3={k3val}")
        continue