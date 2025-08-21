import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
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

###########################
# Line with design dipole #
###########################

madx = Madx()
madx.call("acc-models-elena/elena.seq")
madx.call("acc-models-elena/elena.dbx")
madx.call("acc-models-elena/tools/splitEle_installBPM.madx")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.str")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.beam")
madx.use("elena")

line_mad = xt.Line.from_madx_sequence(madx.sequence.elena)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line_mad.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')

line_design = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    line_design[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_angle = theta_E
    line_design[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_angle = theta_E
    line_design[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_fint = fint
    line_design[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_fint = fint
tw_design = line_design.twiss4d()

save_twiss(tw_design, "twissresults/twiss_design")