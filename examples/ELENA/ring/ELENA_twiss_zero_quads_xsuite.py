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

##################################
# Vary k1, k2, k3 of quadrupoles #
##################################    

k1val, k2val, k3val = 0, 0, 0

line_xsuite = line_mad.copy()
line_xsuite.vars["lnr_kq1"] = k1val
line_xsuite.vars["lnr_kq2"] = k2val
line_xsuite.vars["lnr_kq3"] = k3val

tw_xsuite = line_xsuite.twiss4d()

# Save to not rerun it every time
save_twiss(tw_xsuite, f"twissresults/MD/twiss_fitted_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
  