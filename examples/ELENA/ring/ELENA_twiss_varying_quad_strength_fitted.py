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
# Magnet fitted parameters #
############################

theta_E = 16.4347/180*np.pi
fint = 0.3918

line_fitted = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    line_fitted[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_angle = theta_E
    line_fitted[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_angle = theta_E
    line_fitted[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_fint = fint
    line_fitted[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_fint = fint

##################################
# Vary k1, k2, k3 of quadrupoles #
##################################    

k2val, k3val = 0, 0
for k1val in np.linspace(-0.25, 0.25, 5):
    try:
        print(k1val, k2val, k3val)
        line_fitted.vars["lnr_kq1"] = k1val
        line_fitted.vars["lnr_kq2"] = k2val
        line_fitted.vars["lnr_kq3"] = k3val

        tw_fitted = line_fitted.twiss4d()
        
        # Save to not rerun it every time
        save_twiss(tw_fitted, f"twissresults/MD/twiss_fitted_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
    except ValueError:
        print(f"Computation failed for k1={k1val}, k2={k2val}, k3={k3val}")
        continue
    
k1val, k3val = 0, 0
for k2val in np.linspace(-0.25, 0.25, 5):
    try:
        print(k1val, k2val, k3val)
        line_fitted.vars["lnr_kq1"] = k1val
        line_fitted.vars["lnr_kq2"] = k2val
        line_fitted.vars["lnr_kq3"] = k3val

        tw_fitted = line_fitted.twiss4d()

        # Save to not rerun it every time
        save_twiss(tw_fitted, f"twissresults/MD/twiss_fitted_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
    except ValueError:
        print(f"Computation failed for k1={k1val}, k2={k2val}, k3={k3val}")
        continue
    

k1val, k2val = 0, 0
for k3val in np.linspace(-0.25, 0.25, 5):
    try:
        print(k1val, k2val, k3val)
        line_fitted.vars["lnr_kq1"] = k1val
        line_fitted.vars["lnr_kq2"] = k2val
        line_fitted.vars["lnr_kq3"] = k3val

        tw_fitted = line_fitted.twiss4d()

        # Save to not rerun it every time
        save_twiss(tw_fitted, f"twissresults/MD/twiss_fitted_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
    except ValueError:
        print(f"Computation failed for k1={k1val}, k2={k2val}, k3={k3val}")
        continue
    
