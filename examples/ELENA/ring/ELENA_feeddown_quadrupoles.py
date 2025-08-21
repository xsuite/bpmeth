import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd

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
line_mad.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')

####################################################
# XSuite line with feeddown quadrupole in the body #
# => 0.5% in y, 0.1% in x, so neglegible?          #
####################################################

feeddown_k1l = 0.000467
line_with_quads = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    line_with_quads[f"lnr.mbhek.{dipole_number}.h1"].knl[1] = feeddown_k1l/2
    line_with_quads[f"lnr.mbhek.{dipole_number}.h2"].knl[1] = feeddown_k1l/2
    
tw_with_quads = line_with_quads.twiss4d()

fig, ax = plt.subplots()
ax.plot(tw.s, (tw_with_quads.bety - tw.bety)/tw.bety, label="beta beating x")
ax.plot(tw.s, (tw_with_quads.betx - tw.betx)/tw.betx, label="beta beating y")
plt.legend()
plt.savefig("figures/ELENA_feeddown_quadrupoles.png", dpi=500)
plt.close()

