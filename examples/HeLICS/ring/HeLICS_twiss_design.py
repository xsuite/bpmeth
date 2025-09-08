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

# Save to not rerun it every time
save_twiss(tw, "twissresults/twiss_design")
