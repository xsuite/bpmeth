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

###################
# Some parameters # 
###################

Fint = 0.45  # Based on tanh model
gap = 0.07

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

for dipole_number in ["1a", "1b", "2a", "2b", "3a", "3b"]:
    line[f"mb{dipole_number}"].edge_entry_fint = Fint
    line[f"mb{dipole_number}"].edge_exit_fint = Fint
    line[f"mb{dipole_number}"].edge_entry_hgap = gap/2
    line[f"mb{dipole_number}"].edge_exit_hgap = gap/2
    
tw = line.twiss4d()

# Save to not rerun it every time
save_twiss(tw, "twissresults/twiss_design_with_fint")
