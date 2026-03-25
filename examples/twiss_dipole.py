import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import bpmeth

#################################################
# Example on Xsuite twiss using bpmeth elements #
#################################################

b1 = 0.1
h = 0.1
length = 1

test_dipole_A_full = bpmeth.DipoleVectorPotential(b1=b1, h=h)
test_dipole_H_full = bpmeth.Hamiltonian(length, h, test_dipole_A_full)

line = xt.Line([test_dipole_H_full])
line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)

t = line.twiss4d(betx=1, bety=1, include_collective=True)
