import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import bpmeth
from bpmeth import poly_fit

dipole_k0 = 1.0787486515641855
dipole_h = 1.0787486515641855
dipole_len = 0.9707521299592461

test_dipole_A_full = bpmeth.DipoleVectorPotential(b1=f"{dipole_k0}", curv=dipole_h)
test_dipole_H_full = bpmeth.Hamiltonian(dipole_len, dipole_h, test_dipole_A_full)

line = xt.Line([test_dipole_H_full])
line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)

t = line.twiss4d(betx=1, bety=1, include_collective=True)
print(t.betx,t.alfx)
print(t.bety,t.alfy)


line2=xt.Line([xt.Bend(length=dipole_len,k0=dipole_k0,h=dipole_h)])
line2.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
t2=line2.twiss4d(betx=1, bety=1)
print(t2.betx,t2.alfx)
print(t2.bety,t2.alfy)

