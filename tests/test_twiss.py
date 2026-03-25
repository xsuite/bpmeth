import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import bpmeth

def test_dipole_twiss():
    b1 = 0.1
    h = 0.1
    length = 1
    
    test_dipole_A_full = bpmeth.DipoleVectorPotential(b1=b1, h=h)
    test_dipole_H_full = bpmeth.Hamiltonian(length, h, test_dipole_A_full)
    
    line = xt.Line([test_dipole_H_full])
    line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    
    t = line.twiss4d(betx=1, bety=1, include_collective=True)
    
    line2=xt.Line([xt.Bend(length=length,k0=b1,angle=h*length)])
    line2.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    t2=line2.twiss4d(betx=1, bety=1)
    
    assert np.allclose(t.betx,t2.betx)
    assert np.allclose(t.alfx,t2.alfx)
    assert np.allclose(t.bety,t2.bety)
    assert np.allclose(t.alfy,t2.alfy)

