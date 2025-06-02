import bpmeth
import numpy as np

def test_hardedge_quad():
    b1 = "0.0"
    b2 = "1.0"
    b3 = "0.0"
    h = "0.0"
    length = 1
    A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1, b2, b3))
    H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)
    x0=1e-2
    sol= H_magnet.solve([x0,0,0,0,0,0])
    b2=float(b2)
    r11=np.cos(np.sqrt(b2)*length) #approximation valid for expanded hamiltonian
    r11_h=sol.y[0][-1]/x0
    assert abs(1-r11/r11_h)<1e-4






