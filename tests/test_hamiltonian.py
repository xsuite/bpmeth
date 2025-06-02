import bpmeth
import numpy as np
import xtrack as xt

def test_hardedge_quad_solve():
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

def test_hardedge_quad_track():
    b1 = "0.0"
    b2 = "s"
    b3 = "0.0"
    h = "0.0"
    length = 1
    A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1, b2, b3))
    H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)

    p0=xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV,x=1e-2)
    p1=p0.copy()
    p2=p0.copy()
    p2.s=3

    H_magnet.track(p1)
    H_magnet.track(p2)

    assert p1.x==p2.x
    assert np.isclose(p1.s,1)
    assert np.isclose(p2.s,4)




