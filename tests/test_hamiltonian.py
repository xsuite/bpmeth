import bpmeth
import numpy as np
import xtrack as xt
from bpmeth import poly_fit


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

def test_hardedge_bend_track():
    b1 = "0.1"
    h = "0.1"
    length = 1
    A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1,))
    H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)

    p0=xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV,
            x=1e-2, px=1e-3, y=1e-2, py=1e-3)
    p0=xt.Particles(energy0=1e9, mass0=xt.PROTON_MASS_EV,
            x=0, px=0, y=0, py=0, zeta=0.1,delta=0.001)
    p1=p0.copy()
    p2=p0.copy()

    line=xt.Line([xt.Bend(k0=float(b1),h=float(h),length=length)])
    line.reset_s_at_end_turn=False

    H_magnet.track(p1)
    line.track(p2)

    assert np.isclose(p1.x,p2.x)
    assert np.isclose(p1.px,p2.px)
    assert np.isclose(p1.y,p2.y)
    assert np.isclose(p1.py,p2.py)
    assert np.isclose(p1.zeta,p2.zeta)
    assert np.isclose(p1.s,[1])
    assert np.isclose(p2.s,[1])


def test_hardedge_drift_track():
    b1 = "0.0"
    h = "0.0"
    length = 1
    A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1,))
    H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)

    p0=xt.Particles(energy0=1e9, mass0=xt.PROTON_MASS_EV,
            x=0, px=0, y=0, py=0, zeta=0.1,delta=0.01)
    p1=p0.copy()
    p2=p0.copy()

    line=xt.Line([xt.Drift(length=length)])
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.reset_s_at_end_turn=False

    H_magnet.track(p1)
    line.track(p2)

    assert np.isclose(p1.x,p2.x)
    assert np.isclose(p1.px,p2.px)
    assert np.isclose(p1.y,p2.y)
    assert np.isclose(p1.py,p2.py)
    assert np.isclose(p1.zeta,p2.zeta)
    assert np.isclose(p1.s,[1])
    assert np.isclose(p2.s,[1])


def test_slice_quad_track():
    k1 = 0.01  # m^-1

    # Defining a smooth function from 0 to 1 of length 1 m
    y = np.array([0, k1 / 2, k1])
    s = np.array([0, 0.5, 1])
    poly_entry = poly_fit.poly_fit(
        N=4,
        xdata=s,
        ydata=y,
        x0=[0, s[-1]],
        y0=[0, y[-1]],
        xp0=[0, s[-1]],
        yp0=[0, 0],
        xpp0=[0],
        ypp0=[0],
    )

    b1 = "0.0"  # k0
    b2 = poly_fit.poly_print(poly_entry, x="s")  # k1
    b3 = "0.0"  # k2
    hs = "0.0"
    length = s[-1]
    A_magnet_entry = bpmeth.GeneralVectorPotential(b=(b1, b2, b3))
    H_magnet_entry = bpmeth.Hamiltonian(length=length, curv=float(hs), vectp=A_magnet_entry)

    # track and twiss through one element
    line = xt.Line([H_magnet_entry])
    line.reset_s_at_end_turn = False
    line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    p0 = line.build_particles(x=[0.1])
    line.track(p0, turn_by_turn_monitor="ONE_TURN_EBE")
    data = line.record_last_track


    # slicing in 3 parts
    H1 = bpmeth.Hamiltonian(
        length=length / 3, s_start=0, curv=float(hs), vectp=A_magnet_entry
    )
    H2 = bpmeth.Hamiltonian(
        length=length / 3, s_start=1 / 3 * length, curv=float(hs), vectp=A_magnet_entry
    )
    H3 = bpmeth.Hamiltonian(
        length=length / 3, s_start=2 / 3 * length, curv=float(hs), vectp=A_magnet_entry
    )

    line3 = xt.Line([H1, H2, H3])
    line3.reset_s_at_end_turn = False
    line3.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    p1 = line3.build_particles(x=[0.1])
    line3.track(p1, turn_by_turn_monitor="ONE_TURN_EBE")
    data3 = line3.record_last_track

    assert(np.isclose(data3.x[0,-1],data.x[0,-1]))
    assert(np.isclose(data3.x[0,-1],0.09990009))



