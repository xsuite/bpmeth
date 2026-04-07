import numpy as np
import xtrack as xt
import bpmeth
import sympy as sp


p0c=1e9
c = 299_792_458.0


def comp_to_sympy(comp):
    s = sp.symbols("s")
    out = []
    for row in comp:
        pol = row[0]
        for n, v in enumerate(row[1:]):
            pol += v * s ** (n + 1)
        out.append(pol)
    bs = out[0]
    b = out[1::2]
    a = out[2::2]
    return bs, b, a


def test_drift_track():
    comp = np.zeros((9, 4), dtype=np.float64)
    bs, b, a = comp_to_sympy(comp)
    h = 0.
    length = 0.1
    
    PolyMagnet_drift = bpmeth.PolySegment(length=length, h=h, comp=comp)
    line=xt.Line([xt.Drift(length=length)])
    line.config.XTRACK_USE_EXACT_DRIFTS = True
    line.reset_s_at_end_turn=False
    
    p0 = xt.Particles(x=0.1, px=0.01, y=0.2, py=0.03, delta=0.1, zeta=0.01)
    p1 = p0.copy()
    p2 = p0.copy()
    
    PolyMagnet_drift.track(p1)
    line.track(p2)
    
    assert np.isclose(p1.x, p2.x)
    assert np.isclose(p1.px, p2.px)
    assert np.isclose(p1.y, p2.y)
    assert np.isclose(p1.py, p2.py)
    assert np.isclose(p1.delta, p2.delta)
    assert np.isclose(p1.zeta, p2.zeta)
    
    
def test_bend_track():
    comp = np.zeros((9, 4), dtype=np.float64)
    comp[1, 0] = 0.1
    bs, b, a = comp_to_sympy(comp)
    h = 0.1
    length = 0.1
    
    PolyMagnet_bend = bpmeth.PolySegment(length=length, h=h, comp=comp)
    line=xt.Line([xt.Bend(k0=b[0], angle=h*length, length=length)])
    line.reset_s_at_end_turn=False
    
    p0 = xt.Particles(x=0.1, px=0.01, y=0.2, py=0.03, delta=0.1, zeta=0.01)
    p1 = p0.copy()
    p2 = p0.copy()
    
    PolyMagnet_bend.track(p1)
    line.track(p2)
    
    assert np.isclose(p1.x, p2.x)
    assert np.isclose(p1.px, p2.px)
    assert np.isclose(p1.y, p2.y)
    assert np.isclose(p1.py, p2.py)
    assert np.isclose(p1.delta, p2.delta)
    assert np.isclose(p1.zeta, p2.zeta)
    
    
def test_quad_track():
    comp = np.zeros((9, 4), dtype=np.float64)
    comp[3, 0] = 0.1
    bs, b, a = comp_to_sympy(comp)
    h = 0.
    length = 0.1
    
    PolyMagnet_quad = bpmeth.PolySegment(length=length, h=h, comp=comp)
    line=xt.Line([xt.Quadrupole(k1=b[1], length=length)])
    line.reset_s_at_end_turn=False
    
    p0 = xt.Particles(x=0.1, px=0.01, y=0.2, py=0.03, delta=0.1, zeta=0.01)
    p1 = p0.copy()
    p2 = p0.copy()
    
    PolyMagnet_quad.track(p1)
    line.track(p2)
    
    assert np.isclose(p1.x, p2.x)
    assert np.isclose(p1.px, p2.px)
    assert np.isclose(p1.y, p2.y)
    assert np.isclose(p1.py, p2.py)
    assert np.isclose(p1.delta, p2.delta)
    assert np.isclose(p1.zeta, p2.zeta)
    
    
def test_combined_function_track():
    comp = np.zeros((9, 4), dtype=np.float64)
    comp[1, 0] = 0.1
    comp[3, 0] = 0.05
    bs, b, a = comp_to_sympy(comp)
    h = 0.15
    length = 0.1
    
    PolyMagnet_combined = bpmeth.PolySegment(length=length, h=h, comp=comp)
    line=xt.Line([xt.Bend(k0=b[0], k1=b[1], angle=h*length, length=length)])
    line.reset_s_at_end_turn=False
    
    p0 = xt.Particles(x=0.1, px=0.01, y=0.2, py=0.03, delta=0.1, zeta=0.01)
    p1 = p0.copy()
    p2 = p0.copy()
    
    PolyMagnet_combined.track(p1)
    line.track(p2)
    
    assert np.isclose(p1.x, p2.x)
    assert np.isclose(p1.px, p2.px)
    assert np.isclose(p1.y, p2.y)
    assert np.isclose(p1.py, p2.py)
    assert np.isclose(p1.delta, p2.delta)
    assert np.isclose(p1.zeta, p2.zeta)
    

def test_sdep_track():
    comp = np.zeros((9, 4), dtype=np.float64)
    comp[1, 0] = 0.1
    comp[1, 1] = 0.05
    bs, b, a = comp_to_sympy(comp)
    h = 0.1
    length = 0.1
    
    PolyMagnet_sdep = bpmeth.PolySegment(length=length, h=h, comp=comp)
    A_magnet = bpmeth.GeneralVectorPotential(h=h, b=b)
    H_magnet = bpmeth.Hamiltonian(length=length, h=h, vectp=A_magnet)
    
    p0 = xt.Particles(x=0.1, px=0.01, y=0.2, py=0.03, delta=0.1, zeta=0.01)
    p1 = p0.copy()
    p2 = p0.copy()
    
    PolyMagnet_sdep.track(p1)
    H_magnet.track(p2)
    
    assert np.isclose(p1.x, p2.x)
    assert np.isclose(p1.px, p2.px)
    assert np.isclose(p1.y, p2.y)
    assert np.isclose(p1.py, p2.py)
    assert np.isclose(p1.delta, p2.delta)
    assert np.isclose(p1.zeta, p2.zeta)
    
    
    
    
    