import matplotlib.pyplot as plt
import xtrack as xt
import scipy as sc
import sympy as sp
import numpy as np
from cpymad.madx import Madx
import sys
import nafflib


##################
# HeLICS lattice #
##################

mad=Madx()
#mad.call("/home/silke/Documents/HeLICS/HeLICS.str")
#mad.call("/home/silke/Documents/HeLICS/HeLICS.seq")
mad.call("/home/silke/Documents/PhD/repos/bpmeth/examples/HeLICS/ring/HeLICS.str")
mad.call("/home/silke/Documents/PhD/repos/bpmeth/examples/HeLICS/ring/HeLICS.seq")
mad.beam()
mad.use("he_ring")

def match_tunes(madx,qx,qy):
    madx.input(f'''
    use, sequence=he_ring;
    MATCH;
    GLOBAL, Q2={qy},Q1={qx};
    vary, name=qd_k1, step=0.0000001;
    vary, name=qf_k1, step=0.0000001;
    simplex, calls=5000, tolerance=1E-10;
    endmatch;''')
    qf = madx.eval('qf_k1')
    qd = madx.eval('qd_k1')
    return qf,qd

def match_disp(madx):
    madx.input(f'''
    use, sequence=he_ring;
    MATCH;
    constraint, sequence=he_ring, range = "qf2b", DX=0.0;
    vary, name=qb_k1, step=0.0000001;
    jacobian, calls=5000, tolerance=1E-10;
    endmatch;''')
    qb = madx.eval('qb_k1') 
    return qb

print("Matched to Qx 2.4, Qy 1.2 with quadrupole strengths qf, qd = ", match_tunes(mad, 2.39, 1.21))



###################
# Track particles #
###################

def calc_physical_coordinates(xn, pxn, beta=1, alpha=0):
    """Convert normalized coordinates to physical coordinates."""
    x = xn * np.sqrt(beta) 
    px = (alpha * xn + pxn) / np.sqrt(beta)
    return x, px


def calc_normalized_coords(x, px, betx=1, alphx=0):
    xn = x/np.sqrt(betx)
    pxn = (x*alphx/np.sqrt(betx) + px*np.sqrt(betx))
    return xn, pxn


def create_particles(line, tw0, nr=80, nphi=40):
    n_part = nr * nphi
    r = np.linspace(0, 0.1, nr)
    phi = np.linspace(0, np.pi/2, nphi)

    R, PHI = np.meshgrid(r, phi)
    R = R.flatten()
    PHI = PHI.flatten()

    Xn = R * np.cos(PHI)
    Yn = R * np.sin(PHI)

    # To physical space 
    X, PX = calc_physical_coordinates(Xn, 0, beta=tw0.betx[0], alpha=tw0.alfx[0])
    Y, PY = calc_physical_coordinates(Yn, 0, beta=tw0.bety[0], alpha=tw0.alfy[0])


    part = line.build_particles(x = X, px = PX, y = Y, py = PY, zeta=0, delta=0)
    print(f"{n_part} particles created")
    return X, Y, part


def track_line(line, part, n_turn = 1000):
    line.track(part, num_turns=n_turn, turn_by_turn_monitor=True)
    mon = line.record_last_track
    return mon

def save_tracks(filename, x, px, y, py):
    np.savez(filename, x=x, px=px, y=y, py=py)

def read_tracks(filename):
    data = np.load(filename)
    x = data['x']
    px = data['px']
    y = data['y']
    py = data['py']
    return x, px, y, py

def calc_h(x, px, beta=1, alpha=0):
    xn, pxn = calc_normalized_coords(x, px, betx=beta, alphx=alpha)
    hxm = xn - 1j*pxn
    return hxm

def calc_tunes(hxm, hym):
    n_part = len(hxm)
    Qx1 = np.zeros(n_part)
    Qx2 = np.zeros(n_part)
    Qy1 = np.zeros(n_part)
    Qy2 = np.zeros(n_part)

    for i in range(n_part):
        Qx1[i] = nafflib.tune(hxm[:1000, i], window_order=2, window_type="hann")
        Qx2[i] = nafflib.tune(hxm[-1000:, i], window_order=2, window_type="hann")
        Qy1[i] = nafflib.tune(hym[:1000, i], window_order=2, window_type="hann")
        Qy2[i] = nafflib.tune(hym[-1000:, i], window_order=2, window_type="hann")
    return Qx1, Qx2, Qy1, Qy2


def tunediff(Q1, Q2):
    """ Tune described over interval [-0.5, 0.5] """
    return np.min(np.array([np.abs(Q2 - Q1), 1 - np.abs(Q2 - Q1)]), axis=0)

def save_data(filename, X, Y, Qx1, Qx2, Qy1, Qy2):
    dQ = np.log10(np.sqrt((tunediff(Qx1, Qx2))**2 + tunediff(Qy1, Qy2)**2))
    np.savetxt(filename, np.array([X, Y, Qx1, Qy1, dQ]).T, header="x [m], y [m], Qx, Qy, log10(sqrt(dQx**2 + dQy**2))")


def calc_FMA(line, fringe, n_turn = 1000, nr=80, nphi=40):
    line.configure_bend_model(edge=fringe, core='adaptive', num_multipole_kicks=30)
    
    tw0 = line.twiss4d()

    X, Y, part = create_particles(line, tw0, nr=nr, nphi=nphi)
    mon = track_line(line, part, n_turn = n_turn)
    save_tracks(f"tracks_{fringe}fringe_{n_turn}turns.npz", mon.x, mon.px, mon.y, mon.py)

    hxm = calc_h(mon.x, mon.px, beta=tw0.betx[0], alpha=tw0.alfx[0])
    hym = calc_h(mon.y, mon.py, beta=tw0.bety[0], alpha=tw0.alfy[0])
    Qx1, Qx2, Qy1, Qy2 = calc_tunes(hxm, hym)
    save_data(f"FMA_{fringe}fringe_{n_turn}turns.txt", X, Y, Qx1, Qx2, Qy1, Qy2)


def plot_spectrum(hxm, ax):
    hfmin = np.fft.fft(hxm)
    hfminabs = np.abs(hfmin)
    hfminabs = np.fft.fftshift(hfminabs)        
    ff = np.fft.fftfreq(len(hxm))
    ff = np.fft.fftshift(ff)
    ax.plot(ff, np.log(hfminabs))