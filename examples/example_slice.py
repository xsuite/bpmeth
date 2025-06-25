import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import bpmeth
from bpmeth import poly_fit


# peak gradient
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

# Plotting function
ss = np.linspace(0, 1, 100)
plt.plot(ss, poly_fit.poly_val(poly_entry, ss))

# Full function
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
