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

# Plotting entry
ss = np.linspace(0, 1, 100)
plt.plot(ss, poly_fit.poly_val(poly_entry, ss))

# Entry
b1 = "0.0"  # k0
b2 = poly_fit.poly_print(poly_entry, x="s")  # k1
b3 = "0.0"  # k2
h = "0.0"
length = s[-1]
A_magnet_entry = bpmeth.GeneralVectorPotential(hs=h, b=(b1, b2, b3))
H_magnet_entry = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet_entry)

# Body
b1 = "0.0"  # k0
b2 = str(k1)  # k1
b3 = "0.0"  # k2
h = "0.0"
length = 5
A_magnet_body = bpmeth.GeneralVectorPotential(hs=h, b=(b1, b2, b3))
H_magnet_body = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet_body)

# Exit

# defining a smooth function from 0 to 1 of length 1 m
y = np.array([k1, k1 / 2, 0])
s = np.array([0, 0.5, 1])
poly = poly_fit.poly_fit(
    N=4,
    xdata=s,
    ydata=y,
    x0=[0, s[-1]],
    y0=[y[0], 0],
    xp0=[0, s[-1]],
    yp0=[0, 0],
    xpp0=[s[-1]],
    ypp0=[0],
)

# plotting exit
ss = np.linspace(0, 1, 100)
plt.plot(ss, poly_fit.poly_val(poly, ss))

b1 = "0.0"  # k0
b2 = poly_fit.poly_print(poly, x="(-s)")  # k1
b3 = "0.0"  # k2
h = "0.0"
length = s[-1]
A_magnet_exit = bpmeth.GeneralVectorPotential(hs=h, b=(b1, b2, b3))
H_magnet_exit = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet_exit)


# track and twiss through one element
line = xt.Line([H_magnet_entry])
line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)

p0 = line.build_particles(x=[0])
line.track(p0, turn_by_turn_monitor="ONE_TURN_EBE")
data = line.record_last_track

t = line.twiss4d(betx=1, bety=1, include_collective=True)
print(t.betx)

# Track through the full element
line = xt.Line([H_magnet_entry, H_magnet_body, H_magnet_exit])
line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)

p0 = line.build_particles(x=[0])
line.track(p0, turn_by_turn_monitor="ONE_TURN_EBE")
data = line.record_last_track
print(f"s={data.s}")

t = line.twiss4d(betx=30, bety=30, include_collective=True)

print(t.s)
print(t.betx)
print(t.bety)

# p = t._initial_particles

# line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
# mon = line.record_last_track
