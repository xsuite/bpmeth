import numpy as np
import sympy as sp
import bpmeth
import matplotlib.pyplot as plt
import xtrack as xt


angle = 10  # degrees
angle_rad = angle/180*np.pi
length = 5

A_fringe = bpmeth.GeneralVectorPotential(b=("0.1*(tanh(5*s)+1)/2",))
A_rotated_fringe = A_fringe.transform(theta_E = angle_rad)

A_fringe.plotfield_xz(Y=0.1, includeby=True, xmin=-0.5, xmax=0.5, xstep=0.01, zmin=-length/4, zmax=length/4, zstep=0.05)
A_rotated_fringe.plotfield_xz(Y=0.1, includeby=True, xmin=-0.5, xmax=0.5, xstep=0.01, zmin=-length/4, zmax=length/4, zstep=0.05)

H_fringe = bpmeth.Hamiltonian(length=length, curv=0, vectp=A_fringe)
H_rotated_fringe = bpmeth.Hamiltonian(length=length, curv=0, vectp=A_rotated_fringe)

p0 = xt.Particles(y=np.linspace(-1e-3, 1e-3, 10), s=-length/2, energy0=10e9, mass0=xt.ELECTRON_MASS_EV)
p0_rotated = p0.copy()

sol = H_fringe.track(p0, return_sol=True)
sol_rotated = H_rotated_fringe.track(p0_rotated, return_sol=True)

plt.figure()
for ss in sol:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.show()

plt.figure()
for ss in sol_rotated:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.show()

plt.figure()
for ss in sol:
    plt.plot(ss.t, ss.y[1])
plt.xlabel('s [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure()
for ss in sol_rotated:
    plt.plot(ss.t, ss.y[1])
plt.xlabel('s [m]')
plt.ylabel('y [m]')
plt.show()
