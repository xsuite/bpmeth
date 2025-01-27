import bpmeth
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the numerical solution for the Hamiltonian
"""

p_sp = bpmeth.SympyParticle()

qp0 = [0.2,0,0,0,0,0]  # Initial conditions x, y, tau, px, py, ptau
p_np = bpmeth.NumpyParticle(qp0)


# Dipole parameters
h, b1 = 1, 1
l = 1

dipole = bpmeth.DipoleVectorPotential(h, b1)

A_dipole = dipole.get_A(p_sp)
H_dipole = bpmeth.Hamiltonian(l, h, dipole)

f_dipole = H_dipole.get_vectorfield()

sol_dipole = H_dipole.solve(qp0)
#H_dipole.plotsol(qp0)


# Fringe in straight frame
h = 0
b1fringe = f"{b1}*(tanh(s)+1)/2"
# Fringe field integral K = 1/(2g)
fringe = bpmeth.FringeVectorPotential(b1fringe)
#fringe = solver.FringeVectorPotential(f"{b1}")

A_fringe = fringe.get_A(p_sp)
H_fringe = bpmeth.Hamiltonian(l, h, fringe)

f_fringe = H_fringe.get_vectorfield()

sol_fringe = H_fringe.solve(qp0)
H_fringe.plotsol(qp0)


# General field
h = 1
field = bpmeth.GeneralVectorPotential(h, b=(f"{b1}",))

A_field = field.get_A(p_sp)
H_field = bpmeth.Hamiltonian(l, h, field)

f_field = H_field.get_vectorfield()

sol_field = H_field.solve(qp0)
#H_field.plotsol(qp0)


# Solenoid
p_sp = bpmeth.SympyParticle(beta0=1)

h = 0
l = 5
bs = 1
solenoid = bpmeth.SolenoidVectorPotential(bs)
#solenoid = bpmeth.GeneralVectorPotential(h,bs="1")

A_solenoid = solenoid.get_A(p_sp)
H_solenoid = bpmeth.Hamiltonian(l, h, solenoid)

f_solenoid = H_solenoid.get_vectorfield()

qp0 = [0,0,0,0.2,0,0]  # Initial conditions x, y, tau, px, py, ptau
sol_solenoid = H_solenoid.solve(qp0)
H_solenoid.plotsol(qp0)



# Compare with the fringe field map
h = 0
len = 5
b1 = 0.01
b1fringe = f"{b1}*(tanh(s)+1)/2" # Fringe field integral K = 1/(2g)
    
# First a backwards drift
drift = bpmeth.DriftVectorPotential()
H_drift = bpmeth.Hamiltonian(len/2, h, drift)

# Then a fringe field map
fringe = bpmeth.FringeVectorPotential(b1fringe)
H_fringe = bpmeth.Hamiltonian(len, h, fringe)

# Then a backwards bend
dipole = bpmeth.DipoleVectorPotential(h, b1)
H_dipole = bpmeth.Hamiltonian(len/2, h, dipole)

xfs = []
yfs = []
pxfs = []
pyfs = []
y0s = np.linspace(0, 2, 10)
for i, y0 in enumerate(y0s):
    print(f"calculating {i}")
    sol_drift = H_drift.solve([0,y0,0,0.5,0,0], s_span=[0, -len/2])
    sol_fringe = H_fringe.solve(sol_drift.y[:, -1], s_span=[-len/2, len/2])
    sol_dipole = H_dipole.solve(sol_fringe.y[:, -1], s_span=[len/2, 0])

    all_s = np.append(sol_drift.t, [sol_fringe.t, sol_dipole.t])
    all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_dipole.y[0]])
    all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_dipole.y[1]])
    all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_dipole.y[3]])
    all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_dipole.y[4]])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(all_s, all_x, all_y)

    # fig1, ax1 = plt.subplots(2)
    # ax1[0].plot(all_s, all_x)
    # ax1[1].plot(all_s, all_y)
    # plt.show()

    xfs.append(all_x[-1])
    yfs.append(all_y[-1])
    pxfs.append(all_px[-1])
    pyfs.append(all_py[-1])

fig, ax = plt.subplots(4)
ax[0].plot(y0s, xfs)
ax[0].set_xlabel('y0')
ax[0].set_ylabel('xf')
ax[1].plot(y0s, yfs)
ax[1].set_xlabel('y0')
ax[1].set_ylabel('yf')
ax[2].plot(y0s, pxfs)
ax[2].set_xlabel('y0')
ax[2].set_ylabel('pxf')
ax[3].plot(y0s, pyfs)
ax[3].set_xlabel('y0')
ax[3].set_ylabel('pyf')
plt.show()


