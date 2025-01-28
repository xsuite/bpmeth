import bpmeth
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the numerical solution for the Hamiltonian
"""

p_sp = bpmeth.SympyParticle()


# Compare with the fringe field map
h = 0
len = 5
b1 = 0.1
aa = 1
b1fringe = f"{b1}*(tanh(s/{aa})+1)/2" # Fringe field integral K = aa/(2g)
KKg = aa/2

# First a backwards drift
drift = bpmeth.DriftVectorPotential()
H_drift = bpmeth.Hamiltonian(len/2, h, drift)

# Then a fringe field map
fringe = bpmeth.FringeVectorPotential(b1fringe, nphi=0) # Only b1 and b1'
H_fringe = bpmeth.Hamiltonian(len, h, fringe)

# Then a backwards bend
dipole = bpmeth.DipoleVectorPotential(h, b1)
H_dipole = bpmeth.Hamiltonian(len/2, h, dipole)

steps = 10
xfs, yfs, pxfs, pyfs = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)
x0, px0, py0, tau0, ptau0 = 0, 0, 0, 0, 0
y0s = np.linspace(-1, 1, steps)
for i, y0 in enumerate(y0s):
    print(f"calculating {i}")
    sol_drift = H_drift.solve([x0, y0, tau0, px0, py0, ptau0], s_span=[0, -len/2])
    sol_fringe = H_fringe.solve(sol_drift.y[:, -1], s_span=[-len/2, len/2])
    sol_dipole = H_dipole.solve(sol_fringe.y[:, -1], s_span=[len/2, 0])

    all_s = np.append(sol_drift.t, [sol_fringe.t, sol_dipole.t])
    all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_dipole.y[0]])
    all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_dipole.y[1]])
    all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_dipole.y[3]])
    all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_dipole.y[4]])

    xfs[i] = all_x[-1]
    yfs[i] = all_y[-1]
    pxfs[i] = all_px[-1]
    pyfs[i] = all_py[-1]


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(all_s, all_x, all_y, color='black')

ax.set_xlabel('s')
ax.set_ylabel('x')
ax.set_zlabel('y')

fig1, ax1 = plt.subplots(2)
ax1[0].plot(all_s, all_x, color='black')
ax1[1].plot(all_s, all_y, color='black')

ax1[0].set_xlabel('s')
ax1[0].set_ylabel('x')
ax1[1].set_xlabel('s')
ax1[1].set_ylabel('y')
plt.show()


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
ax[3].plot(y0s, py0 + KKg * b1**2 * yfs)
ax[3].set_xlabel('y0')
ax[3].set_ylabel('pyf')

plt.show()

diffs = pyfs - py0 - KKg * b1**2 * yfs
params = np.polyfit(y0s, diffs, 3)
plt.plot(y0s, np.polyval(params, y0s), label='fit')
plt.plot(y0s, diffs)
plt.xlabel('y0')
plt.ylabel('Difference')
plt.legend()
plt.show()
