import bpmeth
import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


"""
Compare the analytical and numerical solution for the exact solenoid
"""


# Parameters
length = 5
angle = 0
h = angle/length

BB = 1 # Only q By / P0 plays a role so lets call this quantity BB
x0 = 0
y0 = 0
tau0 = 0
px0 = 0.2
py0 = 0.2
ptau0 = 0
beta0 = 1
qp0 = [x0, y0, tau0, px0, py0, ptau0]


# EXACT SOLUTION
# Plot a trajectory with solenoid
delta0 = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
omega = BB / np.sqrt((1+delta0)**2 - (px0 + BB/2*y0)**2 - (py0 - BB/2*x0)**2) 

s = np.linspace(0,length,2500)
cc = np.cos(omega*s)
ss = np.sin(omega*s)

x = 1/2*(cc+1)*x0 + 1/BB*ss*px0 + 1/2*ss*y0 - 1/BB*(cc-1)*py0
y = -1/2*ss*x0 + 1/BB*(cc-1)*px0 + 1/2*(cc+1)*y0 + 1/BB*ss*py0


# NUMERICAL SOLUTION
p_sp = bpmeth.SympyParticle()

solenoid = bpmeth.SolenoidVectorPotential(BB)
H_solenoid = bpmeth.Hamiltonian(length, h, solenoid)
sol_solenoid = H_solenoid.solve(qp0, ivp_opt={'t_eval':s, 'atol':1e-5, 'rtol':1e-8})

solenoid2 = bpmeth.GeneralVectorPotential(h,bs=f"{BB}")
H_solenoid2 = bpmeth.Hamiltonian(length, h, solenoid2)
sol_solenoid2 = H_solenoid2.solve(qp0, ivp_opt={'t_eval':s, 'atol':1e-5, 'rtol':1e-8})


# CANVAS TO PLOT
canvas_zx = bpmeth.CanvasZX()

# Frame in the origin
fr = bpmeth.Frame()
# Curvilinear coordinates in global frame
fb = bpmeth.BendFrame(fr,length,angle)

fb.plot_trajectory_zx(s,x,y, canvas=canvas_zx, color="blue")
H_solenoid.plotsol(qp0, canvas_zx=canvas_zx)


# PLOT THE DIFFERENCE
fig, ax = plt.subplots()
ax.plot(s, x-sol_solenoid.y[0])
ax.set_xlabel('s')
ax.set_ylabel('absolute error x')
plt.show()


