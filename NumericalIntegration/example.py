import solver
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the numerical solution for the Hamiltonian of a curved dipole
"""

p = solver.SympyParticle()

qp0 = [0.2,0,0,0,0,0]  # Initial conditions x, y, tau, px, py, ptau

# Dipole parameters
h, b1 = 1, 1
l = 1

dipole = solver.DipoleVectorPotential(h, b1)

A_dipole = dipole.get_A(p)
H_dipole = solver.Hamiltonian(l, h, dipole)

f_dipole = H_dipole.get_vectorfield()

sol_dipole = H_dipole.solve(qp0)
H_dipole.plotsol(qp0)

# Fringe in straight frame
h = 0
b1fringe = b1*"(tanh(s)+1)/2"
fringe = solver.FringeVectorPotential(b1fringe)
#fringe = solver.FringeVectorPotential(f"{b1}")

A_fringe = fringe.get_A(p)
H_fringe = solver.Hamiltonian(l, h, fringe)

f_fringe = H_fringe.get_vectorfield()

sol_dipole = H_fringe.solve(qp0)
H_fringe.plotsol(qp0)


# General field
h = 1
field = solver.GeneralVectorPotential(h, b=(f"{b1}",))

A_field = field.get_A(p)
H_field = solver.Hamiltonian(l, h, field)

f_field = H_field.get_vectorfield()

sol_field = H_field.solve(qp0)
H_field.plotsol(qp0)

