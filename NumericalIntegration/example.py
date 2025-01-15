import solver
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the numerical solution for the Hamiltonian of a curved dipole
"""

p = solver.SympyParticle()

# Dipole parameters
h, b1 = 1, 1
l = 1

dipole = solver.DipoleVectorPotential(h, b1)
A = dipole.get_A(p)
H = solver.Hamiltonian(l, h, dipole)

f = H.get_vectorfield()

qp0 = [0.2,0.3,0,0.1,0.2,0]  # Initial conditions x, y, tau, px, py, ptau

sol = H.solve(qp0)
H.plotsol(qp0)