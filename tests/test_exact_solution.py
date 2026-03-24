import bpmeth
import numpy as np
import matplotlib.pyplot as plt

"""
Compare the analytical and numerical solution for the exact dipole
"""

def test_exact_dipole():
    # Parameters
    length = 1
    angle = 0.1
    h = angle/length

    b1 = 0.1
    x0 = 0.01
    y0 = 0
    tau0 = 0
    px0 = 0.03
    py0 = 0.02
    ptau0 = 0
    beta0 = 1
    qp0 = [x0, y0, tau0, px0, py0, ptau0]

    # EXACT SOLUTION
    delta = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
    theta0 = np.arcsin(px0 / np.sqrt((1+delta)**2 - py0**2))

    s = np.linspace(0,1,100)
    px = px0*np.cos(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*np.sin(s*h)
    pxder = -px0*h*np.sin(s*h) + ( np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*h*np.cos(s*h)
    x = 1/h * 1/b1 * ( h*np.sqrt((1+delta)**2-px**2-py0**2) - pxder - b1)
    y = h*py0/b1*s - py0/b1 * ( np.arcsin(px/np.sqrt((1+delta)**2)) - np.arcsin(px0/np.sqrt((1+delta)**2)) ) + y0
    py = py0 + 0*s
    ptau = ptau0 + 0*s
    tau = s/beta0 - 1/b1 * (1/beta0 + ptau) * (h*s - np.arcsin(px/np.sqrt((1+delta)**2-py**2)) + np.arcsin(px0/np.sqrt((1+delta)**2-py**2)) ) + tau0

    # NUMERICAL SOLUTION
    dipole = bpmeth.DipoleVectorPotential(h, b1)
    H_dipole = bpmeth.Hamiltonian(length, h, dipole)
    sol_dipole = H_dipole.solve(qp0, ivp_opt={'t_eval':s})
    
    assert np.allclose(x, sol_dipole.y[0])
    assert np.allclose(y, sol_dipole.y[1])
    assert np.allclose(tau, sol_dipole.y[2])
    assert np.allclose(px, sol_dipole.y[3])
    assert np.allclose(py, sol_dipole.y[4])
    assert np.allclose(ptau, sol_dipole.y[5])
    assert np.allclose(s, sol_dipole.t)
