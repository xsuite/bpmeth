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
    
    px = px0*np.cos(s*h) + (np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*np.sin(s*h)
    pxder = -px0*h*np.sin(s*h) + (np.sqrt((1+delta)**2-px0**2-py0**2) - (1/h + x0)*b1 )*h*np.cos(s*h)
    x = 1/h * 1/b1 * (h*np.sqrt((1+delta)**2-px**2-py0**2) - pxder - b1)
    y = h*py0/b1*s - py0/b1 * (np.arcsin(px/np.sqrt((1+delta)**2)) - np.arcsin(px0/np.sqrt((1+delta)**2)) ) + y0
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


def test_exact_solenoid():
    # Parameters
    length = 5
    angle = 0
    h = angle/length

    bs = 1 
    x0 = 0
    y0 = 0
    tau0 = 0
    px0 = 0.2
    py0 = 0.2
    ptau0 = 0
    beta0 = 1
    qp0 = [x0, y0, tau0, px0, py0, ptau0]

    # EXACT SOLUTION
    delta0 = np.sqrt(1 + 2*ptau0/beta0 + ptau0**2) - 1
    omega = bs / np.sqrt((1+delta0)**2 - (px0 + bs/2*y0)**2 - (py0 - bs/2*x0)**2) 
    
    s = np.linspace(0,length,2500)
    cc = np.cos(omega*s)
    ss = np.sin(omega*s)
    
    x = 1/2*(cc+1)*x0 + 1/bs*ss*px0 + 1/2*ss*y0 - 1/bs*(cc-1)*py0
    y = -1/2*ss*x0 + 1/bs*(cc-1)*px0 + 1/2*(cc+1)*y0 + 1/bs*ss*py0

    # NUMERICAL SOLUTION
    p_sp = bpmeth.SympyParticle()

    solenoid = bpmeth.SolenoidVectorPotential(bs)
    H_solenoid = bpmeth.Hamiltonian(length, h, solenoid)
    sol_solenoid = H_solenoid.solve(qp0, ivp_opt={'t_eval':s, 'atol':1e-5, 'rtol':1e-8})

    solenoid2 = bpmeth.GeneralVectorPotential(h=h, bs=bs)
    H_solenoid2 = bpmeth.Hamiltonian(length, h, solenoid2)
    sol_solenoid2 = H_solenoid2.solve(qp0, ivp_opt={'t_eval':s, 'atol':1e-5, 'rtol':1e-8})

# leave absolute tolerance at 1e-8, but since curve goes through zero relative tolerance is not good measure
    assert np.allclose(x, sol_solenoid.y[0], rtol=1)
    assert np.allclose(y, sol_solenoid.y[1], rtol=1)
    assert np.allclose(sol_solenoid.y[0], sol_solenoid2.y[0], rtol=1)
    assert np.allclose(sol_solenoid.y[1], sol_solenoid2.y[1], rtol=1)
    assert np.allclose(sol_solenoid.y[2], sol_solenoid2.y[2], rtol=1)
    assert np.allclose(sol_solenoid.y[3] - solenoid.get_A(lambdify=True)[0]
                       (sol_solenoid.y[0], sol_solenoid.y[1], sol_solenoid.t), 
                       sol_solenoid2.y[3] - solenoid2.get_A(lambdify=True)[0]
                       (sol_solenoid2.y[0], sol_solenoid2.y[1], sol_solenoid2.t), rtol=1)
    assert np.allclose(sol_solenoid.y[4] - solenoid.get_A(lambdify=True)[1]
                       (sol_solenoid.y[0], sol_solenoid.y[1], sol_solenoid.t), 
                       sol_solenoid2.y[4] - solenoid2.get_A(lambdify=True)[1]
                       (sol_solenoid2.y[0], sol_solenoid2.y[1], sol_solenoid2.t), rtol=1)
    assert np.allclose(sol_solenoid.y[5], sol_solenoid2.y[5], rtol=1)
    

    