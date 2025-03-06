import bpmeth
import numpy as np
import matplotlib.pyplot as plt

p_sp = bpmeth.SympyParticle()

qp0 = [0,0,0,0,0,0]  # Initial conditions x, y, tau, px, py, ptau
p_np = bpmeth.NumpyParticle(qp0)

curv=0  # Curvature of the reference frame
length=10
k=2
amp=0.3
b1 = f"{amp}*sin({k}*s)" 

wiggler = bpmeth.GeneralVectorPotential(curv, b=(b1,))

#A_wiggler = wiggler.get_A(p_sp)
H_wiggler = bpmeth.Hamiltonian(length, curv, wiggler)

ivp_opt = {"rtol":1e-4, "atol":1e-7}
sol_wiggler = H_wiggler.solve(qp0, ivp_opt=ivp_opt)
H_wiggler.plotsol(qp0, ivp_opt=ivp_opt)