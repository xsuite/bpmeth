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

wiggler = bpmeth.GeneralVectorPotential(hs=f"{curv}", b=(b1,))
wiggler.plotfield()

#A_wiggler = wiggler.get_Aval(p_sp)
H_wiggler = bpmeth.Hamiltonian(length, curv, wiggler)

sol_wiggler = H_wiggler.solve(qp0, ivp_opt={"rtol":1e-4, "atol":1e-7})
H_wiggler.plotsol(qp0, ivp_opt=ivp_opt)