import bpmeth
import xtrack as xt
import sympy
import numpy as np
import matplotlib.pyplot as plt

###################################################################################
# Track a combined function dipole with expanded Hamiltonian, compare with Xsuite #
###################################################################################

b1 = 0.4
b2 = 0.1
b3 = 0.05
h = 0.3
length = 0.5

# Vector potential (bs, b1, a1, b2, a2, b3, a3, ,... as function of s)
A_magnet = bpmeth.GeneralVectorPotential(h=h, b=(b1,b2,b3))

class MyHamiltonian(bpmeth.Hamiltonian):
    def get_H(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.h
        one_plus_delta_allsq = 1+2*ptau/beta0+ptau**2
        sqrt = coords._m.sqrt
        term1 = ptau / beta0
        term2 = 0.5 * (px**2 + py**2) / sqrt(one_plus_delta_allsq)
        term3 = b1 * (x - (h * x**2) / (2 * (1 + h * x)))
        term4 = b2 * (x**2 - y**2) / 2

        H = term1 + term2 + term3 + term4
        return H

# Make Hamiltonian for a defined length
H_magnet = bpmeth.Hamiltonian(length=length, h=h, vectp=A_magnet)

# This guy is able to track an Xsuite particle!
p0 = xt.Particles(x=np.linspace(-1e-3, 1e-3, 10), energy0=10e9, mass0=xt.ELECTRON_MASS_EV)

p_bpmeth = p0.copy()
H_magnet.track(p_bpmeth, ivp_opt= {"rtol":1e-10, "atol":1e-12})

p_xsuite = p0.copy()
bb = xt.Bend(angle=length*h, length=length, k0=b1, k1=b2)
bb.edge_entry_active = False
bb.edge_exit_active = False
bb.num_multipole_kicks = 100
bb.integrator = 'yoshida4'
bb.model =  'expanded'

bb.track(p_xsuite)

# Print the max difference
print(f'Max difference on x: {np.max(np.abs(p_bpmeth.x - p_xsuite.x))}')
print(f'Max difference on px: {np.max(np.abs(p_bpmeth.px - p_xsuite.px))}')
print(f'Max difference on y: {np.max(np.abs(p_bpmeth.y - p_xsuite.y))}')
print(f'Maz difference on py: {np.max(np.abs(p_bpmeth.py - p_xsuite.py))}')
print(f'Max difference on tau: {np.max(np.abs(p_bpmeth.zeta - p_xsuite.zeta))}')
print(f'Max difference on delta: {np.max(np.abs(p_bpmeth.delta - p_xsuite.delta))}')