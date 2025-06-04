import bpmeth
import numpy as np
import xtrack as xt

b1="1"
length=0.5
x_init = np.linspace(-0.5, 0.5, 11)

dipole = bpmeth.GeneralVectorPotential(b=(b1,))
H_dipole = bpmeth.Hamiltonian(length, 0, dipole)

p = xt.Particles(x=x_init)
H_dipole.track(p, ivp_opt={"atol":1e-8})
assert np.isclose(p.s, length).all()

H_dipole.track(p, ivp_opt={"atol":1e-8}, backtrack=True)
assert np.isclose(p.s, 0).all()
assert np.isclose(p.x, x_init, atol=1e-6).all()


