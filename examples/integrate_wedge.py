import bpmeth
import sympy as sp
import xtrack as xt


class WedgeHamiltonian(bpmeth.Hamiltonian):
    # Might not work for higher than quadrupole, use with caution!
    def __init__(self, vectp):
        super().__init__(length=0, curv=vectp.hs, vectp=vectp)

    def get_H(self, coords):
        """
        The hamiltonian is actually a function of theta, so s has to be read as theta
        in this function to reuse the existing class.
        """
        
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        
        beta0 = coords.beta0
        h = self.curv
        A = self.vectp.get_A()
        A = [Ai.subs({self.vectp.x: x, self.vectp.y: y, self.vectp.s: s}) for Ai in A]
        assert A[0] == 0
        assert A[1] == 0
        
        sqrt = coords._m.sqrt
        tmp1 = sqrt(
            1 + 2 * ptau / beta0 + ptau**2 - px ** 2 - py ** 2
        )
        rho = sp.symbols('rho')
        tmp2 = ((1 + h*x) / h * A[2]).subs({h:1/rho}).cancel()
        tmp2 = tmp2.subs({rho:0})
        
        H = -x*tmp1 - tmp2
        
        return H
    
    def track(self, particle, theta):
        super().track(particle, s_span=[0, theta], ivp_opt={"rtol": 1e-10, "atol": 1e-12})
    
# To inspect equations symbolically and check if correct
s = sp.symbols('s')
b1 = sp.symbols('b1')
b2 = sp.symbols('b2')
h = sp.symbols('h')

A_dipole = bpmeth.GeneralVectorPotential(hs=h, b=(b1,b2), nphi=2)
H_wedge = WedgeHamiltonian(vectp=A_dipole)

p = bpmeth.SympyParticle()
H_wedge.get_H(p) 


# Track XSuite particles
b1 = "0"
b2 = "10"
angle = -0.1
h = sp.symbols('h')

A_dipole = bpmeth.GeneralVectorPotential(hs=h, b=(b1,b2), nphi=2)
H_wedge = WedgeHamiltonian(vectp=A_dipole)

x0 = 7.193443e-02
y0 = 7.967069e-02
px0 = 2.832321e-01
py0 = 1.998590e-02

p0 = xt.Particles(x=x0, px=px0, y=y0, py=py0)
H_wedge.track(p0, theta=angle)

print(p0.x, p0.y, p0.px, p0.py)