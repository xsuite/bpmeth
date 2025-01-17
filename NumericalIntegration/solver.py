
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import bpmeth


class Hamiltonian:
    def __init__(self, length, curv, vectp):
        self.length = length
        self.curv = curv
        self.vectp = vectp
        self.angle = curv * length
    

    def get_H(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.curv
        A = self.vectp.get_A(coords)
        sqrt = coords.math.sqrt
        tmp1 = sqrt(1+2*ptau/beta0+ptau**2-px**2-py**2)
        H = ptau/beta0 - (1+h*x)*(tmp1 + A[2])
        return H
    

    def get_vectorfield(self, coords=None, lambdify=True, beta0=1):
        if coords is None:
            coords = SympyParticle(beta0=beta0)
        x, y, tau = coords.x, coords.y, coords.tau
        px, py, ptau = coords.px, coords.py, coords.ptau
        H = self.get_H(coords)
        fx = H.diff(px)
        fy = H.diff(py)
        ftau = H.diff(ptau)
        fpx = - H.diff(x)
        fpy = - H.diff(y)
        fptau = - H.diff(tau)
        qpdot = [fx, fy, ftau, fpx, fpy, fptau]
        if lambdify:
            qp = (x, y, tau, px, py, ptau)
            s = coords.s
            return sp.lambdify((s,qp), qpdot, modules='numpy')
        else:
            return qpdot


    def solve(self, qp0, s_span=None, ivp_opt=None):
        if s_span is None:
            s_span = [0, self.length]
        if ivp_opt is None:
            ivp_opt = {}
            ivp_opt['t_eval'] = np.linspace(s_span[0], s_span[1], 100)
        f = self.get_vectorfield()
        sol = solve_ivp(f, s_span, qp0, **ivp_opt)
        return sol
    

    def plotsol(self, qp0, s_span=None, ivp_opt=None, figname_zx=None, figname_zxy=None):
        sol = self.solve(qp0, s_span, ivp_opt)
        s = sol.t
        x, y, tau, px, py, ptau = sol.y

        fr = bpmeth.Frame()
        fb = bpmeth.BendFrame(fr, self.length, self.angle)

        fb.plot_trajectory_zx(s, x, y, figname=figname_zx)
        fb.plot_trajectory_zxy(s, x, y, figname=figname_zxy)

        plt.show()
    
    def __repr__(self):
        return f'Hamiltonian({self.length}, {self.curv}, {self.vectp})'


class DipoleVectorPotential:
    def __init__(self, curv, b1):
        self.curv = curv
        self.b1 = b1

    def get_A(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        h = self.curv
        As = -(x+h/2*x**2)/(1+h*x) * self.b1
        return [0, 0, As]
    

class FringeVectorPotential:  # In a straight coordinate frame
    def __init__(self, b1, nphi=5):
        self.b1 = b1
        self.nphi = nphi
    
    def get_A(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        fringe = bpmeth.FieldExpansion(b=(self.b1,), nphi=self.nphi)
        Ax, Ay, As = fringe.get_A()
        return [
            Ax.subs({fringe.x: x, fringe.y: y, fringe.s: s}).evalf(),
            Ay.subs({fringe.x: x, fringe.y: y, fringe.s: s}).evalf(),
            As.subs({fringe.x: x, fringe.y: y, fringe.s: s}).evalf()
        ]
    

class GeneralVectorPotential:  # In bent coordinate frame
    def __init__(self, curv, a=("0*s",), b=("0*s",), bs="0", nphi=5):
        self.curv = f"{curv}"
        self.a = a
        self.b = b
        self.bs = bs
        self.nphi = nphi
    
    def get_A(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        field = bpmeth.FieldExpansion(a=self.a, b=self.b, bs=self.bs, hs=self.curv, nphi=self.nphi)
        Ax, Ay, As = field.get_A()
        return [
            Ax.subs({field.x: x, field.y: y, field.s: s}).evalf(),
            Ay.subs({field.x: x, field.y: y, field.s: s}).evalf(),
            As.subs({field.x: x, field.y: y, field.s: s}).evalf()
        ]


class SympyParticle:
    def __init__(self, beta0=1):
        self.x, self.y, self.tau= sp.symbols('x y tau')
        self.s, self.h = sp.symbols('s h')
        self.px, self.py, self.ptau = sp.symbols('px py ptau')
        self.beta0 = beta0
        self.math = sp

