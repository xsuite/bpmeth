import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from .generate_expansion import FieldExpansion
from .frames import Frame, BendFrame


class Hamiltonian:
    def __init__(self, length, curv, vectp):
        self.length = length
        self.curv = curv
        self.vectp = vectp
        self.angle = curv * length

        self.vectorfield = self.get_vectorfield()
    

    def get_H(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.curv
        A = self.vectp.get_Aval(coords)
        sqrt = coords._m.sqrt
        tmp1 = sqrt(1+2*ptau/beta0+ptau**2-(px-A[0])**2-(py-A[1])**2)
        H = ptau/beta0 - (1+h*x)*(tmp1 + A[2])
        return H
    

    def get_vectorfield(self, coords=None, lambdify=True, beta0=1):
        if coords is None:
            coords = SympyParticle(beta0=beta0)
        x, y, tau = coords.x, coords.y, coords.beta0 * coords.zeta
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
            ivp_opt = {'t_eval': np.linspace(s_span[0], s_span[1], 500), "rtol":1e-4, "atol":1e-7}
        elif 't_eval' not in ivp_opt:
            ivp_opt['t_eval'] = np.linspace(s_span[0], s_span[1], 500)
            
        f = self.vectorfield
        sol = solve_ivp(f, s_span, qp0, **ivp_opt)
        return sol
    

    def track(self, particle):
        if isinstance(particle.x, np.ndarray) or isinstance(particle.x, list):
            results = []
            for i in range(len(particle.x)):
                qp0 = [particle.x[i], particle.y[i], particle.beta0[i] * particle.zeta[i], particle.px[i], particle.py[i], particle.ptau[i]]
                sol = self.solve(qp0)
                s = sol.t
                x, y, tau, px, py, ptau = sol.y
                results.append({
                    'x': x[-1],
                    'y': y[-1],
                    'zeta': tau[-1] / particle.beta0[i],
                    'px': px[-1],
                    'py': py[-1],
                    'ptau': ptau[-1],
                })
            particle.x = [res['x'] for res in results]
            particle.y = [res['y'] for res in results]
            particle.zeta = [res['zeta'] for res in results]
            particle.px = [res['px'] for res in results]
            particle.py = [res['py'] for res in results]
            particle.ptau = [res['ptau'] for res in results]
            particle.s += self.length
        else:
            qp0 = [particle.x, particle.y, particle.beta0 * particle.zeta, particle.px, particle.py, particle.ptau]
            sol = self.solve(qp0)
            s = sol.t
            x, y, tau, px, py, ptau = sol.y
            particle.x = x[-1]
            particle.y = y[-1]
            particle.zeta = tau[-1] / particle.beta0
            particle.px = px[-1]
            particle.py = py[-1]
            particle.ptau = ptau[-1]
            particle.s += self.length


    def plotsol(self, qp0, s_span=None, ivp_opt=None, figname_zx=None, figname_zxy=None, canvas_zx=None, canvas_zxy=None):
        sol = self.solve(qp0, s_span, ivp_opt)
        s = sol.t
        x, y, tau, px, py, ptau = sol.y


        fr = Frame()
        fb = BendFrame(fr, self.length, self.angle)

        fb.plot_trajectory_zx(s, x, y, figname=figname_zx, canvas=canvas_zx)
        #fb.plot_trajectory_zxy(s, x, y, figname=figname_zxy, canvas=canvas_zxy)


    def __repr__(self):
        return f'Hamiltonian({self.length}, {self.curv}, {self.vectp})'


class DriftVectorPotential(FieldExpansion):  
    def __init__(self):
        super().__init__(nphi=0)
        
    def get_Aval(self, coords):
        return [0, 0, 0]


class DipoleVectorPotential(FieldExpansion):
    def __init__(self, curv, b1):
        """
        Dipoles without s-dependence, in this case the vector potential is known analytically.
        
        :param curv (float): Curvature of the reference frame.
        :param b1 (float): Dipole field strength.
        """
        
        self.curv = curv
        self.b1 = b1
        super().__init__(b=(f"{b1}",), hs=f"{curv}", nphi=0)

    def get_Aval(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        h = self.curv
        As = -(x+h/2*x**2)/(1+h*x) * self.b1
        return [0, 0, As]
    

class SolenoidVectorPotential(FieldExpansion):
    def __init__(self, bs):
        """ 
        Solenoids without s-dependence, in this case the vector potential is known analytically.
        Curvature of the reference frame is not implemented, use GeneralVectorPotential instead.
        
        :param bs (float): Solenoid field strength.
        """
        
        self.bs = bs
        super().__init__(bs=f"{bs}", nphi=0)

    def get_Aval(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        bs = self.bs
        return [-bs*y/2, bs*x/2, 0]


class FringeVectorPotential(FieldExpansion):
    def __init__(self, b1, nphi=5):
        """
        Fringe fields in a straight coordinate frame.
        
        :param b1 (str): Fringe field b1 as a function of s.
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """
        
        super().__init__(b=(b1,), nphi=nphi)
    
    def get_Aval(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        Ax, Ay, As = self.get_A()
        return [
            Ax.subs({self.x: x, self.y: y, self.s: s}).evalf(),
            Ay.subs({self.x: x, self.y: y, self.s: s}).evalf(),
            As.subs({self.x: x, self.y: y, self.s: s}).evalf()
        ]
        
    
class GeneralVectorPotential(FieldExpansion):
    def __init__(self, a=("0*s",), b=("0*s",), bs="0", hs="0", nphi=5):
        """ 
        General field expansion.
        
        :param a (tuple of str): a coefficients as a function of s.
        :param b (tuple of str): b coefficients as a function of s.
        :param bs (str): bs coefficient as a function of s.
        :param hs (str): Curvature of the reference frame as a function of s. (to be tested for not constant hs)
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """
        
        super().__init__(a=a, b=b, bs=bs, hs=hs, nphi=nphi)
          
    def get_Aval(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        Ax, Ay, As = self.get_A()
        return [
            Ax.subs({self.x: x, self.y: y, self.s: s}).evalf(),
            Ay.subs({self.x: x, self.y: y, self.s: s}).evalf(),
            As.subs({self.x: x, self.y: y, self.s: s}).evalf()
        ]



class SympyParticle:
    def __init__(self, beta0=1):
        self.x, self.y, self.tau= sp.symbols('x y tau')
        self.zeta = sp.symbols('zeta')
        self.s = sp.symbols('s')
        self.px, self.py, self.ptau = sp.symbols('px py ptau')
        self.beta0 = beta0
        self._m = sp

class NumpyParticle:
    def __init__(self, qp0, s=0, beta0=1):
        self.beta0 = beta0
        self.x, self.y, self.tau, self.px, self.py, self.tau = qp0
        self.zeta = self.tau * self.beta0
        self.s = s
        self._m = np
        