import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from .generate_expansion import FieldExpansion
from .frames import Frame, BendFrame
import numba as nb


class Hamiltonian:
    isthick = True  # To be able to work with Xsuite tracking

    def __init__(self, length, h, vectp, s_start=0):
        """
        :param length (float): Length of the element.
        :param h (float): Curvature of the reference frame.
        :param vectp (FieldExpansion): Field expansion of the element.
        :param s_start (float): Starting position of the element where the particle will enter.
        Important in case of s-dependent fields.
        """

        assert vectp.h == h, "Curvature of the vector potential must match the curvature of the Hamiltonian"

        self.length = length
        self.s_start = s_start
        self.curv = h
        self.vectp = vectp
        self.angle = h * length

        self.vectorfield = self.get_vectorfield()


    def get_H(self, coords):
        """
        Get the Hamiltonian of the system. This can be replaced with a custom Hamiltonian.
        :param coords: Particle object to evaluate the Hamiltonian on, must have attributes x, y, s, px, py, ptau, beta0.
        :return: Hamiltonian expression in the given coordinates.
        """

        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.curv
        A = self.vectp.get_A()
        A = [Ai.subs({self.vectp.x: x, self.vectp.y: y, self.vectp.s: s}) for Ai in A]

        sqrt = coords._m.sqrt
        tmp1 = sqrt(
            1 + 2 * ptau / beta0 + ptau**2 - (px - A[0]) ** 2 - (py - A[1]) ** 2
        )
        H = ptau / beta0 - (1 + h * x) * (tmp1 + A[2])

        return H


    def get_A(self, x,y,s):
        """
        Get the vector potential at a given position.
        :param x: x coordinate.
        :param y: y coordinate.
        :param s: s coordinate.
        :return: Ax, Ay, As at the given position.
        """

        Ax, Ay, As = self.vectp.get_A(lambdify=True)
        return Ax(x, y, s), Ay(x, y, s), As(x, y, s)


    def get_vectorfield(self, lambdify=True):
        """
        Determine the flow of the system from hamilton equations. Not the vector potential!
        :param lambdify (bool): Whether to return a numerical function or the symbolic expression of the vector field.
        :return: Vector field of the system, either as a numerical function or as a symbolic expression.
        """

        coords = SympyParticle()
        x, y, tau = coords.x, coords.y, coords.tau
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        H = self.get_H(coords)
        fx = H.diff(px)
        fy = H.diff(py)
        ftau = H.diff(ptau)
        fpx = -H.diff(x)
        fpy = -H.diff(y)
        fptau = -H.diff(tau)
        qpdot = [fx, fy, ftau, fpx, fpy, fptau]

        if lambdify:
            qp = (x, y, tau, px, py, ptau)
            s = coords.s
            f = sp.lambdify((s, qp, beta0), qpdot, modules="numpy")

            f_numba = nb.njit(f)
            return f_numba
        return qpdot


    def solve(self, qp0, s_span=None, ivp_opt=None, backtrack=False, beta0=1):
        """
        Solve the Hamilton equations for the given initial conditions
        :param qp0: Initial conditions for the integration, must be a list of 6 elements [x, y, tau, px, py, ptau].
        :param s_span: Span of integration in s, must be a list of 2 elements [s_start, s_end]. If None, it will be set
        based on the start of the element and its length.
        :param ivp_opt: Options for the IVP solver, such as tolerance.
        :param backtrack: Whether to integrate backwards in s.
        :param beta0: Reference beta of the particle, used for the Hamiltonian. Default is 1 (ultrarelativistic).
        :return: Solution of the IVP solver, containing the trajectory of the particle through the element. 
        The solution will have attributes t (s values) and y (trajectory in phase space). The trajectory will be a 
        list of 6 elements [x, y, tau, px, py, ptau] for each s value.
        """

        if s_span is None:
            if backtrack:
                s_span = [self.length + self.s_start, self.s_start]
            else:
                s_span = [self.s_start, self.length + self.s_start]
                
        if backtrack:
            assert s_span[0] > s_span[1], "s_span not compatible with backtracking"

        if ivp_opt is None:
            ivp_opt = {}
        if "t_eval" not in ivp_opt:
            ivp_opt["t_eval"] = np.linspace(s_span[0], s_span[1], 500)
        if "rtol" not in ivp_opt:
            ivp_opt["rtol"] = 1e-5
        if "atol" not in ivp_opt:
            ivp_opt["atol"] = 1e-8

        f = self.vectorfield
        sol = solve_ivp(f.py_func, s_span, qp0, args=(beta0,), **ivp_opt)
        return sol
    

    def track(self, particle, s_span=None, return_sol=False, ivp_opt=None, backtrack=False):
        """
        Track a particle through the element by solving the Hamilton equations. Works with NumpyParticle or Xsuite particles.
        :param particle: Particle to track, must have attributes x, y, zeta, px, py, ptau, beta0, s.
        :param s_span: Span of integration in s, must be a list of 2 elements [s_start, s_end]. If None, it will be set
        based on the start of the element and its length.
        :parma return_sol: Whether to return the solution of the IVP solver, in this case the function will return a list of solutions for each particle.
        :param ivp_opt: Options for the IVP solver, such as tolerance.
        :param backtrack: Whether to integrate backwards in s.
        :return: If return_sol=True, return solution of the IVP solver, containing the trajectory of the particle through the element. 
        The solution will have attributes t (s values) and y (trajectory in phase space). The trajectory will be a 
        list of 6 elements [x, y, tau, px, py, ptau] for each s value.
        """

        if s_span is None:
            if backtrack:
                s_span = [self.length + self.s_start, self.s_start]
            else:
                s_span = [self.s_start, self.length + self.s_start]

        if backtrack:
            assert s_span[0] > s_span[1], "s_span not compatible with backtracking"


        is_array = isinstance(particle.x, (list, np.ndarray))

        def to_array(v):
            return np.array(v) if isinstance(v, (list, np.ndarray)) else np.array([v])

        x     = to_array(particle.x)
        y     = to_array(particle.y)
        zeta  = to_array(particle.zeta)
        px    = to_array(particle.px)
        py    = to_array(particle.py)
        ptau  = to_array(particle.ptau)
        beta0 = to_array(particle.beta0)
        ax_p  = to_array(particle.ax)
        ay_p  = to_array(particle.ay)


        out = []
        n = len(x)
        x_out, y_out, zeta_out = np.empty(n), np.empty(n), np.empty(n)
        px_out, py_out, ptau_out = np.empty(n), np.empty(n), np.empty(n)
        for i in range(len(x)):
            # Adjust vector potential for each particle at the beginning of the segment
            xi, yi, zetai, pxi, pyi, ptaui = x[i], y[i], zeta[i], px[i], py[i], ptau[i]
            axi, ayi = ax_p[i], ay_p[i]
            beta0i = beta0[i]

            ax, ay, _ = self.get_A(xi, yi, s_span[0])
            kin_px = pxi - axi
            kin_py = pyi - ayi
            pxi = kin_px + ax
            pyi = kin_py + ay

            qp0 = [xi, yi, zetai/beta0i, pxi, pyi, ptaui]
            sol = self.solve(qp0, s_span=s_span, beta0=beta0i, ivp_opt=ivp_opt)

            xf, yf, tauf, pxf, pyf, ptauf = sol.y
            x_out[i]     = xf[-1]
            y_out[i]     = yf[-1]
            zeta_out[i]  = tauf[-1] * beta0[i]
            px_out[i]    = pxf[-1]
            py_out[i]    = pyf[-1]
            ptau_out[i]  = ptauf[-1]
            if return_sol:
                out.append(sol)

        particle.x    = x_out if is_array else x_out[0]
        particle.y    = y_out if is_array else y_out[0]
        particle.zeta = zeta_out if is_array else zeta_out[0]
        particle.px   = px_out if is_array else px_out[0]
        particle.py   = py_out if is_array else py_out[0]
        particle.ptau = ptau_out if is_array else ptau_out[0]

        particle.s += -self.length if backtrack else self.length

        # Manage vector potential for each particle at the exit of the segment
        ax, ay, _ = self.get_A(particle.x, particle.y, s_span[1])
        particle.ax = ax 
        particle.ay = ay

        if return_sol:
            return out if is_array else out[0]


    def plotsol(self, qp0, s_span=None, ivp_opt=None, figname_zx=None, figname_zxy=None, canvas_zx=None, canvas_zxy=None):
        """
        Plot the solution of the Hamilton equations for the given initial conditions
        :param qp0: Initial conditions for the integration, must be a list of 6 elements [x, y, tau, px, py, ptau].
        :param s_span: Span of integration in s, must be a list of 2 elements [s_start, s_end]. If None, it will be set
        based on the start of the element and its length.
        :param ivp_opt: Options for the IVP solver, such as tolerance.
        :param figname_zx: Name of the figure for the zx plot. If None, the plot will not be saved.
        :param figname_zxy: Name of the figure for the zxy plot. If None, the plot will not be saved.
        :param canvas_zx: Canvas for the zx plot. If None, a new canvas will be created.
        :param canvas_zxy: Canvas for the zxy plot. If None, a new canvas will be created.
        """

        sol = self.solve(qp0, s_span, ivp_opt)
        s = sol.t
        x, y, tau, px, py, ptau = sol.y

        fr = Frame()
        fb = BendFrame(fr, self.length, self.angle)

        fb.plot_trajectory_zx(s, x, y, figname=figname_zx, canvas=canvas_zx)
        fb.plot_trajectory_zxy(s, x, y, figname=figname_zxy, canvas=canvas_zxy)

    def __repr__(self):
        return f"Hamiltonian({self.length}, {self.curv}, {self.vectp})"


###########################################
# A few common cases, to make life easier #
###########################################


class DriftVectorPotential(FieldExpansion):
    def __init__(self):
        super().__init__(nphi=0)

    def get_A(self):
        return [sp.S.Zero, sp.S.Zero, sp.S.Zero]


class DipoleVectorPotential(FieldExpansion):
    def __init__(self, h, b1):
        """
        Dipoles without s-dependence, in this case the vector potential is known analytically.
        :param h (float): Curvature of the reference frame.
        :param b1 (float): Dipole field strength.
        """

        self.h = h
        self.b1 = b1
        super().__init__(b=(b1,), h=h, nphi=0)

    def get_A(self, lambdify=False):
        h = self.h
        x, y, s = self.x, self.y, self.s
        As = -(x + h / 2 * x**2) / (1 + h * x) * self.b[0]
        if lambdify:
            return [
                np.vectorize(sp.lambdify([x, y, s], 0, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], 0, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], As, "numpy")),
            ]
        return sp.Integer(0), sp.Integer(0), As


class SolenoidVectorPotential(FieldExpansion):
    def __init__(self, bs):
        """
        Solenoids without s-dependence, in this case the vector potential is known analytically.
        Curvature of the reference frame is not implemented, use GeneralVectorPotential instead.
        :param bs (float): Solenoid field strength.
        """

        self.bs = bs
        super().__init__(bs=f"{bs}", nphi=0)

    def get_A(self, coords):
        x, y = coords.x, coords.y
        bs = self.bs
        return [-bs * y / 2, bs * x / 2, 0]


class FringeVectorPotential(FieldExpansion):
    def __init__(self, b1, nphi=5):
        """
        Fringe fields in a straight coordinate frame.
        :param b1 (str): Fringe field b1 as a function of s.
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """

        super().__init__(b=(b1,), nphi=nphi)


class GeneralVectorPotential(FieldExpansion):
    def __init__(self, a=(0,), b=(0,), bs=0, h=0, nphi=5):
        """
        General field expansion.
        :param a (tuple of str): a coefficients as a function of s.
        :param b (tuple of str): b coefficients as a function of s.
        :param bs (str): bs coefficient as a function of s.
        :param h (str): Curvature of the reference frame as a function of s.  Has to be constant.
        :param nphi (int): Number of terms in the expansion of the scalar potential.
        """

        super().__init__(a=a, b=b, bs=bs, h=h, nphi=nphi)


########################################
# Particles that can be used to track. #
# One can also use an Xsuite particle. #
########################################

class SympyParticle:
    def __init__(self):
        self.x, self.y, self.tau = sp.symbols("x y tau")
        self.zeta = sp.symbols("zeta") ## to be checked
        self.s = sp.symbols("s")
        self.px, self.py, self.ptau = sp.symbols("px py ptau")
        self.beta0 = sp.symbols("beta0", real=True, positive=True)
        self._m = sp
        self.npart = 1

class NumpyParticle:
    def __init__(self, qp0, s=0, beta0=1):
        self.beta0 = beta0
        self.x, self.y, self.tau, self.px, self.py, self.ptau = qp0
        self.zeta = self.tau * self.beta0
        self.s = s
        self._m = np
        self.npart = 1

class MultiParticle:
    def __init__(self, npart, x=0, y=0, tau=0, px=0, py=0, ptau=0, s=0, beta0=1):
        if isinstance(x, (int, float)):
            x = np.full(npart, x)
        if isinstance(y, (int, float)):
            y = np.full(npart, y)
        if isinstance(tau, (int, float)):
            tau = np.full(npart, tau)
        if isinstance(px, (int, float)):
            px = np.full(npart, px)
        if isinstance(py, (int, float)):
            py = np.full(npart, py)
        if isinstance(ptau, (int, float)):
            ptau = np.full(npart, ptau)
        if isinstance(s, (int, float)):
            s = np.full(npart, s)

        assert len(x) == npart, "Invalid x"
        assert len(y) == npart, "Invalid y"
        assert len(tau) == npart, "Invalid tau"
        assert len(px) == npart, "Invalid px"
        assert len(py) == npart, "Invalid py"
        assert len(ptau) == npart, "Invalid ptau"
        assert len(s) == npart, "Invalid s"
        assert isinstance(beta0, (int, float)), "Invalid beta0"
        
        self.beta0 = beta0
        self.npart = npart
        
        self.x = np.array(x)
        self.y = np.array(y)
        self.tau = np.array(tau)
        self.px = np.array(px)
        self.py = np.array(py)
        self.ptau = np.array(ptau)
        self.zeta = self.tau * self.beta0
        self.s = np.array(s)
        
        self._m = np
        
    def copy(self):
        return MultiParticle(self.npart, x=self.x.copy(), y=self.y.copy(), tau=self.tau.copy(), 
                             px=self.px.copy(), py=self.py.copy(), ptau=self.ptau.copy(), s=self.s.copy())
        
