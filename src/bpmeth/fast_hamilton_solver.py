from .polyhamiltonian_4_3_h import vectorfield, afield
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit(cache=True)
def rk4_step(vectorfield, s, z, ds, beta0, h, comp):
    """
    Perform one RK4 step for the system z' = f(s, z).
    :param vectorfield (function): f(s, x, y, tau, px, py, ptau, beta0, h, fieldder) that returns dx/ds, dy/ds, dtau/ds, dpx/ds, dpy/ds, dptau/ds
    :param s (float): Longitudinal coordinate, acts as time
    :param z (6d array): Current state vector (x, y, tau, px, py, ptau)
    :param ds (float): Step size
    :param h (float): Curvature of the reference trajectory
    :param comp (ndarray): Coefficients specifying the magnetic field
    :return: Updated state after one RK4 step as 6d array
    """
    
    x, y, tau, px, py, ptau = z

    # --- k1 ---
    k1x, k1y, k1tau, k1px, k1py, k1ptau = vectorfield(
        s, x, y, tau, px, py, ptau, beta0, h, comp
    )

    # --- k2 ---
    k2x, k2y, k2tau, k2px, k2py, k2ptau = vectorfield(
        s + 0.5*ds,
        x    + 0.5*ds*k1x,
        y    + 0.5*ds*k1y,
        tau  + 0.5*ds*k1tau,
        px   + 0.5*ds*k1px,
        py   + 0.5*ds*k1py,
        ptau + 0.5*ds*k1ptau,
        beta0, h, comp
    )

    # --- k3 ---
    k3x, k3y, k3tau, k3px, k3py, k3ptau = vectorfield(
        s + 0.5*ds,
        x    + 0.5*ds*k2x,
        y    + 0.5*ds*k2y,
        tau  + 0.5*ds*k2tau,
        px   + 0.5*ds*k2px,
        py   + 0.5*ds*k2py,
        ptau + 0.5*ds*k2ptau,
        beta0, h, comp
    )

    # --- k4 ---
    k4x, k4y, k4tau, k4px, k4py, k4ptau = vectorfield(
        s + ds,
        x    + ds*k3x,
        y    + ds*k3y,
        tau  + ds*k3tau,
        px   + ds*k3px,
        py   + ds*k3py,
        ptau + ds*k3ptau,
        beta0, h, comp
    )

    # --- final update ---
    xn    = x    + (ds/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    yn    = y    + (ds/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    taun  = tau  + (ds/6.0)*(k1tau + 2*k2tau + 2*k3tau + k4tau)
    pxn   = px   + (ds/6.0)*(k1px + 2*k2px + 2*k3px + k4px)
    pyn   = py   + (ds/6.0)*(k1py + 2*k2py + 2*k3py + k4py)
    ptaun = ptau + (ds/6.0)*(k1ptau + 2*k2ptau + 2*k3ptau + k4ptau)

    return xn, yn, taun, pxn, pyn, ptaun

 
    
@njit(cache=True)
def integrate_rk4(vectorfield, s, z, ds, n_steps, beta0, h, comp):
    """
    Integrate the system using RK4 over n_steps.
    :param s0 (float): Starting longitudinal coordinate
    :param z0 (6d array): Initial state vector (x, y, tau, px, py, ptau)
    :param ds (float): Step size
    :param n_steps (int): Number of integration steps
    :return: new state vector
    """
    
    for i in range(n_steps):
        z = rk4_step(vectorfield, s, z, ds, beta0, h, comp)
        s += ds
    return z


@njit(cache=True)
def euler_step(vectorfield, s, z, ds, beta0, h, comp):
    """
    Perform one Euler step for the system z' = f(s, z).
    :param vectorfield: function f(s, x, y, tau, px, py, ptau, beta0, h, comp)
    :param s: Longitudinal coordinate (acts like time)
    :param z: Current state vector (x, y, tau, px, py, ptau)
    :param ds: Step size
    :param beta0, h, comp: parameters for the vectorfield
    :return: Updated state after one Euler step
    """
    
    x, y, tau, px, py, ptau = z
    dx, dy, dtau, dpx, dpy, dptau = vectorfield(s, x, y, tau, px, py, ptau, beta0, h, comp)

    xn    = x    + ds * dx
    yn    = y    + ds * dy
    taun  = tau  + ds * dtau
    pxn   = px   + ds * dpx
    pyn   = py   + ds * dpy
    ptaun = ptau + ds * dptau
    
    z = xn, yn, taun, pxn, pyn, ptaun


@njit(cache=True)
def integrate_euler(vectorfield, s, z, ds, n_steps, beta0, h, comp):
    """
    Integrate the system using simple Euler over n_steps.
    :param s0: Starting longitudinal coordinate
    :param z0: Initial state vector
    :param ds: Step size
    :param n_steps: Number of integration steps
    :return: New state vector after integration
    """
    
    for i in range(n_steps):
        euler_step(vectorfield, s, z, ds, beta0, h, comp)
        s += ds
    return z


class PolySegment:
    isthick = True

    def __init__(self, length, h, comp, s_start=0., integrator=integrate_rk4):
        """
        :param length (float): Length of the element
        :param h (float): Curvature of the reference trajectory
        :param comp (ndarray): Coefficients specifying the magnetic field
        :param s_start (float): Starting s position of the element, default 0
        """
        
        self.length = length
        self.h = h
        self.comp = comp
        self.s_start = s_start
        self.integrator = integrator
        
        
    def track(self, particle, ds=0.01):
        """
        Track a particle through the element using RK4 integration.
        :param particle: Particle object like the one from xsuite, has x, px, y, py, zeta, ptau, beta0, s ...
            XSuite does not support tau, but this is important for us to have canonical conjugates.
        """
        
        n_steps = int(np.ceil(self.length / ds))
        ds_actual = self.length / n_steps

        ax, ay, _ = afield(particle.x, particle.y, self.s_start, self.h, self.comp)
        px = particle.px - particle.ax + ax
        py = particle.py - particle.ay + ay


        # Loop over all particles
        for i in range(len(particle.x)):
            z = (particle.x[i], particle.y[i], particle.zeta[i]/particle.beta0[i], px[i], py[i], particle.ptau[i])
            z = self.integrator(vectorfield, self.s_start, z, ds_actual, n_steps, particle.beta0[i], self.h, self.comp)
            particle.x[i], particle.y[i], tau, particle.px[i], particle.py[i], particle.ptau[i] = z
            particle.zeta[i] = tau * particle.beta0[i]
            particle.s[i] += self.length
       

    
    def track_step_by_step(self, particle, ds=0.01):
        """
        Track a particle through the element using RK4 integration and save every step on the way
        :param particle: Particle object like the one from xsuite, has x, px, y, py, zeta, ptau, beta0, s ...
            XSuite does not support tau, but this is important for us to have canonical conjugates.
        """
        
        tpart = particle.copy()
        self.track(tpart, ds=ds)  # Final points for check
        n_steps = np.ceil(self.length / ds)
        
        mag = PolySegment(length=ds, h=self.h, comp=self.comp, s_start=self.s_start)
        out = [particle.copy()]
        for _ in range(int(n_steps)):
            mag.track(particle, ds=ds)
            out.append(particle.copy())
            mag.s_start += ds
        
        # assert np.allclose(tpart.x, particle.x)
        # assert np.allclose(tpart.y, particle.y)
        # assert np.allclose(tpart.zeta, particle.zeta)
        # assert np.allclose(tpart.s, particle.s)

        return out

    def plot_x(self, part, ax=None, ds=0.01):
        out = self.track_step_by_step(part.copy(), ds=ds)
        s_vals = [p.s[0] for p in out]
        x_vals = [p.x[0] for p in out]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(s_vals, x_vals, label="x vs s")
        ax.set_xlabel("s (m)")
        ax.set_ylabel("x (m)")
        ax.set_title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()

    def plot_y(self, part, ax=None, ds=0.01):
        out = self.track_step_by_step(part.copy(), ds=ds)
        s_vals = [p.s[0] for p in out]
        y_vals = [p.y[0] for p in out]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(s_vals, y_vals, label="y vs s", color="orange")
        ax.xlabel("s (m)")
        ax.ylabel("y (m)")
        ax.title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()

        
        
        

        
        
