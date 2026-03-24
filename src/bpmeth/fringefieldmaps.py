import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import warnings
from .numerical_solver import *
from .track4d import Trajectory, MultiTrajectory

# All types of fringe field maps


class ThinDipoleFringe:   
    def __init__(self, b1, shape, length=1, nphi=5, ivp_opt=None, theta_E=0):
        """    
        Thin numerical dipole fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards bend.
        :param b1: Amplitude of the dipole field.
        :param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" 
        normalized between zero and one.
        :param length: Integration range.
        :param nphi: Number of terms included in expansion.
        :param ivp_opt: Options for the Hamiltonian solver.
        :param theta_E: Angle of the dipole edge, if not zero, the dipole is rotated by this angle.
        """
        
        self.b1 = b1
        self.shape = shape        
        self.b1fringe = f"{b1}*{shape}"
        self.len = length  # Renamed because this is not the actual lenght of the element, the element is thin
        self.nphi = nphi
        if ivp_opt is None:
            ivp_opt = {}
        else:
            self.ivp_opt = ivp_opt
        
        # First a backwards drift
        self.drift = DriftVectorPotential()
        self.H_drift = Hamiltonian(self.len/2, 0, self.drift)

        # Then a fringe field map
        self.fringe = FringeVectorPotential(self.b1fringe, nphi=nphi) # Only b1 and b1'
        if theta_E != 0:
            self.fringe = self.fringe.transform(theta_E)
        self.H_fringe = Hamiltonian(self.len, 0, self.fringe)

        # Then a backwards bend
        self.dipole = DipoleVectorPotential(0, self.b1)
        self.H_dipole = Hamiltonian(self.len/2, 0, self.dipole)
        
    
    def track(self, part, return_trajectories=False):
        """
        :param part: Particles with part.x, part.px etc the coordinates for all part.npart particles.
        :return: A list of trajectory elements for all particles.
        """
        
        npart = len(part.x)
        beta0 = part.beta0
        x, px, y, py, tau, ptau = part.x, part.px, part.y, part.py, part.zeta/beta0, part.ptau

        trajectories = MultiTrajectory(trajectories=[])
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
            sol_drift = self.H_drift.solve(qp0, s_span=[0, -self.len/2], ivp_opt=self.ivp_opt)
            sol_fringe = self.H_fringe.solve(sol_drift.y[:, -1], s_span=[-self.len/2, self.len/2], ivp_opt=self.ivp_opt)
            sol_dipole = self.H_dipole.solve(sol_fringe.y[:, -1], s_span=[self.len/2, 0], ivp_opt=self.ivp_opt)

            # Save all trajectory points
            all_s = np.append(sol_drift.t, [sol_fringe.t, sol_dipole.t])
            all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_dipole.y[0]])
            all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_dipole.y[1]])
            all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_dipole.y[3]])
            all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_dipole.y[4]])
        
            trajectories.add_trajectory(Trajectory(all_s, all_x, all_px, all_y, all_py))
            
            # Update particle position
            xi, yi, taui, pxi, pyi, ptaui = sol_dipole.y[:, -1]
            part.x[i], part.y[i], part.zeta[i], part.px[i], part.py[i], part.ptau[i] = xi, yi, taui*beta0[i], pxi, pyi, ptaui
            
        if return_trajectories:
            return trajectories
    
    
class MADXQuadFringe:
    def __init__(self, b2, angle=0):
        """
        MADX quadrupole fringe field (Hard edge)
        :param b2: Quadrupole design value
        """
        
        self.b2 = b2
        self.angle = angle
        
    def track(self, part):
        xi, pxi, yi, pyi, zetai, ptaui = part.x, part.px, part.y, part.py, part.zeta, part.ptau
        beta0 = part.beta0        
        m = part._m        
       
        trajectories = MultiTrajectory(trajectories = [])
        
        singleparticle=False
        npart = part.npart
        if npart == 1 and not (isinstance(xi, list) or isinstance(xi, np.ndarray)):
            xi, pxi, yi, pyi, zetai, ptaui = [xi], [pxi], [yi], [pyi], [zetai], [ptaui]
            deltai, betai, li = [deltai], [betai], [li]
            singleparticle = True
            
        for i in range(npart):
            xf = xi
            pxf = pxi + self.b2 * np.tan(self.angle) * (xi**2 - yi**2)
            yf = yi
            pyf = pyi - 2 * self.b2 * np.tan(self.angle) * xi * yi
            zetaf = zetai
            ptauf = ptaui
                                
            # Save all trajectory points (thin map so only initial and final position)        
            trajectories.add_trajectory(Trajectory([0, 0], [xi[i], xf], [pxi[i], pxi[i]], [yi[i], yf], [pyi[i], pyf]))
        
            # Update particle position
            if singleparticle:
                part.px, part.py = pxf, pyf
            else:
                part.px[i], part.py[i]  = pxf, pyf
            
        return trajectories
       

class ThinQuadFringe:   
    def __init__(self, b2, b2amp, b3, length=1, nphi=5, ivp_opt={}):
        """    
        Thin numerical quadrupole fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards quad.
        :param b2: Quadrupole fringe field shape
        :param b2amp: Amplitude of the quadrupole component
        :param b3: Sextupole fringe field shape (originating from edge angle)
        :param length: Integration range.
        :param nphi: Number of terms included in expansion.
        :param ivp_opt: Options for the Hamiltonian solver.
        """
        
        self.b2 = b2
        self.b2amp = b2amp
        self.b3 = b3
        self.len = length  # Not actual length of the element, the element is thin
        self.nphi = nphi
        self.ivp_opt = ivp_opt
        
        # First a backwards drift
        self.drift = DriftVectorPotential()
        self.H_drift = Hamiltonian(self.len/2, 0, self.drift)

        # Then a fringe field map
        self.fringe = GeneralVectorPotential(b=("0", self.b2, self.b3), nphi=nphi)
        self.H_fringe = Hamiltonian(self.len, 0, self.fringe)

        # Then a backwards bend
        self.quad = GeneralVectorPotential(b=("0", self.b2amp))
        self.H_quad = Hamiltonian(self.len/2, 0, self.quad)
            
    
    def track(self, part):
        """
        :param part: Particles with part.x, part.px etc the coordinates for all part.npart particles.
        :return: A list of trajectory elements for all particles.
        """
        
        xi, pxi, yi, pyi, zetai, ptaui = part.x, part.px, part.y, part.py, part.zeta, part.ptau
        beta0 = part.beta0        
        deltai = np.sqrt(1 + 2*ptaui / beta0 + ptaui**2) - 1
        betai = np.sqrt(1-(1-beta0)/(1+beta0*ptaui)**2)
        li = -betai * zetai / beta0  
        
        if isinstance(xi, list) or isinstance(xi, np.ndarray):
            npart = len(xi)
            singleparticle = False
        else:
            npart = 1
            xi, pxi, yi, pyi, zetai, ptaui = [xi], [pxi], [yi], [pyi], [zetai], [ptaui]
            deltai, betai, li = [deltai], [betai], [li]
            beta0 = [beta0]
            singleparticle = True

        trajectories = MultiTrajectory(trajectories=[])
        for i in range(npart):
            qp0 = [xi[i], yi[i], 0, pxi[i], pyi[i], 0]
            sol_drift = self.H_drift.solve(qp0, s_span=[0, -self.len/2], ivp_opt=self.ivp_opt)
            sol_fringe = self.H_fringe.solve(sol_drift.y[:, -1], s_span=[-self.len/2, self.len/2], ivp_opt=self.ivp_opt)
            sol_quad = self.H_quad.solve(sol_fringe.y[:, -1], s_span=[self.len/2, 0], ivp_opt=self.ivp_opt)
            
            # Update particle position
            xf, yf, tauf, pxf, pyf, ptauf = sol_quad.y[:, -1]                 
            betaf = np.sqrt(1-(1-beta0[i])/(1+beta0[i]*ptauf)**2)
            zetaf = -betaf * tauf
                   
            # Save all trajectory points
            all_s = np.append(sol_drift.t, [sol_fringe.t, sol_quad.t])
            all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_quad.y[0]])
            all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_quad.y[1]])
            all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_quad.y[3]])
            all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_quad.y[4]])
        
            trajectories.add_trajectory(Trajectory(all_s, all_x, all_px, all_y, all_py))
            
            # Update particle position
            if singleparticle:
                part.x, part.y, part.zeta, part.px, part.py, part.ptau = xf, yf, zetaf, pxf, pyf, ptauf
            else:
                part.x[i], part.y[i], part.zeta[i], part.px[i], part.py[i], part.ptau[i] = xf, yf, zetaf, pxf, pyf, ptauf            
                
        return trajectories 