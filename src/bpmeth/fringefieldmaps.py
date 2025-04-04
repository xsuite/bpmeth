import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import warnings
from .numerical_solver import *
from .track4d import Trajectory, MultiTrajectory

# All types of fringe field maps


class ThinNumericalFringe:   
    def __init__(self, b1, shape, length=1, nphi=5, ivp_opt={}):
        """    
        Thick numerical fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards bend.
        
        :param b1: Amplitude of the dipole field.
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" 
            normalized between zero and one.
        :param len: Integration range.
        :param nphi: Number of terms included in expansion.
        :param ivp_opt: Options for the Hamiltonian solver.
        """
        
        self.b1 = b1
        self.shape = shape        
        self.b1fringe = f"{b1}*{shape}"
        self.length = length
        self.nphi = nphi
        self.ivp_opt = ivp_opt
        
        # First a backwards drift
        self.drift = DriftVectorPotential()
        self.H_drift = Hamiltonian(self.length/2, 0, self.drift)

        # Then a fringe field map
        self.fringe = FringeVectorPotential(self.b1fringe, nphi=nphi) # Only b1 and b1'
        self.H_fringe = Hamiltonian(self.length, 0, self.fringe)

        # Then a backwards bend
        self.dipole = DipoleVectorPotential(0, self.b1)
        self.H_dipole = Hamiltonian(self.length/2, 0, self.dipole)
            
    
    def track(self, part):
        """
        :param part: Particles with part.x, part.px etc the coordinates for all part.npart particles.
        :return: A list of trajectory elements for all particles.
        """
        
        npart = part.npart
        x, px, y, py, tau, ptau = part.x, part.px, part.y, part.py, part.tau, part.ptau

        trajectories = MultiTrajectory(trajectories=[])
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
            sol_drift = self.H_drift.solve(qp0, s_span=[0, -self.length/2], ivp_opt=self.ivp_opt)
            sol_fringe = self.H_fringe.solve(sol_drift.y[:, -1], s_span=[-self.length/2, self.length/2], ivp_opt=self.ivp_opt)
            sol_dipole = self.H_dipole.solve(sol_fringe.y[:, -1], s_span=[self.length/2, 0], ivp_opt=self.ivp_opt)
            
            # Update particle position
            part.x[i], part.px[i], part.tau[i], part.px[i], part.py[i], part.ptau[i] = sol_dipole.y[:, -1]

            # Save all trajectory points
            all_s = np.append(sol_drift.t, [sol_fringe.t, sol_dipole.t])
            all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_dipole.y[0]])
            all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_dipole.y[1]])
            all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_dipole.y[3]])
            all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_dipole.y[4]])
        
            trajectories.add_trajectory(Trajectory(all_s, all_x, all_px, all_y, all_py))
            
        return trajectories



class ThickNumericalFringe:
    def __init__(self,b1,shape,length=1,nphi=5):
        """    
        Thick numerical fringe field at the edge of the magnet. 
        It makes a map for the correction field to the hard edge dipole.
        
        THIS MAP DOES NOT REPRESENT REALITY, BUT ASSUMES THAT THE MAPS COMMUTE!
        IT SHOULD NOT BE USED FOR REAL SIMULATIONS!
        
        :param b1: Amplitude of the dipole field.
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" 
            normalized between zero and one.
        :param len: Integration range.
        :param nphi: Number of terms included in expansion.
        """

        warnings.warn("This map does not represent reality, but assumes that the maps commute! \
            Are you sure you want to use this map?")
        
        self.b1 = b1
        self.shape = shape
        self.b1fringe = f"{b1}*{shape}"
        self.length = length
        self.nphi = nphi
        
        # Backwards drift
        self.drift = DriftVectorPotential()
        self.H_drift = Hamiltonian(self.length/2, 0, self.drift)
        
        # Fringe up to the dipole edge
        self.fringe1 = FringeVectorPotential(self.b1fringe, nphi=nphi)
        self.H_fringe1 = Hamiltonian(self.length/2, 0, self.fringe1)

        # Fringe inside dipole
        self.fringe2 = FringeVectorPotential(f"{self.b1fringe} - {self.b1}", nphi=nphi)
        self.H_fringe2 = Hamiltonian(self.length/2, 0, self.fringe2)
        

    def track(self, part, makethin=True):
        """
        :param part: Particles with part.x, part.px etc the coordinates for all part.npart particles.
        :param makethin: If True, backwards drifts are included 0->-L/2 and L/2->0 
            to have a thin map at the edge.
        :return: A list of trajectory elements for all particles. 

        """
        
        npart = part.npart
        x, px, y, py, tau, ptau = part.x, part.px, part.y, part.py, part.tau, part.ptau
        
        p_sp = SympyParticle()
        trajectories = MultiTrajectory(trajectories = [])
        for i in range(npart):
            qp0 = [x[i], y[i], tau[i], px[i], py[i], ptau[i]]
            
            if makethin:
                sol_drift1 = self.H_drift.solve(qp0, s_span=[0, -self.length/2])
                qp0 = sol_drift1.y[:,-1]
                
            sol_fringe1 = self.H_fringe1.solve(qp0, s_span=[-self.length/2, 0])
            
            # Changing the canonical momenta: 
            # x' and y' are continuous in changing vector potential, px and py are not! 
            # Usually this is not an issue as a_x, a_y are typically zero in the regions between maps
            xval, yval, tauval, pxval, pyval, ptauval = sol_fringe1.y[:, -1]

            ax1 = self.fringe1.get_A(p_sp)[0].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()
            ax2 = self.fringe2.get_A(p_sp)[0].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()

            ay1 = self.fringe1.get_A(p_sp)[1].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()
            ay2 = self.fringe2.get_A(p_sp)[1].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()

            # assert (h==0), "h is not zero, this is not implemented yet"
            pxval += (ax2 - ax1)
            pyval += (ay2 - ay1)
            qp = [xval, yval, tauval, pxval, pyval, ptauval]
            
            sol_fringe2 = self.H_fringe2.solve(qp, s_span=[0, self.length/2])
            xsol, ysol, tausol, pxsol, pysol, ptausol = sol_fringe2.y[:,-1]
            
            if makethin:
                sol_drift2 = self.H_drift.solve([xsol, ysol, tausol, pxsol, pysol, ptausol], 
                                                s_span=[self.length/2, 0])
                xsol, ysol, tausol, pxsol, pysol, ptausol = sol_drift2.y[:,-1]

            # Update particle position
            part.x[i], part.px[i], part.tau[i], part.px[i], part.py[i], part.ptau[i] = xsol, pxsol, tausol, pxsol, pysol, ptausol
            
            # Save all trajectory points
            if makethin:
                all_s = np.append(sol_drift1.t, [sol_fringe1.t, sol_fringe2.t, sol_drift2.t])
                all_x = np.append(sol_drift1.y[0], [sol_fringe1.y[0], sol_fringe2.y[0], sol_drift2.y[0]])
                all_y = np.append(sol_drift1.y[1], [sol_fringe1.y[1], sol_fringe2.y[1], sol_drift2.y[1]])
                all_px = np.append(sol_drift1.y[3], [sol_fringe1.y[3], sol_fringe2.y[3], sol_drift2.y[3]])
                all_py = np.append(sol_drift1.y[4], [sol_fringe1.y[4], sol_fringe2.y[4], sol_drift2.y[4]])
            else:
                all_s = np.append(sol_fringe1.t, sol_fringe2.t)
                all_x = np.append(sol_fringe1.y[0], sol_fringe2.y[0])
                all_y = np.append(sol_fringe1.y[1], sol_fringe2.y[1])
                all_px = np.append(sol_fringe1.y[3], sol_fringe2.y[3])
                all_py = np.append(sol_fringe1.y[4], sol_fringe2.y[4])
            
            trajectories.add_trajectory(Trajectory(all_s, all_x, all_px, all_y, all_py))
        
        return trajectories
    
    

class ForestFringe:
    def __init__(self, b1, Kg, K0gg=0, closedorbit=False, sadistic=False, improved=False):
        """
        :param b1: The design value for the magnet.
        :param Kg: The fringe field integral K times gap height multiplied with the gap height of the magnet.
        "param K0gg: Fringe field integral K0 times gap height squared for the closed orbit effect
        :param sadistic: Include the sadistic term.
            (To be checked, is this actually symplectic?)
        :param closedorbit: Include the closed orbit distortion as a shift in x.
            (This is symplectic as long as the shift is independent of the coordinates.)
        :param improved: If True, use the improved formula with the tangent for the generating function.
        """
        
        self.b1 = b1
        self.Kg = Kg
        self.K0gg = K0gg
        
        x, y, px, py, tau, ptau = sp.symbols("x y px py tau ptau")
        self.x, self.y, self.px, self.py = x, y, px, py
        delta, l = sp.symbols("delta l")
        self.delta, self.l = delta, l
        
        pz = sp.sqrt((1+delta)**2 - px**2 - py**2)
        xp = px/pz
        yp = py/pz
        
        if improved:
            self.phi = self.b1 * sp.tan(sp.atan(xp / (1+(yp)**2)) - self.Kg*self.b1 * (1+xp**2*(2+yp**2))*pz)
        else:
            self.phi = self.b1*xp / (1+(yp)**2) - self.Kg*self.b1**2 * (((1+delta)**2-py**2)/pz**3 + (px/pz)**2*((1+delta)**2-px**2)/pz**3)
            
        self.dphidpx = self.phi.diff(px)
        self.dphidpy = self.phi.diff(py)
        self.dphiddelta = self.phi.diff(delta)
        
        self.closedorbit = closedorbit
        self.sadistic = sadistic
        if self.sadistic:
            warnings.warn("Not sure if this map is symplectic")
            
        
    def track(self, part):
        """
        No longitudinal coordinates yet, only 4d
        
        :param coord: List of coordinates [x, px, y, py], 
            with x = coord[0] etc lists of coordinates for all N particles.
        :return: A list of trajectory elements for all particles.
        """
                
        xi, pxi, yi, pyi, zetai, ptaui = part.x, part.px, part.y, part.py, part.zeta, part.ptau
        beta0 = part.beta0
        
        m = part._m
        deltai = m.sqrt(1 + 2*ptaui / beta0 + ptaui**2) - 1
        betai = m.sqrt(1-(1-beta0)/(1+beta0*ptaui)**2)
        li = -betai * zetai / beta0
        
        trajectories = MultiTrajectory(trajectories = [])
        
        singleparticle=False
        npart = part.npart
        if npart == 1 and not (isinstance(xi, list) or isinstance(xi, np.ndarray)):
            xi, pxi, yi, pyi, zetai, ptaui = [xi], [pxi], [yi], [pyi], [zetai], [ptaui]
            deltai, betai, li = [deltai], [betai], [li]
            singleparticle = True

            
        for i in range(npart):
            x, y, px, py = self.x, self.y, self.px, self.py
            delta, l = self.delta, self.l
            
            phi = self.phi.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i], delta:deltai[i], l:li[i]})
            dphidpx = self.dphidpx.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i], delta:deltai[i], l:li[i]})
            dphidpy = self.dphidpy.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i], delta:deltai[i], l:li[i]})
            dphiddelta = self.dphiddelta.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i], delta:deltai[i], l:li[i]})
            
            if part._m == np:
                phi = float(phi)
                dphidpx = float(dphidpx)
                dphidpy = float(dphidpy)
                dphiddelta = float(dphiddelta)
            
            yf = 2*yi[i] / (1 + m.sqrt(1 - 2*dphidpy*yi[i]))
            xf = xi[i] + 1/2*dphidpx*yf**2
            pyf = pyi[i] - phi*yf
            lf = li[i] - 1/2*dphiddelta*yf**2
            
            if self.closedorbit:
                assert(self.K0gg != 0), "K0gg must be non-zero for closed orbit distortion"
                warnings.warn("With closed orbit distortion the map is not symplectic")
                xf -= self.K0gg / m.sqrt(1 - pxi[i]**2 - pyi[i]**2)
            
            if self.sadistic:
                pyf -= 4*self.b1**2 /(36*self.Kg) * yf**3 #/(1+delta)
            
            # Required x, y, zeta, px, py, ptau
            ptauf = ptaui[i]
            betaf = m.sqrt(1-(1-beta0)/(1+beta0*ptauf)**2)
            zetaf = -beta0 * lf / betaf

            # Update particle position
            if singleparticle:
                part.x, part.y, part.zeta, part.py = xf, yf, zetaf, pyf
            else:
                part.x[i], part.y[i], part.zeta[i], part.py[i],  = xf, yf, zetaf, pyf

            # Save all trajectory points (thin map so only initial and final position)        
            trajectories.add_trajectory(Trajectory([0, 0], [xi[i], xf], [pxi[i], pxi[i]], [yi[i], yf], [pyi[i], pyf]))
            
        return trajectories
