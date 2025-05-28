import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import warnings
from .numerical_solver import *
from .track4d import Trajectory, MultiTrajectory

# All types of fringe field maps


class ThinNumericalFringe:   
    def __init__(self,b1,shape,length=1,nphi=5):
        """    
        Thick numerical fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards bend.
        
        :param b1: Amplitude of the dipole field.
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" 
            normalized between zero and one.
        :param len: Integration range.
        :param nphi: Number of terms included in expansion.
        """
        
        self.b1 = b1
        self.shape = shape        
        self.b1fringe = f"{b1}*{shape}"
        self.length = length
        self.nphi = nphi      
        
        # First a backwards drift
        self.drift = DriftVectorPotential()
        self.H_drift = Hamiltonian(self.length/2, 0, self.drift)

        # Then a fringe field map
        self.fringe = FringeVectorPotential(self.b1fringe, nphi=nphi) # Only b1 and b1'
        self.H_fringe = Hamiltonian(self.length, 0, self.fringe)

        # Then a backwards bend
        self.dipole = DipoleVectorPotential(0, self.b1)
        self.H_dipole = Hamiltonian(self.length/2, 0, self.dipole)
            
    
    def track(self, coord):
        """
        :param coord: List of coordinates [x, px, y, py], 
            with x = coord[0] etc lists of coordinates for all N particles.
        :return: A list of trajectory elements for all particles.
        """
        
        ncoord,npart=coord.shape
        
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]

        trajectories = MultiTrajectory(trajectories=[])
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
            sol_drift = self.H_drift.solve(qp0, s_span=[0, -self.length/2])
            sol_fringe = self.H_fringe.solve(sol_drift.y[:, -1], s_span=[-self.length/2, self.length/2])
            sol_dipole = self.H_dipole.solve(sol_fringe.y[:, -1], s_span=[self.length/2, 0])
            
            coord[0, i] = sol_dipole.y[0][-1]
            coord[1, i] = sol_dipole.y[3][-1]
            coord[2, i] = sol_dipole.y[1][-1]
            coord[3, i] = sol_dipole.y[4][-1]

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
        It consists of a backwards drift, a fringe field map and a backwards bend.
        
        :param b1: Amplitude of the dipole field.
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" 
            normalized between zero and one.
        :param len: Integration range.
        :param nphi: Number of terms included in expansion.
        """
        
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
        
    def track(self, coord, makethin=True):
        """
        :param coord: List of coordinates [x, px, y, py], 
            with x = coord[0] etc lists of coordinates for all N particles.
        :param makethin: If True, backwards drifts are included 0->-L/2 and L/2->0 
            to have a thin map at the edge.
        :return: A list of trajectory elements for all particles. 

        """
        ncoord,npart=coord.shape
        
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]
        
        p_sp = SympyParticle()
        trajectories = MultiTrajectory(trajectories = [])
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
            
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

            coord[0, i] = xsol
            coord[1, i] = pxsol
            coord[2, i] = ysol
            coord[3, i] = pysol
            
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
    def __init__(self, b1, Kg, K0gg=0, closedorbit=False, sadistic=False):
        """
        :param b1: The design value for the magnet.
        :param Kg: The fringe field integral K times gap height multiplied with the gap height of the magnet.
        "param K0gg: Fringe field integral K0 times gap height squared for the closed orbit effect
        :param sadistic: Include the sadistic term.
        :param closedorbit: Include the closed orbit distortion as a shift in x.
            (This is symplectic as long as the shift is independent of the coordinates.)
        """
        
        self.b1 = b1
        self.Kg = Kg
        self.K0gg = K0gg
        x, y, px, py = sp.symbols("x y px py")
        pz = sp.sqrt(1 - px**2 - py**2)
        self.phi = self.b1*px/pz / (1+(py/pz)**2) - self.Kg*self.b1**2 * ((1-py**2)/pz**3 + (px/pz)**2*(1-px**2)/pz**3)
        self.dphidpx = self.phi.diff(px)
        self.dphidpy = self.phi.diff(py)
        self.x, self.y, self.px, self.py = x, y, px, py
        self.closedorbit = closedorbit
        self.sadistic = sadistic        
        
        
    def track(self, coord):
        """
        No longitudinal coordinates yet, only 4d
        
        :param coord: List of coordinates [x, px, y, py], 
            with x = coord[0] etc lists of coordinates for all N particles.
        :return: A list of trajectory elements for all particles.
        """
        
        warnings.warn("This is a 4d map, no longitudinal coordinates yet")
        
        ncoord,npart=coord.shape

        xi = coord[0]
        pxi = coord[1]
        yi = coord[2]
        pyi = coord[3]
        
        trajectories = MultiTrajectory(trajectories = [])
        
        for i in range(npart):
            x, y, px, py = self.x, self.y, self.px, self.py
            phi = float(self.phi.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i]}))
            dphidpx = float(self.dphidpx.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i]}))
            dphidpy = float(self.dphidpy.subs({x: xi[i], y: yi[i], px: pxi[i], py: pyi[i]}))
            
            yf = 2*yi[i] / (1 + np.sqrt(1 - 2*dphidpy*yi[i]))
            xf = xi[i] + 1/2*dphidpx*yf**2
            pyf = pyi[i] - phi*yf
            
            coord[0, i] = xf
            coord[2, i] = yf
            coord[3, i] = pyf
            
            if self.closedorbit:
                assert(self.K0gg != 0), "K0gg must be non-zero for closed orbit distortion"
                warnings.warn("Closed orbit distortion as currently implemented is not symplectic - it depends on px and py")
                xf -= self.K0gg / np.sqrt(1 - pxi[i]**2 - pyi[i]**2)
                coord[0, i] = xf
            
            if self.sadistic:
                pyf -= 4*self.b1**2 /(36*self.Kg) * yf**3 #/(1+delta)
                coord[3, i] = pyf
            
            trajectories.add_trajectory(Trajectory([0, 0], [xi[i], xf], [pxi[i], pxi[i]], [yi[i], yf], [pyi[i], pyf]))
            
        return trajectories
