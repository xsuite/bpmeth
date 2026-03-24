from .fieldmaps import Fieldmap, get_derivative, spTanh, spEnge, Enge
from .generate_expansion import FieldExpansion
from .numerical_solver import Hamiltonian, DipoleVectorPotential
from .frames import CanvasZX, Frame, BendFrame
from .poly_fit import poly_print
import numpy as np
import sympy as sp
import warnings
import matplotlib.pyplot as plt
import scipy as sc
import math
    

class MagnetFromFieldmap:
    isthick=True
    def __init__(self, data, h, l_magn, design_field=None, order=3, hgap=0.05, apt=0.05, radius=0.05, nphi=3, plot=False, step=50, in_FS_coord=False, symmetric=False,
                 smin=None, smax=None):
        """
        :param data: Fieldmap points as columns x, y, z, Bx, By, Bz. Data in Tesla
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, the fieldmap is first converted in Frenet-Serrat coordinates consisting of a straight part before the 
            magnetic length, then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
        :param design_field: Desired value for the average field of lowest component, this is probably 1/rho
        :param order (optional): Highest order multipole (dipole=1)
        :param hgap (optional): Half the gap of the magnet. Only used to estimate a reasonable range of the fringe field.
        :param apt (optional): Total aperture of the magnet (over which the fieldmap should also be given, otherwise give maximal 
            range of the fieldmap). Used to fit the multipoles.
        :param radius (optional): Interpolation radius for determination of field in Frenet-Serrat coordinates.
        :param nphi: Number of phi terms in the expansion, at least phi0 and phi1.
        :param plot: If True, plot the fieldmap and the fitted multipoles when creating the element.
        :param step: Number of points in one step of spline.
        :param in_FS_coord: If True, the fieldmap is already in Frenet-Serrat coordinates.
        :param symmetric: If True, the fieldmap is symmetrized.
        """

        self.data = data

        self.h = h 
        self.l_magn = l_magn
        self.angle = self.h*self.l_magn
        
        self.design_field = design_field

        self.order=order
        self.fieldmap = Fieldmap(data)

        self.xmin = -apt/2
        self.xmax = apt/2
        if smin is None:
            self.smin = -(7.5*hgap + l_magn/2)
        else:
            self.smin = smin
        if smax is None:
            self.smax = 7.5*hgap + l_magn/2 
        else:
            self.smax = smax
        self.sedge = l_magn/2

        self.length = self.smax - self.smin

        self.nphi = nphi

        self.step = step
                
        assert self.h != 0, "This method is only implemented for curved magnets with h!=0"
        if not h==0:
            self.rho = 1/h
            self.phi = l_magn*h

            if not in_FS_coord:
                xFS = np.linspace(self.xmin, self.xmax, 31)
                yFS = [0]
                ns = math.ceil(((self.smax - self.smin) / 0.001) / step) * step + 1
                sFS = np.linspace(self.smin, self.smax, ns) 

                self.fieldmap = self.fieldmap.calc_FS_coords(xFS, yFS, sFS, self.rho, self.phi, radius=radius)
            
            else:
                # We still need the correct number of points in s to work with the splines, 
                # otherwise the length would change
                s = np.unique(self.fieldmap.src['z'])
                smask = (s >= self.smin) & (s <= self.smax)
                s = s[smask]
                x = np.unique(self.fieldmap.src['x'])
                y = np.unique(self.fieldmap.src['y'])

                ds = s[1] - s[0]
                ns = math.ceil(((self.smax - self.smin) / ds) / step) * step + 1
                ss = np.linspace(self.smin, self.smax, ns)
                
                X, Y, S = np.meshgrid(x, y, ss)
                

                self.fieldmap = self.fieldmap.interpolate_points(X, Y, S, radius=radius)

        if symmetric:
            self.fieldmap = self.fieldmap.symmetrize(radius=radius)

        if self.design_field is not None:
            scalefactor = self.design_field / (self.fieldmap.integratedfield(3)[0] / self.l_magn)
            self.fieldmap.rescale(scalefactor)
            
        self.create_Hamiltonian(plot=plot)
        

    def fit_multipoles(self, plot=False):
        print("Fitting multipoles")
        
        if plot:
            fig, ax = plt.subplots()
        else:
            ax=None

        segments, all_pols = self.fieldmap.fit_spline_multipoles(components=np.arange(1,self.order+1,1), ax=ax, step=self.step, smin=self.smin, smax=self.smax)
        bfuncs = [[poly_print(all_pols[i][j], x="s") for j in range(len(all_pols[i]))] for i in range(len(all_pols))]

        self.segments = segments
        self.bfuncs = bfuncs

        
    def create_fieldexpansion(self, plot=False):
        self.fit_multipoles(plot=plot)

        print("Creating field expansion...")
        
        expansions_in = []
        segments_in = [] 
        expansions_body = []
        segments_body = []
        expansions_out = []
        segments_out = []

        for i in range(len(self.segments)): 
            # Loop over all segments and create expansions
            bb = [bbs[i] for bbs in self.bfuncs]
            
            # split in straight and bent parts
            if self.segments[i][0] < -self.sedge and self.segments[i][1] > self.smin:  # Incoming drift
                expansion = FieldExpansion(b=bb, hs="0", nphi=self.nphi)
                expansions_in.append(expansion)
                segments_in.append([max(self.segments[i][0], self.smin), min(self.segments[i][1], -self.sedge)])

            if self.segments[i][1] > -self.sedge and self.segments[i][0] < self.sedge:  # Body part
                expansion = FieldExpansion(b=bb, hs=f"{self.h}", nphi=self.nphi)
                expansions_body.append(expansion)
                segments_body.append([max(self.segments[i][0], -self.sedge), min(self.segments[i][1], self.sedge)])

            if self.segments[i][1] > self.sedge and self.segments[i][0] < self.smax:  # Outgoing drift
                expansion = FieldExpansion(b=bb, hs="0", nphi=self.nphi)
                expansions_out.append(expansion)
                segments_out.append([max(self.segments[i][0], self.sedge), min(self.segments[i][1], self.smax)])
                
        self.expansions_in = expansions_in
        self.segments_in = segments_in
        self.expansions_body = expansions_body
        self.segments_body = segments_body
        self.expansions_out = expansions_out
        self.segments_out = segments_out
        

    def create_Hamiltonian(self, plot=False):
        self.create_fieldexpansion(plot=plot)
        
        print("Creating Hamiltonian...")

        hamiltonians_in = []
        hamiltonians_body = []
        hamiltonians_out = []
        
        for i in range(len(self.expansions_in)):
            hamiltonians_in.append(Hamiltonian(self.segments_in[i][1]-self.segments_in[i][0], 0, self.expansions_in[i], s_start=self.segments_in[i][0]))
        for i in range(len(self.expansions_body)):
            hamiltonians_body.append(Hamiltonian(self.segments_body[i][1]-self.segments_body[i][0], self.h, self.expansions_body[i], s_start=self.segments_body[i][0]))
        for i in range(len(self.expansions_out)):
            hamiltonians_out.append(Hamiltonian(self.segments_out[i][1]-self.segments_out[i][0], 0, self.expansions_out[i], s_start=self.segments_out[i][0]))
            
        self.hamiltonians_in = hamiltonians_in
        self.hamiltonians_body = hamiltonians_body
        self.hamiltonians_out = hamiltonians_out


    def track(self, particles, ivp_opt=None, return_sol=False):        
        sols = []
        for hamiltonian in self.hamiltonians_in:
            sol = hamiltonian.track(particles, ivp_opt=ivp_opt, return_sol=True)
            sols.append(sol)
        for hamiltonian in self.hamiltonians_body:
            sol = hamiltonian.track(particles, ivp_opt=ivp_opt, return_sol=True)
            sols.append(sol)
        for hamiltonian in self.hamiltonians_out:
            sol = hamiltonian.track(particles, ivp_opt=ivp_opt, return_sol=True)
            sols.append(sol)

        if return_sol:
            return sols

        if plot:
            fig, ax = plt.subplots(2, sharex=True)
            for i in range(len(particles.x)):
                tlist = np.concatenate([sol[i].t for sol in sols])
                xlist = np.concatenate([sol[i].y[0] for sol in sols])
                ylist = np.concatenate([sol[i].y[1] for sol in sols])
                ax[0].scatter(tlist, xlist, marker='.')
                ax[1].scatter(tlist, ylist, marker='.')

            ax[0].set_xlabel("s")
            ax[1].set_xlabel("s")
            ax[0].set_ylabel("x")
            ax[1].set_ylabel("y")


