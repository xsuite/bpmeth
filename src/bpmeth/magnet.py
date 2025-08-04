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
    


# A dipole element for XSuite with explicit solving created from a fieldmap
class DipoleFromFieldmap:

    isthick=True

    def __init__(self, data, h, l_magn, design_field, order=3, nparams=5, hgap=0.05, apt=0.05, radius=0.05, shape="enge", plot=False, symmetric=True, nphi=4, guess=None):
        """
        :param data: Fieldmap points as columns x, y, z, Bx, By, Bz. Data in Tesla
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, the fieldmap is first converted in Frenet-Serrat coordinates consisting of a straight part before the 
            magnetic length, then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
        :param design_field: Desired value for the average field, if used for XSuite this is probably 1/rho
        :param order (optional): Highest order multipole (dipole=1)
        :param nparams (optional): Number of parameters in the function and its derivatives used to fit the multipole shapes. 
            nparams-1 is the order of the polynomial in an Enge function.
        :param hgap (optional): Half the gap of the magnet. Only used to estimate a reasonable range of the fringe field.
        :param apt (optional): Total aperture of the magnet (over which the fieldmap should also be given, otherwise give maximal 
            range of the fieldmap). Used to fit the multipoles.
        :param radius (optional): Interpolation radius for determination of field in Frenet-Serrat coordinates.
        :param shape: Shape of the fieldmap. "enge" or "tanh". Default is "enge".
        :param plot: If True, plot the fieldmap and the fitted multipoles when creating the element.
        :param symmetric: If True, the fieldmap is assumed to be symmetric. The multipole fit is mirrored
        :param nphi: Number of phi terms in the expansion, at least phi0 and phi1. If set to zero, use DipoleVectorPotential object
            which doesn't include any field derivatives.
        :param guess: Initial guess for the lowest order fit. 
        """

        self.data = data

        self.h = h 
        self.l_magn = l_magn
        self.angle = self.h*self.l_magn
        self.design_field = design_field

        self.nparams = nparams
        self.order=order
        self.fieldmap = Fieldmap(data)

        self.xmin = -apt/2
        self.xmax = apt/2
        self.smin = -(7.5*hgap + l_magn/2)
        self.smax = 7.5*hgap + l_magn/2  
        self.sedge = l_magn/2
        self.shape = shape

        self.length = self.smax - self.smin

        self.symmetric = symmetric
        self.nphi = nphi
        self.guess = guess
        
        if not h==0:
            self.rho = 1/h
            self.phi = l_magn*h

            xFS = np.linspace(self.xmin, self.xmax, 31)
            yFS = [0]
            sFS = np.arange(self.smin, self.smax, 0.001)  # Better a region that is a bit too large to have freedom :)

            self.fieldmap = self.fieldmap.calc_FS_coords(xFS, yFS, sFS, self.rho, self.phi, radius=radius)
            scalefactor = self.design_field / (self.fieldmap.integratedfield(3)[0] / self.l_magn)
            self.fieldmap.rescale(scalefactor)
            
        #self.fit_multipoles(plot=plot, symmetric=self.symmetric)
        self.create_Hamiltonian(plot=plot, symmetric=self.symmetric)
        

    def fit_multipoles(self, plot=False, symmetric=True, guess=None):
        # Calculate the dipole, quadrupole, sextupole component for the two halves
        # BE CAREFUL: The parameters are given for the edge at zero

        print("Fitting multipoles")

        if self.shape == "tanh":
            self.shapefun = spTanh
            self.degree = 1
          
        elif self.shape == "enge":
            self.shapefun = spEnge

        ax = None
        if plot:
            fig, ax = plt.subplots()
        
        params_out_list, cov_out_list = self.fieldmap.fit_multipoles(self.shapefun, components=np.arange(1,self.order+1,1), design=1, 
                                                                     nparams=self.nparams, zmin=0, zmax=self.smax, zedge=self.sedge, ax=ax,
                                                                     guess=guess)


        self.out_Bfield = params_out_list[0,0]
        print("Central field in the magnet: ", self.out_Bfield)

        ss = np.linspace(-self.sedge, self.smax-self.sedge, 500)
        integrated_field = 2*np.trapz(get_derivative(0, self.nparams, func=self.shapefun, lambdify=True)(ss, *params_out_list[0]), ss)
        average_field = integrated_field / self.l_magn
        print("Average field in the magnet: ", average_field)

        print("Rescaling field to match integrated field with design field")
        scalefactor = self.design_field / average_field
        params_out_list[:, 0] = params_out_list[:, 0] * scalefactor

        # scalefactor_quad = -0.31851618736281756/-0.524262748442634
        # params_out_list[1, 0] = params_out_list[1, 0] * scalefactor_quad

        self.out_Bfield = params_out_list[0,0]
        print("Central field in the magnet: ", self.out_Bfield)

        ss = np.linspace(-self.sedge, self.smax-self.sedge, 500)
        integrated_field = 2*np.trapz(get_derivative(0, self.nparams, func=self.shapefun, lambdify=True)(ss, *params_out_list[0]), ss)
        average_field = integrated_field / self.l_magn
        print("Average field in the magnet: ", average_field)

        if not symmetric:
            params_in_list, cov_in_list = self.fieldmap.fit_multipoles(self.shapefun, components=np.arange(1,self.order+1,1), design=1, 
                                                                   nparams=self.nparams, zmin=self.smin, zmax=0, zedge=-self.sedge, ax=ax)
            # The first parameter is the amplitude.
            self.in_Bfield = params_in_list[0,0]
            params_in_list[:, 0] = params_in_list[:, 0] / self.Bfield / self.rho * scalefactor
            warnings.warn("be careful with asymmetric magnets, scalefactor might not be correct")
        else:
            params_in_list = params_out_list

        self.params_in_list = params_in_list
        self.params_out_list = params_out_list
        
        return params_in_list, params_out_list


    def calc_integrated_field(self, component):
        """
        :param component: 1 is a dipole, 2 is a quadrupole etc.
        """

        warnings.warn("Not implemented yet for asymmetric fields")

        ss = np.linspace(-self.sedge, self.smax-self.sedge, 500)
        integrated_field = 2*np.trapz(get_derivative(component-1, self.nparams, func=self.shapefun, lambdify=True)(ss, *self.params_out_list[component-1]), ss)
        return integrated_field

    
    
    def create_fieldexpansion(self, plot=False, symmetric=True):
        params_in_list, params_out_list = self.fit_multipoles(plot=plot, symmetric=symmetric,guess=self.guess)

        print("Creating field expansion...")
        # Make a field expansion
        b_in = []
        b_out = []
        s = sp.symbols("s")
        for i in range(self.order):
            b_in.append(get_derivative(i, nparams=self.nparams, func=self.shapefun, lambdify=False)(-s-self.sedge if symmetric else (s-self.sedge), *params_in_list[i]))
            b_out.append(get_derivative(i, nparams=self.nparams, func=self.shapefun, lambdify=False)(s-self.sedge, *params_out_list[i]))

        self.b_in = b_in
        self.b_out = b_out
        self.s = s
        
        self.straight_in = FieldExpansion(b=b_in, hs="0", nphi=self.nphi)
        self.straight_out = FieldExpansion(b=b_out, hs="0", nphi=self.nphi)
        
        self.A_straight_in = self.straight_in.get_A(lambdify=True)
        self.A_straight_out = self.straight_out.get_A(lambdify=True)

        if not self.h==0:
            self.bent_in = FieldExpansion(b=b_in, hs=f"{self.h}", nphi=self.nphi)
            self.bent_out = FieldExpansion(b=b_out, hs=f"{self.h}", nphi=self.nphi)
        
            self.A_bent_in = self.bent_in.get_A(lambdify=True)
            self.A_bent_out = self.bent_out.get_A(lambdify=True)


    def get_Bfield(self, x, y, s):
        x = np.asarray(x)
        y = np.asarray(y)
        s = np.asarray(s)
    
        Bx_out = np.empty_like(s)
        By_out = np.empty_like(s)
        Bs_out = np.empty_like(s)
    
        # Regions
        mask1 = s < -self.sedge
        mask2 = (s >= -self.sedge) & (s < 0)
        mask3 = (s >= 0) & (s < self.sedge)
        mask4 = s >= self.sedge
    
        for mask, region in zip(
            [mask1, mask2, mask3, mask4],
            [self.straight_in, self.bent_in, self.bent_out, self.straight_out]
        ):
            if not np.any(mask):
                continue
            Bx_func, By_func, Bs_func = region.get_Bfield()
            Bx_out[mask] = Bx_func(x[mask], y[mask], s[mask])
            By_out[mask] = By_func(x[mask], y[mask], s[mask])
            Bs_out[mask] = Bs_func(x[mask], y[mask], s[mask])
    
        return Bx_out, By_out, Bs_out


    def create_Hamiltonian(self, plot=False, symmetric=True):
        self.create_fieldexpansion(plot=plot, symmetric=symmetric)
        
        print("Creating Hamiltonian...")
        # Make a Hamiltonian
        if self.h==0: 
            self.H_straight_in = Hamiltonian(-self.smin, 0, self.straight_in, s_start=self.smin)
            self.H_straight_out = Hamiltonian(self.smax, 0, self.straight_out, s_start=0)

        else:
            self.H_straight_in = Hamiltonian(self.smax-self.sedge, 0, self.straight_in, s_start=self.smin)
            self.H_bent_in = Hamiltonian(self.sedge, self.h, self.bent_in, s_start=-self.sedge)
            self.H_bent_out = Hamiltonian(self.sedge, self.h, self.bent_out, s_start=0)
            self.H_straight_out = Hamiltonian(self.smax-self.sedge, 0, self.straight_out, s_start=self.sedge)


    def track(self, particle, ivp_opt={}, plot=False):
        ivp_opt = {"rtol": 1e-12, "atol": 1e-12}

        if self.h==0:
            self.H_straight_in.track(particle, ivp_opt=ivp_opt) 
            self.h_straight_out.track(particle, ivp_opt=ivp_opt)
        
        else:
            # Incoming straight part
            sol_sin = self.H_straight_in.track(particle, ivp_opt=ivp_opt, return_sol=True) 

            # Incoming bent part
            sol_bin = self.H_bent_in.track(particle, ivp_opt=ivp_opt, return_sol=True)

            # Outgoing bent part
            sol_bout = self.H_bent_out.track(particle, ivp_opt=ivp_opt, return_sol=True)

            # Outgoing straight part
            sol_sout = self.H_straight_out.track(particle, ivp_opt=ivp_opt, return_sol=True)
        
            if plot:
                fig, ax = plt.subplots(2, sharex=True)
                for i in range(len(particle.x)):
                    tlist = np.concatenate([sol_sin[i].t, sol_bin[i].t, sol_bout[i].t, sol_sout[i].t])
                    xlist = np.concatenate([sol_sin[i].y[0], sol_bin[i].y[0], sol_bout[i].y[0], sol_sout[i].y[0]])
                    ylist = np.concatenate([sol_sin[i].y[1], sol_bin[i].y[1], sol_bout[i].y[1], sol_sout[i].y[1]])
                    ax[0].scatter(tlist, xlist, marker='.')
                    ax[1].scatter(tlist, ylist, marker='.')
                ax[0].set_xlabel("s")
                ax[1].set_xlabel("s")
                ax[0].set_ylabel("x")
                ax[1].set_ylabel("y")

                # canvas = CanvasZX()
                # fr = Frame()
                # fr2 = fr.copy().arc_by(self.sedge, -self.angle/2)
                # fr3 = fr2.copy().move_by([(self.smin+self.sedge)*np.sin(self.angle/2), 0, (self.smin + self.sedge)*np.cos(self.angle/2)]).plot_zx(canvas=canvas)
                # fb = BendFrame(fr2, self.l_magn, self.angle).plot_zx(canvas=canvas)
                # fr4 = fb.end
                # fr5 = fr4.copy().move_by([-(self.smax-self.sedge)*np.sin(self.angle/2), 0, (self.smax - self.sedge)*np.cos(self.angle/2)]).plot_zx(canvas=canvas)
                # for i in range(len(particle.x)):
                #     fr3.plot_trajectory_zx(sol_sin[i].t-self.smin, sol_sin[i].y[0], sol_sin[i].y[1], canvas=canvas)
                #     fb.plot_trajectory_zx(sol_bin[i].t+self.sedge, sol_bin[i].y[0], sol_bin[i].y[1], canvas=canvas)
                #     fb.plot_trajectory_zx(sol_bout[i].t+self.sedge, sol_bout[i].y[0], sol_bout[i].y[1], canvas=canvas)
                #     fr4.plot_trajectory_zx(sol_sout[i].t-self.sedge, sol_sout[i].y[0], sol_sout[i].y[1], canvas=canvas)

                  
class MagnetFromFieldmap:
    isthick=True
    def __init__(self, data, h, l_magn, design_field, order=3, hgap=0.05, apt=0.05, radius=0.05, nphi=4, plot=False, step=50, in_FS_coord=False, symmetric=False):
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
        self.smin = -(7.5*hgap + l_magn/2)
        self.smax = 7.5*hgap + l_magn/2 
        self.sedge = l_magn/2

        self.length = self.smax - self.smin

        self.nphi = nphi

        self.step = step
                
        assert self.h != 0, "This method is only implemented for curved magnets with h!=0"
        if not h==0:
            self.rho = 1/h
            self.phi = l_magn*h

            xFS = np.linspace(self.xmin, self.xmax, 31)
            yFS = [0]
            ns = math.ceil(((self.smax - self.smin) / 0.001) / step) * step + 2  # I didn't figure out why but like this the length is correct...
            sFS = np.linspace(self.smin, self.smax, ns) 

            if not in_FS_coord:
                self.fieldmap = self.fieldmap.calc_FS_coords(xFS, yFS, sFS, self.rho, self.phi, radius=radius)
            
                scalefactor = self.design_field / (self.fieldmap.integratedfield(3)[0] / self.l_magn)
                self.fieldmap.rescale(scalefactor)

        if symmetric:
            self.fieldmap = self.fieldmap.symmetrize(radius=radius)
            
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


    def track(self, particles, ivp_opt={}, plot=False):
        ivp_opt = {"rtol": 1e-12, "atol": 1e-12}
        
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



            
       
       
       
        

class FringeFromFint:
    def __init__(self, b0, fint, center=-0.5, gap=0.05):
        """
        :param b0: Central field of the dipole magnet.
        :param fint: Desired fringe field integral
        :param sedge: The edge of the magnet, such that the integrated field will have the correct value
        :param gap: The gap of the magnet. Used to estimate a reasonable range of the fringe field and calculate the fringe field Integral.
        """

        self.b0 = b0
        self.gap = gap
        self.fint = fint
        
        self.center = center
        
        self.smin = -5*self.gap
        self.smax = 5*self.gap

        self.ELENA_enge_params = [self.b0, -2.19581377e+02, 8.88031231e+02, -1.46018510e+02, 2.61985464e+01, 5.35675107e-01]  # Exit fringe
        assert self.smax < 3.5, "ELENA fringe field parameters are not suited for this choice"
        # Remains at zero until around s=3.5, then it increases again. I cannot guarantee correct behaviour in this case!

        self.scalefactor = self.fit_fint()
        self.sedge = self.calc_edge()
        
    def fit_fint(self):
        ss = np.linspace(self.smin, self.smax, 200)
        By = Enge(ss, *self.ELENA_enge_params)
        ELENA_fint = np.trapz(By*(self.b0-By) / (self.b0**2 * self.gap), ss)
        scalefactor = self.fint / ELENA_fint

        By = Enge(ss/scalefactor, *self.ELENA_enge_params)
        new_fint = np.trapz(By*(self.b0-By) / (self.b0**2 * self.gap), ss)
        assert self.smax/scalefactor < 3.5, "ELENA fringe field parameters are not suited for this choice"
        return scalefactor

    def calc_edge(self):
        ss = np.linspace(self.center, self.smax, 500)
        By = Enge(ss/self.scalefactor, *self.ELENA_enge_params)
        By_int = np.trapz(By, ss)
        sedge = (-self.b0*self.center - By_int) / self.b0
        return sedge

    def plot_fringe(self, ax=None):
        ss = np.linspace(self.smin, self.smax, 200)
        By = Enge(ss/self.scalefactor, *self.ELENA_enge_params)
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(ss, By, label="Fringe field")
        ax.plot(ss, np.heaviside(-ss, self.sedge), label="Hard edge")
        ax.set_xlabel("x")
        ax.set_ylabel("By")
    
    def exitfringevalue(self, s, lambdify=False):
        """
        Returns the fringe field value at s, shifted such that the hard edge is at zero.
        :param x: Position in the fringe field region
        """
        
        return get_derivative(0, 6, spEnge, lambdify=lambdify)((s-self.sedge)/self.scalefactor, *self.ELENA_enge_params)
        
    def entrancefringevalue(self, s, lambdify=False):
        """
        Returns the fringe field value at s, shifted such that the hard edge is at zero.
        :param x: Position in the fringe field region
        """
        
        return self.exitfringevalue(-s, lambdify=lambdify)


class DipoleFromFint:
    def __init__(self, b0, h, l_magn, fint, gap=0.05, nphi=5):
        """
        :param b0: Central field of the dipole magnet.
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, Frenet-Serrat coordinates consisting of a straight part before the magnetic length, 
            then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
        :param fint: Desired fringe field integral.
        :param gap: The gap of the magnet. Used to estimate a reasonable range of the fringe field and calculate the fringe field Integral.
        :param nphi: Number of phi terms in the expansion, at least phi0 and phi1. If set to zero, use DipoleVectorPotential object 
            which doesn't include any field derivatives.
        """
        
        self.b0 = b0
        self.h = h
        self.l_magn = l_magn
        self.angle = self.h*self.l_magn
        self.fint = fint
        self.gap = gap
        self.nphi = nphi

        self.sedge = self.l_magn/2
        self.smin = -self.sedge - 5*self.gap
        self.smax = self.sedge + 5*self.gap

        self.fringe = FringeFromFint(b0, fint, gap=gap, center=-self.sedge)

        self.create_Hamiltonian()
            
    def create_fieldexpansion(self):
        print("Creating field expansion...")

        # Make a field expansion
        b_in = (self.fringe.entrancefringevalue(sp.symbols("s")+self.sedge), )
        b_out = (self.fringe.exitfringevalue(sp.symbols("s")-self.sedge), )
        
        if self.nphi>0:
            self.straight_in = FieldExpansion(b=b_in, hs="0", nphi=self.nphi)
            self.straight_out = FieldExpansion(b=b_out, hs="0", nphi=self.nphi)
        else:
            self.straight_in = DipoleVectorPotential(curv=0, b1=b_in[0])
            self.straight_out = DipoleVectorPotential(curv=0, b1=b_out[0])

        self.A_straight_in = self.straight_in.get_A(lambdify=True)
        self.A_straight_out = self.straight_out.get_A(lambdify=True)


        if not self.h==0:
            if self.nphi>0:
                self.bent_in = FieldExpansion(b=b_in, hs=f"{self.h}")
                self.bent_out = FieldExpansion(b=b_out, hs=f"{self.h}")
            else:
                self.bent_in = DipoleVectorPotential(curv=self.h, b1=b_in[0])
                self.bent_out = DipoleVectorPotential(curv=self.h, b1=b_out[0])

            self.A_bent_in = self.bent_in.get_A(lambdify=True)
            self.A_bent_out = self.bent_out.get_A(lambdify=True)

    def create_Hamiltonian(self, plot=False, symmetric=True):
        print("Creating Hamiltonian...")
        self.create_fieldexpansion()
        
        # Make a Hamiltonian
        if self.h==0: 
            self.H_straight_in = Hamiltonian(-self.smin, 0, self.straight_in)
            self.H_straight_out = Hamiltonian(self.smax, 0, self.straight_out)

        else:
            self.H_straight_in = Hamiltonian(self.smax-self.sedge, 0, self.straight_in, s_start=self.smin)
            self.H_bent_in = Hamiltonian(self.sedge, self.h, self.bent_in, s_start=-self.sedge)
            self.H_bent_out = Hamiltonian(self.sedge, self.h, self.bent_out, s_start=0)
            self.H_straight_out = Hamiltonian(self.smax-self.sedge, 0, self.straight_out, s_start=self.sedge)
        
    def plot_components(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if self.h==0:
            self.straight_in.plot_components(smin=self.smin, smax=0, plot_bs=False, ax=ax)
            self.straight_out.plot_components(smin=0, smax=self.smax, plot_bs=False, ax=ax)
            integral = 0
            integral += np.trapz([self.straight_in.b[0].subs({self.straight_in.s:sval}) for sval in np.linspace(self.smin, 0, 300)], np.linspace(self.smin, 0, 300))
            integral += np.trapz([self.straight_out.b[0].subs({self.straight_out.s:sval}) for sval in np.linspace(0, self.smax, 300)], np.linspace(0, self.smax, 300))
            assert np.isclose(integral, self.b0*self.l_magn, rtol=1e-4), f"Integral of the field does not match the expected value! It is {integral} instead of {self.b0*self.l_magn}!"
        else:
            self.straight_in.plot_components(smin=self.smin, smax=-self.sedge, plot_bs=False, ax=ax)
            self.bent_in.plot_components(smin=-self.sedge, smax=0, plot_bs=False, ax=ax)
            self.bent_out.plot_components(smin=0, smax=self.sedge, plot_bs=False, ax=ax)
            self.straight_out.plot_components(smin=self.sedge, smax=self.smax, plot_bs=False, ax=ax)
            integral = 0
            integral += np.trapz([self.straight_in.b[0].subs({self.straight_in.s:sval}) for sval in np.linspace(self.smin, -self.sedge, 300)], np.linspace(self.smin, -self.sedge, 300))
            integral += np.trapz([self.bent_in.b[0].subs({self.bent_in.s:sval}) for sval in np.linspace(-self.sedge, 0, 300)], np.linspace(-self.sedge, 0, 300))
            integral += np.trapz([self.bent_out.b[0].subs({self.bent_out.s:sval}) for sval in np.linspace(0, self.sedge, 300)], np.linspace(0, self.sedge, 300))
            integral += np.trapz([self.straight_out.b[0].subs({self.straight_out.s:sval}) for sval in np.linspace(self.sedge, self.smax, 300)], np.linspace(self.sedge, self.smax, 300))
            assert np.isclose(integral, self.b0*self.l_magn, rtol=1e-4), f"Integral of the field does not match the expected value! It is {integral} instead of {self.b0*self.l_magn}!"
    
    def track(self, particle, ivp_opt={}, amplify_plot=1):
        if self.h==0:
            sol_in = self.H_straight_in.track(particle, s_span=[self.smin, 0], ivp_opt=ivp_opt) 
            
            sol_out = self.H_straight_out.track(particle, s_span=[0, self.smax], ivp_opt=ivp_opt)
            
            fig, ax = plt.subplots(2, sharex=True)
            for i in range(len(particle.x)):
                tlist = np.concatenate([sol_in[i].t, sol_out[i].t])
                xlist = np.concatenate([sol_in[i].y[0], sol_out[i].y[0]])
                ylist = np.concatenate([sol_in[i].y[1], sol_out[i].y[1]])
                ax[0].scatter(tlist, xlist)
                ax[1].scatter(tlist, ylist)
            ax[1].set_xlabel("s")
            ax[0].set_ylabel("x")
            ax[1].set_ylabel("y")
            plt.show()
            
            
        else:
            sol_sin = self.H_straight_in.track(particle, ivp_opt=ivp_opt, return_sol=True) 

            sol_bin = self.H_bent_in.track(particle, ivp_opt=ivp_opt, return_sol=True)

            sol_bout = self.H_bent_out.track(particle, ivp_opt=ivp_opt, return_sol=True)

            sol_sout = self.H_straight_out.track(particle, ivp_opt=ivp_opt, return_sol=True)

            fig, ax = plt.subplots(2, sharex=True)
            for i in range(len(particle.x)):
                slist = np.concatenate([sol_sin[i].t, sol_bin[i].t, sol_bout[i].t, sol_sout[i].t])
                xlist = np.concatenate([sol_sin[i].y[0], sol_bin[i].y[0], sol_bout[i].y[0], sol_sout[i].y[0]])
                ylist = np.concatenate([sol_sin[i].y[1], sol_bin[i].y[1], sol_bout[i].y[1], sol_sout[i].y[1]])
                ax[0].scatter(slist, xlist)
                ax[1].scatter(slist, ylist)
            ax[1].set_xlabel("s")
            ax[0].set_ylabel("x")
            ax[1].set_ylabel("y")
            plt.show()

            canvas = CanvasZX()
            fr = Frame()
            fr2 = fr.copy().arc_by(self.sedge, -self.angle/2)
            fr3 = fr2.copy().move_by([(self.smin+self.sedge)*np.sin(self.angle/2), 0, (self.smin + self.sedge)*np.cos(self.angle/2)]).plot_zx(canvas=canvas)
            fb = BendFrame(fr2, self.l_magn, self.angle).plot_zx(canvas=canvas)
            fr4 = fb.end
            fr5 = fr4.copy().move_by([-(self.smax-self.sedge)*np.sin(self.angle/2), 0, (self.smax - self.sedge)*np.cos(self.angle/2)]).plot_zx(canvas=canvas)
            for i in range(len(particle.x)):
                fr3.plot_trajectory_zx(sol_sin[i].t-self.smin, sol_sin[i].y[0]*amplify_plot, sol_sin[i].y[1], canvas=canvas)
                fb.plot_trajectory_zx(sol_bin[i].t+self.sedge, sol_bin[i].y[0]*amplify_plot, sol_bin[i].y[1], canvas=canvas)
                fb.plot_trajectory_zx(sol_bout[i].t+self.sedge, sol_bout[i].y[0]*amplify_plot, sol_bout[i].y[1], canvas=canvas)
                fr4.plot_trajectory_zx(sol_sout[i].t-self.sedge, sol_sout[i].y[0]*amplify_plot, sol_sout[i].y[1], canvas=canvas)
        

