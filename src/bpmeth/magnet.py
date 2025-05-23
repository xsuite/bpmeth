from .fieldmaps import Fieldmap, get_derivative, spTanh, spEnge
from .generate_expansion import FieldExpansion
from .numerical_solver import Hamiltonian
import numpy as np
import sympy as sp
import warnings
import matplotlib.pyplot as plt

# A dipole element for XSuite with explicit solving created from a fieldmap


class DipoleFromFieldmap:
    def __init__(self, data, h, l_magn, order=3, nparams=6, hgap=0.05, apt=0.05, radius=0.05, shape="enge", plot=False, symmetric=True, scalefactor=1):
        """
        :param data: Fieldmap points as columns x, y, z, Bx, By, Bz. Data in Tesla
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, the fieldmap is first converted in Frenet-Serrat coordinates consisting of a straight part before the 
            magnetic length, then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
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
        :param scalefactor: If you want to rescale the dipole field to ex. match the closed orbit
        """

        self.data = data

        self.h = h 
        self.l_magn = l_magn

        self.nparams = nparams
        self.order=order
        self.fieldmap = Fieldmap(data)

        self.xmin = -apt/2
        self.xmax = apt/2
        self.smin = -(3.8*hgap + l_magn/2)
        self.smax = 3.8*hgap + l_magn/2
        self.sedge = l_magn/2
        self.shape = shape

        self.length = self.smax - self.smin
        self.scalefactor = scalefactor
        
        if not h==0:
            self.rho = 1/h
            self.phi = l_magn*h

            xFS = np.linspace(self.xmin, self.xmax, 31)
            yFS = [0]
            sFS = np.linspace(-0.75*l_magn, 0.75*l_magn, 201)

            self.fieldmap = self.fieldmap.calc_FS_coords(xFS, yFS, sFS, self.rho, self.phi, radius=radius)
            
        self.create_Hamiltonian(plot=plot, symmetric=symmetric)

    def create_Hamiltonian(self, plot=False, symmetric=True):
        # Calculate the dipole, quadrupole, sextupole component for the two halves
        # BE CAREFUL: The parameters are given for the edge at zero

        if self.shape == "tanh":
            shape = spTanh
            self.degree = 1
          
        elif self.shape == "enge":
            shape = spEnge

        ax = None
        if plot:
            fig, ax = plt.subplots()

        params_out_list, cov_out_list = self.fieldmap.fit_multipoles(shape, components=np.arange(1,self.order+1,1), design=1, 
                                                                     nparams=self.nparams, zmin=0, zmax=self.smax, zedge=self.sedge, ax=ax)
        ss = np.linspace(0, self.smax, 100)

        integrated_field = np.trapz(get_derivative(0, self.nparams, func=shape, lambdify=True)(ss, *params_out_list[0]), ss)
        params_out_list[:, 0] = params_out_list[:, 0] / 2*integrated_field / self.rho * self.scalefactor
        
        if not symmetric:
            warnings.warn("Not correctly normalized for asymmetric magnets")
            params_in_list, cov_in_list = self.fieldmap.fit_multipoles(shape, components=np.arange(1,self.order+1,1), design=1, 
                                                                   nparams=self.nparams, zmin=self.smin, zmax=0, zedge=-self.sedge, ax=ax)
            params_in_list[:, 0] = params_in_list[:, 0] / self.Bfield / self.rho * self.scalefactor
        
        
        # Typically the first parameter is the amplitude.

        print("Creating field expansion...")

        # Make a field expansion
        b_in = []
        b_out = []
        s = sp.symbols("s")
        for i in range(self.order):
            if not symmetric:
                b_in.append(get_derivative(i, nparams=self.nparams, func=shape, lambdify=False)(s, *params_in_list[i]))
            else:
                b_in.append(get_derivative(i, nparams=self.nparams, func=shape, lambdify=False)(-s, *params_out_list[i]))
            b_out.append(get_derivative(i, nparams=self.nparams, func=shape, lambdify=False)(s, *params_out_list[i]))
        
        self.straight_in = FieldExpansion(b=b_in, hs="0")
        self.straight_out = FieldExpansion(b=b_out, hs="0")
        
        self.A_straight_in = self.straight_in.get_A(lambdify=True)
        self.A_straight_out = self.straight_out.get_A(lambdify=True)

        if not self.h==0:
            self.bent_in = FieldExpansion(b=b_in, hs=f"{self.h}")
            self.bent_out = FieldExpansion(b=b_out, hs=f"{self.h}")
        
            self.A_bent_in = self.bent_in.get_A(lambdify=True)
            self.A_bent_out = self.bent_out.get_A(lambdify=True)
        

        print("Creating Hamiltonian...")

        # Make a Hamiltonian
        if self.h==0: 
            self.H_straight_in = Hamiltonian(-self.smin, 0, straight_in)
            self.H_straight_out = Hamiltonian(self.smax, 0, straight_out)

        else:
            self.H_straight_in = Hamiltonian(self.smax-self.sedge, 0, self.straight_in)
            self.H_bent_in = Hamiltonian(self.sedge, self.h, self.bent_in)
            self.H_bent_out = Hamiltonian(self.sedge, self.h, self.bent_out)
            self.H_straight_out = Hamiltonian(self.smax-self.sedge, 0, self.straight_out)



    
    def track(self, particle, ivp_opt={}):
       
        if self.h==0:
            self.H_straight_in.track(particle, s_span=[-self.smax, 0], ivp_opt=ivp_opt) 
            # Correct for discontinuity in vector potential, velocity continuous!
            #particle.px = particle.px - self.A_straight_in[0](particle.x, particle.y, particle.s) + self.A_straight_out[0](particle.x, particle.y, particle.s)
            #particle.py = particle.py - self.A_straight_in[1](particle.x, particle.y, particle.s) + self.A_straight_out[1](particle.x, particle.y, particle.s)

            self.h_straight_out.track(particle, s_span=[0, self.smax], ivp_opt=ivp_opt)
        
        else:
            sol_sin = self.H_straight_in.track(particle, s_span=[-self.smax+self.sedge, 0], ivp_opt=ivp_opt, return_sol=True) 

            Ax_in = self.A_straight_in[0](particle.x, particle.y, particle.s)
            Ay_in = self.A_straight_in[1](particle.x, particle.y, particle.s)
            Ax_out = self.A_bent_in[0](particle.x, particle.y, particle.s)
            Ay_out = self.A_bent_in[1](particle.x, particle.y, particle.s)
            Ax_in = np.where(np.isnan(Ax_in), 0, Ax_in)
            Ay_in = np.where(np.isnan(Ay_in), 0, Ay_in)
            Ax_out = np.where(np.isnan(Ax_out), 0, Ax_out)
            Ay_out = np.where(np.isnan(Ay_out), 0, Ay_out)
            particle.px = particle.px - Ax_in + Ax_out
            particle.py = particle.py - Ay_in + Ay_out

            sol_bin = self.H_bent_in.track(particle, s_span=[0, self.sedge], ivp_opt=ivp_opt, return_sol=True)

            Ax_in = self.A_bent_in[0](particle.x, particle.y, particle.s)
            Ay_in = self.A_bent_in[1](particle.x, particle.y, particle.s)
            Ax_out = self.A_bent_out[0](particle.x, particle.y, particle.s)
            Ay_out = self.A_bent_out[1](particle.x, particle.y, particle.s)
            Ax_in = np.where(np.isnan(Ax_in), 0, Ax_in)
            Ay_in = np.where(np.isnan(Ay_in), 0, Ay_in)
            Ax_out = np.where(np.isnan(Ax_out), 0, Ax_out)
            Ay_out = np.where(np.isnan(Ay_out), 0, Ay_out)
            particle.px = particle.px - Ax_in + Ax_out
            particle.py = particle.py - Ay_in + Ay_out

            sol_bout = self.H_bent_out.track(particle, s_span=[-self.sedge, 0], ivp_opt=ivp_opt, return_sol=True)

            Ax_in = self.A_bent_out[0](particle.x, particle.y, particle.s)
            Ay_in = self.A_bent_out[1](particle.x, particle.y, particle.s)
            Ax_out = self.A_straight_out[0](particle.x, particle.y, particle.s)
            Ay_out = self.A_straight_out[1](particle.x, particle.y, particle.s)
            Ax_in = np.where(np.isnan(Ax_in), 0, Ax_in)
            Ay_in = np.where(np.isnan(Ay_in), 0, Ay_in)
            Ax_out = np.where(np.isnan(Ax_out), 0, Ax_out)
            Ay_out = np.where(np.isnan(Ay_out), 0, Ay_out)
            particle.px = particle.px - Ax_in + Ax_out 
            particle.py = particle.py - Ay_in + Ay_out
            
            sol_sout = self.H_straight_out.track(particle, s_span=[0, self.smax-self.sedge], ivp_opt=ivp_opt, return_sol=True)

            fig, ax = plt.subplots(2, sharex=True)
            for i in range(len(particle.x)):
                tlist = np.concatenate([sol_sin[i].t-self.sedge, sol_bin[i].t-self.sedge, sol_bout[i].t+self.sedge, sol_sout[i].t+self.sedge])
                xlist = np.concatenate([sol_sin[i].y[0], sol_bin[i].y[0], sol_bout[i].y[0], sol_sout[i].y[0]])
                ylist = np.concatenate([sol_sin[i].y[1], sol_bin[i].y[1], sol_bout[i].y[1], sol_sout[i].y[1]])
                ax[0].scatter(tlist, xlist)
                ax[1].scatter(tlist, ylist)
            ax[1].set_xlabel("s")
            ax[0].set_ylabel("x")
            ax[1].set_ylabel("y")
            plt.show()
        



        






