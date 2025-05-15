from .fieldmaps import Fieldmap, get_derivative, spTanh, spEnge
from .generate_expansion import FieldExpansion
from .numerical_solver import Hamiltonian
import numpy as np
import sympy as sp
import warnings

# A dipole element for XSuite with explicit solving created from a fieldmap


class DipoleFromFieldmap:
    def __init__(self, data, h, l_magn, shape, order=3, degree=5, hgap=0.05, apt=0.05, radius=0.01):
        """
        :param data: Fieldmap points as columns x, y, z, Bx, By, Bz.
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, the fieldmap is first converted in Frenet-Serrat coordinates consisting of a straight part before the 
            magnetic length, then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
        :param shape: string specifying the fringe field shape.
            options: "tanh", "enge"
        :param order: Highest order multipole (dipole=1)
        :param degree: Degree of the polynomial in the Enge function and its derivatives used to fit the multipole shapes.
        :param hgap (optional): Half the gap of the magnet. Only used to estimate a reasonable range of the fringe field.
        :param apt (optional): Total aperture of the magnet (over which the fieldmap should also be given, otherwise give maximal 
            range of the fieldmap). Used to fit the multipoles.
        :param radius: Interpolation radius for determination of field in Frenet-Serrat coordinates.
        """

        self.data = data
        self.shape = shape

        self.h = h 
        self.l_magn = l_magn
        self.order = order
        self.degree = degree
        self.fieldmap = Fieldmap(data)

        self.xmin = -apt/2
        self.xmax = apt/2
        self.smin = -(3.8*hgap + l_magn/2)
        self.smax = 3.8*hgap + l_magn/2
        self.sedge = l_magn/2
        
        if not h==0:
            self.rho = 1/h
            self.phi = l_magn*h

            xFS = np.linspace(self.xmin, self.xmax, 31)
            yFS = [0]
            sFS = np.linspace(self.smin, self.smax, 201)

            self.fieldmap = self.fieldmap.calc_FS_coords(xFS, yFS, sFS, self.rho, self.phi, radius=radius)
            
        self.create_Hamiltonian()

    def create_Hamiltonian(self):
        # Calculate the dipole, quadrupole, sextupole component for the two halves
        # BE CAREFUL: The parameters are given for the edge at zero

        print("shape:", self.shape)
        assert self.shape in ["tanh", "enge"]
        
        if self.shape == "tanh":
            shape = spTanh
            guess = [np.max(self.fieldmap.src["By"][(self.fieldmap.src["x"]==0) & (self.fieldmap.src["y"]==0)]), 0.1]
            self.degree = 1
        elif self.shape == "enge":
            shape = spEnge
            guess = [np.max(self.fieldmap.src["By"][(self.fieldmap.src["x"]==0) & (self.fieldmap.src["y"]==0)]), 0, 0, 0, 0, 0]

        params_in_list, cov_in_list = self.fieldmap.fit_multipoles(shape, guess = guess, components=np.arange(1,self.order+1,1), design=1, 
                                                                   degree=self.degree, zmin=self.smin, zmax=0, zedge=-self.sedge)
        guess[1] = -guess[1]
        params_out_list, cov_out_list = self.fieldmap.fit_multipoles(shape, guess = guess, components=np.arange(1,self.order+1,1), design=1, 
                                                                     degree=self.degree, zmin=0, zmax=self.smax, zedge=self.sedge)

        # Make a field expansion
        b_in = []
        b_out = []
        s = sp.symbols("s")
        for i in range(self.order):
            b_in.append(get_derivative(i, poly_order=5, func=shape, lambdify=False)(s, *params_in_list[i]))
            b_out.append(get_derivative(i, poly_order=5, func=shape, lambdify=False)(s, *params_out_list[i]))
        
        straight_in = FieldExpansion(b=b_in, hs="0")
        straight_out = FieldExpansion(b=b_in, hs="0")

        if not self.h==0:
            bent_in = FieldExpansion(b=b_in, hs=f"{self.h}")
            bent_out = FieldExpansion(b=b_out, hs=f"{self.h}")

        # Make a Hamiltonian
        if self.h==0: 
            self.h_straight_in = Hamiltonian(-self.smin, 0, straight_in)
            self.h_straight_out = Hamiltonian(self.smax, 0, straight_out)

        else:
            self.h_straight_in = Hamiltonian(self.smax-self.sedge, 0, straight_in)
            self.h_bent_in = Hamiltonian(self.sedge, self.h, bent_in)
            self.h_bent_out = Hamiltonian(self.sedge, self.h, bent_out)
            self.h_straight_out = Hamiltonian(self.smax-self.sedge, 0, straight_out)

    
    def track(self, particle, ivp_opt={}):
        # NEED TO CORRECT FOR DISCONTINUOUS VECTOR POTENTIALS!!!
        warnings.warn("Not correct for discontinuous vector potentials. This will not give the correct physics - to be implemented.")
        
        if self.h==0:
            self.h_straight_in.track(particle, s_span=[-self.smax, 0], ivp_opt=ivp_opt) 
            self.h_straight_out.track(particle, s_span=[0, self.smax], ivp_opt=ivp_opt)
        
        else:
            self.h_straight_in.track(particle, s_span=[-self.smax+self.sedge, 0], ivp_opt=ivp_opt) 
            self.h_bent_in.track(particle, s_span=[0, self.sedge], ivp_opt=ivp_opt)
            self.h_bent_out.track(particle, s_span=[-self.sedge, 0], ivp_opt=ivp_opt)
            self.h_straight_out.track(particle, s_span=[0, self.smax-self.sedge], ivp_opt=ivp_opt)



        






