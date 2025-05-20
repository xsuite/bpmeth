from .fieldmaps import Fieldmap, get_derivative, spTanh, spEnge
from .generate_expansion import FieldExpansion
from .numerical_solver import Hamiltonian
import numpy as np
import sympy as sp
import warnings
import matplotlib.pyplot as plt

# A dipole element for XSuite with explicit solving created from a fieldmap


class DipoleFromFieldmap:
    def __init__(self, data, h, l_magn, shape, Brho=1, order=3, degree=5, hgap=0.05, apt=0.05, radius=0.01):
        """
        :param data: Fieldmap points as columns x, y, z, Bx, By, Bz. Data in Tesla
        :param h: Curvature of the frame.
            If h==0, the magnet is treated as a straight magnet with magnetic length l_magn
            If h!=0, the fieldmap is first converted in Frenet-Serrat coordinates consisting of a straight part before the 
            magnetic length, then a bent part with bending radius 1/h, then a straight part after the magnetic length.
        :param l_magn: Total magnetic length. For curved magnets, the bending angle is given by l_magn*h.
        :param shape: string specifying the fringe field shape.
            options: "tanh", "enge"
        :param Brho (optional): Beam rigidity to normalize the field. If not specified, the field is given in Tesla.
        :param order (optional): Highest order multipole (dipole=1)
        :param degree (optional): Degree of the polynomial in the Enge function and its derivatives used to fit the multipole shapes.
        :param hgap (optional): Half the gap of the magnet. Only used to estimate a reasonable range of the fringe field.
        :param apt (optional): Total aperture of the magnet (over which the fieldmap should also be given, otherwise give maximal 
            range of the fieldmap). Used to fit the multipoles.
        :param radius (optional): Interpolation radius for determination of field in Frenet-Serrat coordinates.
        """

        self.data = data

        self.h = h 
        self.l_magn = l_magn
        self.shape = shape

        self.Brho = Brho

        self.order = order
        self.degree = degree
        self.fieldmap = Fieldmap(data)

        self.xmin = -apt/2
        self.xmax = apt/2
        self.smin = -(3.8*hgap + l_magn/2)
        self.smax = 3.8*hgap + l_magn/2
        self.sedge = l_magn/2

        self.length = self.smax - self.smin
        
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

        params_in_list, cov_in_list = self.fieldmap.fit_multipoles(shape, guess=guess, components=np.arange(1,self.order+1,1), design=1, 
                                                                   degree=self.degree, zmin=self.smin, zmax=0, zedge=-self.sedge)
        guess[1] = -guess[1]
        params_out_list, cov_out_list = self.fieldmap.fit_multipoles(shape, guess=guess, components=np.arange(1,self.order+1,1), design=1, 
                                                                     degree=self.degree, zmin=0, zmax=self.smax, zedge=self.sedge)
        
        
        # Typically the first parameter is the amplitude.
        params_in_list[:, 0] = params_in_list[:, 0] / self.Brho
        params_out_list[:, 0] = params_out_list[:, 0] / self.Brho

        print("Creating field expansion...")

        # Make a field expansion
        b_in = []
        b_out = []
        s = sp.symbols("s")
        for i in range(self.order):
            b_in.append(get_derivative(i, poly_order=5, func=shape, lambdify=False)(s, *params_in_list[i]))
            b_out.append(get_derivative(i, poly_order=5, func=shape, lambdify=False)(s, *params_out_list[i]))
        
        self.straight_in = FieldExpansion(b=b_in, hs="0")
        self.straight_out = FieldExpansion(b=b_in, hs="0")
        
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
        # NEED TO CORRECT FOR DISCONTINUOUS VECTOR POTENTIALS!!!
        warnings.warn("Not correct for discontinuous vector potentials. This will not give the correct physics - to be implemented.")
        
        if self.h==0:
            self.H_straight_in.track(particle, s_span=[-self.smax, 0], ivp_opt=ivp_opt) 
            # Correct for discontinuity in vector potential, velocity continuous!
            #particle.px = particle.px - self.A_straight_in[0](particle.x, particle.y, particle.s) + self.A_straight_out[0](particle.x, particle.y, particle.s)
            #particle.py = particle.py - self.A_straight_in[1](particle.x, particle.y, particle.s) + self.A_straight_out[1](particle.x, particle.y, particle.s)

            self.h_straight_out.track(particle, s_span=[0, self.smax], ivp_opt=ivp_opt)
        
        else:
            fig, ax = plt.subplots()
            sol = self.H_straight_in.track(particle, s_span=[-self.smax+self.sedge, 0], ivp_opt=ivp_opt, return_sol=True) 
            particle.px = particle.px - self.A_straight_in[0](particle.x, particle.y, particle.s) + self.A_bent_in[0](particle.x, particle.y, particle.s)
            particle.py = particle.py - self.A_straight_in[1](particle.x, particle.y, particle.s) + self.A_bent_in[1](particle.x, particle.y, particle.s)
            print("Ax straight in:", self.A_straight_in[0](particle.x, particle.y, particle.s))
            print("Ax bent in:", self.A_bent_in[0](particle.x, particle.y, particle.s))
            print("Ay straight in:", self.A_straight_in[1](particle.x, particle.y, particle.s))
            print("Ay bent in:", self.A_bent_in[1](particle.x, particle.y, particle.s))
            ax.scatter(sol[0].t-self.sedge, sol[0].y[0], color="red")

            sol = self.H_bent_in.track(particle, s_span=[0, self.sedge], ivp_opt=ivp_opt, return_sol=True)
            particle.px = particle.px - self.A_bent_in[0](particle.x, particle.y, particle.s) + self.A_bent_out[0](particle.x, particle.y, particle.s)
            particle.py = particle.py - self.A_bent_in[1](particle.x, particle.y, particle.s) + self.A_bent_out[1](particle.x, particle.y, particle.s)
            print("Ax bent in:", self.A_bent_in[0](particle.x, particle.y, particle.s))
            print("Ax bent out:", self.A_bent_out[0](particle.x, particle.y, particle.s))
            print("Ay bent in:", self.A_bent_in[1](particle.x, particle.y, particle.s))
            print("Ay bent out:", self.A_bent_out[1](particle.x, particle.y, particle.s))
            ax.scatter(sol[0].t-self.sedge, sol[0].y[0], color="orange")


            sol = self.H_bent_out.track(particle, s_span=[-self.sedge, 0], ivp_opt=ivp_opt, return_sol=True)
            particle.px = particle.px - self.A_bent_out[0](particle.x, particle.y, particle.s) + self.A_straight_out[0](particle.x, particle.y, particle.s)
            particle.py = particle.py - self.A_bent_out[1](particle.x, particle.y, particle.s) + self.A_straight_out[1](particle.x, particle.y, particle.s)
            print("Ax bent out:", self.A_bent_out[0](particle.x, particle.y, particle.s))
            print("Ax straight out:", self.A_straight_out[0](particle.x, particle.y, particle.s))
            print("Ay bent out:", self.A_bent_out[1](particle.x, particle.y, particle.s))
            print("Ay straight out:", self.A_straight_out[1](particle.x, particle.y, particle.s))
            ax.scatter(sol[0].t+self.sedge, sol[0].y[0], color="yellow")
            
            sol = self.H_straight_out.track(particle, s_span=[0, self.smax-self.sedge], ivp_opt=ivp_opt, return_sol=True)
            ax.scatter(sol[0].t-self.sedge, sol[0].y[0], color="green")
            plt.show()




        






