import numpy as np
import matplotlib.pyplot as plt
import bpmeth
import math
import sympy as sp
import warnings

class Phase4d:
    def __init__(self, phase_x, phase_y):
        self.phase_x = phase_x
        self.phase_y = phase_y

    def track(self, coord):
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]

        arg_x = 2*np.pi*self.phase_x
        cx = np.cos(arg_x)
        sx = np.sin(arg_x)
        x1 = x*cx + px*sx
        px1 = -x*sx + px*cx

        arg_y = 2*np.pi*self.phase_y
        cy = np.cos(arg_y)
        sy = np.sin(arg_y)
        y1 = y*cy + py*sy
        py1 = -y*sy + py*cy

        coord[0] = x1
        coord[1] = px1
        coord[2] = y1
        coord[3] = py1

    def __repr__(self):
        return f"Phase4d({self.phase_x}, {self.phase_y})"

class Kick_x:
    def __init__(self, kick, order=1):
        self.kick = kick
        self.order = order

    def track(self, coord):
        x = coord[0]
        coord[1] += self.kick*x**self.order/math.factorial(self.order)

    def __repr__(self):
        return f"Kick_x({self.kick})"

class Kick_y:
    def __init__(self, kick, order=1):
        self.kick = kick
        self.order = order

    def track(self, coord):
        y = coord[2]
        coord[3] += self.kick*y**self.order/math.factorial(self.order)

    def __repr__(self):
        return f"Kick_y({self.kick})"


class Sextupole:
    def __init__(self, b3):
        self.b3 = b3

    def track(self, coord):
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]

        coord[1] += self.b3/2 * x**2
        coord[3] += self.b3/2 * x*y


class NumericalSextupole:
    def __init__(self, b3, len):
        self.b3 = b3
        self.len = len

    def track(self, coord):
        ncoord,npart=coord.shape

        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]

        sextupole = bpmeth.GeneralVectorPotential(0, b=("0", "0", f"{self.b3}"))
        H_sextupole = bpmeth.Hamiltonian(self.len, 0, sextupole)
        
        for i in range(npart):
            sol = H_sextupole.solve([x[i], y[i], 0, px[i], py[i], 0], s_span=[0, self.len])

            coord[0, i] = sol.y[0][-1]
            coord[1, i] = sol.y[3][-1]
            coord[2, i] = sol.y[1][-1]
            coord[3, i] = sol.y[4][-1]


class ThinNumericalFringe:   
    def __init__(self,b1,shape,len=1,nphi=5):
        """    
        Thick numerical fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards bend.
        
        :param b1: Amplitude of the dipole field
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" normalized between zero and one
        :param len: Integration range
        :param nphi: Number of terms included in expansion
        """
        
        self.b1 = b1
        self.shape = shape        
        self.b1fringe = f"{b1}*{shape}"
        self.len = len
        self.nphi = nphi      
        
        # First a backwards drift
        self.drift = bpmeth.DriftVectorPotential()
        self.H_drift = bpmeth.Hamiltonian(self.len/2, 0, self.drift)

        # Then a fringe field map
        self.fringe = bpmeth.FringeVectorPotential(self.b1fringe, nphi=0) # Only b1 and b1'
        self.H_fringe = bpmeth.Hamiltonian(self.len, 0, self.fringe)

        # Then a backwards bend
        self.dipole = bpmeth.DipoleVectorPotential(0, self.b1)
        self.H_dipole = bpmeth.Hamiltonian(self.len/2, 0, self.dipole)

    
    def track(self, coord):
        """
        :param coord: List of coordinates [x, px, y, py], 
                      with x = coord[0] etc lists of coordinates for all N particles.
        :return: A list of trajectory elements for all particles
        """
        ncoord,npart=coord.shape
        
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]

        trajectories = []
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
                        
            sol_drift = self.H_drift.solve(qp0, s_span=[0, -self.len/2])
            sol_fringe = self.H_fringe.solve(sol_drift.y[:, -1], s_span=[-self.len/2, self.len/2])
            sol_dipole = self.H_dipole.solve(sol_fringe.y[:, -1], s_span=[self.len/2, 0])
            
            coord[0, i] = sol_dipole.y[0][-1]
            coord[1, i] = sol_dipole.y[3][-1]
            coord[2, i] = sol_dipole.y[1][-1]
            coord[3, i] = sol_dipole.y[4][-1]
        
            all_s = np.append(sol_drift.t, [sol_fringe.t, sol_dipole.t])
            all_x = np.append(sol_drift.y[0], [sol_fringe.y[0], sol_dipole.y[0]])
            all_y = np.append(sol_drift.y[1], [sol_fringe.y[1], sol_dipole.y[1]])
            all_px = np.append(sol_drift.y[3], [sol_fringe.y[3], sol_dipole.y[3]])
            all_py = np.append(sol_drift.y[4], [sol_fringe.y[4], sol_dipole.y[4]])
        
            trajectories.append(Trajectory(all_s, all_x, all_px, all_y, all_py))
            
        return trajectories


class ThickNumericalFringe:
    def __init__(self,b1,shape,len=1,nphi=5):
        """    
        Thick numerical fringe field at the edge of the magnet. 
        It consists of a backwards drift, a fringe field map and a backwards bend.
        
        :param b1: Amplitude of the dipole field
        "param shape: The shape of the fringe field as a string, example "(tanh(s/0.1)+1)/2" normalized between zero and one
        :param len: Integration range
        :param nphi: Number of terms included in expansion
        """
        
        self.b1 = b1
        self.shape = shape
        self.b1fringe = f"{b1}*{shape}"
        self.len = len
        self.nphi = nphi
        
        # Fringe up to the dipole edge
        self.fringe1 = bpmeth.FringeVectorPotential(self.b1fringe, nphi=5)
        self.H_fringe1 = bpmeth.Hamiltonian(len/2, 0, self.fringe1)

        # Fringe inside dipole
        self.fringe2 = bpmeth.FringeVectorPotential(f"{self.b1fringe} - {self.b1}", nphi=5)
        self.H_fringe2 = bpmeth.Hamiltonian(len/2, 0, self.fringe2)
        
    def track(self, coord):
        """
        :param coord: List of coordinates [x, px, y, py], 
                      with x = coord[0] etc lists of coordinates for all N particles.
        :return: A list of trajectory elements for all particles
        """
        
        ncoord,npart=coord.shape
        
        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]
        
        p_sp = bpmeth.SympyParticle()
        trajectories = []
        for i in range(npart):
            qp0 = [x[i], y[i], 0, px[i], py[i], 0]
            print("qp0:", qp0)
            
            sol_fringe1 = self.H_fringe1.solve(qp0, s_span=[-self.len/2, 0])
            
            # Changing the canonical momenta: 
            # x' and y' are continuous in changing vector potential, px and py are not! 
            # Usually this is not an issue as a_x, a_y are typically zero in the regions between maps
            xval, yval, tauval, pxval, pyval, ptauval = sol_fringe1.y[:, -1]

            ax1 = self.fringe1.get_A(p_sp)[0].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()
            ax2 = self.fringe2.get_A(p_sp)[0].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()

            ay1 = self.fringe1.get_A(p_sp)[1].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()
            ay2 = self.fringe2.get_A(p_sp)[1].subs({p_sp.s:0, p_sp.x:xval, p_sp.y:yval}).evalf()
            print("Vector potentials:", ax1, ax2, ay1, ay2)

            # assert (h==0), "h is not zero, this is not implemented yet"
            pxval += (ax2 - ax1)
            pyval += (ay2 - ay1)
            qp = [xval, yval, tauval, pxval, pyval, ptauval]
            
            sol_fringe2 = self.H_fringe2.solve(qp, s_span=[0, self.len/2])

            coord[0, i] = sol_fringe2.y[0][-1]
            coord[1, i] = sol_fringe2.y[3][-1]
            coord[2, i] = sol_fringe2.y[1][-1]
            coord[3, i] = sol_fringe2.y[4][-1]
            
            all_s = np.append(sol_fringe1.t, sol_fringe2.t)
            all_x = np.append(sol_fringe1.y[0], sol_fringe2.y[0])
            all_y = np.append(sol_fringe1.y[1], sol_fringe2.y[1])
            all_px = np.append(sol_fringe1.y[3], sol_fringe2.y[3])
            all_py = np.append(sol_fringe1.y[4], sol_fringe2.y[4])
            
            trajectories.append(Trajectory(all_s, all_x, all_px, all_y, all_py))
        
        return trajectories
    
    

class ForestFringe:
    def __init__(self, b1, Kg):
        """
       :param b1: The design value for the magnet
       :param Kg: The fringe field integral multiplied with the gap height of the magnet
        """
        
        self.b1 = b1
        self.Kg = Kg
        x, y, px, py = sp.symbols("x y px py")
        pz = sp.sqrt(1 - px**2 - py**2)
        self.phi = self.b1*px/pz / (1+(py/pz)**2) - self.Kg*self.b1**2 * ((1-py**2)/pz**3 + (px/pz)**2*(1-px**2)/pz**3)
        self.dphidpx = self.phi.diff(px)
        self.dphidpy = self.phi.diff(py)
        self.x, self.y, self.px, self.py = x, y, px, py
        
        
    def track(self, coord):
        """
        No longitudinal coordinates yet, only 4d
        """
        
        ncoord,npart=coord.shape

        xi = coord[0]
        pxi = coord[1]
        yi = coord[2]
        pyi = coord[3]
        
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
        

class Trajectory:
    def __init__(self, s, x, px, y, py):
        self.s = s
        self.x = x
        self.px = px
        self.y = y
        self.py = py

    def plot_3d(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.plot3D(self.s, self.x, self.y)
        ax.set_xlabel('s')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        
    def plot_x(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.x, label=label)
        ax.set_xlabel('s')
        ax.set_ylabel('x')
        plt.legend()
    
    def plot_y(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.y, label=label)
        ax.set_xlabel('s')
        ax.set_ylabel('y')
        plt.legend()
    
    
    

class Line4d:
    def __init__(self,elements):
        self.elements = elements

    def track(self, coord, num_turns=1):
        tcoord=coord.copy()
        ncoord,npart=coord.shape
        nelem=len(self.elements)
        output=np.zeros((num_turns,nelem,ncoord,npart))
        for nturn in range(num_turns):
            if nturn % 10 == 0:
                print(f"Turn {nturn}")
            for ielem,element in enumerate(self.elements):
                element.track(tcoord)
                output[nturn,ielem]=tcoord
        return Output4d(coord.copy(),output)


class Output4d:
    def __init__(self, init, output, cut=None):
        self.init = init
        self.output = output
        self.cut = cut

    def tbt(self):
        return np.r_[self.init,self.output[:,-1,:,:]]

    def x(self,part=0,elem=-1):
        return self.output[:,elem,0,part]

    def px(self,part=0,elem=-1):
        return self.output[:,elem,1,part]
    
    def y(self,part=0,elem=-1):
        return self.output[:,elem,2,part]
    
    def py(self,part=0,elem=-1):
        return self.output[:,elem,3,part]

    def hxp(self,part=0,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        return x+1j*px
    
    def hxm(self,part=0,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        return x-1j*px
    
    def hyp(self,part=0,elem=-1):
        y=self.y(part,elem)
        py=self.py(part,elem)
        return y+1j*py
    
    def hym(self,part=0,elem=-1):
        y=self.y(part,elem)
        py=self.py(part,elem)
        return y-1j*py

    def plot_xpx(self,elem=0,ax=None,xlims=None,ylims=None,savepath=None):
        if ax is None:
            fig = plt.figure()
            ax=fig.add_subplot(aspect='equal')
        x=self.output[:,elem,0,:]
        px=self.output[:,elem,1,:]
        if self.cut is not None:
            cut=np.all((abs(x)<self.cut),axis=0)
            ax.plot(x[:,cut],px[:,cut],'.')
        else:
            ax.plot(x,px,'.')
        ax.set_xlabel('x')
        ax.set_ylabel('px')
        if not xlims is None:
            ax.set_xlim(xlims)
        if not ylims is None:
            ax.set_ylim(ylims)
        if not savepath is None:
            plt.savefig(savepath,dpi=300)
            plt.close()

    def plot_ypy(self,elem=0,ax=None,xlims=None,ylims=None,savepath=None):
        if ax is None:
            fig = plt.figure()
            ax=fig.add_subplot(aspect='equal')
        y=self.output[:,elem,2,:]
        py=self.output[:,elem,3,:]
        if self.cut is not None:
            cut=np.all((abs(y)<self.cut),axis=0)
            ax.plot(y[:,cut],py[:,cut],'.')
        else:
            ax.plot(y,py,'.')
        ax.set_xlabel('y')
        ax.set_ylabel('py')
        if not xlims is None:
            ax.set_xlim(xlims)
        if not ylims is None:
            ax.set_ylim(ylims)
        if not savepath is None:
            plt.savefig(savepath,dpi=300)
            plt.close()


    def plot_loss(self,elem=0):
        x=self.output[:,elem,0,:]
        cut=np.any((abs(x)>self.cut),axis=0)
        plt.plot(self.init[0],cut,'.')


    def plot_spectrum_x(self,part,elem=-1,ax=None,log=True,padding=2**16,plot_hxmin=True,plot_hxplus=False):
        if ax is None:
            fig, ax = plt.subplots()
        x=np.r_[self.x(part,elem),np.zeros(padding-len(self.x(part,elem)))]
        px=np.r_[self.px(part,elem),np.zeros(padding-len(self.px(part,elem)))]
        
        hfplus=np.abs(np.fft.fft(x+1j*px))
        hfplus=np.fft.fftshift(hfplus)
        hfmin=np.abs(np.fft.fft(x-1j*px))
        hfmin=np.fft.fftshift(hfmin)
        ff=np.fft.fftfreq(len(x))
        ff=np.fft.fftshift(ff)

        if plot_hxmin:
            if log:
                ax.plot(ff,np.log(hfmin), label="h-")
            else:
                ax.plot(ff,hfmin, label="h-")

        if plot_hxplus:
            if log:
                ax.plot(ff,np.log(hfplus), label="h+")
            else:
                ax.plot(ff,hfplus, label="h+")
        
        plt.legend()


    def plot_spectrum_y(self,part,elem=-1,ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        y=self.y(part,elem)
        py=self.py(part,elem)
        hfplus=np.abs(np.fft.fft(y+1j*py))
        ff=np.fft.fftfreq(len(y))
        hfplus=np.fft.fftshift(hfplus)
        ff=np.fft.fftshift(ff)
        ax.plot(ff,np.log(hfplus), label="h+")


class NormalForms4d:
    def __init__(self, h, phi_x, phi_y, Qx, Qy, num_turns):
        self.h = h
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.Qx = Qx
        self.Qy = Qy
        self.num_turns = num_turns
        #self.line = Line([Phase(tune)])

    @property
    def f(self):
        h = self.h
        f = np.zeros_like(h, dtype=complex)
        phi_x = self.phi_x
        phi_y = self.phi_y
        Qx = self.Qx
        Qy = self.Qy
        for p in range(len(h)):
            for q in range(len(h[p])):
                for r in range(len(h[p, q])):
                    for t in range(len(h[p, q, r])):
                        if h[p, q, r, t] != 0:
                            f[p, q, r, t] = (h[p, q, r, t] * np.exp(1j*(p-q)*phi_x + 1j*(r-t)*phi_y) /
                                             (1 - np.exp(2*np.pi*1j * ((p-q)*Qx+(r-t)*Qy))))
        return f

    def calc_coords(self, part):
        Ix = (part[0]**2 + part[1]**2)/2
        Iy = (part[2]**2 + part[3]**2)/2
        psi0x = np.where(part[0]!=0, np.arctan(part[1]/part[0]), 0)
        print(part[2])
        psi0y = np.where(part[2]!=0, np.arctan(part[3]/part[2]), 0)
        Qx = self.Qx
        Qy = self.Qy
        f = self.f

        hxplus = np.array([np.sqrt(2*Ix) * np.exp(-1j*(2*np.pi*Qx*N + psi0x)) for N in range(self.num_turns)])
        for p in range(len(f)):
            for q in range(len(f[p])):
                for r in range(len(f[p,q])):
                    for t in range(len(f[p,q,r])):
                        if f[p,q,r,t] != 0:
                            corr = np.array([2j * q*f[p,q,r,t] * (2*Ix)**((p+q-1)/2) * (2*Iy)**((r+t)/2) 
                                            * np.exp(1j*(q-p-1)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r)*(2*np.pi*Qy*N + psi0y)) 
                                            for N in range(self.num_turns)])
                            hxplus += corr 
        
     

        hxmin = np.array([np.sqrt(2*Ix) * np.exp(1j*(2*np.pi*Qx*N + psi0x)) for N in range(self.num_turns)])
        for p in range(len(f)):
            for q in range(len(f[p])):
                for r in range(len(f[p,q])):
                    for t in range(len(f[p,q,r])):
                        if f[p,q,r,t] != 0:
                            corr = np.array([-2j * p*f[p,q,r,t] * (2*Ix)**((p+q-1)/2) * (2*Iy)**((r+t)/2) 
                                            * np.exp(1j*(q-p+1)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r)*(2*np.pi*Qy*N + psi0y))
                                            for N in range(self.num_turns)])  

                            hxmin += corr
                            print(f"Spectral line: ({q-p+1}, {t-r})")

        hyplus = np.array([np.sqrt(2*Iy) * np.exp(-1j*(2*np.pi*Qy*N + psi0y)) for N in range(self.num_turns)])
        for p in range(len(f)):
            for q in range(len(f[p])):
                for r in range(len(f[p,q])):
                    for t in range(len(f[p,q,r])):
                        if f[p,q,r,t] != 0:
                            corr = np.array([2j * t*f[p,q,r,t] * (2*Ix)**((p+q)/2) * (2*Iy)**((r+t-1)/2) 
                                            * np.exp(1j*(q-p)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r-1)*(2*np.pi*Qy*N + psi0y))
                                            for N in range(self.num_turns)])
                            hyplus += corr

        hymin = np.array([np.sqrt(2*Iy) * np.exp(1j*(2*np.pi*Qy*N + psi0y)) for N in range(self.num_turns)])
        for p in range(len(f)):
            for q in range(len(f[p])):
                for r in range(len(f[p,q])):
                    for t in range(len(f[p,q,r])):
                        if f[p,q,r,t] != 0:
                            corr = np.array([-2j * r*f[p,q,r,t] * (2*Ix)**((p+q)/2) * (2*Iy)**((r+t-1)/2) 
                                            * np.exp(1j*(q-p)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r+1)*(2*np.pi*Qy*N + psi0y))
                                            for N in range(self.num_turns)])
                            hymin += corr


        x = (hxplus + hxmin) / 2
        if any(abs(pp.imag) > 1e-10 for tt in x for pp in tt):
            warnings.warn("x is not real, are you sure you gave a physical Hamiltonian?")
            
        px = -1j * (hxplus - hxmin) / 2
        if any(abs(pp.imag) > 1e-10 for tt in px for pp in tt):
            warnings.warn("px is not real, are you sure you gave a physical Hamiltonian?")

        y = (hyplus + hymin) / 2
        if any(abs(pp.imag) > 1e-10 for tt in y for pp in tt):
            warnings.warn("y is not real, are you sure you gave a physical Hamiltonian?")

        py = -1j * (hyplus - hymin) / 2
        if any(abs(pp.imag) > 1e-10 for tt in py for pp in tt):
            warnings.warn("py is not real, are you sure you gave a physical Hamiltonian?")

        
        output = np.zeros((self.num_turns, 1, 4, len(part[0])), dtype=complex)
        
        for nturn in range(self.num_turns):
            output[nturn, 0] = np.array([x[nturn], px[nturn], y[nturn], py[nturn]])
            
        return Output4d(part.copy(),output)

        

