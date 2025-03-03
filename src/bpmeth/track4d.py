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
    def __init__(self, b3, length):
        self.b3 = b3
        self.length = length
        
        self.sextupole = bpmeth.GeneralVectorPotential(0, b=("0", "0", f"{self.b3}"))
        self.H_sextupole = bpmeth.Hamiltonian(self.length, 0, self.sextupole)

    def track(self, coord):
        ncoord,npart=coord.shape

        x = coord[0]
        px = coord[1]
        y = coord[2]
        py = coord[3]       
        
        for i in range(npart):
            sol = self.H_sextupole.solve([x[i], y[i], 0, px[i], py[i], 0], s_span=[0, self.length])

            coord[0, i] = sol.y[0][-1]
            coord[1, i] = sol.y[3][-1]
            coord[2, i] = sol.y[1][-1]
            coord[3, i] = sol.y[4][-1]


        

class Trajectory:
    def __init__(self, s, x, px, y, py):
        self.s = s
        self.x = x
        self.px = px
        self.y = y
        self.py = py

    def plot_3d(self, ax=None, label=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.plot3D(self.s, self.x, self.y, label=None)
        ax.set_xlabel('s')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        plt.legend()
        return ax
    
    def plot_x(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.x, label=label, ls='', marker='.')
        ax.set_xlabel('s')
        ax.set_ylabel('x')
        plt.legend()
        return ax
    
    def plot_y(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.y, label=label, ls='', marker='.')
        ax.set_xlabel('s')
        ax.set_ylabel('y')
        plt.legend()
        return ax
    
    def plot_px(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.px, label=label, ls='', marker='.')
        ax.set_xlabel('s')
        ax.set_ylabel('px')
        plt.legend()
        return ax
    
    def plot_py(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.s, self.py, label=label, ls='', marker='.')
        ax.set_xlabel('s')
        ax.set_ylabel('py')
        plt.legend()
        return ax
    
    
class MultiTrajectory:
    def __init__(self, trajectories=[]):
        """
        :param trajectories: a list of Trajectory objects for all particles
        """
        
        self.trajectories = trajectories
    
    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
        
    def get_trajectory(self, index):
        return self.trajectories[index]
    
    def plot_3d(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for trajectory in self.trajectories:
            trajectory.plot_3d(ax)
            
    def plot_final(self, ax=None, label=None):
        """
        Plot trends in final coordinates to compare different maps.
        They will simply be plotted against the coordinate index.
        """
        
        if ax is None:
            fig, ax = plt.subplots(4)
            ax[0].set_xlabel('Particle index')
            ax[0].set_ylabel('x')
            ax[1].set_xlabel('Particle index')
            ax[1].set_ylabel('px')
            ax[2].set_xlabel('Particle index')
            ax[2].set_ylabel('y')
            ax[3].set_xlabel('Particle index')
            ax[3].set_ylabel('py')
            
        ax[0].plot([trajectory.x[-1] for trajectory in self.trajectories], label=label)
        ax[1].plot([trajectory.px[-1] for trajectory in self.trajectories], label=label)
        ax[2].plot([trajectory.y[-1] for trajectory in self.trajectories], label=label)
        ax[3].plot([trajectory.py[-1] for trajectory in self.trajectories], label=label)
        plt.legend()
        
        return ax
    
    
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


    def plot_spectrum_x(self,part,elem=-1,ax=None,log=True,padding=2**16,plot_hxmin=True,
                        plot_hxplus=False,plot_phase=False,unwrap=True,ax_phase=None,label=None,color="black"):
        if ax is None:
            fig, ax = plt.subplots()
        if plot_phase and ax_phase is None:
            fig_phase, ax_phase = plt.subplots()
            
        x = np.r_[self.x(part,elem),np.zeros(padding-len(self.x(part,elem)))]
        px = np.r_[self.px(part,elem),np.zeros(padding-len(self.px(part,elem)))]
        
        hfplus = np.fft.fft(x+1j*px)
        hfplusabs = np.abs(hfplus)
        hfplusabs = np.fft.fftshift(hfplusabs)
        hfplusangle = np.angle(hfplus)
        hfplusangle = np.fft.fftshift(hfplusangle) 
        if unwrap:
            hfplusangle = np.unwrap(hfplusangle)
        
        hfmin = np.fft.fft(x-1j*px)
        hfminabs = np.abs(hfmin)
        hfminabs = np.fft.fftshift(hfminabs)
        hfminangle = np.angle(hfmin)
        hfminangle = np.fft.fftshift(hfminangle)
        if unwrap:
            hfminangle = np.unwrap(hfminangle)
        
        ff = np.fft.fftfreq(len(x))
        ff = np.fft.fftshift(ff)

        if plot_hxmin:
            if log:
                ax.plot(ff,np.log(hfminabs), label=f"h- {label}", color=color)
            else:
                ax.plot(ff,hfminabs, label=f"h- {label}", color=color)
        
            if plot_phase:
                ax_phase.plot(ff,hfminangle, label=f"Phase h- {label}", color=color, linestyle='--')  

        if plot_hxplus:
            if log:
                ax.plot(ff,np.log(hfplusabs), label=f"h+ {label}", color=color)
            else:
                ax.plot(ff,hfplusabs, label=f"h+ {label}", color=color)
                
            if plot_phase:
                ax_phase.plot(ff,hfplusangle, label=f"Phase h+ {label}", color=color, linestyle='--')
                
        plt.legend()


    def plot_spectrum_y(self,part,elem=-1,ax=None,log=True,padding=2**16,plot_hxmin=True,
                        plot_hxplus=False,plot_phase=False,unwrap=True,ax_phase=None,label=None,color="black"):
        if ax is None:
            fig, ax = plt.subplots()
        if plot_phase and ax_phase is None:
            fig_phase, ax_phase = plt.subplots()
            
        y = np.r_[self.y(part,elem),np.zeros(padding-len(self.y(part,elem)))]
        py = np.r_[self.py(part,elem),np.zeros(padding-len(self.py(part,elem)))]
              
        hfplus = np.fft.fft(y+1j*py)
        hfplusabs = np.abs(hfplus)
        hfplusabs = np.fft.fftshift(hfplusabs)
        hfplusangle = np.angle(hfplus)
        hfplusangle = np.fft.fftshift(hfplusangle)   
        if unwrap:
            hfplusangle = np.unwrap(hfplusangle)

        hfmin = np.fft.fft(y-1j*py)
        hfminabs = np.abs(hfmin)
        hfminabs = np.fft.fftshift(hfminabs)
        hfminangle = np.angle(hfmin)
        hfminangle = np.fft.fftshift(hfminangle)
        if unwrap:
            hfminangle = np.unwrap(hfminangle)
        
        ff = np.fft.fftfreq(len(y))
        ff = np.fft.fftshift(ff)
        
        if plot_hxmin:
            if log:
                ax.plot(ff,np.log(hfminabs), label=f"h- {label}", color=color)
            else:
                ax.plot(ff,hfminabs, label=f"h- {label}", color=color)
                
            if plot_phase:
                ax_phase.plot(ff,hfminangle, label=f"Phase h- {label}", color=color, linestyle='--')
                
        if plot_hxplus:
            if log:
                ax.plot(ff,np.log(hfplusabs), label=f"h+ {label}", color=color)
            else:
                ax.plot(ff,hfplusabs, label=f"h+ {label}", color=color)
                
            if plot_phase:
                ax_phase.plot(ff,hfplusangle, label=f"Phase h+ {label}", color=color, linestyle='--')
                
        plt.legend()


class NormalForms4d:
    def __init__(self, h, phi_x, phi_y, Qx, Qy, num_turns):
        """
        :param h: four or five dimensional array with the Hamiltonian coefficients. Each h[p,q,r,t] is either a 
            complex number or a numpy array of complex numbers corresponding to different locations
        :param phi_x: horizontal phase advance of the perturbation, either a float or a numpy array of floats
        :param phi_y: vertical phase advance of the perturbation, either a float or a numpy array of floats
        :param Qx: horizontal tune
        :param Qy: vertical tune
        :param num_turns: number of turns to track
        """
        
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
        f = np.zeros((5,5,5,5), dtype=complex)
        phi_x = self.phi_x
        phi_y = self.phi_y
        Qx = self.Qx 
        Qy = self.Qy

        if len(h.shape) == 4:
            # Only one s-position with perturbation
            assert(isinstance(phi_x, float))
            assert(isinstance(phi_y, float))
        
            for p in range(len(h)):
                for q in range(len(h[p])):
                    for r in range(len(h[p, q])):
                        for t in range(len(h[p, q, r])):
                            if h[p, q, r, t] != 0:
                                f[p, q, r, t] = (h[p, q, r, t] * np.exp(1j*(p-q)*phi_x + 1j*(r-t)*phi_y) /
                                                (1 - np.exp(2*np.pi*1j * ((p-q)*Qx+(r-t)*Qy))))
        else:
            assert(isinstance(phi_x, np.ndarray) and phi_x.shape[0] == h.shape[4])
            assert(isinstance(phi_y, np.ndarray) and phi_y.shape[0] == h.shape[4])
            for p in range(len(h)):
                for q in range(len(h[p])):
                    for r in range(len(h[p, q])):
                        for t in range(len(h[p, q, r])):
                            if p!=q or r!=t:  # Drop the detuning terms
                                f[p, q, r, t] = np.nansum(h[p, q, r, t] * np.exp(1j*(p-q)*phi_x + 1j*(r-t)*phi_y) /
                                                    (1 - np.exp(2*np.pi*1j * ((p-q)*Qx+(r-t)*Qy))), axis=0)
            
        return f
    
    def calc_deltaQ(self, Jx, Jy):
        """
        :param Jx: Linear horizontal action
        :param Jy: Linear vertical action
        """
        h = self.h
        delta_Qx, delta_Qy = 0, 0
        for p in range(len(h)):
            for r in range(len(h[p, p])):
                # Terms pprr give the detuning with amplitude
                delta_Qx += np.nansum(-1/(2*np.pi) * h[p, p, r, r]) * p/2*Jx**(p/2-1) * Jy**(r/2)
                delta_Qy += np.nansum(-1/(2*np.pi) * h[p, p, r, r]) * Jx**(p/2) * r/2*Jy**(r/2-1)
        return delta_Qx, delta_Qy

    def calc_coords(self, part, detuning=False):
        # Nonlinear action, but here calculated neglecting nonlinear terms (so as if it was linear action)
        Ix = (part[0]**2 + part[1]**2)/2
        Iy = (part[2]**2 + part[3]**2)/2
        psi0x = np.where(part[0]!=0, np.arctan(part[1]/part[0]), 0)
        psi0y = np.where(part[2]!=0, np.arctan(part[3]/part[2]), 0)
        Qx = self.Qx
        Qy = self.Qy
        f = self.f
        
        if detuning:
            deltaQ = self.calc_deltaQ(Ix, Iy)
            Qx += deltaQ[0]
            Qy += deltaQ[1]
            print("Detuning with amplitude included")

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

        

