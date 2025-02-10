import numpy as np
import matplotlib.pyplot as plt

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
        coord[1] += self.kick*x**self.order

    def __repr__(self):
        return f"Kick_x({self.kick})"

class Kick_y:
    def __init__(self, kick, order=1):
        self.kick = kick
        self.order = order

    def track(self, coord):
        y = coord[2]
        coord[3] += self.kick*y**self.order

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


class Line4d:
    def __init__(self,elements):
        self.elements = elements

    def track(self, coord, num_turns=1):
        tcoord=coord.copy()
        ncoord,npart=coord.shape
        nelem=len(self.elements)
        output=np.zeros((num_turns,nelem,ncoord,npart))
        for nturn in range(num_turns):
            for ielem,element in enumerate(self.elements):
                element.track(tcoord)
                output[nturn,ielem]=tcoord
        return Output4d(coord.copy(),output)


class Output4d:
    def __init__(self, init, output, cut=10):
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

    def plot_xpx(self,elem=0,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        x=self.output[:,elem,0,:]
        px=self.output[:,elem,1,:]
        cut=np.all((abs(x)<self.cut),axis=0)
        ax.plot(x[:,cut],px[:,cut],'.')
        ax.set_xlabel('x')
        ax.set_ylabel('px')

    def plot_ypy(self,elem=0,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        y=self.output[:,elem,2,:]
        py=self.output[:,elem,3,:]
        cut=np.all((abs(y)<self.cut),axis=0)
        ax.plot(y[:,cut],py[:,cut],'.')
        ax.set_xlabel('y')
        ax.set_ylabel('py')

    def plot_loss(self,elem=0):
        x=self.output[:,elem,0,:]
        cut=np.any((abs(x)>self.cut),axis=0)
        plt.plot(self.init[0],cut,'.')

    def plot_spectrum_x(self,part,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        hfplus=np.abs(np.fft.fft(x+1j*px))
        ff=np.fft.fftfreq(len(x))
        hfplus=np.fft.fftshift(hfplus)
        ff=np.fft.fftshift(ff)
        plt.plot(ff,np.log(hfplus), label="h+")
        
        # hfmin=np.abs(np.fft.fft(x-1j*px))
        # hfmin=np.fft.fftshift(hfmin)
        # plt.plot(ff,np.log(hfmin), label="h-")
        # plt.legend()

    def plot_spectrum_y(self,part,elem=-1):
        y=self.y(part,elem)
        py=self.py(part,elem)
        hfplus=np.abs(np.fft.fft(y+1j*py))
        ff=np.fft.fftfreq(len(y))
        hfplus=np.fft.fftshift(hfplus)
        ff=np.fft.fftshift(ff)
        plt.plot(ff,np.log(hfplus), label="h+")


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
                            f[p, q, r, t] = h[p, q, r, t] * np.exp(1j*(p-q)*phi_x + 1j*(r-t)*phi_y) / (1 - np.exp(2*np.pi*1j * ((p-q)*Qx+(r-t)*Qy)))
        return f

    def calc_coords(self, part):
        Ix = (part[0]**2 + part[1]**2)/2
        Iy = (part[2]**2 + part[3]**2)/2
        psi0x = np.where(part[0]!=0, np.arctan(part[1]/part[0]), 0)
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
                            corr = np.array([2j * p*f[p,q,r,t] * (2*Ix)**((p+q-1)/2) * (2*Iy)**((r+t)/2) 
                                            * np.exp(1j*(q-p+1)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r)*(2*np.pi*Qy*N + psi0y))
                                            for N in range(self.num_turns)])
                            hxplus -= corr

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
                            corr = np.array([2j * r*f[p,q,r,t] * (2*Ix)**((p+q)/2) * (2*Iy)**((r+t-1)/2) 
                                            * np.exp(1j*(q-p)*(2*np.pi*Qx*N + psi0x) + 1j*(t-r+1)*(2*np.pi*Qy*N + psi0y))
                                            for N in range(self.num_turns)])
                            hyplus -= corr


        x = (hxplus + hxmin) / 2
        px = -1j * (hxplus - hxmin) / 2

        y = (hyplus + hymin) / 2
        py = -1j * (hyplus - hymin) / 2

        output = np.zeros((self.num_turns, 1, 4, len(part[0])))

        for nturn in range(self.num_turns):
            output[nturn, 0] = np.array([x[nturn], px[nturn], y[nturn], py[nturn]])

        return Output4d(part.copy(),output)

        

