import numpy as np
import matplotlib.pyplot as plt

class Phase:
    def __init__(self, phase):
        self.phase = phase

    def track(self, coord):
        x=coord[0]
        px=coord[1]
        arg=2*np.pi*self.phase
        cx=np.cos(arg)
        sx=np.sin(arg)
        x1=x*cx+px*sx
        px1=-x*sx+px*cx
        coord[0]=x1
        coord[1]=px1

    def __repr__(self):
        return f"Phase({self.phase})"

class Kick:
    def __init__(self, kick, order=1):
        self.kick = kick
        self.order = order

    def track(self, coord):
        x=coord[0]
        coord[1]+=self.kick*x**self.order

    def __repr__(self):
        return f"Kick({self.kick})"


class Line:
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
        return Output(coord.copy(),output)

class Output:
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

    def hxp(self,part=0,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        return x+1j*px

    def hxm(self,part=0,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        return x-1j*px

    def plot_xpx(self,elem=0,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        x=self.output[:,elem,0,:]
        px=self.output[:,elem,1,:]
        cut=np.all((abs(x)<self.cut),axis=0)
        ax.plot(x[:,cut],px[:,cut],'.')
        ax.set_xlabel('x')
        ax.set_ylabel('px')

    def plot_loss(self,elem=0):
        x=self.output[:,elem,0,:]
        cut=np.any((abs(x)>self.cut),axis=0)
        plt.plot(self.init[0],cut,'.')

    def plot_spectrum(self,part,elem=-1):
        x=self.x(part,elem)
        px=self.px(part,elem)
        hfplus=np.abs(np.fft.fft(x+1j*px))
        ff=np.fft.fftfreq(len(x))
        hfplus=np.fft.fftshift(hfplus)
        ff=np.fft.fftshift(ff)
        plt.plot(ff,np.log(hfplus), label="h+")
        #plt.plot(ff,hfplus, label="h+")
        
        # hfmin=np.abs(np.fft.fft(x-1j*px))
        # hfmin=np.fft.fftshift(hfmin)
        # plt.plot(ff,np.log(hfmin), label="h-")
        # plt.legend()


class NormalForms:
    def __init__(self, h, phi, Qx, num_turns):
        self.h = h
        self.phi = phi
        self.Qx = Qx
        self.num_turns = num_turns
        #self.line = Line([Phase(tune)])

    @property
    def f(self):
        h = self.h
        f = np.zeros((len(h), len(h[0])), dtype=complex)
        phi = self.phi
        for p in range(len(h)):
            for q in range(len(h[p])):
                if h[p, q] != 0:
                    f[p, q] = h[p, q] * np.exp(1j*(p-q)*phi) / (1 - np.exp(2*np.pi*1j*(p-q)*self.Qx))
        return f

    def calc_xpx(self, part):
        Ix = (part[0]**2 + part[1]**2)/2
        psi0 = np.arctan(part[1]/part[0])
        Qx = self.Qx
        f = self.f

        hxplus = np.array([np.sqrt(2*Ix) * np.exp(-1j*(2*np.pi*Qx*N + psi0)) for N in range(self.num_turns)])
        for p in range(len(f)):
            for q in range(len(f[p])):
                corr = np.array([2j * q*f[p,q] * (2*Ix)**((p+q-1)/2) * np.exp(1j*(q-p-1)*(2*np.pi*Qx*N + psi0)) for N in range(self.num_turns)])
                hxplus += corr
        
        hxmin = np.array([np.sqrt(2*Ix) * np.exp(1j*(2*np.pi*Qx*N + psi0)) for N in range(self.num_turns)])
        print("Spectrum for hxmin...")
        for p in range(len(f)):
            for q in range(len(f[p])):
                corr = np.array([-2j * p*f[p,q] * (2*Ix)**((p+q-1)/2) * np.exp(1j*(q-p+1)*(2*np.pi*Qx*N + psi0)) for N in range(self.num_turns)])
                hxmin += corr
                if f[p, q] != 0:
                    print(f"({q-p+1}, 0)")
                

        x = (hxplus + hxmin) / 2
        px = -1j * (hxplus - hxmin) / 2


        output = np.zeros((self.num_turns, 1, 2, len(part[0])))

        for nturn in range(self.num_turns):
            output[nturn, 0] = np.array([x[nturn], px[nturn]])

        return Output(part.copy(),output)

        

