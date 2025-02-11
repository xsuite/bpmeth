
import track2d as xt2d
import numpy as np
import matplotlib.pyplot as plt

def get_freq(x):
   n=len(x)
   x_f=np.abs(np.fft.fft(x))/n
   k=np.argmax(x_f)
   return np.fft.fftfreq(n)[k]

npart=100
part=np.zeros((2,npart))
part[0]=np.linspace(0,1,part)
Qx = 0.33
nturns = 2**14
xlims=[-1,1] 
ylims=[-1,1]

# Kicks
line=xt2d.Line([xt2d.Phase(Qx), xt2d.Kick(0.1,2)])
o_kicks2d=line.track(part, num_turns=nturns)
o_kicks2d.plot_xpx(xlims=xlims, ylims=ylims)


# Normal forms
h = np.zeros((4,4))
h[3,0] = 0.1/3/8
h[0,3] = 0.1/3/8
h[1,2] = 0.1/8
h[2,1] = 0.1/8

sext2d = xt2d.NormalForms(h, 2*np.pi*Qx, Qx, nturns)
o_norm2d = sext2d.calc_xpx(part)
o_norm2d.plot_xpx(xlims=xlims, ylims=ylims)

index=95
o_kicks2d.plot_spectrum(index)
o_norm2d.plot_spectrum(index)
