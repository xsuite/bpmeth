
import track2d as xt
import numpy as np
import matplotlib.pyplot as plt

def get_freq(x):
   n=len(x)
   x_f=np.abs(np.fft.fft(x))/n
   k=np.argmax(x_f)
   return np.fft.fftfreq(n)[k]

npart=100
part=np.zeros((2,npart))
part[0]=np.linspace(0,1,npart)

line=xt.Line([xt.Phase(0.111), xt.Kick(0.1,2)])
o=line.track(part, num_turns=2**14)
o.plot_xpx()

h = np.zeros((4,4))
h[3,0] = 0.1/3/8
h[0,3] = 0.1/3/8
h[1,2] = 0.1/8
h[2,1] = 0.1/8

plt.figure()
sext = xt.NormalForms(h, 0, 0.111, 2**14)
o2 = sext.calc_xpx(part)

index=95
o.plot_spectrum(index)
o2.plot_spectrum(index)




import track4d as xt

npart = 100
part = np.zeros((4, npart))
part[0] = np.linspace(0, 1, npart)
part[3] = np.linspace(0, 1, npart)

line = xt.Line4d([xt.Phase4d(0.311, 0.112), xt.Kick_x(0.1, 2)])
line_sext = xt.Line4d([xt.Phase4d(0.311, 0.112), xt.Sextupole(0.1)])
o = line_sext.track(part, num_turns=2**14)
o.plot_xpx()

h = np.zeros((4,4,4,4))
h[3,0,0,0] = 0.1/3/8
h[0,3,0,0] = 0.1/3/8
h[1,2,0,0] = 0.1/8
h[2,1,0,0] = 0.1/8
h[1,0,2,0] = 0.1/2/8
h[0,1,2,0] = 0.1/2/8
h[1,0,0,2] = 0.1/2/8
h[0,1,0,2] = 0.1/2/8
h[1,0,1,1] = 0.1/2/4
h[0,1,1,1] = 0.1/2/4
sext = xt.NormalForms4d(h, 0, 0, 0.311, 0.112, 2**14)
o2 = sext.calc_coords(part)

index=95
o.plot_spectrum_x(index)
o2.plot_spectrum_x(index)