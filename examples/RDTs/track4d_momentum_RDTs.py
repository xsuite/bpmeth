import numpy as np
import matplotlib.pyplot as plt
import math
import bpmeth

"""
Analysis of a ring with a map coming from H = a x
Delta px = :-H: px - px = -a [x, px] = -a
RDTs H = a x = a * sqrt(beta_x)/2 * (hx+ + hx-) => h1000 = h0100 = -a * sqrt(beta_x)/2
"""

npart = 1
xvals = np.linspace(0.02, 0.05, npart)
part = bpmeth.MultiParticle(npart, x=xvals, y=0)  

Qx = 0.123
Qy = 0.211
nturns = 100
xlims=[-0.1,0.1] 
ylims=[-0.1,0.1]    


fig, ax = plt.subplots()
a = 0.01


phix = 0.01
phiy = 0.
# Solution with tracking
line_shift = bpmeth.Line4d([bpmeth.Phase4d(Qx-phix, Qy-phiy), bpmeth.Kick_x(-a, order=0), bpmeth.Phase4d(phix, phiy)])
o_track = line_shift.track(part, num_turns=nturns)
o_track.plot_xpx(xlims=xlims, ylims=ylims, elem=2, ax=ax) # observation at end of the line


# Solution with normal forms
betax=1
alphax=0
h = np.zeros((4, 4, 4, 4), dtype=complex)
h[1,0,0,0] = -a * np.sqrt(betax)/2 
h[0,1,0,0] = -a * np.sqrt(betax)/2 

ff = bpmeth.NormalForms4d(h, 2*np.pi*(-phix) if -phix > 0 else 2*np.pi*(-phix+Qx), 2*np.pi*(-phiy) if -phiy > 0 else 2*np.pi*(-phix+Qy), Qx, Qy, nturns)  # source - observation (+2piQ)
o_norm = ff.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)

ax.plot(-a * betax/2 * np.cos(np.pi*Qx)/np.sin(np.pi*Qx), -a/2 * (1-alphax * np.cos(np.pi*Qx)/np.sin(np.pi*Qx)), 'ro')


fig, ax = plt.subplots()
a = 0.01
betax=0.5
betay=0.5
alphax=0.5
alphay=0.5

# Solution with tracking for beta, alpha not equal to 1,0
line = bpmeth.Line4d([bpmeth.OneTurnMap(Qx, Qy, betax, betay, alphax, alphay), bpmeth.Kick_x(-a, order=0)])
o_track = line.track(part, num_turns=nturns)
o_track.plot_xpx(xlims=xlims, ylims=ylims, ax=ax, elem=1)  # observation after element 1

ax.plot(-a * betax/2 * np.cos(np.pi*Qx)/np.sin(np.pi*Qx), -a/2 * (1-alphax * np.cos(np.pi*Qx)/np.sin(np.pi*Qx)), 'ro')


"""
Analysis of a ring with a map coming from H = a px
Induced shift Delta x = :-H: x - x = -a [px, x] = a
RDTs: 
"""

class Shift_x:
    def __init__(self, shift):
        self.shift = shift

    def track(self, part):
        part.x += self.shift

fig, ax = plt.subplots()
a = 0.01

# Solution with tracking
line_shift = bpmeth.Line4d([bpmeth.Phase4d(Qx-phix, Qy-phiy), Shift_x(a), bpmeth.Phase4d(phix, phiy)])
o_track = line_shift.track(part, num_turns=nturns)
o_track.plot_xpx(xlims=xlims, ylims=ylims, ax=ax, elem=2)  # observation end of line

# Solution with normal forms
betax=1
alphax=0
h = np.zeros((4, 4, 4, 4), dtype=complex)
h[1,0,0,0] = a / (2*np.sqrt(betax)) * (alphax + 1j)
h[0,1,0,0] = a / (2*np.sqrt(betax)) * (alphax - 1j)

ff = bpmeth.NormalForms4d(h, 2*np.pi*phix, 2*np.pi*phiy, Qx, Qy, nturns)  # observation end of line
o_norm = ff.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)

# Theoretical value fixed point
ax.plot(a/2, -a/2*np.cos(np.pi*Qx)/np.sin(np.pi*Qx), 'ro')



fig, ax = plt.subplots()
a = 0.01
betax=0.5
betay=0.5
alphax=0.5
alphay=0.5

# Solution with tracking for beta, alpha not equal to 1,0
line = bpmeth.Line4d([bpmeth.OneTurnMap(Qx, Qy, betax, betay, alphax, alphay), Shift_x(a)])
o_track = line.track(part, num_turns=nturns)
o_track.plot_xpx(xlims=xlims, ylims=ylims, ax=ax, elem=1)  # observation after element 1

# Theoretical value fixed point
ax.plot(a/2 * (1 + alphax*np.cos(np.pi*Qx)/np.sin(np.pi*Qx)), -(1+alphax**2) * a/2/betax*np.cos(np.pi*Qx)/np.sin(np.pi*Qx), 'ro')


