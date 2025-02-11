
import track4d as xt4d
import numpy as np
import matplotlib.pyplot as plt

npart = 20
part = np.zeros((4, npart))
part[0] = np.linspace(0, 1, npart)
Qx = 0.33
Qy = 0
nturns = 150
xlims=[-1,1] 
ylims=[-1,1]

fig, ax = plt.subplots()

# Solution with kicks
line_kick = xt4d.Line4d([xt4d.Phase4d(Qx, Qy), xt4d.Kick_x(0.1, 2)])
o_kick = line_kick.track(part, num_turns=nturns)
o_kick.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)

line_sext = xt4d.Line4d([xt4d.Phase4d(Qx, Qy), xt4d.Sextupole(0.1)])
o_sext = line_sext.track(part, num_turns=nturns)
o_sext.plot_xpx(xlims=xlims, ylims=ylims)

for i,tune in enumerate(np.arange(0.332,0.334,0.0001)):
   npart = 100
   part = np.zeros((4, npart))
   part[0] = np.linspace(0, 1, npart)
   line_sext = xt4d.Line4d([xt4d.Phase4d(tune, Qy), xt4d.Sextupole(0.1)])
   o_kick = line_sext.track(part, num_turns=nturns)
   o_kick.plot_xpx(xlims=xlims, ylims=ylims, savepath=f"tune_{i}.png")

# Solution with normal forms
h = np.zeros((4,4,4,4))
h[3,0,0,0] = 0.1/3/8
h[0,3,0,0] = 0.1/3/8
h[1,2,0,0] = 0.1/8
h[2,1,0,0] = 0.1/8
# h[1,0,2,0] = -0.1/2/8
# h[0,1,2,0] = -0.1/2/8
# h[1,0,0,2] = -0.1/2/8
# h[0,1,0,2] = -0.1/2/8
# h[1,0,1,1] = -0.1/2/4
# h[0,1,1,1] = -0.1/2/4
sext4d = xt4d.NormalForms4d(h, 2*np.pi*Qx, 2*np.pi*Qy, Qx, Qy, nturns)
o_norm = sext4d.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims, ax=ax)


# Solution with numerical integration
line_num = xt4d.Line4d([xt4d.Phase4d(Qx, Qy), xt4d.NumericalSextupole(-10, 0.01)])
o_num = line_num.track(part, num_turns=nturns)
o_num.plot_xpx(xlims=xlims, ylims=ylims)

index=95
o_kick.plot_spectrum_x(index)
o_norm.plot_spectrum_x(index)
o_num.plot_spectrum_x(index)