import bpmeth
import numpy as np
import matplotlib.pyplot as plt

quadfringe = bpmeth.FieldExpansion(b=("0", "0.1*(tanh(s)+1)/2",))

rot_quadfringe = quadfringe.transform(30/180*np.pi)

"""
To compare the two methods, the b2 component of the fringe field in the beam frame has to be the same.
For the rotated field, this component is b2(s cos(theta)) - b2'(s cos(theta)) s sin(theta)^2
"""

quadfringe2 = bpmeth.FieldExpansion(b=("0", rot_quadfringe.b[1]))

cut_quadfringe = quadfringe2.cut_at_angle(30/180*np.pi)

fig1, ax1 = plt.subplots()
cut_quadfringe.plot_components(ax=ax1, smin=-5, smax=5)
ax1.set_ylim(-0.03, 0.1)

fig2, ax2 = plt.subplots()
rot_quadfringe.plot_components(ax=ax2, smin=-5, smax=5)
ax2.set_ylim(-0.03, 0.1)

ss = np.linspace(-5, 5, 100)
fig, ax = plt.subplots()
for i in range(len(rot_quadfringe.b)):
    ax.plot(ss, [(rot_quadfringe.b[i]-cut_quadfringe.b[i]).subs({quadfringe.s:sval}).evalf() for sval in ss], label=f"b{i+1}")
    print(f"Integral for b{i+1}:")
    print(f"- Integral for cut fringe: \t {np.trapezoid([cut_quadfringe.b[i].subs({quadfringe.s:sval}).evalf() for sval in ss], ss)}")
    print(f"- Integral for rotated fringe: \t {np.trapezoid([rot_quadfringe.b[i].subs({quadfringe.s:sval}).evalf() for sval in ss], ss)}")
ax.legend()