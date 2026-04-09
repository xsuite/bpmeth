import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import bpmeth
import scipy as sc


data = np.loadtxt("../fieldmaps/quadrupole_toymodels/field_map_straight_quad_resol_2mm.txt", skiprows=9)
quad = bpmeth.Fieldmap(data).mirror(left=False)

data = np.loadtxt("../fieldmaps/quadrupole_toymodels/field_map_quad_edge_30deg_resol_2mm.txt", skiprows=9)
cutquad = bpmeth.Fieldmap(data).mirror(left=False)

######################
# Fit Enge functions #
######################

svals, coeffs, coeffsstd = cutquad.s_multipoles(3)
npoints = len(svals)
params, cov = sc.optimize.curve_fit(bpmeth.Enge, svals[:npoints//2], coeffs[:npoints//2, 1], 
                                    sigma=coeffsstd[:npoints//2, 1], p0=np.ones(5))

svals, coeffs, coeffsstd = quad.s_multipoles(3)
npoints = len(svals)
strparams, strcov = sc.optimize.curve_fit(bpmeth.Enge, svals[:npoints//2], coeffs[:npoints//2, 1], 
                                    sigma=coeffsstd[:npoints//2, 1], p0=np.ones(5))

ss = np.linspace(-0.8, 0, 200)
sedge = -np.trapezoid(bpmeth.Enge(ss, *params), x=ss) / params[0]  # Entrance edge


##############################
# Model based on cut at edge #
##############################

fig, ax = plt.subplots()
cutquad.translate(0, 0, -sedge).s_multipoles(3, ax=ax)

s = sp.symbols('s')
b2 = sp.Function('b2')
model = bpmeth.FieldExpansion(b=("0", b2(s)))

cut_model = model.cut_at_angle(30/180*np.pi)

# Substitute b2 with actual profile
cut_enge = bpmeth.FieldExpansion(b=[cut_model.b[i].replace(b2, lambda s : bpmeth.spEnge(s+sedge, *params)).doit() for i in range(3)])

colors=["skyblue", "navajowhite", "limegreen"]
ss = np.linspace(-0.25, -sedge, 200)
s = cut_enge.s
for i, bb in enumerate(cut_enge.b):
    ax.plot(ss, [sp.sympify(bb).subs(s,sval).evalf() for sval in ss], label=f"Model b{i+1}", color=colors[i], ls="--", linewidth=0.8)

ax.set_xlabel("s [m]", fontsize=13)
ax.set_ylabel(r"Multipole strength [T/m$^n$]", fontsize=13)
ax.set_xlim(-0.5, -sedge)
plt.legend(fontsize=13)
plt.savefig("cut_model_comparison.png", dpi=300)


###############################
# Model based on rotated edge #
###############################

fig, ax = plt.subplots()
cutquad.translate(0, 0, -sedge).s_multipoles(3, ax=ax)

s = sp.symbols('s')
b2 = sp.Function('b2')
model = bpmeth.FieldExpansion(b=("0", b2(s)))

rot_model = model.transform(30/180*np.pi)

# Substitute b2 with actual profile
rot_enge = bpmeth.FieldExpansion(b=[rot_model.b[i].replace(b2, lambda s : bpmeth.spEnge(s+sedge, *strparams)).doit() for i in range(3)])

colors=["skyblue", "navajowhite", "limegreen"]
ss = np.linspace(-0.25, -sedge, 200)
s = cut_enge.s
for i, bb in enumerate(rot_enge.b):
    ax.plot(ss, [sp.sympify(bb).subs(s,sval).evalf() for sval in ss], label=f"Model b{i+1}", color=colors[i], ls="--", linewidth=0.8)

ax.set_xlabel("s [m]", fontsize=13)
ax.set_ylabel(r"Multipole strength [T/m$^n$]", fontsize=13)
ax.set_xlim(-0.5, -sedge)
plt.legend(fontsize=13)
plt.savefig("rotated_model_comparison.png", dpi=300)
