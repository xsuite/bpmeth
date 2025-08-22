import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
import bpmeth
import mpmath

def K0ggtanh(b1, a, L):
    """    
    K0 fringe field integral for a tanh times the gap height squared.
    
    :param b1: fringe field coefficient.
    :param a: fringe field range.
    :param L: integral around the fringe field ranges from -L to L.
        (In theory the value of K0gg should not depend on L, otherwise L is chosen too small.)
    """

    return b1* ( L**2/2 - L*a/2*np.log((1+np.exp(-2*L/a))*(1+np.exp(2*L/a))) \
        + a**2/4*(mpmath.polylog(2, -np.exp(-2*L/a))-mpmath.polylog(2, -np.exp(2*L/a))) )
    
# Properties
h = 0
b1 = 1
aa = 0.1
length = 10*aa
b1shape = f"(tanh(s/{aa})+1)/2"
Kg = aa/2
hgap = 0.1
gap = 2*hgap
K0gg = K0ggtanh(b1, aa, length/2)
angle = 0.5
print("cofint = ", K0gg / gap**2 / b1)

fringe_xsuite = xt.Bend(k0=b1, length=0, edge_entry_model='dipole-only', edge_entry_fint=Kg/gap, edge_entry_angle=angle, edge_entry_hgap=hgap, edge_exit_model='suppressed')  # dipole entrance, no exit
line = xt.Line(elements=[fringe_xsuite])
fringe_bpmeth = bpmeth.ThinNumericalFringe(str(b1), b1shape, length=length, nphi=5, ivp_opt={"atol":1e-10}, theta_E=angle)
fringe_forest = bpmeth.ForestFringe(b1, Kg, K0gg=K0gg, sadistic=True, closedorbit=True)  # To double check implementation in xsuite and have some more freedom

# Tracking
# X dependence
x0 = np.linspace(-1e-1, 1e-1, 101)
p = xt.Particles(x=x0)
p_bpmeth = p.copy()
p_xsuite = p.copy()
p_forest = p.copy()

line.build_tracker(use_prebuilt_kernels=False)  # Build tracker for xsuite fringe
line.track(p_xsuite)
trajectories_bpmeth = fringe_bpmeth.track(p_bpmeth)
trajectories_forest = fringe_forest.track(p_forest)

fig, ax = plt.subplots()
ax.plot(x0, p_xsuite.px, label='px xsuite')
ax.plot(x0, p_bpmeth.px, label='px bpmeth')
ax.plot(x0, p_forest.px, label='px forest')
ax.plot(x0, p_xsuite.py, label='py xsuite')
ax.plot(x0, p_bpmeth.py, label='py bpmeth')
ax.plot(x0, p_forest.py, label='py forest')
ax.set_xlabel('x')
ax.legend()

# Y dependence
y0 = np.linspace(-1e-1, 1e-1, 101)
p = xt.Particles(y=y0)
p_bpmeth = p.copy()
p_xsuite = p.copy()
p_forest = p.copy()

line.track(p_xsuite)
trajectories_bpmeth = fringe_bpmeth.track(p_bpmeth, return_trajectories=True)
trajectories_forest = fringe_forest.track(p_forest)

fig, ax = plt.subplots()
ax.plot(y0, p_xsuite.py, label='py xsuite')
ax.plot(y0, p_bpmeth.py, label='py bpmeth')
# ax.plot(y0, p_forest.py, label='py forest')
ax.plot(y0, p_xsuite.px, label='px xsuite')
ax.plot(y0, p_bpmeth.px, label='px bpmeth')
# ax.plot(y0, p_forest.px, label='px forest')
ax.set_xlabel('y')
ax.legend()

fig, ax = plt.subplots()
ax.plot(y0, p_xsuite.x, label='x xsuite')
ax.plot(y0, p_bpmeth.x, label='x bpmeth')
# ax.plot(y0, p_forest.x, label='x forest')
ax.plot(y0, p_xsuite.y, label='y xsuite')
ax.plot(y0, p_bpmeth.y, label='y bpmeth')
# ax.plot(y0, p_forest.y, label='y forest')
ax.set_xlabel('y')
ax.legend()

poly_bpmeth = np.polyfit(y0, p_bpmeth.py, 3)  # Highest power first
poly_xsuite = np.polyfit(y0, p_xsuite.py, 3)  # Highest power first
poly_forest = np.polyfit(y0, p_forest.py, 3)  # Highest power first

print("Zeroth order from bpmeth:", poly_bpmeth[-1])
print("First order from bpmeth:", poly_bpmeth[-2])
print("Second order from bpmeth:", poly_bpmeth[-3])
print("Third order from bpmeth:", poly_bpmeth[-4])
print("Zeroth order from xsuite:", poly_xsuite[-1])
print("First order from xsuite:", poly_xsuite[-2])
print("Second order from xsuite:", poly_xsuite[-3])
print("Third order from xsuite:", poly_xsuite[-4])

# X Y dependence
xx = np.linspace(-1e-1, 1e-1, 11)
yy = np.linspace(-1e-1, 1e-1, 11)
XX, YY = np.meshgrid(xx, yy)

p = xt.Particles(x = XX.flatten(), y = YY.flatten())
p_bpmeth = p.copy()
p_xsuite = p.copy()
p_forest = p.copy()

line.track(p_xsuite)
trajectories_bpmeth = fringe_bpmeth.track(p_bpmeth, return_trajectories=True)
trajectories_forest = fringe_forest.track(p_forest)

fig, ax = plt.subplots(subplot_kw={"projection":'3d'})
ax.plot_surface(XX, YY, p_xsuite.py.reshape(XX.shape), label='px xsuite', alpha=0.5)
ax.plot_surface(XX, YY, p_bpmeth.py.reshape(XX.shape), label='px bpmeth', alpha=0.5)



# Plot individiual trajectories
index=0
trajectory_bpmeth = trajectories_bpmeth.get_trajectory(index)

fig, ax = plt.subplots()
trajectory_bpmeth.plot_x(ax=ax, label='bpmeth')
ax.plot(p.s[index], p.x[index], label='initial condition', marker='o')
ax.plot(p_xsuite.s[index], p_xsuite.x[index], marker='o', label='xsuite')
ax.plot(p_forest.s[index], p_forest.x[index], marker='o', label='forest with closed orbit')
ax.set_xlabel('s')
ax.set_ylabel('x')
ax.legend()

fig, ax = plt.subplots()
trajectory_bpmeth.plot_y(ax=ax, label='bpmeth')
ax.plot(p.s[index], p.y[index], label='initial condition', marker='o')
ax.plot(p_xsuite.s[index], p_xsuite.y[index], marker='o', label='xsuite')
ax.plot(p_forest.s[index], p_forest.y[index], marker='o', label='forest with closed orbit')
ax.set_xlabel('s')
ax.set_ylabel('y')
ax.legend()