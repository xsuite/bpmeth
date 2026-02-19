#!/usr/bin/env python
# coding: utf-8


import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
import scipy as sc
import os



# # ELENA lattice


########################################
# Clone acc-models repository of ELENA #
########################################

repo_url = "https://gitlab.cern.ch/acc-models/acc-models-elena.git"
folder_name = "acc-models-elena"

if os.path.isdir(folder_name):
    print(f"Folder '{folder_name}' already exists. Skipping clone.")
else:
    print(f"Folder '{folder_name}' not found. Cloning repository...")
    ret = os.system(f"git clone {repo_url}")
    if ret != 0:
        raise RuntimeError("git clone failed")
    print("Clone completed.")



##########################
# ELENA lattice in MAD-X #
##########################

madx = Madx()
madx.call("acc-models-elena/elena.seq")
madx.call("acc-models-elena/elena.dbx")
madx.call("acc-models-elena/tools/splitEle_installBPM.madx")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.str")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.beam")
madx.use("elena")


#########################
# ELENA model in XSuite #
#########################

line = xt.Line.from_madx_sequence(madx.sequence.elena, deferred_expressions=True)
start_elem = "lnr.vvgbf.0114"
line.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line.particle_ref = xt.Particles(p0c=0.1e9, mass0=0.938272e9+2*511e3, q0=-1)
line.configure_bend_model(core='adaptive', edge='full')
line.configure_drift_model("exact")
tw = line.twiss4d()
tw.plot()


# # Dipole

############################
# Magnet design parameters #
############################

phi = 60/180*np.pi
rho = 0.927  # https://edms.cern.ch/ui/file/1311860/2.0/LNA-MBHEK-ER-0001-20-00.pdf
dipole_h = 1/rho
l_magn = rho*phi
gap=0.076
theta_E = 17/180*np.pi
fint = 0.424
B_dip_T = 0.42881  # Design dipole field in Tesla


############################################# 
# bpmeth dipole fieldmap model with splines #
#############################################

data = np.loadtxt("../dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]  # Fieldmap in Tesla
Brho = B_dip_T * rho  # T.m
dipole = bpmeth.Fieldmap(data)
dipole.rescale(1/Brho)

radius=0.0025
dipole_FS = dipole.calc_FS_coords(XFS=np.linspace(-0.025, 0.025, 51), YFS=[0], SFS=np.arange(-l_magn, l_magn, 0.001), rho=rho, phi=phi, radius=radius)
dipole_FS = dipole_FS.symmetrize(radius=radius)
fig, ax = plt.subplots(figsize=(15, 3))
zvals, coeffs, coeffsstd = dipole_FS.z_multipoles(3, ax=ax)  # Fitted only normal multipoles! 
ax.set_ylim(-15, 20)


plt.scatter(dipole.src['z'], dipole.src["x"], label="Fieldmap")
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.xlabel("z")
plt.ylabel('x')




allowed_length = line["drift_18"].length*2 + line["lnr.mbhek.0245.h1"].length*2  # This is the space available in the ring for the dipole until the next element, so the fringes overlap!
print("Total allowed length of the dipole:", allowed_length)
print("Half the allowed magnet length: ", allowed_length/2)


max_multipole = 3  # Include up to sextupoles in the dipole. Higher only supported when modified in source code.
fringe1_start = -0.76
fringe1_end = -0.35
magnet_start = -0.78

fringe2_start = -fringe1_end
fringe2_end = -fringe1_start

n_splits = 11  # number of polynomial pieces in fringe field
step = (np.abs(zvals-fringe1_end).argmin() - np.abs(zvals-fringe1_start).argmin()) // n_splits
split1_inds = np.sort([np.abs(zvals-fringe1_start).argmin() + i*step for i in range(n_splits+1)])
split2_inds = np.sort([np.abs(zvals-fringe2_end).argmin() - i*step for i in range(n_splits+1)])

start1_ind = split1_inds[0]
end1_ind = split1_inds[-1]
start2_ind = split2_inds[0]
end2_ind = split2_inds[-1]
magnet_start_ind = np.abs(zvals-magnet_start).argmin()
magnet_end_ind = np.abs(zvals+magnet_start).argmin()


# Calculate average body fields, to have constant body. For dipole component the design field is chosen instead.
i1, i2 = split1_inds[-1], split2_inds[0]
body_fields = [np.trapezoid(coeffs[i1:i2+1, 0], zvals[i1:i2+1]) / (zvals[i2]-zvals[i1]), 
               np.trapezoid(coeffs[i1:i2+1, 1], zvals[i1:i2+1]) / (zvals[i2]-zvals[i1]), 
               np.trapezoid(coeffs[i1:i2+1, 2], zvals[i1:i2+1]) / (zvals[i2]-zvals[i1]),
               np.trapezoid(coeffs[i1:i2+1, 3], zvals[i1:i2+1]) / (zvals[i2]-zvals[i1])
]
print(body_fields)


################################
# Fit spline to each multipole #
################################

all_pols = []
for index in range(max_multipole):
    # The actual data
    b = coeffs[:, index]
    dz = np.diff(zvals, prepend=zvals[0])
    bp = np.nan_to_num(np.diff(b,prepend=b[0])/dz, copy=True)

    # The points to fit
    x = zvals.copy()
    y = b.copy()
    y[start1_ind] = 0  # Force field to be zero at the start of the fringe
    y[end1_ind] = body_fields[index]  # Force field to be equal to body value at the end of the fringe
    y[start2_ind] = body_fields[index]  
    y[end2_ind] = 0

    yp = bp.copy()
    yp[start1_ind] = 0  # Force derivative to be zero at the start of the fringe
    yp[end1_ind] = 0  # Force derivative to be zero at the end of the fringe
    yp[start2_ind] = 0
    yp[end2_ind] = 0

    # Fit the different pieces
    pols = [np.zeros(4)]
    for ia,ib in zip(split1_inds[:-1],split1_inds[1:]):
        pol=bpmeth.fit_segment(ia,ib,x,y,yp)        
        pols.append(pol)
    pols.append(np.array([body_fields[index], 0, 0, 0]))
    for ia,ib in zip(split2_inds[:-1],split2_inds[1:]):
        pol=bpmeth.fit_segment(ia,ib,x,y,yp)        
        pols.append(pol)
    pols.append(np.zeros(4))
    all_pols.append(pols)

# Save the segment boundaries
segments = []
segment_inds = []
segments.append((zvals[magnet_start_ind], zvals[start1_ind]))
segment_inds.append([magnet_start_ind, start1_ind])
for ia,ib in zip(split1_inds[:-1],split1_inds[1:]):
    segment_inds.append([ia, ib])
    segments.append([zvals[ia], zvals[ib]])
segment_inds.append([end1_ind, start2_ind])
segments.append((zvals[end1_ind], zvals[start2_ind]))
for ia,ib in zip(split2_inds[:-1],split2_inds[1:]):
    segment_inds.append([ia, ib])
    segments.append([zvals[ia], zvals[ib]])
segments.append((zvals[end2_ind], zvals[magnet_end_ind]))
segment_inds.append([end2_ind, magnet_end_ind])

all_pols = np.array(all_pols)  # Shape (n_multipoles, n_segments, order+1=4)

fig, ax = plt.subplots(figsize=(15, 3))
plt.axvline(x=-l_magn/2, color="gray", linestyle="--")
for index in range(max_multipole):
    pols = []

    # The actual data
    b = coeffs[:, index]

    for (ia, ib), pol in zip(segment_inds, all_pols[index]):
        bpmeth.plot_fit(ia,ib,zvals,b,pol,ax=ax,data=True)


def pol_to_comp(pols):
    """
    :param part: xt.Particles
    :param pols: Polynomial coefficients from the fit for this segment, 
        first index is the multipole order (only normal components) b1, b2, b3..., 
        second index is the order of the term in the cubic polynomial (zero = constant)
    :return: comp, the input needed for the segment tracker
    """

    # comp has bs, b1, a1, b2, a2, ... as first index, and the order of the term in the cubic polynomial as second index (zero = constant)
    comp = np.zeros((9, 4), dtype=np.float64) 

    for mult in range(len(pols)):
        comp[2*mult+1] = pols[mult]

    return comp


class Magnet:
    isthick=True
    def __init__(self, segments, all_pols, z_edge, rho, ds=0.01):
        """
        :param segments: List of (a,b) tuples defining the segments along z, entrance half of the magnet.
        :param all_pols: For each multipole, list of lists of polynomials for each segment, entrance half 
            of the magnet. The order of the multipoles is len(all_pols).
        :param z_edge: Longitudinal position of the edge of the magnet, to determine curvature.
        :param rho: Bending radius of the magnet.
        """
        self.segments = segments
        self.all_pols = all_pols

        self.z_edge = z_edge
        self.rho = rho

        self.order = len(all_pols)  

        self.length = segments[-1][1] - segments[0][0]
        self.build_tracker()
        self.ds = ds

    def build_tracker(self):
        self.polysegments = []
        for i, segment in enumerate(self.segments):
            comp = pol_to_comp(self.all_pols[:, i])

            if segment[0] < -self.z_edge:
                h = 0.  # Magic, this has to be a float otherwise it is extremely slow
                s_start = segment[0]
                s_end = np.min([segment[1], -self.z_edge])
                self.polysegments.append(bpmeth.PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))
            if segment[1] > -self.z_edge and segment[0] < self.z_edge:
                h = 1/self.rho
                s_start = np.max([segment[0], -self.z_edge])
                s_end = np.min([segment[1], self.z_edge])
                self.polysegments.append(bpmeth.PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))
            if segment[1] > self.z_edge:
                h = 0.
                s_start = np.max([segment[0], self.z_edge])
                s_end = segment[1]
                self.polysegments.append(bpmeth.PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))

    def rescale(self, scalefactor):
        self.all_pols = np.array([[pol*scalefactor for pol in pols] for pols in self.all_pols])
        self.build_tracker()
        return self

    def copy(self):
        return Magnet(self.segments, self.all_pols, self.z_edge, self.rho)

    def track(self, particles):
        for i, polysegment in enumerate(self.polysegments):
            polysegment.track(particles, ds=self.ds)

    def track_step_by_step(self, particles):
        out = []
        for polysegment in self.polysegments:
            seg_out = polysegment.track_step_by_step(particles, ds=self.ds)
            out.extend(seg_out)
        return out

    def track_segment_by_segment(self, particles):
        out = [particles.copy()]
        for polysegment in self.polysegments:
            polysegment.track(particles, ds=self.ds)
            out.append(particles.copy())
        return out

    def to_dict(self):
        out = {"segments": self.segments, "all_pols": self.all_pols, "z_edge": self.z_edge, "rho": self.rho}
        return out


Melvin = Magnet(segments, all_pols, l_magn/2, rho, 0.001)


###################################################
# Rescale magnet to get zero closed orbit at exit #
################################################### 

def objective(scalefactor):
    print("Trying scal factor:", scalefactor, end='\r')
    pp = xt.Particles(s=magnet_start)
    magnet = Melvin.copy()
    magnet.rescale(scalefactor)
    magnet.track(pp)
    return abs(np.mean(pp.x))

bounds = (1.0088, 1.0095)
result = sc.optimize.minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol":1e-8})
scalefactor = result.x
print(f"Best scale factor: {scalefactor}")
print(f"Resulting (absolute value) closed orbit at exit: {objective(scalefactor)} m")

Melvin.rescale(scalefactor)

pp = xt.Particles(s=magnet_start, x=[-0.01, 0, 0.01])
out = Melvin.track_segment_by_segment(pp)
plt.plot([p.s for p in out], [p.x for p in out], marker='o', ls='--')
plt.axvline(x=-l_magn/2, color="gray", ls="--")
plt.axvline(x=l_magn/2, color="gray", ls="--")


# # Twiss ELENA with dipole fieldmap

###################
# Replace dipoles #
###################

tw0 = line.twiss4d()
tw0.plot()

for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line.remove(f"lnr.mbhek.{dipole_number}.h1")
    line.remove(f"lnr.mbhek.{dipole_number}.m")
    line.remove(f"lnr.mbhek.{dipole_number}.h2")
    line.insert(f"Melvin_{dipole_number}", Melvin, at=dipole_s)
tab = line.get_table()


tab.rows[10:20]

tw = line.twiss4d(include_collective=True)

tw.plot()

fig, ax = plt.subplots(2, figsize=(9.5, 6))
pp1 = tw.plot(ax=ax[1])
pp1.left.set_ylim((0, 15))
pp1.right.set_ylim((0, 1.6))
ax[1].set_title("Fieldmap dipoles")
pp0 = tw0.plot(ax=ax[0])
pp0.left.set_ylim((0, 15))
pp0.right.set_ylim((0, 1.6))
ax[0].set_title("Current acc-models dipoles")
plt.tight_layout()
plt.savefig("Twiss comparison.png", dpi=300)






