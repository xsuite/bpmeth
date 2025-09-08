
import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
import json

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

###################
# Line from MAD-X #
###################

madx = Madx()
madx.call("acc-models-elena/elena.seq")
madx.call("acc-models-elena/elena.dbx")
madx.call("acc-models-elena/tools/splitEle_installBPM.madx")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.str")
madx.call("acc-models-elena/scenarios/highenergy/highenergy.beam")
madx.use("elena")

line_mad = xt.Line.from_madx_sequence(madx.sequence.elena, deferred_expressions=True)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice

# Flat 1 (injection = flat top): constant B = 0.3598 T (corresponding to p = 100 MeV/c 
# and a bending curvature radius of 0.927 m). 
# https://cds.cern.ch/record/1694484/files/CERN-2014-002.pdf
line_mad.particle_ref = xt.Particles(p0c=0.1e6, mass0=0.938272e6, q0=-1)
line_mad.configure_bend_model(core='adaptive', edge='full')

circum = 30.4055
c = 299792458
beta0 = line_mad.particle_ref.beta0[0]
frev = beta0 * c / circum

h = 1  # Injection harmonic number (section 3.4.3)
frf = h*frev
line_mad["lnr.acwo2.0530"].frequency = frf

voltage = 15
line_mad["lnr.acwo2.0530"].voltage = voltage

line_mad.twiss()

####################
# Zero quadrupoles #
####################

line = line_mad.copy()

k1val, k2val, k3val = 0, 0, 0

line.vars["lnr_kq1"] = k1val
line.vars["lnr_kq2"] = k2val
line.vars["lnr_kq3"] = k3val

#####################################
# Lumped map with fitted parameters #
#####################################

fitted_fint = 0.3918485
fitted_theta_E = 16.4347/180*np.pi
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")

    line[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_angle = fitted_theta_E
    line[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_angle = fitted_theta_E
    line[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_fint = fitted_fint
    line[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_fint = fitted_fint
    
tw_zero = line.twiss()
print(f"Tunes with zero quads: Qx = {tw_zero.qx:.6f}, Qy = {tw_zero.qy:.6f}")

##############
# Vary delta #
##############

ndd = 15
dds = np.linspace(-0.01, 0.01, ndd)
qxs = np.zeros(ndd)  # Horizontal tune
qys = np.zeros(ndd)  # Vertical tune
cos = np.zeros(ndd)  # Closed orbit
pts = np.zeros(ndd)  # pt of closed orbit

fig, ax = plt.subplots()
for i, dd in enumerate(dds):
    line.vars["mydelta"] = dd
    for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
        line[f'lnr.mbhek.{dipole_number}.h1'].k0 = f"{dipole_h} / (1+mydelta)"
        line[f'lnr.mbhek.{dipole_number}.h2'].k0 = f"{dipole_h} / (1+mydelta)"
        
    tw = line.twiss()
    
    qxs[i] = tw.qx
    qys[i] = tw.qy
    cos[i] = np.mean(tw.x) 
    pts[i] = np.mean(tw.ptau)
        
    ax.plot(tw.s, tw.x, label=f"delta={dd:.3f}")
ax.set_ylabel("x [m]")
ax.set_xlabel("s [m]")
ax.legend()
plt.savefig("figures/ELENA_zero_quads_with_perturbations_orbits.png", dpi=300)
plt.close()

deltas = np.sqrt(1 + 2*pts/beta0 + pts**2) - 1
print("Deltas: ", deltas)
    
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(dds, qxs, label="Qx", marker='.')
ax[1].plot(dds, qys, label="Qy", marker='.')
ax[0].set_ylabel("Qx")
ax[1].set_ylabel("Qy")
ax[1].set_xlabel("delta")    
plt.savefig("figures/ELENA_zero_quads_with_perturbations_tunes.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.plot(dds, cos, label="Closed orbit", marker='.')
ax.set_ylabel("Closed orbit [m]")
ax.set_xlabel("delta")
plt.savefig("figures/ELENA_zero_quads_with_perturbations_average_closed_orbit.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.plot(dds, pts, label="pt of closed orbit", marker='.')
ax.set_ylabel("pt of closed orbit")
ax.set_xlabel("delta")
plt.savefig("figures/ELENA_zero_quads_with_perturbations_average_pt.png", dpi=300)
plt.close()

    

