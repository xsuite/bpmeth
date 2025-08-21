import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
import json
from save_read_twiss import save_twiss

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

line_mad = xt.Line.from_madx_sequence(madx.sequence.elena)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line_mad.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')

############################################# 
# bpmeth dipole fieldmap model with splines #
#############################################

# My dipole with splines, matched such that the closed orbit remains zero
data = np.loadtxt("../dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

kf_splines=1.000447  # No closed orbit
# nphi=3 includes up to second order derivatives of b.
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=3, hgap=gap/2, nphi=3, plot=False, symmetric=True, step=50, radius=0.0025)
fringelen = (dipole_splines.smax - dipole_splines.smin - l_magn)/2

###############################################
# Line with lumped map with fitted parameters #
###############################################

fitted_fint = 0.3918485
fitted_theta_E = 16.4347/180*np.pi
fitted_bend = xt.Bend(length=l_magn, h=dipole_h, k0=dipole_h, edge_entry_fint=fitted_fint, edge_exit_fint=fitted_fint,
                      edge_entry_angle=fitted_theta_E, edge_exit_angle=fitted_theta_E, model='bend-kick-bend', 
                      edge_entry_hgap=gap/2, edge_exit_hgap=gap/2)
dd = xt.Drift(length=fringelen)
line_fitted_dipole = xt.Line(elements=[dd, fitted_bend, dd])
line_fitted_dipole.particle_ref = xt.Particles()
line_fitted_dipole.build_tracker()
p0 = line_fitted_dipole.build_particles()
mat_fitted = line_fitted_dipole.compute_one_turn_matrix_finite_differences(p0)['R_matrix']

line_mydipole = xt.Line(elements=[dipole_splines])
line_mydipole.particle_ref = xt.Particles()
line_mydipole.build_tracker()
p0 = line_mydipole.build_particles()
mat_mydipole = line_mydipole.compute_one_turn_matrix_finite_differences(p0, include_collective=True)['R_matrix']

# First determine the edge angle such that R21 is the same
print("fitted R21:   ", mat_fitted[1,0])
print("mydipole R21: ", mat_mydipole[1,0])
# Matched to 1e-8

# Then determine the fringe field integral such that R43 is the same
print("fitted R43:   ", mat_fitted[3,2])
print("mydipole R43: ", mat_mydipole[3,2])
# Matched to 1e-8

line_fitted = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_fitted.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")

    line_fitted[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_angle = fitted_theta_E
    line_fitted[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_angle = fitted_theta_E
    line_fitted[f"lnr.mbhek.{dipole_number}.h1"].edge_entry_fint = fitted_fint
    line_fitted[f"lnr.mbhek.{dipole_number}.h2"].edge_exit_fint = fitted_fint
    
tw_fitted = line_fitted.twiss4d()

save_twiss(tw_fitted, "twissresults/twiss_fitted")


################################
# # Include b3 in lumped model #
################################

# body_b3 = dipole_splines.fieldmap.integratedfield(3, zmin=s_max-l_magn/2, zmax=l_magn/2-s_max)[2] / (l_magn-2*s_max)
# body_b3l = body_b3 * l_magn
# # edge_b3l = dipole_splines.fieldmap.integratedfield(3, zmin=-s_min-l_magn/2, zmax=s_max-l_magn/2)[2]
# edge_b3l = dipole_edge.integratedfield(3, zmin=-0.2, zmax=0.2)[2] - body_b3 * s_max
# print("Integrated b3 body: ", body_b3l)
# print("Integrated b3 edge: ", edge_b3l)

# for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
#     dipole_s = line_fitted.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")

#     line_fitted[f"lnr.mbhek.{dipole_number}.h1"].knl[2] = body_b3l/2
#     line_fitted[f"lnr.mbhek.{dipole_number}.h2"].knl[2] = body_b3l/2
#     line_fitted.insert(f"sextupole.{dipole_number}.h1", xt.Multipole(knl=[0,0,edge_b3l]), at=dipole_s - line_fitted[f"lnr.mbhek.{dipole_number}.h1"].length)
#     line_fitted.insert(f"sextupole.{dipole_number}.h2", xt.Multipole(knl=[0,0,edge_b3l]), at=dipole_s + line_fitted[f"lnr.mbhek.{dipole_number}.h2"].length)
    
# tw_fitted = line_fitted.twiss4d()

# print("Horizontal chromaticity lumped map with sextupole component: ", tw_fitted.dqx)
# print("Vertical chromaticity lumped map with sextupole component: ", tw_fitted.dqy)

# This does NOT explain the chromaticity change we see!

# # without sextupole component in fieldmap?
# fitted_fint_without_sext = 0.39198805
# fitted_theta_E_without_sext = 16.4255725/180*np.pi
# fitted_bend_without_sext = xt.Bend(length=l_magn, h=dipole_h, k0=dipole_h, edge_entry_fint=fitted_fint_without_sext,
#                                    edge_exit_fint=fitted_fint_without_sext, edge_entry_angle=fitted_theta_E_without_sext, 
#                                    edge_exit_angle=fitted_theta_E_without_sext, model='bend-kick-bend',
#                                    edge_entry_hgap=gap/2, edge_exit_hgap=gap/2)
# line_fitted_without_sext = xt.Line(elements=[dd, fitted_bend_without_sext, dd])
# line_fitted_without_sext.particle_ref = xt.Particles()
# line_fitted_without_sext.build_tracker()
# p0 = line_fitted_without_sext.build_particles()
# mat_fitted_without_sext = line_fitted_without_sext.compute_one_turn_matrix_finite_differences(p0)['R_matrix']

# line_mydipole_without_sext = xt.Line(elements=[dipole_splines_without_sext])
# line_mydipole_without_sext.particle_ref = xt.Particles()
# line_mydipole_without_sext.build_tracker()
# p0 = line_mydipole_without_sext.build_particles()
# mat_mydipole_without_sext = line_mydipole_without_sext.compute_one_turn_matrix_finite_differences(p0, include_collective=True)['R_matrix']

# # First determine the edge angle such that R21 is the same
# print("fitted R21:   ", mat_fitted_without_sext[1,0])
# print("mydipole R21: ", mat_mydipole_without_sext[1,0])
# # Matched to 1e-8

# # Then determine the fringe field integral such that R43 is the same
# print("fitted R43:   ", mat_fitted_without_sext[3,2])
# print("mydipole R43: ", mat_mydipole_without_sext[3,2])
# # Matched to 1e-8