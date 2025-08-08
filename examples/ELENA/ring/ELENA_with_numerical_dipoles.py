import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp


# ELENA lattice in MAD-X
madx = Madx()
madx.call("ring/acc-models-elena/elena.seq")
madx.call("ring/acc-models-elena/elena.dbx")
madx.call("ring/acc-models-elena/tools/splitEle_installBPM.madx")
madx.call("ring/acc-models-elena/scenarios/highenergy/highenergy.str")
madx.call("ring/acc-models-elena/scenarios/highenergy/highenergy.beam")
madx.use("elena")


# ELENA model in XSuite
line_mad = xt.Line.from_madx_sequence(madx.sequence.elena)
start_elem = "lnr.vvgbf.0114"
line_mad.cycle(name_first_element=start_elem, inplace=True)  # Such that dipole is not spanning over end-beginning of lattice
line_mad.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)
line_mad.configure_bend_model(core='adaptive', edge='full')
tw = line_mad.twiss4d()
tw.plot()

# Magnet parameters
phi = 60/180*np.pi
rho = 0.927  # https://edms.cern.ch/ui/file/1311860/2.0/LNA-MBHEK-ER-0001-20-00.pdf
dipole_h = 1/rho
l_magn = rho*phi
gap=0.076
theta_E = 17/180*np.pi

# My dipole with splines, matched such that the closed orbit remains zero
data = np.loadtxt("dipole/ELENA_fieldmap.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]

kf_splines=1.000447
dipole_splines = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=3, hgap=gap/2, nphi=2, plot=True, symmetric=True, step=50, radius=0.0025)
dipole_splines_without_sext = bpmeth.MagnetFromFieldmap(data, 1/rho, l_magn, design_field=1/rho*kf_splines, order=2, hgap=gap/2, nphi=2, plot=True, symmetric=True, step=50, radius=0.0025)

# Edge of magnet fieldmap
xmin, xmax, n_x = -0.4*gap, 0.4*gap, 51
zmin, zmax, n_z = -0.2*l_magn, 0.2*l_magn, 201
xarr = np.linspace(xmin, xmax, n_x)
yarr = [0]
zarr = np.linspace(zmin, zmax, n_z)

data_edge = np.loadtxt("dipole/ELENA_edge.csv", skiprows=1, delimiter=",")[:, [0,1,2,7,8,9]]
dipole_edge = bpmeth.Fieldmap(data_edge)
central_field_from_enge_fit = 4.28304565e-01  # Determined from directly fitting fieldmap without rescaling

dipole_edge.rescale(dipole_h /central_field_from_enge_fit)
dipole_edge = dipole_edge.calc_edge_frame(xarr, yarr, zarr, theta_E, rho, phi, radius=0.005)
fig, ax = plt.subplots()
guess = [dipole_h,  5.45050869e+02, -1.34905910e+02, 2.84624272e+01,  5.49796478e-01]
# guess = [5.375e-7, -3.85190565e+03, 1.29508736e+03, -1.29954076e+02, 2.50076967e+01, 5.22942000e-01]
engeparams, engecov = dipole_edge.fit_multipoles(bpmeth.spEnge, components=[1], design=1, nparams=5, ax=ax,
                                                 guess = guess, padding=True, entrance=False, design_field=dipole_h)

# Compare fieldmap with model based on edge enge and rotation
s = sp.symbols('s')
b1 = sp.Function('b1')(s)
b1model = bpmeth.spEnge(-s, *engeparams[0])

dipole_model = bpmeth.GeneralVectorPotential(b=(b1model,))
dipole_model_FS = dipole_model.transform(theta_E=theta_E, maxpow=3)

fig, ax = plt.subplots()
dipole_splines.fieldmap.z_multipoles(2, ax=ax)
dipole_model_FS.plot_components(smin=-0.5, smax=0.5, s_shift=-l_magn/2, ax=ax)


# Line with replaced dipoles with sextupole component
line_splines = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_mad.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.h1")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.m")
    line_splines.remove(f"lnr.mbhek.{dipole_number}.h2")
    line_splines.insert(f"spline_{dipole_number}", dipole_splines, at=dipole_s)
tw_splines = line_splines.twiss4d(include_collective=True)

# Line with replaced dipoles without sextupole component
line_splines_without_sext = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    dipole_s = line_mad.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.h1")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.m")
    line_splines_without_sext.remove(f"lnr.mbhek.{dipole_number}.h2")
    line_splines_without_sext.insert(f"spline_{dipole_number}", dipole_splines, at=dipole_s)
tw_splines_without_sext = line_splines_without_sext.twiss4d(include_collective=True)

# XSuite line with same body sextupole as the fieldmap of the magnet
body_smin = -0.35
body_smax = 0.35
k2 = dipole_splines.fieldmap.integratedfield(3, zmin=body_smin, zmax=body_smax)[2] / (body_smax - body_smin) # Average sextupole component of the body
k2l = k2 * l_magn

line_mad_with_sext = line_mad.copy()
for dipole_number in ["0135", "0245", "0335", "0470", "0560", "0640"]:
    line_mad_with_sext[f"lnr.mbhek.{dipole_number}.h1"].knl[2] = k2l/2
    line_mad_with_sext[f"lnr.mbhek.{dipole_number}.h2"].knl[2] = k2l/2
tw_with_sext = line_mad_with_sext.twiss4d()


# Print results to compare. Main difference is in chromaticity, not in the linear optics
print("XSuite qx: ", tw.qx)
print("Xsuite with sextupole qx:", tw_with_sext.qx)
print("Fieldmap qx:", tw_splines.qx)
print("Fieldmap without sextupole component qx:", tw_splines_without_sext.qx)
print("XSuite qy: ", tw.qy)
print("XSuite with sextupole qy:", tw_with_sext.qy)
print("Fieldmap qy:", tw_splines.qy)
print("Fieldmap without sextupole component qy:", tw_splines_without_sext.qy)
print("XSuite dqx: ", tw.dqx)
print("XSuite with sextupole dqx:", tw_with_sext.dqx)
print("Fieldmap dqx:", tw_splines.dqx)
print("Fieldmap without sextupole component dqx:", tw_splines_without_sext.dqx)
print("XSuite dqy: ", tw.dqy)
print("XSuite with sextupole dqy:", tw_with_sext.dqy)
print("Fieldmap dqy:", tw_splines.dqy)
print("Fieldmap without sextupole component dqy:", tw_splines_without_sext.dqy)


# # Dipole fitted with enge function
# # No longer used since it does not describe the sextupole component properly
# kf_enge=1.0008781
# guess = [ 1.09058298e+00, 3.19608461e+02, -1.03315497e+02, 2.80836370e+01, 4.23617617e-01]
# dipole_enge = bpmeth.DipoleFromFieldmap(data, 1/rho, l_magn,
#                                    design_field=dipole_h*kf_enge, shape="enge", hgap=gap/2, apt=gap,
#                                    radius=0.0025, order=2, plot=True, nphi=2,
#                                    guess = guess )
# 
# line_enge = line_mad.copy()
# for dipole_number in ["0135"]:#, "0245", "0335", "0470", "0560", "0640"]:
#     dipole_s = line_mad.get_s_position(at_elements=f"lnr.mbhek.{dipole_number}.m")
#     line_enge.remove(f"lnr.mbhek.{dipole_number}.h1")
#     line_enge.remove(f"lnr.mbhek.{dipole_number}.m")
#     line_enge.remove(f"lnr.mbhek.{dipole_number}.h2")
# 
# tw_enge = line_enge.twiss4d(include_collective=True,compute_chromatic_properties=False)