from cpymad.madx import Madx

madx = Madx()

madx.call("acc-models-elena/elena.seq")
# Apertures:
madx.call("acc-models-elena/elena.dbx")
# Split elements to allow inserting BPMs:
madx.call("acc-models-elena/tools/splitEle_installBPM.madx")
# call strenghts for a given scenario
madx.call("acc-models-elena/scenarios/highenergy/highenergy.str")
# define beam
madx.call("acc-models-elena/scenarios/highenergy/highenergy.beam")


def get_madx_chrom(madx, chrom=False):
    madx.use("elena")
    madx.twiss(chrom=chrom)
    if chrom:  # Chromaticity as dQ/ddelta
        # Method with delta_s variation (a parameter to change the machine - not delta. 
        # Rescales the Hamiltonian, but is incorrect for small machines due to the bend where k != h, 
        # which is not the case since k is rescaled with (1+delta_s)
        corr = madx.sequence.elena.beam.beta
    else:
        corr = 1
    return (
        madx.table.twiss.summary.q1,
        madx.table.twiss.summary.q2,
        madx.table.twiss.summary.dq1/corr,
        madx.table.twiss.summary.dq2/corr,
    )


def get_madx_ptc(madx, time=False, exact=True):
    madx.use("elena")
    madx.input(
        f"""ptc_create_universe;
ptc_create_layout, time={time}, model=2, method=6, nst=5, exact={exact};
ptc_twiss, closed_orbit, icase=56, no=4, slice_magnets;
ptc_end;"""
    )
    if time:  # Longitudinal coordinate ptau
        corr = 1
    else:  # Longitudinal coordinate delta => correct dQ with ddelta/dptau=1/beta
        corr = madx.sequence.elena.beam.beta
        
    # Return dQ = dQ/dptau
    return (
        madx.table.ptc_twiss.summary.q1,
        madx.table.ptc_twiss.summary.q2,
        madx.table.ptc_twiss.summary.dq1 / corr,
        madx.table.ptc_twiss.summary.dq2 / corr,
    )


get_madx_chrom(madx,chrom=False)
get_madx_chrom(madx,chrom=True)  #????
get_madx_ptc(madx,time=False, exact=True)
get_madx_ptc(madx,time=True, exact=True)


import xtrack as xt

line = xt.Line.from_madx_sequence(madx.sequence.elena)
line.particle_ref = xt.Particles(p0c=0.1, mass0=0.938272, q0=1)

bend_fint = line["lnr.mbhek.0640.h2"].edge_exit_fint
print(f"Fringe field integral in lattice: {bend_fint}")
print(f"Fringe field integral from fieldmap: 0.42356")

tw = line.twiss4d()

print(f"{tw.qx}")
print(f"{tw.qy}")
print(f"{tw.dqx/madx.sequence.elena.beam.beta}")
print(f"{tw.dqy/madx.sequence.elena.beam.beta}")

line2 = line.copy()
for dipolename in ["lnr.mbhek.0135", "lnr.mbhek.0245", "lnr.mbhek.0335", "lnr.mbhek.0470", 
                   "lnr.mbhek.0560", "lnr.mbhek.0640"]:
    line2[f"{dipolename}.h1"].edge_entry_fint = 0
    line2[f"{dipolename}.h2"].edge_exit_fint = 0

print("Fringe field integral set to zero")

tw2 = line2.twiss4d()

print(f"{tw2.qx}")
print(f"{tw2.qy}")
print(f"{tw2.dqx/madx.sequence.elena.beam.beta}")
print(f"{tw2.dqy/madx.sequence.elena.beam.beta}")






