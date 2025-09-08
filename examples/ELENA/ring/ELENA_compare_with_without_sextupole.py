import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_read_twiss import load_twiss

###########################################################################
# Compare twiss with splines_without_sext twiss (tw_splines_without_sext) #
###########################################################################

df_splines, scalars_splines = load_twiss("twissresults/twiss_splines")
df_splines_without_sext, scalars_splines_without_sext = load_twiss("twissresults/twiss_splines_without_sext")
df_merged = pd.merge_asof(df_splines_without_sext, df_splines, on="s", direction="nearest", tolerance=1e-10, suffixes=("_splines_without_sext", "_splines"))
df_merged = df_merged.dropna(subset=["betx_splines_without_sext", "betx_splines", "bety_splines_without_sext", "bety_splines"])

# Plotting the beta functions
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.betx_splines_without_sext, label='betx splines_without_sext')
ax.plot(df_merged.s, df_merged.betx_splines, label='betx splines')
ax.plot(df_merged.s, df_merged.bety_splines_without_sext, label='bety splines_without_sext')
ax.plot(df_merged.s, df_merged.bety_splines, label='bety splines')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta function (m)')
plt.legend()
plt.savefig("figures/ELENA_betas_splines_without_sextVSreal.png", dpi=500)
plt.close()

# Plotting beta beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.betx_splines - df_merged.betx_splines_without_sext) / df_merged.betx_splines_without_sext, label='Beta beating x')
ax.plot(df_merged.s, (df_merged.bety_splines - df_merged.bety_splines_without_sext) / df_merged.bety_splines_without_sext, label='Beta beating y')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta beating')
plt.legend() 
plt.savefig("figures/ELENA_betabeating_splines_without_sextVSreal.png", dpi=500)
plt.close()

print("Horizontal tune shift: ", scalars_splines['qx'] - scalars_splines_without_sext['qx'])
print("Vertical tune shift: ", scalars_splines['qy'] - scalars_splines_without_sext['qy'])

# Plotting the dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines, label='dispersion splines')
ax.plot(df_merged.s, df_merged.dx_splines_without_sext, label='dispersion splines_without_sext')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_dispersion_splines_without_sextVSreal.png", dpi=500)
plt.close()

# Plotting the normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines / df_merged.betx_splines, label='Normalized dispersion splines')
ax.plot(df_merged.s, df_merged.dx_splines_without_sext / df_merged.betx_splines_without_sext, label='Normalized dispersion splines_without_sext')
ax.set_xlabel('s (m)')
ax.set_ylabel('Normalized dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersion_splines_without_sextVSreal.png", dpi=500)
plt.close()

# Plotting dispersion beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines - df_merged.dx_splines_without_sext) / df_merged.dx_splines_without_sext, label='Dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_dispersionbeating_splines_without_sextVSreal.png", dpi=500)
plt.close()

# Dispersion beating in normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_splines_without_sext/df_merged.betx_splines_without_sext) / np.max(df_merged.dx_splines_without_sext/df_merged.betx_splines_without_sext), label='Normalized dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersionbeating_splines_without_sextVSreal.png", dpi=500)
plt.close()

print("Maximal normalized dispersion beating: ", np.max(np.abs((df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_splines_without_sext/df_merged.betx_splines_without_sext) / np.max(df_merged.dx_splines_without_sext/df_merged.betx_splines_without_sext))))

print("Horizontal chromaticity fieldmap: ", scalars_splines['dqx'])
print("Horizontal chromaticity splines_without_sext: ", scalars_splines_without_sext['dqx'])
print("Vertical chromaticity fieldmap: ", scalars_splines['dqy'])
print("Vertical chromaticity splines_without_sext: ", scalars_splines_without_sext['dqy'])

print("Horizontal chromaticity change: ", scalars_splines['dqx'] - scalars_splines_without_sext['dqx'])
print("Vertical chromaticity change: ", scalars_splines['dqy'] - scalars_splines_without_sext['dqy'])