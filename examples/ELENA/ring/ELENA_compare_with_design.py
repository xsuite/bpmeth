import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_read_twiss import load_twiss

###############################################
# Compare twiss with design twiss (tw_design) #
###############################################

df_splines, scalars_splines = load_twiss("twissresults/twiss_splines")
df_design, scalars_design = load_twiss("twissresults/twiss_design")
df_merged = pd.merge_asof(df_design, df_splines, on="s", direction="nearest", tolerance=1e-10, suffixes=("_design", "_splines"))
df_merged = df_merged.dropna(subset=["betx_design", "betx_splines", "bety_design", "bety_splines"])

# Plotting the beta functions
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.betx_design, label='betx design')
ax.plot(df_merged.s, df_merged.betx_splines, label='betx splines')
ax.plot(df_merged.s, df_merged.bety_design, label='bety design')
ax.plot(df_merged.s, df_merged.bety_splines, label='bety splines')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta function (m)')
plt.legend()
plt.savefig("figures/ELENA_betas_designVSreal.png", dpi=500)
plt.close()

# Plotting beta beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.betx_splines - df_merged.betx_design) / df_merged.betx_design, label='Beta beating x')
ax.plot(df_merged.s, (df_merged.bety_splines - df_merged.bety_design) / df_merged.bety_design, label='Beta beating y')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta beating')
plt.legend() 
plt.savefig("figures/ELENA_betabeating_designVSreal.png", dpi=500)
plt.close()

print("Horizontal tune shift: ", scalars_splines['qx'] - scalars_design['qx'])
print("Vertical tune shift: ", scalars_splines['qy'] - scalars_design['qy'])

# Plotting the dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines, label='dispersion splines')
ax.plot(df_merged.s, df_merged.dx_design, label='dispersion design')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_dispersion_designVSreal.png", dpi=500)
plt.close()

# Plotting the normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines / df_merged.betx_splines, label='Normalized dispersion splines')
ax.plot(df_merged.s, df_merged.dx_design / df_merged.betx_design, label='Normalized dispersion design')
ax.set_xlabel('s (m)')
ax.set_ylabel('Normalized dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersion_designVSreal.png", dpi=500)
plt.close()

# Plotting dispersion beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines - df_merged.dx_design) / df_merged.dx_design, label='Dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_dispersionbeating_designVSreal.png", dpi=500)
plt.close()

# Dispersion beating in normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_design/df_merged.betx_design) / np.max(df_merged.dx_design/df_merged.betx_design), label='Normalized dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersionbeating_designVSreal.png", dpi=500)
plt.close()

print("Maximal normalized dispersion beating: ", np.max(np.abs((df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_design/df_merged.betx_design) / np.max(df_merged.dx_design/df_merged.betx_design))))

print("Horizontal chromaticity fieldmap: ", scalars_splines['dqx'])
print("Horizontal chromaticity design: ", scalars_design['dqx'])
print("Vertical chromaticity fieldmap: ", scalars_splines['dqy'])
print("Vertical chromaticity design: ", scalars_design['dqy'])

print("Horizontal chromaticity change: ", scalars_splines['dqx'] - scalars_design['dqx'])
print("Vertical chromaticity change: ", scalars_splines['dqy'] - scalars_design['dqy'])