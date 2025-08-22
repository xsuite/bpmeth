import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_read_twiss import load_twiss

###########################################
# Compare twiss with tanh twiss (tw_tanh) #
###########################################

df_splines, scalars_splines = load_twiss("twissresults/twiss_splines")
df_tanh, scalars_tanh = load_twiss("twissresults/twiss_tanh")
df_merged = pd.merge_asof(df_tanh, df_splines, on="s", direction="nearest", tolerance=1e-10, suffixes=("_tanh", "_splines"))
df_merged = df_merged.dropna(subset=["betx_tanh", "betx_splines", "bety_tanh", "bety_splines"])

# Plotting the beta functions
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.betx_tanh, label='betx tanh')
ax.plot(df_merged.s, df_merged.betx_splines, label='betx splines')
ax.plot(df_merged.s, df_merged.bety_tanh, label='bety tanh')
ax.plot(df_merged.s, df_merged.bety_splines, label='bety splines')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta function (m)')
plt.legend()
plt.savefig("figures/ELENA_betas_tanhVSreal.png", dpi=500)
plt.close()

# Plotting beta beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.betx_splines - df_merged.betx_tanh) / df_merged.betx_tanh, label='Beta beating x')
ax.plot(df_merged.s, (df_merged.bety_splines - df_merged.bety_tanh) / df_merged.bety_tanh, label='Beta beating y')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta beating')
plt.legend() 
plt.savefig("figures/ELENA_betabeating_tanhVSreal.png", dpi=500)
plt.close()

print("Horizontal tune shift: ", scalars_splines['qx'] - scalars_tanh['qx'])
print("Vertical tune shift: ", scalars_splines['qy'] - scalars_tanh['qy'])

# Plotting the dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines, label='dispersion splines')
ax.plot(df_merged.s, df_merged.dx_tanh, label='dispersion tanh')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_dispersion_tanhVSreal.png", dpi=500)
plt.close()

# Plotting the normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines / df_merged.betx_splines, label='Normalized dispersion splines')
ax.plot(df_merged.s, df_merged.dx_tanh / df_merged.betx_tanh, label='Normalized dispersion tanh')
ax.set_xlabel('s (m)')
ax.set_ylabel('Normalized dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersion_tanhVSreal.png", dpi=500)
plt.close()

# Plotting dispersion beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines - df_merged.dx_tanh) / df_merged.dx_tanh, label='Dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_dispersionbeating_tanhVSreal.png", dpi=500)
plt.close()

# Dispersion beating in normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_tanh/df_merged.betx_tanh) / np.max(df_merged.dx_tanh/df_merged.betx_tanh), label='Normalized dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersionbeating_tanhVSreal.png", dpi=500)
plt.close()

print("Maximal normalized dispersion beating: ", np.max(np.abs((df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_tanh/df_merged.betx_tanh) / np.max(df_merged.dx_tanh/df_merged.betx_tanh))))

print("Horizontal chromaticity fieldmap: ", scalars_splines['dqx'])
print("Horizontal chromaticity tanh: ", scalars_tanh['dqx'])
print("Vertical chromaticity fieldmap: ", scalars_splines['dqy'])
print("Vertical chromaticity tanh: ", scalars_tanh['dqy'])

print("Horizontal chromaticity change: ", scalars_splines['dqx'] - scalars_tanh['dqx'])
print("Vertical chromaticity change: ", scalars_splines['dqy'] - scalars_tanh['dqy'])