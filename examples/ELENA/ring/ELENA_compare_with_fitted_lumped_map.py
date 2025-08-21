import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_read_twiss import load_twiss

##################################
# Compare with fitted lumped map #
##################################

df_splines, scalars_splines = load_twiss("twissresults/twiss_splines")
df_fitted, scalars_fitted = load_twiss("twissresults/twiss_fitted")
df_merged = pd.merge_asof(df_fitted, df_splines, on="s", direction="nearest", tolerance=1e-10, suffixes=("_fitted", "_splines"))
df_merged = df_merged.dropna(subset=["betx_fitted", "betx_splines", "bety_fitted", "bety_splines"])

# Plotting the beta functions to compare
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.betx_fitted, label='betx fitted')
ax.plot(df_merged.s, df_merged.betx_splines, label='betx splines')
ax.plot(df_merged.s, df_merged.bety_fitted, label='bety fitted')
ax.plot(df_merged.s, df_merged.bety_splines, label='bety splines')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta function (m)')
plt.legend()
plt.savefig("figures/ELENA_betas_fittedVSreal.png", dpi=500)
plt.close()

# Beta beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.betx_splines - df_merged.betx_fitted) / df_merged.betx_fitted, label='Beta beating x')
ax.plot(df_merged.s, (df_merged.bety_splines - df_merged.bety_fitted) / df_merged.bety_fitted, label='Beta beating y')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta beating')
plt.legend() 
plt.savefig("figures/ELENA_betabeating_fittedVSreal.png", dpi=500)
plt.close()

print("Maximal value horizontal beta beating: ", np.max(np.abs((df_merged.betx_splines - df_merged.betx_fitted) / df_merged.betx_fitted)))
print("Maximal value vertical beta beating: ", np.max(np.abs((df_merged.bety_splines - df_merged.bety_fitted) / df_merged.bety_fitted)))

print("Horizontal tune: ", scalars_fitted['qx'])
print("Vertical tune: ", scalars_fitted['qy'])
print("Horizontal tune shift: ", scalars_splines['qx'] - scalars_fitted['qx'])
print("Vertical tune shift: ", scalars_splines['qy'] - scalars_fitted['qy'])

# Plotting the dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines, label='dispersion splines')
ax.plot(df_merged.s, df_merged.dx_fitted, label='dispersion fitted')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_dispersion_fittedVSreal.png", dpi=500)
plt.close()

# plotting the normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_splines / df_merged.betx_splines, label='Normalized dispersion splines')
ax.plot(df_merged.s, df_merged.dx_fitted / df_merged.betx_fitted, label='Normalized dispersion fitted')
ax.set_xlabel('s (m)')
ax.set_ylabel('Normalized dispersion (m)')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersion_fittedVSreal.png", dpi=500)
plt.close()

# Dispersion beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines - df_merged.dx_fitted) / df_merged.dx_fitted, label='Dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_dispersionbeating_fittedVSreal.png", dpi=500)
plt.close()

# Dispersion beating in normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_fitted/df_merged.betx_fitted) / np.max(df_merged.dx_fitted/df_merged.betx_fitted), label='Normalized dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/ELENA_normalizeddispersionbeating_fittedVSreal.png", dpi=500)
plt.close()

print("Maximal normalized dispersion beating: ", np.max(np.abs((df_merged.dx_splines/df_merged.betx_splines - df_merged.dx_fitted/df_merged.betx_fitted) / np.max(df_merged.dx_fitted/df_merged.betx_fitted))))

print("Horizontal chromaticity fieldmap: ", scalars_splines['dqx'])
print("Horizontal chromaticity fitted: ", scalars_fitted['dqx'])
print("Vertical chromaticity fieldmap: ", scalars_splines['dqy'])
print("Vertical chromaticity fitted: ", scalars_fitted['dqy'])

print("Horizontal chromaticity change: ", scalars_splines['dqx'] - scalars_fitted['dqx'])
print("Vertical chromaticity change: ", scalars_splines['dqy'] - scalars_fitted['dqy'])



