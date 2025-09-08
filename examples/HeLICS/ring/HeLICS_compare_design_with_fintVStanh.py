import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("../../ELENA/ring"))
from save_read_twiss import load_twiss

###################################################################
# Compare twiss with design_with_fint twiss (tw_design_with_fint) #
###################################################################

df_tanh, scalars_tanh = load_twiss("twissresults/twiss_tanh")
df_design_with_fint, scalars_design_with_fint = load_twiss("twissresults/twiss_design_with_fint")
df_merged = pd.merge_asof(df_design_with_fint, df_tanh, on="s", direction="nearest", tolerance=1e-10, suffixes=("_design_with_fint", "_tanh"))
df_merged = df_merged.dropna(subset=["betx_design_with_fint", "betx_tanh", "bety_design_with_fint", "bety_tanh"])

# Plotting the beta functions
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.betx_design_with_fint, label='betx design_with_fint')
ax.plot(df_merged.s, df_merged.betx_tanh, label='betx tanh')
ax.plot(df_merged.s, df_merged.bety_design_with_fint, label='bety design_with_fint')
ax.plot(df_merged.s, df_merged.bety_tanh, label='bety tanh')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta function (m)')
plt.legend()
plt.savefig("figures/HeLICS_betas_design_with_fintVStanh.png", dpi=500)
plt.close()

# Plotting beta beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.betx_tanh - df_merged.betx_design_with_fint) / df_merged.betx_design_with_fint, label='Beta beating x')
ax.plot(df_merged.s, (df_merged.bety_tanh - df_merged.bety_design_with_fint) / df_merged.bety_design_with_fint, label='Beta beating y')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta beating')
plt.legend() 
plt.savefig("figures/HeLICS_betabeating_design_with_fintVStanh.png", dpi=500)
plt.close()

print("Horizontal tune shift: ", scalars_tanh['qx'] - scalars_design_with_fint['qx'])
print("Vertical tune shift: ", scalars_tanh['qy'] - scalars_design_with_fint['qy'])

# Plotting the dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_tanh, label='dispersion tanh')
ax.plot(df_merged.s, df_merged.dx_design_with_fint, label='dispersion design_with_fint')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion (m)')
plt.legend()
plt.savefig("figures/HeLICS_dispersion_design_with_fintVStanh.png", dpi=500)
plt.close()

# Plotting the normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, df_merged.dx_tanh / df_merged.betx_tanh, label='Normalized dispersion tanh')
ax.plot(df_merged.s, df_merged.dx_design_with_fint / df_merged.betx_design_with_fint, label='Normalized dispersion design_with_fint')
ax.set_xlabel('s (m)')
ax.set_ylabel('Normalized dispersion (m)')
plt.legend()
plt.savefig("figures/HeLICS_normalizeddispersion_design_with_fintVStanh.png", dpi=500)
plt.close()

# Plotting dispersion beating
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_tanh - df_merged.dx_design_with_fint) / df_merged.dx_design_with_fint, label='Dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/HeLICS_dispersionbeating_design_with_fintVStanh.png", dpi=500)
plt.close()

# Dispersion beating in normalized dispersion
fig, ax = plt.subplots()
ax.plot(df_merged.s, (df_merged.dx_tanh/df_merged.betx_tanh - df_merged.dx_design_with_fint/df_merged.betx_design_with_fint) / np.max(df_merged.dx_design_with_fint/df_merged.betx_design_with_fint), label='Normalized dispersion beating')
ax.set_xlabel('s (m)')
ax.set_ylabel('Dispersion beating')
plt.legend()
plt.savefig("figures/HeLICS_normalizeddispersionbeating_design_with_fintVStanh.png", dpi=500)
plt.close()

print("Maximal normalized dispersion beating: ", np.max(np.abs((df_merged.dx_tanh/df_merged.betx_tanh - df_merged.dx_design_with_fint/df_merged.betx_design_with_fint) / np.max(df_merged.dx_design_with_fint/df_merged.betx_design_with_fint))))

print("Horizontal chromaticity fieldmap: ", scalars_tanh['dqx'])
print("Horizontal chromaticity design_with_fint: ", scalars_design_with_fint['dqx'])
print("Vertical chromaticity fieldmap: ", scalars_tanh['dqy'])
print("Vertical chromaticity design_with_fint: ", scalars_design_with_fint['dqy'])

print("Horizontal chromaticity change: ", scalars_tanh['dqx'] - scalars_design_with_fint['dqx'])
print("Vertical chromaticity change: ", scalars_tanh['dqy'] - scalars_design_with_fint['dqy'])