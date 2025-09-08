import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from save_read_twiss import load_twiss

models = ["splines", "tanh", "design", "fitted"]

dfs = {}
scalars = {}

# Load twiss data for different models
for model in models:
    df, scalar = load_twiss(f"twissresults/twiss_{model}")
    dfs[model] = df
    scalars[model] = scalar

# Plot beta functions
fig, ax = plt.subplots()
for model in models:
    ax.plot(dfs[model].s, dfs[model].betx, label=f'betx {model}')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta x (m)')
plt.legend()
plt.savefig("figures/ELENA_betx_all_models.png", dpi=500)

fig, ax = plt.subplots()
for model in models:
    ax.plot(dfs[model].s, dfs[model].bety, label=f'bety {model}')
ax.set_xlabel('s (m)')
ax.set_ylabel('Beta y (m)')
plt.legend()
plt.savefig("figures/ELENA_bety_all_models.png", dpi=500)

for model in models:
    print("Horizontal tune for", model, ":", scalars[model]['qx'])
    
print()
for model in models:
    print("Vertical tune for", model, ":", scalars[model]['qy'])
    
print()
for model in models:
    print("Horizontal chromaticity for", model, ":", scalars[model]['dqx'])
    
print()
for model in models:
    print("Vertical chromaticity for", model, ":", scalars[model]['dqy'])