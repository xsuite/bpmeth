import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
from save_read_twiss import load_twiss

fig, ax = plt.subplots(3, figsize=(12, 10))

for model in ["splines", "design", "fitted"]:
    k2val, k3val = 0, 0
    qxlist = []
    qylist = []
    klist = []
    for k1val in [-0.25, -0.125, -0.01, 0, 0.01, 0.125, 0.25]:
        df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
        klist.append(k1val)
        qxlist.append(scalar['qx'])
        qylist.append(scalar['qy'])
    
    ax[0].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[0].plot(klist, qylist, label=f'Qy {model}', marker='.')
    
    k1val, k3val = 0, 0
    qxlist = []
    qylist = []
    klist = []
    for k2val in [-0.25, -0.125, -0.01, 0, 0.01, 0.125, 0.25]:
        df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
        klist.append(k2val)
        qxlist.append(scalar['qx'])
        qylist.append(scalar['qy'])
    
    ax[1].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[1].plot(klist, qylist, label=f'Qy {model}', marker='.')

    k1val, k2val = 0, 0
    qxlist = []
    qylist = []
    klist = []
    for k3val in [-0.25, -0.125, -0.01, 0, 0.01, 0.125, 0.25]:
        df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
        klist.append(k3val)
        qxlist.append(scalar['qx'])
        qylist.append(scalar['qy'])
    
    ax[2].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[2].plot(klist, qylist, label=f'Qy {model}', marker='.')

ax[0].set_xlabel('k1, with k2=0, k3=0')
ax[0].set_ylabel('Tunes')
ax[0].legend()
ax[1].set_xlabel('k2, with k1=0, k3=0')
ax[1].set_ylabel('Tunes')
ax[1].legend()
ax[2].set_xlabel('k3, with k1=0, k2=0')
ax[2].set_ylabel('Tunes')
ax[2].legend()
plt.savefig("figures/ELENA_tunes_vs_quad_strength.png", dpi=500)
plt.close()