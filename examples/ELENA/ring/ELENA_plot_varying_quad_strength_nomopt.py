import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
from save_read_twiss import load_twiss

fig, ax = plt.subplots(3, figsize=(12, 10))

k1nom, k2nom, k3nom = 2.7423, -1.9514, 0.6381

for model in ["splines", "design", "fitted"]:
    k2val, k3val = k2nom, k3nom
    qxlist = []
    qylist = []
    klist = []
    for deltak1 in np.linspace(-0.25, 0.25, 5):
        k1val = k1nom + deltak1
        try:
            df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
            klist.append(k1val)
            qxlist.append(scalar['qx'])
            qylist.append(scalar['qy'])
        except FileNotFoundError:
            print(f"File not found for k1={k1val}, k2={k2val}, k3={k3val}")
            continue
    
    ax[0].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[0].plot(klist, qylist, label=f'Qy {model}', marker='.')
    
    k1val, k3val = k1nom, k3nom
    qxlist = []
    qylist = []
    klist = []
    for deltak2 in np.linspace(-0.25, 0.25, 5):
        k2val = k2nom + deltak2
        try:
            df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
            klist.append(k2val)
            qxlist.append(scalar['qx'])
            qylist.append(scalar['qy'])
        except FileNotFoundError:
            print(f"File not found for k1={k1val}, k2={k2val}, k3={k3val}")
            continue
    
    ax[1].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[1].plot(klist, qylist, label=f'Qy {model}', marker='.')

    k1val, k2val = k1nom, k2nom
    qxlist = []
    qylist = []
    klist = []
    for deltak3 in np.linspace(-0.25, 0.25, 5):
        k3val = k3nom + deltak3
        try:
            df, scalar = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3val:.2f}")
            klist.append(k3val)
            qxlist.append(scalar['qx'])
            qylist.append(scalar['qy'])
        except FileNotFoundError:
            print(f"File not found for k1={k1val}, k2={k2val}, k3={k3val}")
            continue
    
    ax[2].plot(klist, qxlist, label=f'Qx {model}', marker='.')
    ax[2].plot(klist, qylist, label=f'Qy {model}', marker='.')

ax[0].set_xlabel(f'k1, with k2={k2nom}, k3={k3nom}')
ax[0].set_ylabel('Tunes')
ax[0].legend()
ax[1].set_xlabel(f'k2, with k1={k1nom}, k3={k3nom}')
ax[1].set_ylabel('Tunes')
ax[1].legend()
ax[2].set_xlabel(f'k3, with k1={k1nom}, k2={k2nom}')
ax[2].set_ylabel('Tunes')
ax[2].legend()
plt.savefig("figures/ELENA_tunes_vs_quad_strength_nomopt.png", dpi=500)
plt.close()