import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
from save_read_twiss import load_twiss

k1nom, k2nom, k3nom = 2.7423, -1.9514, 0.6381
for model in ["design", "fitted", "splines"]:
    k2val, k3val = k2nom, k3nom
    k1_left = k1nom-0.01
    k1_right = k1nom+0.01
    try:
        df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1_left:.2f}_{k2val:.2f}_{k3val:.2f}")
        df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1_right:.2f}_{k2val:.2f}_{k3val:.2f}")
        
        qx_left = scalar_left['qx']
        qx_right = scalar_right['qx']
        dqx_dk1 = (qx_right - qx_left) / (k1_right - k1_left)
        print(f"dqx/dk1 for {model}: {dqx_dk1}")
    
        qy_left = scalar_left['qy']
        qy_right = scalar_right['qy']
        dqy_dk1 = (qy_right - qy_left) / (k1_right - k1_left)
        print(f"dqy/dk1 for {model}: {dqy_dk1}")
        
    except FileNotFoundError:
        print(f"File not found for k1={k1_left} or k1={k1_right}, k2={k2val}, k3={k3val}")
        continue
    
    k1val, k3val = k1nom, k3nom
    k2_left = k2nom-0.01
    k2_right = k2nom+0.01
    try:
        df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2_left:.2f}_{k3val:.2f}")
        df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2_right:.2f}_{k3val:.2f}")
        
        qx_left = scalar_left['qx']
        qx_right = scalar_right['qx']
        dqx_dk2 = (qx_right - qx_left) / (k2_right - k2_left)
        print(f"dqx/dk2 for {model}: {dqx_dk2}")
        
        qy_left = scalar_left['qy']
        qy_right = scalar_right['qy']
        dqy_dk2 = (qy_right - qy_left) / (k2_right - k2_left)
        print(f"dqy/dk2 for {model}: {dqy_dk2}")
    
    except FileNotFoundError:
        print(f"File not found for k2={k2_left} or k2={k2_right}, k1={k1val}, k3={k3val}")
        continue
    
    k1val, k2val = k1nom, k2nom
    k3_left = k3nom-0.01
    k3_right = k3nom+0.01
    try:
        df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3_left:.2f}")
        df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3_right:.2f}")
        
        qx_left = scalar_left['qx']
        qx_right = scalar_right['qx']
        dqx_dk3 = (qx_right - qx_left) / (k3_right - k3_left)
        print(f"dqx/dk3 for {model}: {dqx_dk3}")
        
        qy_left = scalar_left['qy']
        qy_right = scalar_right['qy']
        dqy_dk3 = (qy_right - qy_left) / (k3_right - k3_left)
        print(f"dqy/dk3 for {model}: {dqy_dk3}")
    
    except FileNotFoundError:
        print(f"File not found for k3={k3_left} or k3={k3_right}, k1={k1val}, k2={k2val}")
        continue
    