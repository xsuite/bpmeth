import xtrack as xt
import numpy as np
import bpmeth
import matplotlib.pyplot as plt
from cpymad.madx import Madx
import sympy as sp
import pandas as pd
from save_read_twiss import load_twiss

for model in ["design", "fitted", "splines"]:
    k2val, k3val = 0, 0
    k1_left = -0.01
    df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1_left:.2f}_{k2val:.2f}_{k3val:.2f}")
    k1_right = 0.01
    df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1_right:.2f}_{k2val:.2f}_{k3val:.2f}")
    
    qx_left = scalar_left['qx']
    qx_right = scalar_right['qx']
    dqx_dk1 = (qx_right - qx_left) / (k1_right - k1_left)
    print(f"dqx/dk1 for {model}: {dqx_dk1}")
    
    qy_left = scalar_left['qy']
    qy_right = scalar_right['qy']
    dqy_dk1 = (qy_right - qy_left) / (k1_right - k1_left)
    print(f"dqy/dk1 for {model}: {dqy_dk1}")
    
    k1val, k3val = 0, 0
    k2_left = -0.01
    df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2_left:.2f}_{k3val:.2f}")
    k2_right = 0.01
    df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2_right:.2f}_{k3val:.2f}")
    
    qx_left = scalar_left['qx']
    qx_right = scalar_right['qx']
    dqx_dk2 = (qx_right - qx_left) / (k2_right - k2_left)
    print(f"dqx/dk2 for {model}: {dqx_dk2}")
    
    qy_left = scalar_left['qy']
    qy_right = scalar_right['qy']
    dqy_dk2 = (qy_right - qy_left) / (k2_right - k2_left)
    print(f"dqy/dk2 for {model}: {dqy_dk2}")
    
    k1val, k2val = 0, 0
    k3_left = -0.01
    df_left, scalar_left = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3_left:.2f}")
    k3_right = 0.01
    df_right, scalar_right = load_twiss(f"twissresults/MD/twiss_{model}_{k1val:.2f}_{k2val:.2f}_{k3_right:.2f}")
    
    qx_left = scalar_left['qx']
    qx_right = scalar_right['qx']
    dqx_dk3 = (qx_right - qx_left) / (k3_right - k3_left)
    print(f"dqx/dk3 for {model}: {dqx_dk3}")
    
    qy_left = scalar_left['qy']
    qy_right = scalar_right['qy']
    dqy_dk3 = (qy_right - qy_left) / (k3_right - k3_left)
    print(f"dqy/dk3 for {model}: {dqy_dk3}")
    
    