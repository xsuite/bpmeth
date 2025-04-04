import bpmeth
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import mpmath

def K0ggtanh(b1, a, L):
    """    
    K0 fringe field integral for a tanh times the gap height squared.
    
    :param b1: fringe field coefficient.
    :param a: fringe field range.
    :param L: integral around the fringe field ranges from -L to L.
        (In theory the value of K0gg should not depend on L, otherwise L is chosen too small.)
    """

    return b1* ( L**2/2 - L*a/2*np.log((1+np.exp(-2*L/a))*(1+np.exp(2*L/a))) \
        + a**2/4*(mpmath.polylog(2, -np.exp(-2*L/a))-mpmath.polylog(2, -np.exp(2*L/a))) )


# Properties
h = 0
length = 5
b1 = 0.1
aa = 1
b1shape = f"(tanh(s/{aa})+1)/2"
b1fringe = f"{b1}*{b1shape}" # Fringe field integral K = aa/(2g)
Kg = aa/2
K0gg = K0ggtanh(b1, aa, length/2)


sp_part = bpmeth.SympyParticle()
xi, yi, zetai, pxi, pyi, ptaui = sp_part.x, sp_part.y, sp_part.zeta, sp_part.px, sp_part.py, sp_part.ptau
beta0 = sp_part.beta0
fringe_forest = bpmeth.ForestFringe(b1, Kg, K0gg)
trajectories_forest = fringe_forest.track(sp_part)
xf, yf, zetaf, pxf, pyf, ptauf = sp_part.x, sp_part.y, sp_part.zeta, sp_part.px, sp_part.py, sp_part.ptau

matrix = sp.Matrix([xf, pxf, yf, pyf, zetaf/beta0, ptauf])
J = matrix.jacobian([xi, pxi, yi, pyi, zetai/beta0, ptaui]).subs({zetai: 0, ptaui: 0}) 

S = sp.Matrix.vstack(
    sp.Matrix.hstack(sp.zeros(3), sp.eye(3)),
    sp.Matrix.hstack(-sp.eye(3), sp.zeros(3)))

symplectic_condition = J.T * S * J  # Has to be equal to S

symplectic_condition.subs({xi:0, yi:0, pxi:0, pyi:0, zetai:0, ptaui:0})