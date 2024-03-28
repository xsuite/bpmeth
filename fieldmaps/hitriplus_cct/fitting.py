import numpy as np
import matplotlib.pyplot as plt
import gzip
import pyvista as pv
import math

data=np.loadtxt("ELBHZ-FIELD-AT-POINTS.txt")

x,y,z,bx,by,bz=data.T


def Bxmod(x, s, params):
    """
    Function for the Bx field on the y=0 axis using a multipole expansion 
    in curved coordinates, Bx(x, y=0, s) = sum(aj(s) x^j/j!, where aj(s) will
    be described by a sum of sigmoids.

    :param x: x position of the field
    :param s: s position of the field
    :param params: two dimensional numpy array, with the first index 
    the order in x and the first index the parameters describing the function aj.
    """

    return sum( aj(s, params[j]) * x**j / math.factorial(j) for j in range(len(params)))


def aj(s, params):
    """
    Function to model the coefficient a(s) in the multipole expansion,
    here modelled as the sum of two arctan.

    :param s: s position of the field
    :param params: one dimensional array of parameters
    """

    return params[0] * (np.arctan(params[1]*(s+params[2])) + np.arctan(params[1]*(s-params[2])))
