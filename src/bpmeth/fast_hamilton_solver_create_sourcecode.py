import sympy as sp
import numba
import numpy as np 
from .numerical_solver import GeneralVectorPotential, Hamiltonian   ## To fix

"""
Write source code for field evaluations in fast hamiltonian solver using these functions

----------------------------------------

fieldder[0] -> bs
fieldder[1] -> b1
fieldder[2] -> a1
fieldder[3] -> b2
fieldder[4] -> a2
fieldder[5] -> b3
fieldder[6] -> a3
fieldder[7] -> b4
fieldder[8] -> a4
"""

comps = []

def mk_s_poly(cpmidx, sorder):
    ss = [sp.Symbol(f"fieldder[{cpmidx},{ii}]", real=True) for ii in range(sorder + 1)]
    s = sp.var("s", real=True)
    return sum(ss[i] * s**i for i in range(len(ss)))


def mk_fieldder_sp(sorder, ab_order):
    comps = [mk_s_poly(0, sorder)]
    for ii in range(ab_order * 2):
        comps.append(mk_s_poly(ii + 1, sorder))
    return comps


def mk_field(ab_order=4, sorder=3, h=True, nphi=5, out=None):
    fd = mk_fieldder_sp(sorder, ab_order)
    if h:
        h = sp.var("h", real=True)
    else:
        h = "0"
    b = fd[1 : ab_order * 2 + 1 : 2]
    a = fd[2 : ab_order * 2 + 1 : 2]
    vp = GeneralVectorPotential(bs=fd[0], b=b, a=a, hs=h, nphi=nphi)
    Bx_sp, By_sp, Bs_sp = vp.get_Bfield(lambdify=False)
    Ax_sp, Ay_sp, As_sp = vp.get_A(lambdify=False)
    
    HH = Hamiltonian(length=0, curv=h, vectp=vp)
    xdot, ydot, taudot, pxdot, pydot, ptaudot = HH.get_vectorfield(lambdify=False)
    
    if out is not None:
        src = ["import numba"]
        src.append("import numpy as np")
        src.append("")
        src.append("@numba.njit(cache=True)")
        src.append(f"def bfield(x,y,s,h,fieldder):")
        src.append(f"  return {Bx_sp},{By_sp},{Bs_sp}")
        src.append("")
        src.append("@numba.njit(cache=True)")
        src.append(f"def afield(x,y,s,h,fieldder):")
        src.append(f"  return {Ax_sp},{Ay_sp},{As_sp}")
        src.append("")
        src.append("@numba.njit(cache=True)")
        src.append(f"def vectorfield(s,x,y,tau,px,py,ptau,beta0,h,fieldder):")
        src.append(f"  return {xdot}, {ydot}, {taudot}, {pxdot}, {pydot}, {ptaudot}".replace("sqrt(", "np.sqrt("))
        src = "\n".join(src)
        open(out, "w").write(src)


