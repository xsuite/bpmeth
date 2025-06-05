import numpy as np
from math import factorial
import scipy.optimize

fname="field_map_straight_quad_resol_2mm.txt"
x,y,z,bx,by,bz=np.loadtxt(fname,skiprows=9).T
nx=len(set(x))
ny=len(set(y))
nz=len(set(z))

nn=5
mm=5

c=np.zeros((nn+1,mm+1,nz))
iz=0

def fitb(c):
    cbx=0
    cby=0
    ia=iz*(nx*ny)
    ib=(iz+1)*(nx*ny)
    xs=x[ia:ib]
    ys=y[ia:ib]
    bxs=bx[ia:ib]
    bys=by[ia:ib]
    cc=c.reshape(nn+1,mm+1,nz)

    for m in range(nn):
        for n in range(nn):
            xynm=xs**n*ys**m
            cbx+=cc[m,n+1,iz]*xynm
            cby+=cc[m+1,n,iz]*xynm

    return ((bxs-cbx)**2+(bys-cby)**2).sum()

fitb(c.flatten())
scipy.optimize.fmin(fitb,c)


