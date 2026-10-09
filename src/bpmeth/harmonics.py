import numpy as np
from math import factorial
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial

def calc_dk(ByiBx, rr, nk, s_index):
    """
    Takes a precomputed field array and computes d_k as a function of r 

    :param ByiBx_rts: complex array, shape (nr, ntheta, ns) representing (By + i*Bx) already evaluated on the (r, theta, s) grid. 
        Theta samples must be uniform in [0, 2*pi).
    :param rr: array of r values, shape (nr,)
    :param nk: max |k|
    :param s_index: index along the s-axis to use
    :return: out : complex array, shape (nr, 2*nk+1)  — same as dkharmonics
    """
    
    ntheta = ByiBx.shape[1]
    b = ByiBx[:, :, s_index]        # (nr, ntheta) at specified s
    d = np.fft.fft(b, axis=1) / ntheta       # FFT along theta axis
    out = np.empty((len(rr), 2*nk+1), dtype=complex)
    out[:, 0]    = d[:, 0]
    out[:, 1::2] = d[:, 1 : nk+1]
    out[:, 2::2] = d[:, ntheta : ntheta-nk-1 : -1]
    return out

def fit_dkl(dkofrarray, rr, nk, nl):
    """
    Numerically stable Chebyshev implementation of radial fit.
    Fits f_k(r) ≈ Σ_l dkl[k,l] r^{|k|+2l}, internally uses a Chebyshev basis in x = r^2.
    
    :param dkofrarray: dk values corresponding to the given radii, shape (nr, 2*nk+1)
    :param rr: array of r values, shape (nr,)
    :param nk: max |k|
    :param nl: max l
    :return: dkl : complex ndarray, shape (2*nk+1, nl)
        f_k(r) ≈ Σ_l dkl[k,l] r^{|k|+2l}
    """
    
    dkl = np.zeros((2*nk + 1, nl), dtype=complex)
    x = rr**2
    xmin = x.min()
    xmax = x.max()
    s = 2*(x - xmin)/(xmax - xmin) - 1 # scale x -> s in [-1,1]
    # Chebyshev Vandermonde in scaled variable
    T = chebvander(s, nl-1)
    # k ordering:
    # [0, +1, -1, +2, -2, ...]
    for ik in range(2*nk + 1):
        if ik == 0:
            kabs = 0
        else:
            kabs = (ik + 1)//2
        # absorb r^|k| into the matrix itself
        A = (rr[:, None]**kabs) * T
        y = dkofrarray[:, ik]
        # stable least squares in Chebyshev basis
        c_re, _, _, _ = np.linalg.lstsq(A, y.real, rcond=None)
        c_im, _, _, _ = np.linalg.lstsq(A, y.imag, rcond=None)
        c = c_re + 1j*c_im
        # Convert:  Σ c_n T_n(s)    into ordinary polynomial in x=r^2: Σ a_l x^l
        cheb = Chebyshev(c, domain=[xmin, xmax])
        poly = cheb.convert(kind=Polynomial)
        coeffs = poly.coef
        # store coefficients of x^l so total basis is    r^|k| * (r^2)^l = r^{|k|+2l}
        ncopy = min(len(coeffs), nl)
        dkl[ik, :ncopy] = coeffs[:ncopy]
    return dkl

def calc_bnian(dkl, nl, nk):
    """ 
    Calculate bn + i an from dkl coefficients.
    
    :param dkl: complex ndarray, shape (2*nk+1, nl)
    :param nl: max l
    :param nk: max |k|
    """
        
    # Precompute factorials
    fact = np.array([factorial(n) for n in range(nk)], dtype=np.int16)
    
    # Build k index dict mapping: row index for k = 0,1,-1,2,-2,...
    k_index = {0: 0}
    for i in range(1, nk+1):
        k_index[i]  = 2*i - 1
        k_index[-i] = 2*i #negative k gets even row
        
    bnian = np.zeros(nk, dtype=complex)

    for n in range(nk):
        temp = 0.0 + 0.0j  # initialize sum
        lmax = min(n//2, nl-1)
        for l in range(lmax + 1):
            k = n - 2*l
            if k == 0:
                temp += dkl[k_index[0], l]
            else:
                temp += dkl[k_index[k], l] + dkl[k_index[-k], l]
        bnian[n] = fact[n] * temp
    return bnian
            
def calc_harmonics(ByiBx, s_index, nk, rr):
    '''
    Complete function for local harmonic analysis from By+iBx to coefficients a,b 
    
    :param ByiBx: complex array, shape (nr, ntheta, ns) representing (By + i*Bx) already evaluated on the (r, theta, s) grid.
    :param s_index: index along the s-axis to use
    :param nk: max |k|
    :param rr: array of r values, shape (nr,)
    '''
    
    dk = calc_dk(ByiBx=ByiBx, rr=rr, nk=nk, s_index=s_index)
    nl = nk//2 + 1
    dkl = fit_dkl(dkofrarray=dk,rr=rr,nk=nk, nl=nl)
    bnian = calc_bnian(dkl, nl, nk)
    an = bnian.imag
    bn = bnian.real
    return an, bn

def print_dkl(dkl):
    nk = (dkl.shape[0]-1) // 2
    nl = nk//2 + 1
    print(f"k  ", end="")
    for ll in range(nl):
        print(f"l={ll}"," "*22, end="")
    print()
    print(f" 0", end="")
    for ll in range(nl):
        print(f" {dkl[0,ll].real:12.7f}, {dkl[0,ll].imag:12.7f}",end="")
    print()
    for kk in range(0,nk-1):
        print(f" {kk+1}", end="")
        for ll in range(nl):
            print(f" {dkl[1+2*kk,ll].real:12.7f}, {dkl[1+2*kk,ll].imag:12.7f}",end="")
        print()
        print(f"-{kk+1}", end="")
        for ll in range(nl):
            print(f" {dkl[2+2*kk,ll].real:12.7f}, {dkl[1+2*kk,ll].imag:12.7f}",end="")
        print()


    