import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import warnings


def phinplus2(phi, x, s, hs):
    """
    Recursion relation for the magnetic scalar potential
    """

    xx = ((1 + hs * x) * phi.diff(x)).diff(x)
    ss = (1 / (1 + hs * x) * phi.diff(s)).diff(s)
    return -1 / (1 + hs * x) * (xx + ss)


class FieldExpansion:
    def __init__(self, a=("0*s",), b=("0*s",), bs="0", hs="0", nphi=5):
        self.x, self.y, self.s = sp.symbols("x y s", real=True)
        self.a = tuple(eval(aa, sp.__dict__, {"s": self.s}) if isinstance(aa, str) else aa.subs(sp.Symbol("s"), self.s) for aa in a) 
        self.b = tuple(eval(bb, sp.__dict__, {"s": self.s}) if isinstance(bb, str) else bb.subs(sp.Symbol("s"), self.s) for bb in b) 
        self.bs = eval(bs, sp.__dict__, {"s": self.s}) if isinstance(bs, str) else bs.subs(sp.Symbol("s"), self.s)
        self.hs = eval(hs, sp.__dict__, {"s": self.s}) if isinstance(hs, str) else hs.subs(sp.Symbol("s"), self.s)
        self.nphi = nphi


    def get_phi(self, tolerance=1e-4, apperture=0.05):
        """
        a=(a1,a2,a3,...)
        b=(b1,b2,b3,...)
        """

        x, y, s = self.x, self.y, self.s
        a = self.a
        b = self.b
        bs = self.bs
        hs = self.hs
        nphi = self.nphi
        
        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(a))) + sp.integrate(bs, s)
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(b)))        
        phiv = [phi0, phi1]
        for i in range(nphi):
            phiv.append(phinplus2(phiv[i], x, s, hs).simplify())
        phi = sum(pp * y**i / sp.factorial(i) for i, pp in enumerate(phiv)).simplify()
    

        """
        Truncating the expansion in phi will violate Maxwell equations. Verify if the 
        derivation is not too large by checking if the laplacian of the potential remains 
        within the allowed tolerances.
        """

        lapl = (
            1/(1 + hs*x) * (1/(1 + hs*x) * phi.diff(s)).diff(s) +
            1/(1 + hs*x) * ((1 + hs*x) * phi.diff(x)).diff(x) +
            phi.diff(y, 2)
        )
        laplval = lapl.subs({x: apperture, y: apperture, s: apperture}).evalf()
        if isinstance(laplval, float) or isinstance(laplval, sp.core.numbers.Float):
            assert laplval < tolerance, \
                f"More terms are needed in the expansion, the laplacian is {laplval}"
        else:
            warnings.warn("The laplacian could not be evaluated, are you doing symbolic calculations?")
        
        return phi
        

    
    def get_A(self):
        x, y, s = self.x, self.y, self.s
        hs = self.hs
        Bx, By, Bs = self.get_Bfield(lambdify=False)

        Ax = - sp.integrate(Bs, (y, 0, y))
        Ay = sp.S(0)
        As = sp.integrate(Bx, (y, 0, y)) - 1/(1+hs*x) * sp.integrate((1+hs*x)*By, (x, 0, x)).subs({y:0})

        return Ax, Ay, As

    
    def get_Bfield(self, lambdify=True):
        x = self.x
        y = self.y
        s = self.s
        phi = self.get_phi()
        hs = self.hs
        Bx, By, Bs = phi.diff(x), phi.diff(y), 1/(1+hs*x)*phi.diff(s)

        if lambdify:
            return (
                np.vectorize(sp.lambdify([x, y, s], Bx, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], By, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], Bs, "numpy")),
            )

        else:
            return Bx, By, Bs

    
    def transform(self, theta_E=0, rho=np.inf, maxpow=None):
        """
        So far only implemented for the entrance of the element. Some signs will differ for the exit!
        """
        assert self.hs==0, "Transform only implemented starting from straight frame!"        
        
        x, y, s = self.x, self.y, self.s
        at = self.a
        bt = self.b
        bst = self.bs
        nphi = self.nphi
        if maxpow is None:
            maxpow = nphi
        
        # Determine the scalar potential 0 and 1 terms and transform to new frame
        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(at))) + sp.integrate(bst, s)
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(bt))) 

        if rho==np.inf:  # Straight frame
            xt = s*sp.sin(theta_E) + x*sp.cos(theta_E)
            st = s*sp.cos(theta_E) - x*sp.sin(theta_E)
            print(f"Rotating over angle {theta_E} to new straight frame...")

        else:  # Curved frame
            xt = (rho+x)*(sp.cos(theta_E-s/rho)) - rho*sp.cos(theta_E)
            st = rho*sp.sin(theta_E) - (rho+x)*(sp.sin(theta_E-s/rho))
            print(f"Rotating over angle {theta_E} to new curved frame with bending radius {rho}...")

        phi0 = phi0.subs([(x,xt), (s,st)])
        phi1 = phi1.subs([(x,xt), (s,st)])
            
        phi0 = phi0.series(x, 0, maxpow).removeO().as_poly(x)
        phi1 = phi1.series(x, 0, maxpow).removeO().as_poly(x)
        
        # Extract the multipole coefficients.
        a = []
        degree0 = 0 if phi0==0 else phi0.degree(x)
        for i in range(1, degree0+1):
            a.append(phi0.coeff_monomial(x**i))
            
        bs = phi0.coeff_monomial(x**0).diff(s)

        b = []
        degree1 = 0 if phi1==0 else phi1.degree(x)
        for i in range(degree1+1):
            b.append(phi1.coeff_monomial(x**i))
            
        return FieldExpansion(a=a, b=b, bs=bs, nphi=nphi)
            

    def plotfield_yz(self, X=0, ax=None, bmin=None, bmax=None, includebx=False):
        ymin, ymax, ystep = -2, 2, 0.05
        zmin, zmax, zstep = -3, 3, 0.05
        Y = np.arange(ymin, ymax, ystep)
        Z = np.arange(zmin, zmax, zstep)
        Z, Y = np.meshgrid(Z, Y)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        By = Byfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)

        bmagn = np.sqrt(By**2 + Bs**2)
        if includebx:
            Bx = Bxfun(X, Y, Z)
            bmagn_color = np.sqrt(bmagn**2 + Bx**2)
        else:
            bmagn_color = bmagn
            
        if bmin is None:
            bmin = bmagn_color.min()
        if bmax is None:
            bmax = bmagn_color.max()

        if ax is None:
            fig, ax = plt.subplots()
        plt.imshow(bmagn_color, extent=(zmin, zmax, ymin, ymax), origin='lower', vmin=bmin, vmax=bmax)
        plt.colorbar()
        skip = 4
        plt.quiver(Z[::skip, ::skip], Y[::skip, ::skip], Bs[::skip, ::skip]/bmagn[::skip, ::skip], 
                By[::skip, ::skip]/bmagn[::skip, ::skip], scale=50, pivot='mid') 
        
        plt.xlabel("s")
        plt.ylabel("y")

        plt.show()
        
    def plotfield_xz(self, Y=0, ax=None, bmin=None, bmax=None, includeby=False):
        xmin, xmax, xstep = -2, 2, 0.05
        zmin, zmax, zstep = -3, 3, 0.05
        X = np.arange(xmin, xmax, xstep)
        Z = np.arange(zmin, zmax, zstep)
        Z, X = np.meshgrid(Z, X)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        Bx = Bxfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)

        bmagn = np.sqrt(Bx**2 + Bs**2)
        if includeby:
            By = Byfun(X, Y, Z)
            bmagn_color = np.sqrt(bmagn**2 + By**2)
        else:
            bmagn_color = bmagn
            
        if bmin is None:
            bmin = bmagn_color.min()
        if bmax is None:
            bmax = bmagn_color.max()

        if ax is None:
            fig, ax = plt.subplots()
        plt.imshow(bmagn_color, extent=(zmin, zmax, xmin, xmax), origin='lower', vmin=bmin, vmax=bmax)
        plt.colorbar()
        skip = 4
        plt.quiver(Z[::skip, ::skip], X[::skip, ::skip], Bs[::skip, ::skip]/bmagn[::skip, ::skip], 
                Bx[::skip, ::skip]/bmagn[::skip, ::skip], scale=50, pivot='mid') 
        
        plt.xlabel("s")
        plt.ylabel("x")

        plt.show()
        
        
    def calc_RDTs(self, n, betx=1, bety=1, alphx=0, alphy=0):
        """
        Calculate the RDTs of the given element symbolically as a function of s
        
        :param n (int): Highest order of the RDTs to calculate.
        :param betx (float or sp.symbol): Beta function in x.
        :param bety (float or sp.symbol): Beta function in y.
        :param alphx (float or sp.symbol): Alpha function in x.
        :param alphy (float or sp.symbol): Alpha function in y.
        :return (List of lists of sp expressions): Symbolic expression of the h values h_pqrt 
        to calculate the RDTs as a function of s.
        """

        x, y, s = self.x, self.y, self.s
        hs = self.hs
        Ax, Ay, As = self.get_A()
        
        hxp, hxm, hyp, hym = sp.symbols("hxp hxm hyp hym")
        xsubs = sp.sqrt(betx)/2*(hxp + hxm)
        ysubs = sp.sqrt(bety)/2*(hyp + hym)
        pxsubs = - 1/(2*sp.sqrt(betx))*((alphx + 1j)*hxp + (alphx - 1j)*hxm)
        pysubs = - 1/(2*sp.sqrt(bety))*((alphy + 1j)*hyp + (alphy - 1j)*hym)
        
        # H = -(1+hx) As
        H_As = ((1+hs*x) * As).series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)])
        H_As_poly = H_As.as_poly(hxp, hxm, hyp, hym)
        
        # H = -(1+hx) px Ax
        H_pxAx = -((1+hs*x) * pxsubs * Ax).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_pxAx_poly = H_pxAx.as_poly(hxp, hxm, hyp, hym)
        
        # H = (1+hx) 1/2 Ax^2
        H_Ax2 = ((1+hs*x) * 1/2 * Ax**2).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_Ax2_poly = H_Ax2.as_poly(hxp, hxm, hyp, hym)

        # H = -(1+hx) py Ay
        H_pyAy = -((1+hs*x) * pysubs * Ay).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_pyAy_poly = H_pyAy.as_poly(hxp, hxm, hyp, hym)
        
        # H = (1+hx) 1/2 Ay^2
        H_Ay2 = ((1+hs*x) * 1/2 * Ay**2).cancel().series(x,0,n).removeO().subs([(x, xsubs), (y, ysubs)]).expand()
        H_Ay2_poly = H_Ay2.as_poly(hxp, hxm, hyp, hym)
        
        h = sp.MutableDenseNDimArray(np.zeros((n+1, n+1, n+1, n+1), dtype=object))

        for p in range(n+1):
            for q in range(n+1-p):
                for r in range(n+1-p-q):
                    for t in range(n+1-p-q-r):
                        h[p,q,r,t] += H_As_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_pxAx_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_Ax2_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_pyAy_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)
                        h[p,q,r,t] += H_Ay2_poly.coeff_monomial(hxp**p*hxm**q*hyp**r*hym**t)

        return h
        
        

    def writefile(self, xarr, yarr, zarr, filename):
        X, Y, Z = np.meshgrid(xarr, yarr, zarr)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        Bx = Bxfun(X, Y, Z)
        By = Byfun(X, Y, Z)
        Bs = Bsfun(X, Y, Z)

        with open(f'{filename}.csv', 'w') as file:
            file.write(f'"X","Y","Z","Bx","By","Bz"\n')
            np.savetxt(file, np.c_[X.flat, Y.flat, Z.flat, Bx.flat, By.flat, Bs.flat], delimiter=',', fmt='%.10f')


    def __call__(self, x, y, z):
        # Return the magnetic field at the point (x, y, z)
        Bx, By, Bs = self.get_Bfield(lambdify=False)
        return (
            Bx.subs({self.x: x, self.y: y, self.s: z}).evalf(),
            By.subs({self.x: x, self.y: y, self.s: z}).evalf(),
            Bs.subs({self.x: x, self.y: y, self.s: z}).evalf(),
        )



