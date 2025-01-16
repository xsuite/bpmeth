import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


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
        self.a  = tuple(eval(aa, sp.__dict__, {"s": self.s}) for aa in a)
        self.b  = tuple(eval(bb, sp.__dict__, {"s": self.s}) for bb in b)
        self.bs = eval(bs, sp.__dict__, {"s": self.s})
        self.hs = eval(hs, sp.__dict__, {"s": self.s})
        self.nphi = nphi


    def get_phi(self):
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
        
        phi0 = sum((an * x ** (n + 1) / sp.factorial(n + 1) for n, an in enumerate(a))) + bs
        phi1 = sum((bn * x**n / sp.factorial(n) for n, bn in enumerate(b)))        
        phiv = [phi0, phi1]
        for i in range(nphi):
            phiv.append(phinplus2(phiv[i], x, s, hs).simplify())
        return sum(pp * y**i / sp.factorial(i) for i, pp in enumerate(phiv)).simplify()

    
    def get_A(self):
        x, y, s = self.x, self.y, self.s
        hs = self.hs
        phi = self.get_phi()
        
        



        

    
    def get_Bfield(self, lambdify=True):
        x = self.x
        y = self.y
        s = self.s
        phi = self.get_phi()
        Bx, By, Bs = phi.diff(x), phi.diff(y), phi.diff(s)

        if lambdify:
            return (
                np.vectorize(sp.lambdify([x, y, s], Bx, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], By, "numpy")),
                np.vectorize(sp.lambdify([x, y, s], Bs, "numpy")),
            )

        else:
            return Bx, By, Bs


    def plotfield(self):
        ymin, ymax, ystep = -2, 2, 0.05
        zmin, zmax, zstep = -3, 3, 0.05
        Y = np.arange(ymin, ymax, ystep)
        Z = np.arange(zmin, zmax, zstep)
        Z, Y = np.meshgrid(Z, Y)

        Bxfun, Byfun, Bsfun = self.get_Bfield()

        By = Byfun(0, Y, Z)
        Bs = Bsfun(0, Y, Z)

        bmagn = np.sqrt(By**2 + Bs**2)

        plt.imshow(bmagn, extent=(zmin, zmax, ymin, ymax), origin='lower')
        plt.colorbar()
        skip = 4
        plt.quiver(Z[::skip, ::skip], Y[::skip, ::skip], Bs[::skip, ::skip]/bmagn[::skip, ::skip], 
                By[::skip, ::skip]/bmagn[::skip, ::skip], scale=50, pivot='mid') 
        
        plt.xlabel("s")
        plt.ylabel("y")

        plt.show()


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



