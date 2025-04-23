import pyvista as pv
import numpy as np
import sympy as sp
import math
import scipy as sc
import matplotlib.pyplot as plt


def Enge(x, *params):
    return params[0] / (1+np.exp(np.poly1d(params[1:])(x)))
        
        
def spEnge(s, *params):
    return params[0] / (1+sp.exp(sp.Poly(params[1:], s).as_expr()))

class Magnet:
    def __init__(self, data):
        self.data = data
        self.src = pv.PolyData(self.data.T[:3].T)
        self.src['Bx'] = self.data.T[3].T
        self.src['By'] = self.data.T[4].T
        self.src['Bz'] = self.data.T[5].T
        self.src['x'] = self.data.T[0].T
        self.src['y'] = self.data.T[1].T
        self.src['z'] = self.data.T[2].T

    def plot(self):
        self.src.plot(scalars="By")
        
    def interpolate_points(self, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, field, radius=0.01):
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        
        X, Y, Z = np.meshgrid(x, y, z)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        
        dst.plot(scalars=field)
        
        dst['x'] = x
        dst['y'] = y
        dst['z'] = z
        
        return dst
        
    def xprofile(self, ypos, zpos, field, ax=None, xmax=None):
        assert ypos in self.src['y'] and zpos in self.src['z'], "These values are not present in the data"
        if xmax is None:
            xmax = np.max(abs(self.src['x']))
        mask = (self.src['y'] == ypos) & (self.src['z'] == zpos) & (abs(self.src['x']) <= xmax)
        
        x = self.src['x'][mask]
        fieldvals = self.src[field][mask]
        
        if ax is not None:
            ax.plot(x, fieldvals, label=f"y={ypos}, z={zpos}")
            ax.set_xlabel('x')
            ax.set_ylabel(field)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        return x, fieldvals
        
    def zprofile(self, xpos, ypos, field, ax=None):
        assert xpos in self.src['x'] and ypos in self.src['y'], "These values are not present in the data"
        mask = (self.src['x'] == xpos) & (self.src['y'] == ypos)
        
        z = self.src['z'][mask]
        fieldvals = self.src[field][mask]

        if ax is not None:
            ax.plot(z, fieldvals, label=f"x={xpos}, y={ypos}")
            ax.set_xlabel('z')
            ax.set_ylabel(field)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        return z, fieldvals

        
    def deriv_xprofile(self, ypos, zpos, field, order):
        x, fieldvals = self.xprofile(ypos, zpos, field)
        
        x0ind = int(np.where(x == 0)[0][0])
        h = 2 * (sorted(x)[1] - sorted(x)[0])  # step size
        
        coeffs = np.zeros(order+1)
        # Central finite difference
        for n in range(order+1):
            for i in range(n+1):
                index = int(2*(n/2-i)+x0ind)  # Each step is h/2
                coeffs[n] += 1/h**n * (-1)**i * math.factorial(n)/math.factorial(n-i)/math.factorial(i) * fieldvals[index]

        return coeffs
    
    def fit_xprofile(self, ypos, zpos, field, order, ax=None, xmax=None):
        assert ypos in self.src['y'] and zpos in self.src['z'], "These values are not present in the data"
        x, fieldvals = self.xprofile(ypos, zpos, field, ax=ax, xmax=xmax)

        paramslist = np.array([np.polyfit(x, fieldvals, order+j)[j:] for j in range(5)])
        params = np.mean(paramslist, axis=0)
        paramsstd = np.std(paramslist, axis=0)
        coeffslist = [paramslist[:, order-i] * math.factorial(i) for i in range(order+1)]
        coeffs = np.mean(coeffslist, axis=1)
        coeffsstd = np.std(coeffslist, axis=1)
        
        if ax:
            ax.scatter(x, fieldvals, label="data")
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(params+paramsstd, xx), label="fit+std", color="gray")
            ax.plot(xx, np.polyval(params-paramsstd, xx), label="fit-std", color="gray")
            ax.plot(xx, np.polyval(params, xx), label="fit", color="black")
            ax.legend()
            
        return coeffs, coeffsstd

    def z_multipoles(self, order, xmax=None):
        zvals = np.unique(self.src['z'])
        coeffs = np.zeros((len(zvals), order+1))
        coeffsstd = np.zeros((len(zvals), order+1))
        for i, zpos in enumerate(zvals):
            coeffs[i], coeffsstd[i] = self.fit_xprofile(0, zpos, "By", order, xmax=xmax)
        return zvals, coeffs, coeffsstd
    
    def get_b1(self, zpos):
        x, by = self.xprofile(0, zpos, "By")
        x0ind = int(np.where(x == 0)[0][0])
        b1s = [ by[x0ind+i] for i in range(-2,3)]
        return np.mean(b1s), np.std(b1s)
    
    def get_b2(self, zpos):
        x, by = self.xprofile(0, zpos, "By")
        x0ind = int(np.where(x == 0)[0][0])
        b2s = [ (by[x0ind+i]-by[x0ind-i])/(x[x0ind+i]-x[x0ind-i]) for i in range(3,8)]
        return np.mean(b2s), np.std(b2s)

    def get_b3(self, zpos):
        x, by = self.xprofile(0, zpos, "By")
        x0ind = int(np.where(x == 0)[0][0])
        
        b2ls = [ (by[x0ind-3+i]-by[x0ind-3-i])/(x[x0ind-3+i]-x[x0ind-3-i]) for i in range(3,8)]
        b2rs = [ (by[x0ind+3+i]-by[x0ind+3-i])/(x[x0ind+3+i]-x[x0ind+3-i]) for i in range(3,8)]
        
        b2l=np.mean(b2ls)
        b2r=np.mean(b2rs)
        b2lstd=np.std(b2ls)
        b2rstd=np.std(b2rs)
        
        b3 = (b2r-b2l)/(x[x0ind+3]-x[x0ind-3])
        b3err = np.sqrt(b2lstd**2 + b2rstd**2) / (x[x0ind+3]-x[x0ind-3])
        
        return b3, b3err
               
               
    def fit_b2_enge(self, degree=5, xmax=None, ax=None):
        """
        The fit is an Enge-function, a function defined as A /(1+exp(c0 + c1 s + c2 s^2 + ...))
        """

        zvals, coeffs, coeffsstd = self.z_multipoles(2, xmax=xmax)
        b2 = coeffs[:, 1]
        b2err = coeffsstd[:, 1]
        
        params, cov = sc.optimize.curve_fit(Enge, zvals, b2, sigma=b2err, p0=np.ones(degree+1))
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(zvals, b2, yerr=b2err, marker='.', capsize=5, label="b2", color="blue", ls='', zorder=0)
        ax.plot(zvals, Enge(zvals, *params), label=f'Enge function fit of degree {degree}', color="black", zorder=10)
        plt.legend()
        plt.show()
        
        return params, cov
        
        