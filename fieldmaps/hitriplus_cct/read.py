import numpy as np
import gzip
import pyvista as pv
import matplotlib.pyplot as plt
import scipy as sc


class Magnet:
    def __init__(self, data):
        self.data = data
        self.src = pv.PolyData(self.data.T[:3].T)
        self.src['Bx'] = self.data.T[3].T
        self.src['By'] = self.data.T[4].T
        self.src['Bz'] = self.data.T[5].T


    def plot(self):
        self.src.plot()
        

    def curved_plot(self, rho, rmax, smax, field, radius=0.01, n_psi=32, n_r=5, n_s=101):
        """
        Plot the field in a curved reference frame
        """ 

        s, psi, r = np.meshgrid(np.linspace(-smax, smax, n_s),
                                np.linspace(0, 2*np.pi, n_psi+1),
                                np.linspace(0.0, rmax, n_r))

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)
        dst.plot(scalars=field)

        return dst


    def calc_x(self, rho, rmax, spos, field, radius=0.01, n_r=25, plot=False):
        """
        Calculate profile of field in x direction at y=0 and given s
        """

        s, psi, r = np.meshgrid([spos],
                                [0.0],
                                np.linspace(-rmax, rmax, n_r))

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)
        dst["xFS"] = np.array(r*np.cos(psi)).flatten()  # Frenet-serret x
       
        if plot:
            plt.scatter(dst.point_data["xFS"], dst.point_data[field])
            plt.show()

        return dst


    def fit_x(self, rho, rmax, spos, field, radius=0.01, n_r=25, degree=10, plot=False):
        """
        Fit a polynomial to the x distribution for y=0 and a given s
        Bx(x, y=0, s) = a_j(s) x^j/j!
        By(x, y=0, s) = b_j(s) x^j/j!
        """

        dst = self.calc_x(rho, rmax, spos, field, radius, n_r)
        coefs = np.polyfit(dst.point_data["xFS"], dst.point_data[field], degree)

        if plot:
            x = np.linspace(min(dst.point_data["xFS"]), max(dst.point_data["xFS"]), 100)
            y = sum([coefs[i] * x**(degree-i) for i in range(len(coefs))])
            plt.scatter(dst.point_data["xFS"], dst.point_data[field])
            plt.plot(x, y) 
            plt.show()

        return coefs
    

    def fit_s(self, rho, rmax, smax, field, radius=0.01, n_r=25, n_s=25, xdegree=10, plot=False):
        """
        Fit a given function model(s, params) to the coefficients aj and bj
        """

        # Collect all aj(si)
        s = np.linspace(-smax, smax, n_s)
        coeflist = np.zeros((n_s, xdegree+1))
        for i, spos in enumerate(s):
            coefs = self.fit_x(rho, rmax, spos, field, radius, n_r, xdegree)
            coeflist[i, :] = coefs
        
        # Fit a model function to every aj 
        params = [0 for j in range(xdegree)]
        for j in range(xdegree):
            params[j] = sc.optimize.curve_fit(model, s, coeflist[:, j])

        return params

        
def model(s, c0, c1, c2, c3, c4):
    """
    Model for aj(s), bj(s), all parameters have to be separate arguments.
    """

    return c0 + c1*s + c2*s**2 + c3*s**3 + c4*s**4





data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
cctmagnet = Magnet(data)

#cctmagnet.plot()

rho = 1.65
rmax = 0.1
smax = 0.8
#cctmagnet.curved_plot(rho, rmax, smax, "Bz")
#cctmagnet.curved_plot(rho, rmax, smax, "Bz", n_psi=2)

spos=0.1
#cctmagnet.fit_x(rho, rmax, spos, "Bz", plot=True)

cctmagnet.fit_s(rho, rmax, smax, "Bz")
