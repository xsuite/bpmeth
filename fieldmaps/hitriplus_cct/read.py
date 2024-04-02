import numpy as np
import gzip
import pyvista as pv
import matplotlib.pyplot as plt


class Magnet:
    def __init__(self, data):
        self.data = data
        self.src = pv.PolyData(self.data.T[:3].T)
        self.src['Bx'] = self.data.T[3].T
        self.src['By'] = self.data.T[4].T
        self.src['Bz'] = self.data.T[5].T


    def plot(self):
        self.src.plot()
        

    def curved_plot(self, rho, rmax, field, radius=0.1, n_psi=32):
        """
        Plot the field in a curved reference frame
        """ 

        s, psi, r = np.meshgrid(np.linspace(-1, 1, 101),
                                np.linspace(0, 2*np.pi, n_psi+1),
                                np.linspace(0.0, rmax, 5))

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)
        dst.plot(scalars=field)

        return dst


    def calc_x(self, spos, rho, rmax, field, radius=0.1, n_r=25, plot=False):
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


    def fit_x(self, spos, rho, rmax, field, radius=0.1, n_r=25, degree=5, plot=False):
        """
        Fit a polynomial to the x distribution for y=0 and a given s
        Bx(x, y=0, s) = a_j(s) x^j/j!
        By(x, y=0, s) = b_j(s) x^j/j!
        """

        dst = self.calc_x(spos, rho, rmax, field, radius, n_r)
        coefs = np.polyfit(dst.point_data["xFS"], dst.point_data[field], degree)
        print(coefs)

        if plot:
            x = np.linspace(min(dst.point_data["xFS"]), max(dst.point_data["xFS"]), 100)
            y = sum([coefs[i] * x**(degree-i) for i in range(len(coefs))])
            plt.scatter(dst.point_data["xFS"], dst.point_data[field])
            plt.plot(x, y) 
            plt.show()

        return coefs
    



data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
cctmagnet = Magnet(data)

cctmagnet.plot()

rho = 1.65
rmax = 0.1
#cctmagnet.curved_plot(rho, rmax, "Bz")
cctmagnet.curved_plot(rho, rmax, "Bz", n_psi=2)
cctmagnet.fit_x(0.1, rho, rmax, "Bz", plot=True)
