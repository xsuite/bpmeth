import numpy as np
import gzip
import pyvista as pv
import matplotlib.pyplot as plt
import scipy as sc
import math


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

        # Frenet-Serret coordinates
        dst["xFS"] = np.array(r*np.cos(psi)).flatten()
        dst["yFS"] = 0 * s.flatten()
        dst["sFS"] = s.flatten()

        # Field in Frenet-Serret coordinates
        dst["BxFS"] = dst["Bx"] * np.cos(s/rho).flatten() + dst["By"] * np.sin(s/rho).flatten()
        dst["ByFS"] = dst["Bz"]
        dst["BsFS"] = dst["Bx"] * np.sin(s/rho).flatten() - dst["By"] * np.cos(s/rho).flatten()
       
        if plot:
            plt.scatter(dst.point_data["xFS"], dst.point_data[field])
            plt.show()

        return dst


    def calc_s(self, rho, smax, field, radius=0.01, n_s=101, plot=False):
        """
        Calculate profile of field in s direction at x=y=0
        """

        s, psi, r = np.meshgrid(np.linspace(-smax, smax, n_s),
                                [0.0],
                                [0.0])

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)

        # Frenet-Serret coordinates
        dst["xFS"] = 0 * s.flatten()
        dst["yFS"] = 0 * s.flatten()
        dst["sFS"] = s.flatten()

        # Field in Frenet-Serret coordinates
        dst["BxFS"] = dst["Bx"] * np.cos(s/rho).flatten() + dst["By"] * np.sin(s/rho).flatten()
        dst["ByFS"] = dst["Bz"]
        dst["BsFS"] = dst["Bx"] * np.sin(s/rho).flatten() - dst["By"] * np.cos(s/rho).flatten()
       
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(dst.point_data["sFS"], dst.point_data[field], label=field)
            ax.legend()
            plt.show()

        return dst


    def fit_x(self, rho, rmax, spos, field, radius=0.01, n_r=25, degree=10, plot=False):
        """
        Fit a polynomial to the x distribution for y=0 and a given s
        Bx(x, y=0, s) = a_j(s) x^j/j!
        By(x, y=0, s) = b_j(s) x^j/j!

        :return: list of coefficients, highest order term first
        """

        # In order to calculate the derivative in zero we want to have interpolation at zero, 
        # so we need n_r to be odd for the interpolation
        n_r = n_r + 1 - n_r%2
        dst = self.calc_x(rho, rmax, spos, field, radius, n_r)

        # Extract the multipole coefficients by fitting polynomial
        params = np.polyfit(dst.point_data["xFS"], dst.point_data[field], degree)
        coefsfit = [params[degree-i] * math.factorial(i) for i in range(degree+1)]

        # Extract the multipole coefficients by calculating derivatives, using central difference
        h = 4*rmax / (n_r-1)  # Twice the interval between two points, since we need half step values
        center = n_r // 2  # x=0
        coefsdiff = np.zeros(degree+1)
        for n in range(degree+1): 
            for i in range(n+1):
                ind = center+2*(n/2-i)
                coefsdiff[n] += 1/h**n * (-1)**i * math.factorial(n)/math.factorial(n-i)/math.factorial(i) * dst.point_data[field][int(ind)]

        if plot:
            x = np.linspace(min(dst.point_data["xFS"]), max(dst.point_data["xFS"]), 100)
            yfit = sum([coefsfit[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            ydiff = sum([coefsdiff[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            fig, ax = plt.subplots()
            ax.scatter(dst.point_data["xFS"], dst.point_data[field], label="Data")
            ax.plot(x, yfit, label="Fit") 
            ax.plot(x, ydiff, label="Derivatives") 
            ax.set_ylim(1.1*min(dst.point_data[field]), 1.1*max(dst.point_data[field]))
            fig.legend()
            plt.show()

        return coefsfit, coefsdiff

    
    def magnetic_length(self, rho, smax, radius=0.01, n_s=101, plot=False):
        """
        Calculate the magnetic length of the magnet
        """
        
        n_s = n_s + 1 - n_s%2
        dst = self.calc_s(rho, smax, "ByFS", radius, n_s)

        Bint = sc.integrate.simpson(dst.point_data["ByFS"], x=dst.point_data["sFS"])
        [Bcenter] = dst.point_data["ByFS"][dst.point_data["sFS"]==0]
        mlen = Bint / Bcenter

        print("Integrated field: ", Bint)
        print("Central field: ", Bcenter)

        if plot:
            fig, ax = plt.subplots()
            ax.axvline(x=mlen/2, color="red")
            ax.axvline(x=-mlen/2, color="red")
            plt.text(mlen/2, 4.3, "$L_{mag}/2$", color="red", horizontalalignment="center")
            plt.text(-mlen/2, 4.3, "$L_{mag}/2$", color="red", horizontalalignment="center")
            #ax.scatter(dst.point_data["sFS"], dst.point_data["BxFS"], label="$B_x(x=y=0, s)$", s=3, color="orange")
            ax.scatter(dst.point_data["sFS"], dst.point_data["ByFS"], label="$B_y(x=y=0, s)$", s=3, color="blue")
            ax.scatter(dst.point_data["sFS"], dst.point_data["BsFS"], label="$B_s(x=y=0, s)$", s=3, color="lightgreen")
            ax.set_xlabel("s (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.show()

        return mlen



data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
cctmagnet = Magnet(data)

#cctmagnet.plot()

field = "Bz"
rho = 1.65
rmax = 0.1
smax = 0.8

#cctmagnet.curved_plot(rho, rmax, smax, field)
#cctmagnet.curved_plot(rho, rmax, smax, field, n_psi=2)
#
#cctmagnet.curved_plot(rho, rmax, smax, "Bx", n_psi=2)
#cctmagnet.curved_plot(rho, rmax, smax, "By", n_psi=2)
#
spos=0.1
#cctmagnet.calc_x(rho, rmax, spos, "Bx", plot=True)
#cctmagnet.calc_x(rho, rmax, spos, "By", plot=True)
#cctmagnet.calc_x(rho, rmax, spos, "Bz", plot=True)
#
#cctmagnet.calc_x(rho, rmax, spos, "BxFS", plot=True)
#cctmagnet.calc_x(rho, rmax, spos, "ByFS", plot=True)
#cctmagnet.calc_x(rho, rmax, spos, "BsFS", plot=True)
#
spos=0.1
#cctmagnet.fit_x(rho, rmax, spos, "ByFS", plot=True)

#cctmagnet.calc_s(rho, 1, "BxFS", plot=True)
#cctmagnet.calc_s(rho, 1, "ByFS", plot=True)
#cctmagnet.calc_s(rho, 1, "BsFS", plot=True)

#cctmagnet.fit_x(rho, rmax, spos, "BxFS", n_r = 55, degree = 20, plot=True)
#cctmagnet.fit_x(rho, rmax, spos, "ByFS", n_r = 55, degree = 20, plot=True)

cctmagnet.magnetic_length(rho, smax, n_s=200, plot=True)






