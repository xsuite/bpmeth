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
        Y = -np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)
        dst.plot(scalars=field)

        return dst


    def calc_x(self, rho, rmax, spos, radius=0.01, n_r=25, plot=False, fields=["Bx", "By", "Bz"]):
        """
        Calculate profile of field in x direction at y=0 and given s

        :param fields: Fields that are desired to plot
        """

        s, psi, r = np.meshgrid([spos],
                                [0.0],
                                np.linspace(-rmax, rmax, n_r))

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = -np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)

        # Frenet-Serret coordinates
        dst["xFS"] = np.array(r*np.cos(psi)).flatten()
        dst["yFS"] = np.array(r*np.sin(psi)).flatten()
        dst["sFS"] = s.flatten()

        # Field in Frenet-Serret coordinates
        dst["BxFS"] = dst["Bx"] * np.cos(s/rho).flatten() - dst["By"] * np.sin(s/rho).flatten()
        dst["ByFS"] = dst["Bz"]
        dst["BsFS"] = dst["Bx"] * np.sin(s/rho).flatten() + dst["By"] * np.cos(s/rho).flatten()
       
        if plot:
            fig, ax = plt.subplots()
            if "Bx" in fields:
                ax.scatter(dst.point_data["xFS"], dst.point_data["BxFS"], color="orange", s=3, label=f"$B_x(x, y=0, s={spos})$")
            if "By" in fields:
                ax.scatter(dst.point_data["xFS"], dst.point_data["ByFS"], color="blue", s=3, label=f"$B_y(x, y=0, s={spos})$")
            if "Bz" in fields:
                ax.scatter(dst.point_data["xFS"], dst.point_data["BsFS"], color="lightgreen", s=3, label=f"$B_s(x, y=0, s={spos})$")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.show()

        return dst


    def calc_s(self, rho, smax, radius=0.01, n_s=101, plot=False, fields=["Bx", "By", "Bs"]):
        """
        Calculate profile of field in s direction at x=y=0

        :param fields: Fields that are desired to plot
        """

        s, psi, r = np.meshgrid(np.linspace(-smax, smax, n_s),
                                [0.0],
                                [0.0])

        X = np.cos(s/rho) * (rho+r*np.cos(psi))
        Y = -np.sin(s/rho) * (rho+r*np.cos(psi))
        Z = r*np.sin(psi)
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ)

        dst = dst.interpolate(self.src, radius=radius)

        # Frenet-Serret coordinates
        dst["xFS"] = np.array(r*np.cos(psi)).flatten()
        dst["yFS"] = np.array(r*np.sin(psi)).flatten()
        dst["sFS"] = s.flatten()

        # Field in Frenet-Serret coordinates
        dst["BxFS"] = dst["Bx"] * np.cos(s/rho).flatten() - dst["By"] * np.sin(s/rho).flatten()
        dst["ByFS"] = dst["Bz"]
        dst["BsFS"] = dst["Bx"] * np.sin(s/rho).flatten() + dst["By"] * np.cos(s/rho).flatten()
       
        if plot:
            fig, ax = plt.subplots()
            if "Bx" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BxFS"], label="$B_x(x=y=0, s)$", s=3, color="orange")
            if "By" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["ByFS"], label="$B_y(x=y=0, s)$", s=3, color="blue")
            if "Bs" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BsFS"], label="$B_s(x=y=0, s)$", s=3, color="lightgreen")
            ax.set_xlabel("s (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.show()

        return dst


    def fit_x(self, rho, rmax, spos, field, radius=0.01, n_r=200, degree=10, plot=False, stepsize = 4):
        """
        Fit a polynomial to the x distribution for y=0 and a given s
        Bx(x, y=0, s) = a_j(s) x^j/j!
        By(x, y=0, s) = b_j(s) x^j/j!

        :return: list of coefficients, highest order term first
        """

        # In order to calculate the derivative in zero we want to have interpolation at zero, 
        # so we need n_r to be odd for the interpolation
        n_r = n_r + 1 - n_r%2
        dst = self.calc_x(rho, rmax, spos, radius, n_r)

        # Extract the multipole coefficients by fitting polynomial
        params = np.polyfit(dst.point_data["xFS"], dst.point_data[field], degree)
        coefsfit = [params[degree-i] * math.factorial(i) for i in range(degree+1)]

        # Extract the multipole coefficients by fitting polynomial only to the central region
        fitmin = 3*n_r//10
        fitmax = n_r - fitmin
        params = np.polyfit(dst.point_data["xFS"][fitmin:fitmax], dst.point_data[field][fitmin:fitmax], degree)
        coefsfitcenter = [params[degree-i] * math.factorial(i) for i in range(degree+1)]

        # Extract the multipole coefficients by calculating derivatives, using central difference
        h = stepsize * 4*rmax / (n_r-1)  # Twice the interval between two points, since we need half step values
        center = n_r // 2  # x=0
        coefsdiff = np.zeros(degree+1)
        for n in range(degree+1): 
            for i in range(n+1):
                ind = center+2*(n/2-i)*stepsize
                coefsdiff[n] += 1/h**n * (-1)**i * math.factorial(n)/math.factorial(n-i)/math.factorial(i) * dst.point_data[field][int(ind)]

        if plot:
            if field == "BxFS":
                direction = "x"
            elif field == "ByFS":
                direction = "y"
            elif field == "BsFS":
                direction = "s"
            else:
                direction = ""

            x = np.linspace(min(dst.point_data["xFS"]), max(dst.point_data["xFS"]), 1000)
            yfit = sum([coefsfit[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            yfitcenter = sum([coefsfitcenter[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            ydiff = sum([coefsdiff[i] * x**(i) / math.factorial(i) for i in range(degree+1)])

            fig, ax = plt.subplots()
            ax.scatter(dst.point_data["xFS"], dst.point_data[field], label=f"$B_{direction}(x, y=0, s={spos})$", s=3, color="orange")
            #ax.plot(x, yfit, label="Fit", color="red") 
            ax.plot(x, yfitcenter, label="Polynomial fit", color="purple") 
            ax.plot(x, ydiff, label="Partial derivative", color="pink") 
            #ax.axvline(x=dst.point_data["xFS"][fitmin], color="gray")
            #ax.axvline(x=dst.point_data["xFS"][fitmax], color="gray")
            ax.set_ylim(1.1*min(dst.point_data[field]), 1.1*max(dst.point_data[field]))
            ax.set_xlabel("x (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.show()

    
    def magnetic_length(self, rho, smax, radius=0.01, n_s=101, plot=False, fields=["Bx", "By", "Bs"]):
        """
        Calculate the magnetic length of the magnet

        :param fields: Fields that are desired to plot
        """
        
        n_s = n_s + 1 - n_s%2
        dst = self.calc_s(rho, smax, radius, n_s)

        Bint = sc.integrate.simpson(dst.point_data["ByFS"], x=dst.point_data["sFS"])
        [Bcenter] = dst.point_data["ByFS"][dst.point_data["sFS"]==0]
        mlen = Bint / Bcenter

        if plot:
            fig, ax = plt.subplots()
            ax.axvline(x=mlen/2, color="red")
            ax.axvline(x=-mlen/2, color="red")
            plt.text(mlen/2, 4.3, "$L_{mag}/2$", color="red", horizontalalignment="center")
            plt.text(-mlen/2, 4.3, "$L_{mag}/2$", color="red", horizontalalignment="center")
            if "Bx" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BxFS"], label="$B_x(x=y=0, s)$", s=3, color="orange")
            if "By" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["ByFS"], label="$B_y(x=y=0, s)$", s=3, color="blue")
            if "Bs" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BsFS"], label="$B_s(x=y=0, s)$", s=3, color="lightgreen")
            ax.set_xlabel("s (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.show()

        return Bcenter, mlen


field = "Bz"
rho = 1.65
rmax = 0.1

data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
cctmagnet = Magnet(data)

# Overview of magnet
cctmagnet.plot()

# Curved frame and slice 
smax = 0.8
cctmagnet.curved_plot(rho, rmax, smax, field)
cctmagnet.curved_plot(rho, rmax, smax, field, n_psi=2)

# Bx and By in slice
smax = 0.8
cctmagnet.curved_plot(rho, rmax, smax, "Bx", n_psi=2, n_r=10, n_s=200)
cctmagnet.curved_plot(rho, rmax, smax, "By", n_psi=2, n_r=10, n_s=200)
cctmagnet.curved_plot(rho, rmax, smax, "Bz", n_psi=2, n_r=10, n_s=200)

# Plot the field in x direction at given s position
spos=0
cctmagnet.calc_x(rho, rmax, spos, n_r=200, plot=True, fields="Bx")
cctmagnet.calc_x(rho, rmax, spos, n_r=200, plot=True, fields="By")

spos=0.1
cctmagnet.calc_x(rho, rmax, spos, n_r=200, plot=True, fields="Bx")
cctmagnet.calc_x(rho, rmax, spos, n_r=200, plot=True, fields="By")

# Plot in s direction of the field on axis
cctmagnet.calc_s(rho, 1, plot=True)

# Fit of fields to determine multipole coefficients
spos=0.1
cctmagnet.fit_x(rho, rmax, spos, "BxFS", degree = 20, plot=True)
cctmagnet.fit_x(rho, rmax, spos, "ByFS", degree = 20, plot=True)

# Plot in s direction including magnetic length
cctmagnet.magnetic_length(rho, smax, n_s=200, plot=True, fields=["By", "Bs"])






