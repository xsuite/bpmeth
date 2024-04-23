import numpy as np
import gzip
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


    def closest_index(self, arr, value):
        """
        Find the index of the element with value closest to the requested value, using binary search
        """

        left, right = 0, len(arr) - 1
        closest_diff = float('inf')
        closest_index = None
    
        while left <= right:
            mid = (left + right) // 2
            current = arr[mid]
            diff = abs(current - value)
    
            if diff < closest_diff:
                closest_diff = diff
                closest_index = mid
    
            if current < value:
                left = mid + 1
            elif current > value:
                right = mid - 1
            else:
                return mid, current  # If 0.4 is found exactly, return index and value
    
        return closest_index



    def fit_x(self, rho, rmax, spos, field, radius=0.01, n_r=200, degree=10, plot=False):
        """
        Fit a polynomial to the x distribution for y=0 and a given s using two methods
        Bx(x, y=0, s) = a_j(s) x^j/j!
        By(x, y=0, s) = b_j(s) x^j/j!

        :return: list of coefficients for both methods, lowest order first
        """

        # In order to calculate the derivative in zero we want to have interpolation at zero, 
        # so we need n_r to be odd for the interpolation
        n_r = n_r + 1 - n_r%2
        dst = self.calc_x(rho, rmax, spos, radius, n_r)

        # Extract the multipole coefficients by fitting polynomial
        params = np.polyfit(dst.point_data["xFS"], dst.point_data[field], degree)
        coefsfit = [params[degree-i] * math.factorial(i) for i in range(degree+1)]

        # Extract the multipole coefficients by fitting polynomial only to the central region
        fitmin = self.closest_index(dst.point_data["xFS"], -0.04)
        fitmax = n_r - fitmin
        params = np.polyfit(dst.point_data["xFS"][fitmin:fitmax], dst.point_data[field][fitmin:fitmax], degree)
        coefsfitcenter = [params[degree-i] * math.factorial(i) for i in range(degree+1)]

        # Extract the multipole coefficients by calculating derivatives, using central difference
        center = n_r // 2  # x=0
        if degree>0:
            stepsize = 2 * (fitmax-center) // degree
        else:
            stepsize=1
        h = stepsize * 4*rmax / (n_r-1)  # Twice the interval between two points, since we need half step values
        coefsdiff = np.zeros(degree+1)
        for n in range(degree+1): 
            for i in range(n+1):
                ind = int(center+2*(n/2-i)*stepsize)
                coefsdiff[n] += 1/h**n * (-1)**i * math.factorial(n)/math.factorial(n-i)/math.factorial(i) * dst.point_data[field][ind]

        if plot:
            x = np.linspace(min(dst.point_data["xFS"]), max(dst.point_data["xFS"]), 1000)
            #yfit = sum([coefsfit[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            yfitcenter = sum([coefsfitcenter[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
            ydiff = sum([coefsdiff[i] * x**(i) / math.factorial(i) for i in range(degree+1)])

            fig, ax = plt.subplots(figsize=(4,3.25))
            ax.scatter(dst.point_data["xFS"], dst.point_data[field], label=f"$B_{field[1]}(x, y=0, s={spos})$", s=3, color="orange")
            #ax.plot(x, yfit, label="Fit", color="red") 
            ax.plot(x, yfitcenter, label="Polynomial fit", color="purple") 
            ax.plot(x, ydiff, label="Partial derivative", color="pink") 

            ymax = max(dst.point_data[field])
            ymin = min(dst.point_data[field])
            ydelta = ymax - ymin
            ax.annotate('', xy=(-0.04, ymin - 0.05*ydelta), xytext=(0.04, ymin - 0.05*ydelta),\
                    arrowprops=dict(arrowstyle='<->', color="gray"))
            ax.text(0, ymin, "Aperture", color="gray", horizontalalignment="center")

            ax.set_ylim(ymin - 0.1*ydelta, 1.6*ymax)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("Field (T)")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"{field[0:2]}fit.png")
            plt.show()

        return dst, coefsfitcenter, coefsdiff
        
    
    def plot_difference(self, rho, rmax, spos, field, radius=0.01, n_r=200, degree=10):
        """
        Fit a polynomial to the x distribution for y=0 and a given s using two different methods, and compare them with the real values
        """

        dst, coefsfitcenter, coefsdiff = self.fit_x(rho, rmax, spos, field, radius, n_r, degree)

        mask = (dst.point_data["xFS"]>-0.04) & (dst.point_data["xFS"]<0.04)
        x = dst.point_data["xFS"][mask]

        yfitcenter = dst.point_data[field][mask] - sum([coefsfitcenter[i] * x**(i) / math.factorial(i) for i in range(degree+1)])
        ydiff = dst.point_data[field][mask] - sum([coefsdiff[i] * x**(i) / math.factorial(i) for i in range(degree+1)])

        fig, ax = plt.subplots(figsize=(4,3.25))
        ax.scatter(x, yfitcenter, label=f"Polynomial fit", s=3, color="purple")
        ax.scatter(x, ydiff, label=f"Partial derivative", s=3, color="pink")

        ymin = min(yfitcenter + ydiff)
        ymax = max(yfitcenter + ydiff)
        ydelta = ymax - ymin

        ax.set_ylim(ymin - 0.1*ydelta, ymax + 0.5*ydelta)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("Deviation from actual field (T)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{field[0:2]}fitdiff_{degree}.png")
        plt.show()

        return


    def compare_methods(self, rho, rmax, spos, field, radius=0.01, n_r=200, maxdegree=20, indices=None):
        """
        Compare the stability of the parameters for the two methods. For a good method the
        parameters do not depend on the degree, only then one can talk to magnet designers!
        """

        allfitcenter = [[] for i in range(maxdegree+1)]
        alldiff = [[] for i in range(maxdegree+1)]
        for degree in range(maxdegree+1):
            _, coefsfitcenter, coefsdiff = self.fit_x(rho, rmax, spos, field, radius, n_r, degree)
            allfitcenter[degree] = coefsfitcenter
            alldiff[degree] = coefsdiff

        if indices is None:
            indices = [i for i in range(maxdegree+1)]

        fig, ax = plt.subplots(len(indices))
        for axindex, index in enumerate(indices):
            fiti = [allfitcenter[i][index] for i in range(index, maxdegree+1)]
            diffi = [alldiff[i][index] for i in range(index, maxdegree+1)]
            xi = [i for i in range(index, maxdegree+1)]
            ax[axindex].plot(xi, fiti, label="Polynomial fit")
            ax[axindex].plot(xi, diffi, label="Partial derivatives")
            ax[axindex].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[axindex].set_xlabel("Degree of calculation")
            ax[axindex].set_ylabel(f"coeff_{index}")
            ax[axindex].set_xlim(0,maxdegree)
        plt.legend()
        plt.show()
        
        return

    
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
            fig, ax = plt.subplots(figsize=(4,3.25))
            ax.axvline(x=mlen/2, color="gray", linewidth=0.75)
            ax.axvline(x=-mlen/2, color="gray", linewidth=0.75)
            plt.text(mlen/2, 6.25, "$L_{mag}/2$", color="gray", horizontalalignment="center")
            plt.text(-mlen/2, 6.25, "$L_{mag}/2$", color="gray", horizontalalignment="center")
            if "Bx" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BxFS"], label="$B_x(x=y=0, s)$", s=3, color="orange")
            if "By" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["ByFS"], label="$B_y(x=y=0, s)$", s=3, color="blue")
            if "Bs" in fields:
                ax.scatter(dst.point_data["sFS"], dst.point_data["BsFS"], label="$B_s(x=y=0, s)$", s=3, color="lightgreen")
            ax.set_xlabel("s (m)")
            ax.set_ylabel("Field (T)")
            ax.set_ylim((-0.5,6))
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig("magneticlength.png")
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
cctmagnet.fit_x(rho, rmax, spos, "BxFS", degree=20, plot=True)
cctmagnet.fit_x(rho, rmax, spos, "ByFS", degree=20, plot=True)
# Plot the difference between the fit and the real value for both methods 
spos=0.1
cctmagnet.plot_difference(rho, rmax, spos, "BxFS", degree=20)
cctmagnet.plot_difference(rho, rmax, spos, "ByFS", degree=20)

# Compare the two methods concerning stability
spos=0.1
cctmagnet.compare_methods(rho, rmax, spos, "BxFS", maxdegree=30, indices=[0,1,2,3,4,5,6])
cctmagnet.compare_methods(rho, rmax, spos, "ByFS", maxdegree=30, indices=[0,1,2,3,4,5,6])



# Plot in s direction including magnetic length
cctmagnet.magnetic_length(rho, smax, n_s=200, plot=True, fields=["By", "Bs"])






