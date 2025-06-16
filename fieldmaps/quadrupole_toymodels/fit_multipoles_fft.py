import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class FieldMap:
    def __init__(self, fname):
        self.fname = fname
        x, y, z, Bx, By, Bz = np.loadtxt(fname, skiprows=9).T
        nx = len(set(x))
        ny = len(set(y))
        nz = len(set(z))
        x_grid = x[:nx]
        y_grid = y[: nx * ny : nx]
        z_grid = z[:: nx * ny]
        Bx = Bx.reshape(nx, ny, nz, order="F")
        By = By.reshape(nx, ny, nz, order="F")
        Bz = Bz.reshape(nx, ny, nz, order="F")

        # Ensure the grid is symmetric about the y-axis
        y_grid = np.concatenate((-y_grid[:0:-1], y_grid))
        ny = len(y_grid)

        Bx = np.concatenate((-Bx[:, :0:-1, :], Bx), axis=1)
        By = np.concatenate((By[:, :0:-1, :], By), axis=1)
        Bz = np.concatenate((Bz[:, :0:-1, :], Bz), axis=1)

        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid
        self.nx = nx
        self.ny = ny
        self.nz = nz

        xyz = (self.x_grid, self.y_grid, self.z_grid)

        # Create interpolators
        self._Bxi = RegularGridInterpolator(
            xyz, Bx.reshape(self.nx, self.ny, self.nz, order="F")
        )
        self._Byi = RegularGridInterpolator(
            xyz, By.reshape(self.nx, self.ny, self.nz, order="F")
        )
        self._Bzi = RegularGridInterpolator(
            xyz, Bz.reshape(self.nx, self.ny, self.nz, order="F")
        )

    def check_interpolation(self):
        assert np.allclose(self._Byi((self.x, self.y, self.z)), self.By)

    def __call__(self, x, y, z):
        bx, by, bz = self._Bxi((x, y, z)), self._Byi((x, y, z)), self._Bzi((x, y, z))
        return bx, by, bz

    def Bx(self, x, y, z):
        """Get Bx at (x, y, z)."""
        bx = self._Bxi((x, y, z))
        return bx

    def By(self, x, y, z):
        """Get By at (x, y, z)."""
        by = self._Byi((x, y, z))
        return by

    def Bz(self, x, y, z):
        """Get Bz at (x, y, z)."""
        bz = self._Bzi((x, y, z))
        return bz

    def plot_z(self, x, y, z=None, n=101):
        if z is None:
            z = np.linspace(self.z_grid[0], self.z_grid[-1], n)

        Bx_val = self.Bx(x, y, z)
        By_val = self.By(x, y, z)
        Bz_val = self.Bz(x, y, z)

        plt.plot(z, Bx_val, label="Bx")
        plt.plot(z, By_val, label="By")
        plt.plot(z, Bz_val, label="Bz")
        plt.xlabel("z")
        plt.ylabel("Magnetic Field")
        plt.legend()
        plt.show()

    def byibx(self, r, z, sample_points=64):
        """Calculate By+iBx along a circle of radius r at height z."""
        t = 2*np.pi/sample_points*np.arange(sample_points) 
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.full_like(t, z)
        Bx_val = self.Bx(x, y, z)
        By_val = self.By(x, y, z)
        return By_val + 1j * Bx_val

    def plot_byibx(self, r, z, sample_points=256):
        """Plot By+iBx along a circle of radius r at height z."""
        byibx = self.byibx(r, z, sample_points=sample_points)
        t = 360 / sample_points * np.arange(sample_points)
        plt.plot(t, byibx.real, label=f"$B_y(r={r},\\phi,z={z})$")
        plt.plot(t, byibx.imag, label=f"$B_x(r={r},\\phi,z={z})$")
        plt.xlabel("Angle (degrees)")
        plt.ylabel(r"$B_y, B_x$")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_multipole(self, r, z, sample_points=256, nmax=15):
        """Calculate the multipole coefficients along a circle of radius r at height z."""
        byibx = self.byibx(r, z, sample_points=sample_points)
        coeffs = np.fft.fft(byibx)[:nmax] / sample_points
        return coeffs/r**np.arange(nmax)

    def plot_bn_r(self, n, z, rmin=None, rmax=None):
        """Plot the multipole coefficients along a circle of radius r at height z."""
        if rmin is None:
            rmin= list(sorted(np.abs(self.x_grid)))[1]
        if rmax is None:
            rmax = self.x_grid.max() / 2
        r = np.linspace(rmin, rmax, 51)
        assert n > 0, "n must be a positive integer"
        bn = [self.get_multipole(r_i, z)[n - 1].real for r_i in r]
        plt.plot(r, bn,label=f"$B_{n}(r)$ at $z={z}$")
        plt.xlabel("Radius r")
        plt.ylabel(f"Multipole Coefficient $B_{n}$")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_bn_z(self, n, r, zmin=None, zmax=None, ax=None):
        """Plot the multipole coefficients along a circle of radius r at height z."""
        if zmin is None:
            zmin = self.z_grid[0]
        if zmax is None:
            zmax = self.z_grid[-1]
        z = np.linspace(zmin, zmax, 51)
        assert n > 0, "n must be a positive integer"
        bn = [self.get_multipole(r, z_i)[n - 1].real for z_i in z]
        if ax is None:
            fig, ax = plt.subplots()
        plt.plot(z, bn,label=f"$B_{n}(z)$ at $r={r}$")
        plt.xlabel("z")
        plt.ylabel(f"Multipole Coefficient")
        plt.grid(True)
        plt.legend()
        #plt.show()
        return z, bn

    def plot_multipole(self, r, z, n=151):
        """Plot the multipole coefficients along a circle of radius r at height z."""
        coeffs = self.get_multipole(r, z, n)
        plt.plot(np.abs(coeffs), label="Magnitude")
        plt.plot(np.angle(coeffs), label="Phase")
        plt.xlabel("Multipole Order")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.title(f"Multipole Coefficients at r={r}, z={z}")
        plt.show()
