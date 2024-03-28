import numpy as np
import gzip
import pyvista as pv


class Magnet:
    def __init__(self, data):
        self.data = data
        self.src = pv.PolyData(self.data.T[:3].T)
        self.src['Bx'] = self.data.T[3].T
        self.src['By'] = self.data.T[4].T
        self.src['Bz'] = self.data.T[5].T


    def plot(self):
        self.src.plot()
        

    def curvedplot(self, rho, rmax, field, radius=0.1, n_psi=32):
        s, psi, r = np.meshgrid(np.linspace(-1, 1, 101),
                                np.linspace(0, 2*np.pi, n_psi+1),
                                np.linspace(0.0, rmax, 5))

        x = np.cos(s/rho) * (rho+r*np.cos(psi))
        y = np.sin(s/rho) * (rho+r*np.cos(psi))
        z = r*np.sin(psi)
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        dst = pv.PolyData(xyz)

        dst=dst.interpolate(self.src, radius=radius)
        dst.plot(scalars=field)





data = np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
cctmagnet = Magnet(data)

cctmagnet.plot()

rho = 1.65
rmax = 0.1
cctmagnet.curvedplot(rho, rmax, "Bz")
cctmagnet.curvedplot(rho, rmax, "Bz", n_psi=2)

