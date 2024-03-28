import numpy as np
import gzip
import pyvista as pv


# Data in global coordinate system
data=np.loadtxt(gzip.open('fieldmap-cct.txt.gz'),skiprows=9)
src=pv.PolyData(data.T[:3].T)
src['Bx']=data.T[3].T
src['By']=data.T[4].T
src['Bz']=data.T[5].T


# Curved coordinate system for the hitriplus cct magnet
s,psi,r=np.meshgrid(np.linspace(-1,1,101),
                    np.linspace(0,2*np.pi,33),
                    np.linspace(0.0,0.1,5))


R=1.65

x=np.cos(s/R)*(R+ r*np.cos(psi))
y=np.sin(s/R)*(R+ r*np.cos(psi))
z=r*np.sin(psi)
xyz=np.array([x.flatten(),y.flatten(),z.flatten()]).T
dst=pv.PolyData(xyz)

#pl=pv.Plotter()
#pl.add_mesh(src)
#pl.add_mesh(dst)
#pl.show()

# Interpolation of data on curved coordinate system
dst=dst.interpolate(src, radius=0.1)
dst.plot(scalars="Bz")
