import bpmeth
import xtrack as xt
import numpy as np

dipole = bpmeth.DipoleFromFint(5, 5, 0.3, 0.8)

p = xt.Particles(x=np.linspace(-1e-3, 1e-3, 5), y=np.linspace(-1e-3, 1e-3, 5), p0c=0.1, mass0=0.938272, q0=1)
dipole.track(p)