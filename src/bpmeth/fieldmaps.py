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

    

    
def get_derivative(deriv_order, nparams, func, lambdify=True):
    # Calculate the derivative
    x = sp.symbols('x')
    params = sp.symbols(f'p0:{nparams}')
    derivative_sym = sp.diff(func(x, *params), x, deriv_order)

    if lambdify:  # Numpy
        def derivative_func(x_val, *param_values):
            func = sp.lambdify((x, *params[:len(param_values)]), derivative_sym, "numpy")
            return func(x_val, *param_values)
    else:  # Sympy
        def derivative_func(s_val, *param_values):
            subs_dict = {x: s_val, **{p: v for p, v in zip(params, param_values)}}
            return derivative_sym.subs(subs_dict)

    return derivative_func


def spTanh(s, *params):
    return 1/2*params[0]*(sp.tanh(s/params[1])+1)




class Fieldmap:
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
        
    def interpolate_points(self, X, Y, Z, radius=0.01):
        """
        Interpolate the field at the given points (meshgrid)
        """
        
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        
        #dst.plot(scalars=field)
        
        dst['x'] = X.flatten()
        dst['y'] = Y.flatten()
        dst['z'] = Z.flatten()
        
        return dst

    def calc_FS_coords(self, XFS, YFS, SFS, rho, phi, radius=0.01):
        """
        Determine a dataframe consisting of a straight piece up to -l_magn/2, then a bent piece up to l_magn/2, 
        then again a straight piece.

        :param XFS: Array of X positions in Frenet-Serrat coordinates.
        :param YFS: Array of Y positions in Frenet-Serrat coordinates.
        :param SFS: Array of S positions in Frenet-Serrat coordinates.
        :param rho: Bending radius of the magnet, in radians.
        :param phi: Angle of the magnet. Related to the magnetic length by l_magn = rho * phi.
        :param radius: Interpolation radius, default 0.01.
        :return: Fieldmap object in FS_coordinates.
        """

        l_magn = rho*phi
        
        # Straight part at negative s
        xarr = XFS
        yarr = YFS
        sarr = SFS[SFS<-l_magn/2] + l_magn/2
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_ns = x
        y_ns = y
        s_ns = s - l_magn/2

        X_ns = (rho + x) * np.cos(phi/2) + s * np.sin(phi/2)
        Y_ns = y
        Z_ns = -(rho + x) * np.sin(phi/2) + s * np.cos(phi/2)

        XYZ = np.array([X_ns.flatten(), Y_ns.flatten(), Z_ns.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_ns = dst["Bx"]*np.cos(phi/2) + dst["Bz"]*np.sin(phi/2)
        By_ns = dst["By"]
        Bs_ns = -dst["Bx"]*np.sin(phi/2) + dst["Bz"]*np.cos(phi/2)

        # Bent part 
        xarr = XFS
        yarr = YFS
        sarr = SFS[abs(SFS)<l_magn/2]
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_b = x
        y_b = y
        s_b = s

        X_b = np.cos(s/rho) * (rho + x)
        Y_b = y
        Z_b = np.sin(s/rho) * (rho + x)

        XYZ = np.array([X_b.flatten(), Y_b.flatten(), Z_b.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_b = dst["Bx"]*np.cos(s.flatten()/rho) - dst["Bz"]*np.sin(s.flatten()/rho)
        By_b = dst["By"]
        Bs_b = dst["Bx"]*np.sin(s.flatten()/rho) + dst["Bz"]*np.cos(s.flatten()/rho)

        # Straight part positive s
        xarr = XFS
        yarr = YFS
        sarr = SFS[SFS>l_magn/2] - l_magn/2
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_ps = x
        y_ps = y
        s_ps = s + l_magn/2

        X_ps = (rho + x) * np.cos(phi/2) - s * np.sin(phi/2)
        Y_ps = y
        Z_ps = (rho + x) * np.sin(phi/2) + s * np.cos(phi/2)

        XYZ = np.array([X_ps.flatten(), Y_ps.flatten(), Z_ps.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_ps = dst["Bx"]*np.cos(phi/2) - dst["Bz"]*np.sin(phi/2)
        By_ps = dst["By"]
        Bs_ps = dst["Bx"]*np.sin(phi/2) + dst["Bz"]*np.cos(phi/2)

        # Combine all
        x = np.concatenate((x_ns.flatten(), x_b.flatten(), x_ps.flatten()))
        y = np.concatenate((y_ns.flatten(), y_b.flatten(), y_ps.flatten()))
        s = np.concatenate((s_ns.flatten(), s_b.flatten(), s_ps.flatten()))

        Bx = np.concatenate((Bx_ns.flatten(), Bx_b.flatten(), Bx_ps.flatten()))
        By = np.concatenate((By_ns.flatten(), By_b.flatten(), By_ps.flatten()))
        Bz = np.concatenate((Bs_ns.flatten(), Bs_b.flatten(), Bs_ps.flatten()))

        data = np.array([x, y, s, Bx, By, Bz]).T

        return Fieldmap(data)

        
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
        """
        So far only normal multipoles
        """
        zvals = np.unique(self.src['z'])
        coeffs = np.zeros((len(zvals), order+1))
        coeffsstd = np.zeros((len(zvals), order+1))
        for i, zpos in enumerate(zvals):
            coeffs[i], coeffsstd[i] = self.fit_xprofile(0, zpos, "By", order, xmax=xmax)
        return zvals, coeffs, coeffsstd


    def integratedfield(self, order, xmax=None):
        zvals, coeffs, coeffsstd = self.z_multipoles(order, xmax=xmax)
        
        integrals = np.zeros(order)
        for i in range(order):
            integrals[i] = np.trapezoid(coeffs[:,i], zvals)
        return integrals

               
               
    def fit_multipoles(self, shape, components=[1,2], design=1, nparams=6, xmax=None, zmin=-9999, zmax=9999, zedge=0, guess=None, ax=None, force_zero=False):
        """
        Fit appropriate functions to the multipoles in the fieldmap, taking into account the design of the magnet.

        :param shape: Fringe field shape - function to fit the lowest order fringe field.
        :param components: List of components to fit, counting starts at one like the multipole coefficients.
        :param design: What is the design of the magnet? Dipole:1, Quadrupole:2 etc. 
                       The design component has an Enge as fringe field, each higher order
                       scales with a higher order derivative for fitting.
                       NO COMBINED FUNCTION ALLOWED 
        :param nparams (optional): Number of parameters in the shape, nparams-1 is the order of the polynomial in the Enge function.
        :param xmax: Range in x to take into account in multipole fitting in transverse plane.
        :param zmin: Longitudinal range.
        :param zmax: Longitudinal range.
        :param zedge: Location of the magnet edge in case only the fringe field shape is fitted, default zero.
        :param guess: Guess for parameters of the Enge function.
        :param ax: If given, plot the fit in these axis.
        :param force_zero: If true, force the field to be zero outside of the magnet, and the maximal field to be kept inside the magnet.
        :return: Parameters and covariances for all components.
        """

        order = max(components) - 1
        zvals, coeffs, coeffsstd = self.z_multipoles(order, xmax=xmax)
        mask = (zvals>zmin) & (zvals<zmax)
        zvals = zvals[mask]

        params_list = np.zeros((len(components), nparams))
        cov_list = np.zeros((len(components), nparams, nparams))
        

        for i, component in enumerate(components):
            print(f"fitting b{component}...")
            b = coeffs[:, component-1]
            berr = coeffsstd[:, component-1]

            b = b[mask]
            berr = berr[mask]

            if guess is None:
                guess = np.zeros(nparams)
                guess[0] = b[np.argmax(np.abs(b))]


            if force_zero:
                zmin = np.min(zvals)
                min_ind = np.argmin(zvals)
                zmax = np.max(zvals)
                max_ind = np.argmax(zvals)
                print(component-design)
                if component-design == 0:  # main field
                    if b[min_ind] < b[max_ind]:  # entrance fringe
                        centralfield = b[max_ind]
                        b = np.append(b, [centralfield, centralfield])
                        zv = np.append(zvals, [zmax+0.05, zmax+0.1])
                        berr = np.append(berr, [1e-12, 1e-12])
                    else:  # exit fringe
                        centralfield = b[min_ind]
                        b = np.append([centralfield, centralfield], b)
                        zv = np.append([zmin-0.05, zmin-0.1], zvals)
                        berr = np.append([1e-12, 1e-12], berr)

                else:
                    zv = zvals


            shape_derivative = get_derivative(component-design, nparams, shape)
            params, cov = sc.optimize.curve_fit(shape_derivative, zv-zedge, b, sigma=berr, p0=guess, maxfev=10000)
            params_list[i] = params
            cov_list[i] = cov
            
            if ax is not None:
                zz = np.linspace(min(zvals), max(zvals), 100)
                ax.errorbar(zv, b, yerr=berr, marker='.', capsize=5, label=f"b{component}", ls='', zorder=0)
                ax.plot(zz, shape_derivative(zz-zedge, *params), label=f'Fit of {component-design}th order derivative with {nparams} parameters to component b{component}', zorder=10)

            guess=params

        if ax is not None:
            plt.legend()
        
        return params_list, cov_list 
