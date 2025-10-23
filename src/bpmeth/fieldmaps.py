import pyvista as pv
import numpy as np
import sympy as sp
import math
import scipy as sc
import matplotlib.pyplot as plt
from .poly_fit import fit_segment, plot_fit


def Enge(x, *params):
    return params[0] / (1+np.exp(np.poly1d(params[1:])(x)))
       
def spEnge(s, *params):
    return params[0] / (1+sp.exp(sp.Poly(params[1:], s).as_expr()))

def spTanh(s, *params):
    return params[0] * (sp.tanh(s/params[1]) + 1)/2


    
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

    
def moving_average(arr, N):
    # Step 1: Compute trimmed moving average (shorter result)
    result = []
    for i in range(len(arr) - N + 1):
        window = arr[i:i+N]
        if N >= 3:
            trimmed = np.sort(window)[1:-1]  # remove min and max
            result.append(np.mean(trimmed))
        else:
            result.append(np.mean(window))
    result = np.array(result)

    # Step 2: Pad the result to match the original length
    pad_width = (N - 1) // 2
    result_padded = np.pad(result, pad_width, mode='edge')  # replicate edges

    return result_padded





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

    def __add__(self, other):
        # Needs to have fieldmaps defined in the same points
        if self.src.n_points != other.src.n_points:
            raise ValueError("Fieldmaps have different number of points.")
        if not np.allclose(self.src.points, other.src.points):
            raise ValueError("Fieldmaps have different point coordinates.")

        x = self.src["x"]
        y = self.src["y"]
        z = self.src["z"]
        Bx = self.src["Bx"] + other.src["Bx"]
        By = self.src["By"] + other.src["By"]
        Bz = self.src["Bz"] + other.src["Bz"]
        
        data = np.array([x, y, z, Bx, By, Bz]).T

        return Fieldmap(data)
    
    def plot(self, field="By"):
        self.src.plot(scalars=field)
        
    def interpolate_points(self, X, Y, Z, radius=0.01):
        """
        Interpolate the field at the given points (meshgrid)
        """
        
        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        
        data = np.array([X.flatten(), Y.flatten(), Z.flatten(), dst['Bx'], dst['By'], dst['Bz']]).T

        return Fieldmap(data)

    def calc_FS_coords(self, XFS, YFS, SFS, rho, phi, radius=0.01):
        """
        Determine a dataframe consisting of a straight piece up to -l_magn/2, then a bent piece up to l_magn/2, 
        then again a straight piece.
        (0,0,0) is center of curvature
        !!! This does not work for negative bending radius !!!

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


    def calc_edge_frame(self, Xedge, Yedge, Zedge, edge_angle, rho, phi, radius=0.01):
        """
        :param Xedge: local X coordinates in the edge frame.
        :param Yedge: local Y coordinates in the edge frame.
        :param Zedge: local Z coordinates in the edge frame.
        :param edge_angle: Angle with respect to a sector bend, in radians
        :param rho: Bending radius of the magnet, in radians.
        :param phi: Angle of the magnet. Related to the magnetic length by l_magn = rho * phi.
        :param radius: Interpolation radius, default 0.01.
        :return: Fieldmap object in FS_coordinates.
        """

        l_magn = rho*phi
        x, y, z = np.meshgrid(Xedge, Yedge, Zedge)

        X = rho * np.cos(phi/2) + x * np.cos(phi/2-edge_angle) - z * np.sin(phi/2-edge_angle)
        Y = y
        Z = rho * np.sin(phi/2) + x * np.sin(phi/2-edge_angle) + z * np.cos(phi/2-edge_angle)

        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)

        Bx = dst["Bx"] * np.cos(phi/2-edge_angle) - dst["Bz"] * np.sin(phi/2-edge_angle)
        By = dst["By"]
        Bz = dst["Bx"] * np.sin(phi/2-edge_angle) + dst["Bz"] * np.cos(phi/2-edge_angle)

        data = np.array([x.flatten(), y.flatten(), z.flatten(), Bx, By, Bz]).T

        return Fieldmap(data)

    def get_data(self):
        data = np.array([self.src['x'], self.src['y'], self.src['z'],
                         self.src['Bx'], self.src['By'], self.src['Bz']]).T
        return data
    
    def rotate(self, theta):
        """
        Rotate the frame around the y-axis by theta radians counterclockwise
        """
        
        x = self.src['x']
        y = self.src['y']
        z = self.src['z']
        
        Bx = self.src['Bx']
        By = self.src['By']
        Bz = self.src['Bz']
        
        x_rotated = x * np.cos(theta) + z * np.sin(theta)
        z_rotated = -x * np.sin(theta) + z * np.cos(theta)
        
        Bx_rotated = Bx * np.cos(theta) + Bz * np.sin(theta)
        Bz_rotated = -Bx * np.sin(theta) + Bz * np.cos(theta)
        
        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([x_rotated, y, z_rotated, Bx_rotated, By, Bz_rotated]).T

        return Fieldmap(data)
        
    def translate(self, dx, dy, dz): 
        """
        Translate the frame by dx, dy, dz in the x, y, z directions respectively
        """
        
        x = self.src['x']
        y = self.src['y']
        z = self.src['z']

        Bx = self.src['Bx']
        By = self.src['By']
        Bz = self.src['Bz']

        x_translated = x + dx 
        y_translated = y + dy 
        z_translated = z + dz 

        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([x_translated, y_translated, z_translated, Bx, By, Bz]).T
    
        return Fieldmap(data)

    def symmetrize(self, radius=0.01):
        sarr = np.unique(self.src['z'])
        xarr = np.unique(self.src['x'])
        yarr = np.unique(self.src['y'])

        s, x, y = np.meshgrid(sarr, xarr, yarr)
        xys1 = np.array([x.flatten(), y.flatten(), s.flatten()]).T
        dst1 = pv.PolyData(xys1).interpolate(self.src, radius=radius)

        xys2 = np.array([x.flatten(), y.flatten(), -s.flatten()]).T
        dst2 = pv.PolyData(xys2).interpolate(self.src, radius=radius)

        Bx = (dst1["Bx"] + dst2["Bx"])/2
        By = (dst1["By"] + dst2["By"])/2
        Bs = (dst1["Bz"] - dst2["Bz"])/2

        data = np.array([x.flatten(), y.flatten(), s.flatten(), Bx, By, Bs]).T

        return Fieldmap(data)
            
    def mirror(self, left=True):
        """
        Mirror the frame
        :param left: If True, take left side and mirror, otherwise right side.
        """
        
        z = self.src['z']
        if left:
            mask = z < 0
        else:
            mask = z > 0

        x = self.src['x'][mask]
        y = self.src['y'][mask]
        z = self.src['z'][mask]

        Bx = self.src['Bx'][mask]
        By = self.src['By'][mask]
        Bz = self.src['Bz'][mask]

        xx = np.concatenate((x, x))
        yy = np.concatenate((y, y))
        zz = np.concatenate((z, -z))
        
        Bxx = np.concatenate((Bx, Bx))
        Byy = np.concatenate((By, By))
        Bzz = np.concatenate((Bz, -Bz))

        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([xx, yy, zz, Bxx, Byy, Bzz]).T
    
        return Fieldmap(data)
    
    def rescale(self, scalefactor):
        self.src["Bx"] = self.src["Bx"] * scalefactor
        self.src["By"] = self.src["By"] * scalefactor
        self.src["Bz"] = self.src["Bz"] * scalefactor

        
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

    def z_multipoles(self, order, xmax=None, ax=None, mov_av=1, colors=None, ls='', marker='.', ms=6, elinewidth=0.5, capsize=1, labels=None):
        """
        So far only normal multipoles
        """

        zvals = np.unique(self.src['z'])
        coeffs = np.zeros((len(zvals), order+1))
        coeffsstd = np.zeros((len(zvals), order+1))
        for i, zpos in enumerate(zvals):
            coeffs[i], coeffsstd[i] = self.fit_xprofile(0, zpos, "By", order, xmax=xmax)
        
        for i in range(order + 1):
            coeffs[:, i] = moving_average(coeffs[:, i], N=mov_av)
            coeffsstd[:, i] = moving_average(coeffsstd[:, i], N=mov_av)            
        
        if ax is not None:
            if labels is None:
                labels = [f"b{i+1}" for i in range(order+1)]
            for i in range(order + 1):
                color=colors[i] if colors is not None else None
                ax.errorbar(zvals, coeffs[:,i], yerr=coeffsstd[:,i], label=labels[i], ls=ls, 
                        capsize=capsize, marker=marker, color=color, elinewidth=elinewidth, ms=ms)
        return zvals, coeffs, coeffsstd


    def integratedfield(self, order, xmax=None, zmin=-9999, zmax=9999):
        zvals, coeffs, coeffsstd = self.z_multipoles(order, xmax=xmax)
        
        mask = (zvals>zmin) & (zvals<zmax)
        
        integrals = np.zeros(order)
        for i in range(order):
            integrals[i] = np.trapezoid(coeffs[:,i][mask], zvals[mask])
        return integrals


    def calc_Fint(self, gap, b0):
        """
        Fringe field integral that specifies the closed orbit distortion.

        :param hgap: The gap of the magnet, used to make K0 dimensionless.
        :param b0: The design field of the body of the magnet. If unknown, one can estimate this as the maximum of b1.
        :return: Fully dimensionless integral: divided by b0 and gap^2
        """

        zvals, coeffs, coeffsstd = self.z_multipoles(2)
        b1 = coeffs[:, 0]

        Fint = 1/gap * np.trapezoid(b1 * (b0 - b1) / b0**2, zvals)

        return Fint


    def calc_K0(self, gap, b0, entrance=True):
        """
        Fringe field integral that specifies the closed orbit distortion.
        Assumes that the edge is located at z=0, as would be returned by the function calc_edge_frame.
        This will change the value of the integral.
        Entrance and exit integrals are connected by a minus sign.

        :param hgap: The gap of the magnet, used to make K0 dimensionless.
        :param b0: The design field of the body of the magnet. If unknown, one can estimate this as the maximum of b1.
        :param entrance: Optional parameter specifying if edge is entrance or exit of the magnet.
        :return: Fully dimensionless integral: divided by b0 and gap^2
        """

        zvals, coeffs, coeffsstd = self.z_multipoles(2)
        b1 = coeffs[:, 0]

        if entrance is False:
            b1 = b1[::-1]

        K0 = 1/gap**2 * np.trapezoid((zvals) * (b0*np.heaviside(zvals, 0.5) - b1) / b0, zvals)

        return K0
               
               
    def fit_multipoles(self, shape, components=[1,2], design=1, nparams=5, xmax=None, zmin=-9999, zmax=9999, zedge=0, guess=None, ax=None, padding=False, entrance=False, design_field=None):
        """
        Fit appropriate functions to the multipoles in the fieldmap, taking into account the design of the magnet.

        :param shape: Fringe field shape - function to fit the lowest order fringe field.
        :param components: List of components to fit, counting starts at one like the multipole coefficients.
        :param design: What is the design of the magnet? Dipole:1, Quadrupole:2 etc. 
                       The design component has an Enge as fringe field, each higher order
                       scales with a higher order derivative for fitting.
                       NO COMBINED FUNCTION ALLOWED 
        :param nparams (optional): Number of parameters in the shape, nparams-1 is the order of the polynomial in the Enge function. 
                                   Notice the following for a fringe field: our fit should remain at one for large negative values of x, 
                                   and go to zero at large positive values of x (or vise versa). The highest order term of the polynomial
                                   in the Enge function should be odd. Hence nparams should be odd as well.
        :param xmax: Range in x to take into account in multipole fitting in transverse plane.
        :param zmin: Longitudinal range.
        :param zmax: Longitudinal range.
        :param zedge: Location of the magnet edge in case only the fringe field shape is fitted, default zero.
        :param guess: Guess for parameters of the Enge function.
        :param ax: If given, plot the fit in these axis.
        :return: Parameters and covariances for all components.
        """

        order = max(components) - 1
        zvals, coeffs, coeffsstd = self.z_multipoles(order, xmax=xmax)
        mask = (zvals>zmin) & (zvals<zmax)
        zvals = zvals[mask]

        params_list = np.zeros((len(components), nparams))
        cov_list = np.zeros((len(components), nparams, nparams))
        
        if design_field is None:
            design_field = b[np.argmax(np.abs(b))]  # Does not work

        for i, component in enumerate(components):        
            print(f"fitting b{component}...")
            b = coeffs[:, component-1]
            berr = coeffsstd[:, component-1]

            b = b[mask]
            berr = berr[mask]
            
            if padding:
                if entrance:
                    b = np.pad(b, (0, 10), mode='linear_ramp', end_values=design_field)
                    b = np.pad(b, (10, 0), mode='linear_ramp', end_values=0)
                else:
                    b = np.pad(b, (10, 0), mode='linear_ramp', end_values=design_field)
                    b = np.pad(b, (0, 10), mode='linear_ramp', end_values=0)
                berr = np.pad(berr, (10, 10), mode='constant', constant_values=np.mean(berr)*10)
                zvals = np.pad(zvals, (10, 0), mode='linear_ramp', end_values=np.min(zvals)-1)
                zvals = np.pad(zvals, (0, 10), mode='linear_ramp', end_values=np.max(zvals)+1)

            if guess is None:
                guess = np.zeros(nparams)
                guess[0] = design_field

            shape_derivative = get_derivative(component-design, nparams, shape)
            params, cov = sc.optimize.curve_fit(shape_derivative, zvals-zedge, b, sigma=berr, p0=guess, maxfev=10000)
            params_list[i] = params.copy()
            cov_list[i] = cov
            
            if ax is not None:
                zz = np.linspace(min(zvals), max(zvals), 100)
                ax.errorbar(zvals, b, yerr=berr, marker='.', capsize=5, label=f"b{component}", ls='', zorder=0)
                ax.plot(zz, shape_derivative(zz-zedge, *params), label=f'Fit of {component-design}th order derivative with {nparams} parameters to component b{component}', zorder=10)

            guess=params.copy()

        if ax is not None:
            plt.legend()
        
        return params_list, cov_list 
    
    
    def fit_spline_multipoles(self, components=[1,2], step=50, ax=None, smin=-9999, smax=9999):
        """
        Fit splines to the multipoles in the fieldmap, taking into account the design of the magnet.

        :param components: List of components to fit, counting starts at one like the multipole coefficients.
        :param ax: If given, plot the fit in these axis.
        """        
        
        order = max(components) - 1
        zvals, coeffs, coeffsstd = self.z_multipoles(order, mov_av = 5)

        mask = (zvals>=smin) & (zvals<=smax)
        zvals = zvals[mask]
        coeffs = coeffs[mask]
        coeffsstd = coeffsstd[mask]

        all_pols = []
        ii=np.arange(0,len(zvals)-step,step)
        segments = []

        for component in components:
            pols = []
            index = component-1
            b = coeffs[:, index]
            dz = np.diff(zvals, prepend=zvals[0])
            bp = np.nan_to_num(np.diff(b,prepend=b[0])/dz, copy=True)
            # bpp = np.nan_to_num(np.diff(bp,prepend=bp[0])/dz, copy=True)            

            for ia,ib in zip(ii,ii+step):
                pol=fit_segment(ia,ib,zvals,b,bp)
                if ax is not None:
                    plot_fit(ia,ib,zvals,b,pol, ax=ax, data=True)
                pols.append(pol)
            all_pols.append(pols)

        for ia,ib in zip(ii,ii+step):
            segments.append([zvals[ia], zvals[ib]])
        
        return segments, all_pols
                
                
                
                
            
            
            

                
        
        
        
