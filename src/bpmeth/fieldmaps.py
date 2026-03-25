import pyvista as pv
import numpy as np
import sympy as sp
import math
import scipy as sc
import matplotlib.pyplot as plt
from .poly_fit import fit_segment, plot_fit


def Enge(x, *params):
    """
    Return the value of the Enge function at the given point, with the specified parameters.
    :param x: Point(s) at which to evaluate the Enge function, can be a scalar or a numpy array.
    :param params: List of parameters for the Enge function, where the first element is the amplitude 
    and the rest are coefficients for the polynomial in the exponent, lowest order first.
    :return: Values of the Enge function at the given point(s).
    """
    
    return params[0] / (1+np.exp(np.poly1d(params[1:])(x)))
    
    
def spEnge(s, *params):
    """
    Return the expression of an Enge function at a given point, symbolicly in s, with the specified parameters.
    :param s: Symbolic variable representing the point at which to evaluate the Enge function.
    :param params: List of parameters for the Enge function, where the first element is the amplitude
    and the rest are coefficients for the polynomial in the exponent, lowest order first. 
    The parameters are compatible with the usual Enge, but here with sympy.
    :return: Expression of the Enge function in terms of the symbolic variable s and the parameters, 
    which can be used for symbolic manipulation and differentiation.
    """
    
    return params[0] / (1+sp.exp(sp.Poly(params[1:], s).as_expr()))


def spTanh(s, *params):
    """
    Return the expression of a typical tanh edge at a given point, symbolicly in s, with the specified parameters.
    :param s: Symbolic variable representing the point at which to evaluate the tanh function.
    :param params: List of parameters for the tanh function, where the first element is the amplitude and 
    the second element is the scale factor for s in the tanh function giving the range of the fringe field.
    :return: Expression of the tanh function in terms of the symbolic variable s and the parameters,
    which can be used for symbolic manipulation and differentiation.
    """
    
    return params[0] * (sp.tanh(s/params[1]) + 1)/2

   
def moving_average(arr, N):
    """ 
    Calculate the moving average of the array with width N.
    :param arr: 1D array of values to average.
    :param N: Width of the moving average window. If N >= 3, a trimmed moving average is calculated by removing 
    the minimum and maximum values in each window before averaging.
    :return: 1D array with moving averages, padded to original length.
    """
    
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
        """
        Initialize the Fieldmap object with the given data, which is expected to be a 2D array where 
        each row corresponds to a point in space and contains the coordinates (x, y, s) and the 
        field components (Bx, By, Bs) at that point.
        The vertical direction is y, the longitudinal direction is s, and the horizontal direction is x. xys is right handed.
        The parameter s is used to generally represent the longitudinal direction, the frame can be straight or bent. 
        When z is used in the following code, the frame explicitly has to be straight.
        """
        
        self.data = data
        self.src = pv.PolyData(self.data.T[:3].T)
        self.src['Bx'] = self.data.T[3].T
        self.src['By'] = self.data.T[4].T
        self.src['Bs'] = self.data.T[5].T
        self.src['x'] = self.data.T[0].T
        self.src['y'] = self.data.T[1].T
        self.src['s'] = self.data.T[2].T
        

    def __add__(self, other):
        """
        Add two fieldmaps together by summing their field components at the same points.
        :param other: Another Fieldmap object to add to this one.
        :return: A new Fieldmap object representing the sum of the two fieldmaps.
        """
        
        # Needs to have fieldmaps defined in the same points, otherwise the user is expected to interpolate the fieldmaps at the desired points.
        if self.src.n_points != other.src.n_points:
            raise ValueError("Fieldmaps have different number of points. Please interpolate them at the same points before adding.")
        if not np.allclose(self.src.points, other.src.points):
            raise ValueError("Fieldmaps have different point coordinates. Please interpolate them at the same points before adding.")

        x = self.src["x"]
        y = self.src["y"]
        s = self.src["s"]
        Bx = self.src["Bx"] + other.src["Bx"]
        By = self.src["By"] + other.src["By"]
        Bs = self.src["Bs"] + other.src["Bs"]
        
        data = np.array([x, y, s, Bx, By, Bs]).T

        return Fieldmap(data)
    
    
    def plot(self, field="By"):
        """ 
        Plot the fieldmap using pyvista, with the specified field component as the scalar field for coloring.
        :param field: The field component to use for coloring the plot, can be "Bx", "By", or "Bs". Default is "By".
        """
        
        self.src.plot(scalars=field)
        
        
    def interpolate_points(self, x, y, s, radius=0.01):
        """
        Interpolate the field at the given points.
        :param x: Array of x coordinates of the points where the field should be interpolated.
        :param y: Array of y coordinates of the points where the field should be interpolated.
        :param s: Array of s coordinates of the points where the field should be interpolated.
        :param radius: Interpolation radius, default 0.01. This parameter is very important for an 
        accurate interpolation, it should be chosen based on the density of the original fieldmap points.
        Uses a Gaussian interpolation kernel, the radius adjusts this kernel. If the radius is too small, 
        the interpolation will be very noisy, if it is too large, the interpolation will be very smooth 
        and may miss important features of the fieldmap.
        :return: A new Fieldmap object containing the interpolated field values at the specified points.
        """
        
        xys = np.array([x.flatten(), y.flatten(), s.flatten()]).T
        dst = pv.PolyData(xys).interpolate(self.src, radius=radius)
        
        data = np.array([x.flatten(), y.flatten(), s.flatten(), dst['Bx'], dst['By'], dst['Bs']]).T

        return Fieldmap(data)


    def FS_frame_plot(self, rho, phi, field, radius=0.01, xmax=0.01, ymax=0.01, nx=51, ny=10, ns=101, smax=1):
        """
        Plot the fieldmap in Frenet-Serret coordinates, but represented in global frame.
        :param rho: Bending radius of the magnet.
        :param phi: Angle of the magnet in radians. Related to the magnetic length by l_magn = rho * phi.
        :param field: Which field component to plot, can be "Bx", "By", or "Bs".
        :param radius: Interpolation radius, default 0.01. See interpolate_points for its function.
        :param xmax: Maximal x value to take into account in the plot, best to exclude any region outside of GFR.
        :param ymax: Maximal y value to take into account in the plot, best to exclude any region outside of GFR.
        :param nx: Number of points in the x direction for the plot.
        :param ny: Number of points in the y direction for the plot.
        :param ns: Number of points in the s direction for the plot.
        :param smax: Maximal s value to take into account in the plot, best to exclude any region outside of GFR. 
        Given as fraction of l_magn
        :return: The fieldmap objects interpolated at FS points.
        """
        
        l_magn = rho*phi

        # ----- Straight part at negative s -----
        s, x, y = np.meshgrid(np.linspace(-l_magn*smax, -l_magn/2, ns//4), 
                              np.linspace(-xmax, xmax, nx), 
                              np.linspace(-ymax, ymax, ny))
        X_ns = (rho + x) * np.cos(phi/2) + (s+l_magn/2) * np.sin(phi/2) 
        Y_ns = y
        Z_ns = -(rho + x) * np.sin(phi/2) + (s+l_magn/2) * np.cos(phi/2)

        # ----- Bent part -----
        s, x, y = np.meshgrid(np.linspace(-l_magn/2, l_magn/2, ns//2), 
                              np.linspace(-xmax, xmax, nx), 
                              np.linspace(-ymax, ymax, ny))
        X_b = np.cos(s/rho) * (rho + x) 
        Y_b = y
        Z_b = np.sin(s/rho) * (rho + x)
        
        # ----- Straight part at positive s -----
        s, x, y = np.meshgrid(np.linspace(l_magn/2, l_magn*smax, ns//4), 
                              np.linspace(-xmax, xmax, nx), 
                              np.linspace(-ymax, ymax, ny))
        X_ps = (rho + x) * np.cos(phi/2) - (s-l_magn/2) * np.sin(phi/2)
        Y_ps = y
        Z_ps = (rho + x) * np.sin(phi/2) + (s-l_magn/2) * np.cos(phi/2)
        
        X = np.concatenate((X_ns.flatten(), X_b.flatten(), X_ps.flatten()))
        Y = np.concatenate((Y_ns.flatten(), Y_b.flatten(), Y_ps.flatten()))
        Z = np.concatenate((Z_ns.flatten(), Z_b.flatten(), Z_ps.flatten()))
        fm = self.interpolate_points(X, Y, Z, radius=radius)
        fm.plot(field)
        return fm

    
    def FS_frame_plot_cilindrical(self, rho, phi, field, radius=0.01, rmax=0.01, ntheta=32, nr=5, ns=101):
        """
        Plot the fieldmap in Frenet-Serret coordinates, but represented in global frame, using cylindrical coordinates.
        :param rho: Bending radius of the magnet.
        :param phi: Angle of the magnet in radians. Related to the magnetic length by l_magn = rho * phi.
        :param field: Which field component to plot, can be "Bx", "By", or "Bs".
        :param radius: Interpolation radius, default 0.01. See interpolate_points for its function.
        :param rmax: Maximal radial distance from the reference trajectory to take into account in the plot, best to exclude any region outside of GFR.
        :param ntheta: Number of points in the angular direction for the plot.
        :param nr: Number of points in the radial direction for the plot.
        :param ns: Number of points in the s direction for the plot.
        :return: The fieldmap objects interpolated at FS points. 
        """
        
        l_magn = rho*phi

        # ----- Straight part at negative s -----
        s, theta, r = np.meshgrid(np.linspace(-l_magn, -l_magn/2, ns//4), 
                                np.linspace(0, 2*np.pi, ntheta+1)[:-1], 
                                np.linspace(0, rmax, nr))
        X_ns = (rho + r*np.cos(theta)) * np.cos(phi/2) + (s+l_magn/2) * np.sin(phi/2) 
        Y_ns = r*np.sin(theta)
        Z_ns = -(rho + r*np.cos(theta)) * np.sin(phi/2) + (s+l_magn/2) * np.cos(phi/2)

        # ----- Bent part -----
        s, theta, r = np.meshgrid(np.linspace(-l_magn/2, l_magn/2, ns//2), 
                                np.linspace(0, 2*np.pi, ntheta), 
                                np.linspace(0, rmax, nr))
        X_b = np.cos(s/rho) * (rho + r*np.cos(theta)) 
        Y_b = r*np.sin(theta)
        Z_b = np.sin(s/rho) * (rho + r*np.cos(theta))
        
        # ----- Straight part at positive s -----
        s, theta, r = np.meshgrid(np.linspace(l_magn/2, l_magn, ns//4),
                                np.linspace(0, 2*np.pi, ntheta), 
                                np.linspace(0, rmax, nr))
        X_ps = (rho + r*np.cos(theta)) * np.cos(phi/2) - (s-l_magn/2) * np.sin(phi/2)
        Y_ps = r*np.sin(theta)
        Z_ps = (rho + r*np.cos(theta)) * np.sin(phi/2) + (s-l_magn/2) * np.cos(phi/2)
        
        X = np.concatenate((X_ns.flatten(), X_b.flatten(), X_ps.flatten()))
        Y = np.concatenate((Y_ns.flatten(), Y_b.flatten(), Y_ps.flatten()))
        Z = np.concatenate((Z_ns.flatten(), Z_b.flatten(), Z_ps.flatten()))
        fm = self.interpolate_points(X, Y, Z, radius=radius)
        fm.plot(field)
        return fm
    

    def calc_FS_coords(self, xFS, yFS, sFS, rho, phi, radius=0.01):
        """
        Determine the fieldmap in a frame consisting of a straight piece up to -l_magn/2, then a bent piece up to l_magn/2, 
        then again a straight piece. Assumes that the original fieldmap is in global frame XYS, 
        with (0,0,0) the center of curvature (and hence typically outside the magnetic fieldmap).
        Y is vertical, the magnet rotated so that a typical magnet is symmetric around S=0 and located 
        at positive X (if the curvature radius is positive), XYS right handed.
        :param xFS: Array of x positions in Frenet-Serret coordinates.
        :param yFS: Array of y positions in Frenet-Serret coordinates.
        :param sFS: Array of s positions in Frenet-Serret coordinates.
        :param rho: Bending radius of the magnet.
        :param phi: Angle of the magnet in radians. Related to the magnetic length by l_magn = rho * phi.
        :param radius: Interpolation radius, default 0.01. See interpolate_points for its function.
        :return: Fieldmap object in Frenet-Serret coordinates.
        """

        l_magn = rho*phi
        
        # ----- Straight part at negative s -----
        xarr = xFS
        yarr = yFS
        sarr = sFS[sFS<-l_magn/2] + l_magn/2
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_ns = x
        y_ns = y
        s_ns = s - l_magn/2

        X_ns = (rho + x) * np.cos(phi/2) + s * np.sin(phi/2)  # Global coordinates
        Y_ns = y
        Z_ns = -(rho + x) * np.sin(phi/2) + s * np.cos(phi/2)

        XYZ = np.array([X_ns.flatten(), Y_ns.flatten(), Z_ns.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_ns = dst["Bx"]*np.cos(phi/2) + dst["Bs"]*np.sin(phi/2)
        By_ns = dst["By"]
        Bs_ns = -dst["Bx"]*np.sin(phi/2) + dst["Bs"]*np.cos(phi/2)

        # ----- Bent part -----
        xarr = xFS
        yarr = yFS
        sarr = sFS[abs(sFS)<l_magn/2]
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_b = x
        y_b = y
        s_b = s

        X_b = np.cos(s/rho) * (rho + x)  # Global coordinates
        Y_b = y
        Z_b = np.sin(s/rho) * (rho + x)

        XYZ = np.array([X_b.flatten(), Y_b.flatten(), Z_b.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_b = dst["Bx"]*np.cos(s.flatten()/rho) - dst["Bs"]*np.sin(s.flatten()/rho)
        By_b = dst["By"]
        Bs_b = dst["Bx"]*np.sin(s.flatten()/rho) + dst["Bs"]*np.cos(s.flatten()/rho)

        # ----- Straight part at positive s -----
        xarr = xFS
        yarr = yFS
        sarr = sFS[sFS>l_magn/2] - l_magn/2
        
        s, x, y = np.meshgrid(sarr, xarr, yarr)
        x_ps = x
        y_ps = y
        s_ps = s + l_magn/2

        X_ps = (rho + x) * np.cos(phi/2) - s * np.sin(phi/2)  # Global coordinates
        Y_ps = y
        Z_ps = (rho + x) * np.sin(phi/2) + s * np.cos(phi/2)

        XYZ = np.array([X_ps.flatten(), Y_ps.flatten(), Z_ps.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)
        Bx_ps = dst["Bx"]*np.cos(phi/2) - dst["Bs"]*np.sin(phi/2)
        By_ps = dst["By"]
        Bs_ps = dst["Bx"]*np.sin(phi/2) + dst["Bs"]*np.cos(phi/2)

        # Combine all
        x = np.concatenate((x_ns.flatten(), x_b.flatten(), x_ps.flatten()))
        y = np.concatenate((y_ns.flatten(), y_b.flatten(), y_ps.flatten()))
        s = np.concatenate((s_ns.flatten(), s_b.flatten(), s_ps.flatten()))

        Bx = np.concatenate((Bx_ns.flatten(), Bx_b.flatten(), Bx_ps.flatten()))
        By = np.concatenate((By_ns.flatten(), By_b.flatten(), By_ps.flatten()))
        Bs = np.concatenate((Bs_ns.flatten(), Bs_b.flatten(), Bs_ps.flatten()))

        data = np.array([x, y, s, Bx, By, Bs]).T

        return Fieldmap(data)
    

    def calc_global_coords(self, rho, phi):
        """
        Transform a fieldmap from Frenet–Serret coordinates back to global coordinates.
        Inverse of calc_FS_coords. Assumes that the original fieldmap is in Frenet–Serret coordinates, 
        with straight frame for s < -l_magn/2, bent frame for abs(s) < l_magn/2, and straight frame for s > l_magn/2.
        :param rho: Bending radius of the magnet.
        :param phi: Deflection angle (in radians).
        :return: Fieldmap object in global coordinates.
        """

        l_magn = rho * phi
    
        x = self.src["x"]
        y = self.src["y"]
        s = self.src["s"]
        Bx = self.src["Bx"]
        By = self.src["By"]
        Bs = self.src["Bs"]

        # Masks per region
        mask_ns = s < -l_magn / 2
        mask_b  = np.abs(s) < l_magn / 2
        mask_ps = s >  l_magn / 2

        # Allocate output
        X, Y, Z = np.zeros_like(x), np.zeros_like(y), np.zeros_like(s)
        Bx_g, By_g, Bz_g = np.zeros_like(Bx), np.zeros_like(By), np.zeros_like(Bs)

        # Negative straight
        s_ns = s[mask_ns] + l_magn/2
        X[mask_ns] = (rho + x[mask_ns]) * np.cos(phi/2) + s_ns * np.sin(phi/2)
        Y[mask_ns] = y[mask_ns]
        Z[mask_ns] = -(rho + x[mask_ns]) * np.sin(phi/2) + s_ns * np.cos(phi/2)
        Bx_g[mask_ns] = Bx[mask_ns]*np.cos(phi/2) - Bs[mask_ns]*np.sin(phi/2)
        By_g[mask_ns] = By[mask_ns]
        Bz_g[mask_ns] = Bx[mask_ns]*np.sin(phi/2) + Bs[mask_ns]*np.cos(phi/2)

        # Bent
        X[mask_b] = np.cos(s[mask_b]/rho) * (rho + x[mask_b])
        Y[mask_b] = y[mask_b]
        Z[mask_b] = np.sin(s[mask_b]/rho) * (rho + x[mask_b])
        Bx_g[mask_b] = Bx[mask_b]*np.cos(s[mask_b]/rho) + Bs[mask_b]*np.sin(s[mask_b]/rho)
        By_g[mask_b] = By[mask_b]
        Bz_g[mask_b] = -Bx[mask_b]*np.sin(s[mask_b]/rho) + Bs[mask_b]*np.cos(s[mask_b]/rho)

        # Positive straight
        s_ps = s[mask_ps] - l_magn/2
        X[mask_ps] = (rho + x[mask_ps]) * np.cos(phi/2) - s_ps * np.sin(phi/2)
        Y[mask_ps] = y[mask_ps]
        Z[mask_ps] = (rho + x[mask_ps]) * np.sin(phi/2) + s_ps * np.cos(phi/2)
        Bx_g[mask_ps] = Bx[mask_ps]*np.cos(phi/2) + Bs[mask_ps]*np.sin(phi/2)
        By_g[mask_ps] = By[mask_ps]
        Bz_g[mask_ps] = -Bx[mask_ps]*np.sin(phi/2) + Bs[mask_ps]*np.cos(phi/2)

        data = np.array([X, Y, Z, Bx_g, By_g, Bs_g]).T
        return Fieldmap(data)


    def calc_edge_frame(self, Xedge, Yedge, Zedge, edge_angle, rho, phi, radius=0.01):
        """
        Straight frame parallel to the edge, including possible edge angle. Needed to determine 
        fringe field shape accurately. Fieldmap should be given in global coordinates.
        :param Xedge: local X coordinates in the edge frame.
        :param Yedge: local Y coordinates in the edge frame.
        :param Zedge: local Z coordinates in the edge frame.
        :param edge_angle: Angle with respect to a sector bend, in radians
        :param rho: Bending radius of the magnet, in radians.
        :param phi: Angle of the magnet. Related to the magnetic length by l_magn = rho * phi.
        :param radius: Interpolation radius, default 0.01.
        :return: Fieldmap object in edge frame.
        """

        l_magn = rho*phi
        x, y, z = np.meshgrid(Xedge, Yedge, Zedge)

        X = rho * np.cos(phi/2) + x * np.cos(phi/2-edge_angle) - z * np.sin(phi/2-edge_angle)
        Y = y
        Z = rho * np.sin(phi/2) + x * np.sin(phi/2-edge_angle) + z * np.cos(phi/2-edge_angle)

        XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        dst = pv.PolyData(XYZ).interpolate(self.src, radius=radius)

        Bx = dst["Bx"] * np.cos(phi/2-edge_angle) - dst["Bs"] * np.sin(phi/2-edge_angle)
        By = dst["By"]
        Bz = dst["Bx"] * np.sin(phi/2-edge_angle) + dst["Bs"] * np.cos(phi/2-edge_angle)

        data = np.array([x.flatten(), y.flatten(), z.flatten(), Bx, By, Bz]).T

        return Fieldmap(data)


    def get_data(self):
        """
        :return: Data of the fieldmap, as np.array([x, y, s, Bx, By, Bs]).T, same shape as needed for 
        the input of a Fieldmap object.
        """
        
        data = np.array([self.src['x'], self.src['y'], self.src['s'],
                         self.src['Bx'], self.src['By'], self.src['Bs']]).T
        return data
    
    
    def rotate(self, theta):
        """
        Rotate the frame around the y-axis. Fieldmap should be given in a straight frame.
        :param theta: Angle to rotate the frame, in radians. Positive values correspond to counterclockwise rotation. 
        :return: New rotated Fieldmap object.
        """
        
        x = self.src['x']
        y = self.src['y']
        z = self.src['s']
        
        Bx = self.src['Bx']
        By = self.src['By']
        Bz = self.src['Bs']
        
        x_rotated = x * np.cos(theta) + z * np.sin(theta)
        z_rotated = -x * np.sin(theta) + z * np.cos(theta)
        
        Bx_rotated = Bx * np.cos(theta) + Bz * np.sin(theta)
        Bz_rotated = -Bx * np.sin(theta) + Bz * np.cos(theta)
        
        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([x_rotated, y, z_rotated, Bx_rotated, By, Bz_rotated]).T

        return Fieldmap(data)
        
        
    def translate(self, dx, dy, ds): 
        """
        Translate the frame by dx, dy, ds in the x, y, s directions respectively. 
        Shifted in the given frame, so if the frame is curved then the shift happens along the curved axes.
        :param dx: Translation in the x direction.
        :param dy: Translation in the y direction.
        :param ds: Translation in the s direction.
        :return: New translated Fieldmap object.
        """
        
        x = self.src['x']
        y = self.src['y']
        s = self.src['s']

        Bx = self.src['Bx']
        By = self.src['By']
        Bs = self.src['Bs']

        x_translated = x + dx 
        y_translated = y + dy 
        s_translated = s + ds

        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([x_translated, y_translated, z_translated, Bx, By, Bs]).T
    
        return Fieldmap(data)


    def symmetrize(self, radius=0.01):
        """
        Symmetrize the fieldmap around s=0, by averaging the field values at (x, y, s) and (x, y, -s).
        This is useful for magnets that are supposed to be symmetric, but the fieldmap has some asymmetry
        due to measurement errors or interpolation artifacts.
        :param radius: Interpolation radius, default 0.01. See interpolate_points for its functionality.
        :return: New symmetrized Fieldmap object.
        """
        
        sarr = np.unique(self.src['s'])
        xarr = np.unique(self.src['x'])
        yarr = np.unique(self.src['y'])

        # Build a new symmetrix s grid
        smax = max(abs(sarr.min()), abs(sarr.max()))
        s_sym = np.linspace(-smax, smax, len(sarr))

        s, x, y = np.meshgrid(s_sym, xarr, yarr)
        xys1 = np.array([x.flatten(), y.flatten(), s.flatten()]).T
        dst1 = pv.PolyData(xys1).interpolate(self.src, radius=radius)

        xys2 = np.array([x.flatten(), y.flatten(), -s.flatten()]).T
        dst2 = pv.PolyData(xys2).interpolate(self.src, radius=radius)

        Bx = (dst1["Bx"] + dst2["Bx"])/2
        By = (dst1["By"] + dst2["By"])/2
        Bs = (dst1["Bs"] - dst2["Bs"])/2

        data = np.array([x.flatten(), y.flatten(), s.flatten(), Bx, By, Bs]).T

        return Fieldmap(data)
    
            
    def mirror(self, left=True):
        """
        Mirror the fieldmap around s=0. Useful when only half of the magnet is given.
        :param left: If True, take left side and mirror, otherwise right side.
        :return: New mirrored Fieldmap object.
        """
        
        s = self.src['s']
        if left:
            mask = s < 0
        else:
            mask = s > 0

        x = self.src['x'][mask]
        y = self.src['y'][mask]
        s = self.src['s'][mask]

        Bx = self.src['Bx'][mask]
        By = self.src['By'][mask]
        Bs = self.src['Bs'][mask]

        xx = np.concatenate((x, x))
        yy = np.concatenate((y, y))
        ss = np.concatenate((s, -s))
        
        Bxx = np.concatenate((Bx, Bx))
        Byy = np.concatenate((By, By))
        Bss = np.concatenate((Bs, -Bs))

        # Create new dataframe to update geometry and not only the coordinates of the points, 
        # needed for interpolation later on, easier than updating it in place.        
        data = np.array([xx, yy, ss, Bxx, Byy, Bss]).T
    
        return Fieldmap(data)
    
    
    def rescale(self, scalefactor):
        """ 
        :param scalefactor: Factor by which to scale the field values.
        :return: New Fieldmap object with rescaled field values.
        """
        
        self.src["Bx"] = self.src["Bx"] * scalefactor
        self.src["By"] = self.src["By"] * scalefactor
        self.src["Bs"] = self.src["Bs"] * scalefactor

        
    def xprofile(self, ypos, spos, field, ax=None, xmax=None):
        """ 
        Return the field values along the horizontal direction at the given vertical and longitudinal positions.
        :param ypos: Vertical position at which to extract the horizontal profile.
        :param spos: Longitudinal position at which to extract the horizontal profile.
        :param field: Which field component to extract, can be "Bx", "By", or "Bs".
        :param ax: If given, plot the horizontal profile on the given matplotlib axis.
        :param xmax: Maximal x value to take into account in the profile, best to exclude any region outside of GFR.
        :return: Tuple of (x, fieldvals), where x is the array of x coordinates and fieldvals is the array of 
        corresponding field values for the specified field component.
        """ 
        
        assert ypos in self.src['y'] and spos in self.src['s'], "These values are not present in the data"
        if xmax is None:
            xmax = np.max(abs(self.src['x']))
        mask = (self.src['y'] == ypos) & (self.src['s'] == spos) & (abs(self.src['x']) <= xmax)
        
        x = self.src['x'][mask]
        fieldvals = self.src[field][mask]
        
        if ax is not None:
            ax.plot(x, fieldvals, label=f"y={ypos}, s={spos}")
            ax.set_xlabel('x')
            ax.set_ylabel(field)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        return x, fieldvals
        
        
    def sprofile(self, xpos, ypos, field, ax=None):
        """
        Return the field values along the longitudinal direction at the given horizontal and vertical positions.
        :param xpos: Horizontal position at which to extract the longitudinal profile.
        :param ypos: Vertical position at which to extract the longitudinal profile.
        :param field: Which field component to extract, can be "Bx", "By", or "Bs".
        :param ax: If given, plot the longitudinal profile on the given matplotlib axis.
        :return: Tuple of (s, fieldvals), where s is the array of s coordinates and fieldvals is the array of
        corresponding field values for the specified field component.
        """
        
        assert xpos in self.src['x'] and ypos in self.src['y'], "These values are not present in the data"
        mask = (self.src['x'] == xpos) & (self.src['y'] == ypos)
        
        s = self.src['s'][mask]
        fieldvals = self.src[field][mask]

        if ax is not None:
            ax.plot(s, fieldvals, label=f"x={xpos}, y={ypos}")
            ax.set_xlabel('s')
            ax.set_ylabel(field)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        return s, fieldvals
        
        
    def fit_xprofile(self, ypos, spos, field, order, ax=None, xmax=None, radius=0.01):
        """
        Fit the field values along the horizontal direction at the given vertical and longitudinal positions 
        with a polynomial of the given order, and return the coefficients of the fit.
        :param ypos: Vertical position at which to fit the horizontal profile.
        :param spos: Longitudinal position at which to fit the horizontal profile.
        :param field: Which vectorfield to fit, typically By for normal multipoles.
        :param order: Maximal order to be deterimed in the fit. Order = 1 must fit b1 only, so a polynomial of degree = order - 1.
        :param ax: If given, plot profile and fit with error region.
        :param xmax: Maximal x value to take into account in fit, best to exclude any region outside of GFR.
        :param radius: Radius for interpolation if the asked position is not present in the data.
        :return: Tuple of (coeffs, coeffsstd), where coeffs is the array of coefficients of the fit, 
        starting with the highest order, and coeffsstd is the array of standard deviations of the coefficients, 
        estimated from fits with different polynomial orders. 
        """
        
        if ypos in self.src['y'] and spos in self.src['s']:
            x, fieldvals = self.xprofile(ypos, spos, field, ax=ax, xmax=xmax)
        else:
            xvals = np.linspace(-xmax, xmax, 51)
            X, Y, S = np.meshgrid(xvals, ypos, zpos)
            fm = self.interpolate_points(X, Y, S, radius=radius)
            x, fieldvals = fm.xprofile(ypos, spos, field, ax=ax, xmax=xmax)

        paramslist = np.array([np.polyfit(x, fieldvals, order-1+j)[j:] for j in range(5)])
        param = np.mean(paramslist, axis=0)
        paramsstd = np.std(paramslist, axis=0)
        # Parameters from fit are given starting by highest order, ex. if order = 2, we have [b2, b1] and we want to return [b1, b2/2]
        coeffslist = [paramslist[:, order-1-i] * math.factorial(i) for i in range(order)]
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


    def s_multipoles(self, order, xmax=None, ax=None, mov_av=1, **kwargs):
        """
        Normal multipoles as a function of s, by fitting the horizontal profile at each s position and taking the coefficients of the fit.
        :param order: Maximal order of the multipoles to be determined. Order = 1 must fit b1 only, so a polynomial of degree = order - 1.
        :param xmax: Maximal x value to take into account in the fit, best to exclude any region outside of GFR.
        :param ax: If given, plot the multipoles as a function of s on the given matplotlib axis, with error bars corresponding to 
        the standard deviation of the coefficients estimated from fits with different polynomial orders.
        :param mov_av: If greater than 1, apply a moving average with the given width to the multipole coefficients as a function of s, to smooth out noise.
        :param **kwargs: Additional parameters to pass to the ax.errorbar function when plotting, such as color or label.
        :return: Tuple of (svals, coeffs, coeffsstd), where svals is the array of s coordinates at which the multipoles were determined.        
        """

        svals = np.unique(self.src['s'])
        
        coeffs = np.zeros((len(svals), order))
        coeffsstd = np.zeros((len(svals), order))
        for i, spos in enumerate(svals):
            coeffs[i], coeffsstd[i] = self.fit_xprofile(0, spos, "By", order, xmax=xmax)
        
        for i in range(order):
            coeffs[:, i] = moving_average(coeffs[:, i], N=mov_av)
            coeffsstd[:, i] = moving_average(coeffsstd[:, i], N=mov_av)            
        
        if ax is not None:
            for i in range(order):
                ax.errorbar(svals, coeffs[:,i], yerr=coeffsstd[:,i], label=f"b{i+1}", **kwargs)
        return svals, coeffs, coeffsstd


    def skew_s_multipoles(self, order, xmax=None, ax=None, mov_av=1, **kwargs):
        """
        Skew multipoles as a function of s, by fitting the horizontal profile at each s position and taking the coefficients of the fit.
        :param order: Maximal order of the multipoles to be determined. Order = 1 must fit a1 only, so a polynomial of degree = order - 1.
        :param xmax: Maximal x value to take into account in the fit, best to exclude any region outside of GFR.
        :param ax: If given, plot the multipoles as a function of s on the given matplotlib axis, with error bars corresponding to
        the standard deviation of the coefficients estimated from fits with different polynomial orders.
        :param mov_av: If greater than 1, apply a moving average with the given width to the multipole coefficients as a function of s, to smooth out noise.
        :param **kwargs: Additional parameters to pass to the ax.errorbar function when plotting, such as color or label.
        :return: Tuple of (svals, coeffs, coeffsstd), where svals is the array of s coordinates at which the multipoles were determined.
        """

        svals = np.unique(self.src['s'])
        coeffs = np.zeros((len(svals), order+1))
        coeffsstd = np.zeros((len(svals), order+1))
        for i, spos in enumerate(svals):
            coeffs[i], coeffsstd[i] = self.fit_xprofile(0, spos, "Bx", order, xmax=xmax)
        
        for i in range(order + 1):
            coeffs[:, i] = moving_average(coeffs[:, i], N=mov_av)
            coeffsstd[:, i] = moving_average(coeffsstd[:, i], N=mov_av)            
        
        if ax is not None:
            if labels is None:
                labels = [f"a{i+1}" for i in range(order+1)]
            for i in range(order + 1):
                color=colors[i] if colors is not None else None
                ax.errorbar(svals, coeffs[:,i], yerr=coeffsstd[:,i], **kwargs)
        return svals, coeffs, coeffsstd


    def integratedfield(self, order, xmax=None, smin=-9999, smax=9999):
        """ 
        Determine integrated normal field components.
        :param order: Maximal order of the multipoles to be determined. Order = 1 must fit a1 only, so a polynomial of degree = order - 1.
        :param xmax: Maximal x value to take into account in the fit, best to exclude any region outside of GFR.
        :param smin: Minimal s value to take into account in the integration region.
        :param smax: Maximal s value to take into account in the integration region.
        :return: numpy array with field integrals.        
        """ 
        
        svals, coeffs, coeffsstd = self.s_multipoles(order, xmax=xmax)
        
        mask = (svals>smin) & (svals<smax)
        
        integrals = np.zeros(order)
        for i in range(order):
            integrals[i] = np.trapezoid(coeffs[:,i][mask], svals[mask])
        return integrals


    def calc_Fint(self, gap, b0):
        """
        Typical fringe field integral that changes the focussing.        
        Assumes that the edge is located at z=0, as would be returned by the function calc_edge_frame.
        :param hgap: The gap of the magnet, used to make K0 dimensionless.
        :param b0: The design field of the body of the magnet. If unknown, one can estimate this as the maximum of b1.
        :return: Fully dimensionless integral, divided by b0**2 and gap.
        """

        svals, coeffs, coeffsstd = self.s_multipoles(2)
        b1 = coeffs[:, 0]

        Fint = 1/gap * np.trapezoid(b1 * (b0 - b1) / b0**2, svals)

        return Fint


    def calc_K0(self, gap, b0, entrance=True):
        """
        Fringe field integral that specifies the closed orbit distortion.
        Assumes that the edge is located at z=0, as would be returned by the function calc_edge_frame.
        Entrance and exit integrals are connected by a minus sign.
        :param hgap: The gap of the magnet, used to make K0 dimensionless.
        :param b0: The design field of the body of the magnet. If unknown, one can estimate this as the maximum of b1.
        :param entrance: Optional parameter specifying if edge is entrance or exit of the magnet.
        :return: Fully dimensionless integral, divided by b0 and gap**2.
        """

        zvals, coeffs, coeffsstd = self.z_multipoles(2)
        b1 = coeffs[:, 0]
        plt.plot(zvals, b1)

        if entrance is False:
            b1 = b1[::-1]

        K0 = 1/gap**2 * np.trapezoid((zvals) * (b0*np.heaviside(zvals, 0.5) - b1) / b0, zvals)

        return K0
               
         
    def fit_spline_multipoles(self, components=[1,2], step=50, ax=None, smin=-9999, smax=9999):
        """
        Fit splines to the multipoles in the fieldmap, taking into account the design of the magnet.
        :param components: List of components to fit, counting starts at one like the multipole coefficients.
        :param ax: If given, plot the fit in these axis.
        """        
        
        order = max(components)
        zvals, coeffs, coeffsstd = self.s_multipoles(order, mov_av = 5)

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