import pandas
import gzip
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import scipy as sc
import math
import numpy as np


def Enge(x, *params):
    return 1 / (1+np.exp(np.poly1d(params)(x)))

def Tanh(x, *params):
    return (np.tanh(-x/params[0])+1)/2

class Magnet:
    def __init__(self, data, localpos, nominalfield):
        self.data = data
        self.localpos = localpos
        self.src = pv.PolyData(self.data.T[:3].T)

        # Field in global coordinates
        self.src['Bx'] = self.data.T[7].T
        self.src['By'] = self.data.T[8].T
        self.src['Bz'] = self.data.T[9].T

        # Local coordinate system around the edge of the magnet
        self.src['xloc'] = self.localpos.T[0].T
        self.src['yloc'] = self.localpos.T[1].T
        self.src['sloc'] = self.localpos.T[2].T

        # Field in local coordinates
        self.src['Bxloc'] = self.src['Bx']*np.cos(13*np.pi/180) - self.src['Bz']*np.sin(13*np.pi/180) 
        self.src['Byloc'] = self.src['By']
        self.src['Bzloc'] = self.src['Bx']*np.sin(13*np.pi/180) + self.src['Bz']*np.cos(13*np.pi/180) 

        # Normalised by the field in the center of the magnet
        self.src['Bxnorm'] = self.src['Bxloc'] / nominalfield
        self.src['Bynorm'] = self.src['Byloc'] / nominalfield
        self.src['Bznorm'] = self.src['Bzloc'] / nominalfield

        

    def plot(self, field):
        """
        Plot the field of the gfield of the given dataset.

        :param field: (string)
            Specifies which field component to plot.
            Options:
                "Bx", "By", "Bz" in lab frame, By is vertical field.
        """

        self.src.plot(scalars=field)

    def plot_x(self, yval, sval, figure=None):
        """
        Plot a cross section in the x direction at a given y and s value

        :param figure: (fig, ax) if given 
        """

        if figure is None:
            fig, ax = plt.subplots()
        else: 
            fig, ax = figure

        ymask = self.src['yloc'] == yval
        smask = self.src['sloc'] == sval
        mask = ymask & smask

        ax.scatter(self.src['xloc'][mask], self.src['Bxloc'][mask], color='blue', label='Bx') 
        ax.scatter(self.src['xloc'][mask], self.src['Byloc'][mask], color='red', label='By') 
        ax.scatter(self.src['xloc'][mask], self.src['Bzloc'][mask], color='green', label='Bz')
        ax.set_xlabel('x')
        plt.legend()
        
        return (fig, ax)


    def plot_y(self, xval, sval, figure=None):
        """
        Plot a cross section in the y direction at a given x and s value

        :param figure: (fig, ax) if given 
        """

        if figure is None:
            fig, ax = plt.subplots()
        else: 
            fig, ax = figure

        xmask = self.src['xloc'] == xval
        smask = self.src['sloc'] == sval
        mask = xmask & smask

        ax.scatter(self.src['yloc'][mask], self.src['Bxloc'][mask], color='blue', label='Bx')
        ax.scatter(self.src['yloc'][mask], self.src['Byloc'][mask], color='red', label='By')
        ax.scatter(self.src['yloc'][mask], self.src['Bzloc'][mask], color='green', label='Bs')
        ax.set_xlabel('y')
        plt.legend()
        
        return (fig, ax)

    
    def plot_s(self, xval, yval, figure=None):
        """
        Plot a cross section in the s direction at a given x and y value

        :param figure: (fig, ax) if given 
        """

        if figure is None:
            fig, ax = plt.subplots()
        else: 
            fig, ax = figure

        xmask = self.src['xloc'] == xval
        ymask = self.src['yloc'] == yval
        mask = xmask & ymask

        ax.scatter(self.src['sloc'][mask], self.src['Bxloc'][mask], color='blue', label='Bx')
        ax.scatter(self.src['sloc'][mask], self.src['Byloc'][mask], color='red', label='By')
        ax.scatter(self.src['sloc'][mask], self.src['Bzloc'][mask], color='green', label='Bz')
        ax.set_xlabel('s')
        plt.legend()
        
        return (fig, ax)

        
    def fit_b1_poly(self, degree=5, ax=None, figname=None):
        """
        b1 is the zeroth order contribution to By, so fit at x=0, y=0.
        The fit is a polynomial

        :param degree: order of the polynomial fit.
        """

        xmask = self.src['xloc'] == 0
        ymask = self.src['yloc'] == 0
        mask = xmask & ymask

        if ax is None:
            fig, ax = plt.subplots()

        smin = np.min(self.src['sloc'][mask])
        smax = np.max(self.src['sloc'][mask])
        sarr = np.linspace(smin, smax, 100)

        params = np.polyfit(self.src['sloc'][mask], self.src['Bynorm'][mask], deg=degree)
        polyn = np.poly1d(params)
        ax.plot(sarr, polyn(sarr), label=f'Polynomial fit of degree {degree}', color="green")

        #ax.scatter(self.src['sloc'][mask], self.src['Bynorm'][mask], color='gray', label='By/B0')
        ax.set_xlabel('s')
        ax.set_ylabel('By/B0')
        plt.legend()
        
        if figname is not None:
            plt.savefig(figname)

        with open('redpoints.txt', 'w') as file:
            file.write(f'"s","By"\n')
            np.savetxt(file, np.c_[self.src['sloc'][mask], self.src['Bynorm'][mask]], delimiter=',', fmt='%.10f')


    def fit_b1_enge(self, degree=5, ax=None, figname=None):
        """
        b1 is the zeroth order contribution to By, so fit at x=0, y=0.
        The fit is an Enge-function, a function defined as 1/(1+exp(c0 + c1 s + c2 s^2 + ...))

        :param degree: order of the polynomial fit.
        """

        xmask = self.src['xloc'] == 0
        ymask = self.src['yloc'] == 0
        mask = xmask & ymask
        
        if ax is None:
            fig, ax = plt.subplots()

        smin = np.min(self.src['sloc'][mask])
        smax = np.max(self.src['sloc'][mask])
        sarr = np.linspace(smin, 20*smax, 500)

        svals = np.array(self.src['sloc'][mask])
        fieldvals = np.array(self.src['Bynorm'][mask])
        svals = np.append(svals, [1.5*smax, 2*smax, 2.5*smax, 3*smax, 3.5*smax, 4*smax, 5*smax, 6*smax, 10*smax, 20*smax])
        fieldvals = np.append(fieldvals, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        params, cov = sc.optimize.curve_fit(Enge, svals, fieldvals, p0=np.ones(degree))
        ax.plot(sarr, Enge(sarr, *params), label=f'Enge function fit of degree {degree}', color="blue")

        ax.scatter(svals, fieldvals, color='gray', label='By/B0')
        ax.set_xlabel('s')
        ax.set_ylabel('By/B0')
        ax.set_xlim(smin, smax)
        plt.legend()
        
        if figname is not None:
            plt.savefig(figname)
        
        return params

    def fit_tanh(self, ax=None):
        """ 
        Fit a tanh function to the By field in the s direction.
        """
        
        xmask = self.src['xloc'] == 0
        ymask = self.src['yloc'] == 0
        mask = xmask & ymask

        if ax is None:
            fig, ax = plt.subplots()

        smin = np.min(self.src['sloc'][mask])
        smax = np.max(self.src['sloc'][mask])
        sarr = np.linspace(smin, smax, 100)

        svals = np.array(self.src['sloc'][mask])
        fieldvals = np.array(self.src['Bynorm'][mask])
        svals = np.append(svals, [2*smax, 3*smax])
        fieldvals = np.append(fieldvals, [0, 0])
        params, cov = sc.optimize.curve_fit(Tanh, svals, fieldvals, p0=[1])
        ax.plot(sarr, Tanh(sarr, *params), label=f'Tanh fit', color="red")

        #ax.scatter(self.src['sloc'][mask], self.src['Bynorm'][mask], color='gray', label='By/B0')
        ax.set_xlabel('s')
        ax.set_ylabel('By/B0')
        plt.legend()
        
        return params
