import numpy.polynomial
import scipy.special

import bpmeth
import xtrack as xt
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from bpmeth import poly_fit

plt.close("all")

########################################################################################################################
# IMPORTING AND PREPARING THE DATA
########################################################################################################################
# Function to parse the data from a csv.
def parse_to_dataframe(file_path):
    """Parse directly into a multi-indexed DataFrame."""
    # Read the entire file into a DataFrame
    df = pd.read_csv(file_path, sep=r'\s+', header=None,
                     names=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'])

    # Set multi-index
    df.set_index(['X', 'Y', 'Z'], inplace=True)

    return df

# Parse the data from knot_map_text.txt
file_path = 'example_data/knot_map_test.txt'
df = parse_to_dataframe(file_path)

# Select at which transverse coordinates (x, y) we want to evaluate the field (usually (0, 0)).
# The subsetz indicates that it takes the z-axis as the independent coordinate.
xy_point = (0, 0)
subsetz = df.xs(xy_point, level=['X', 'Y'])

# Extract the transverse fields and the longitudinal axis as numpy arrays.
# The B_z is on the order of 10⁻⁷, compared to B_x and B_y so can be neglected.
z_values = subsetz.index.to_numpy()
z_values = z_values * 0.001             # Convert to meters, more stable for the polynomials.
bx_values = subsetz['Bx'].to_numpy()
by_values = subsetz['By'].to_numpy()
#bz_values = subsetz['Bz'].to_numpy()
bt_values = np.sqrt(bx_values**2 + by_values**2)


########################################################################################################################
# DETERMINING BORDERS
########################################################################################################################

# Finds the peaks in B_x and B_y
bx_peaks   = find_peaks(bx_values)
by_peaks   = find_peaks(by_values)
bx_valleys = find_peaks(-bx_values)
by_valleys = find_peaks(-by_values)

# Reassign to only include our regions of interest.
# The find_peaks also picks up peaks in the flat regions, so this is to exclude those.
# The 99's are to actually include one peak which is exactly at 100.
bx_peaks   = bx_peaks[0][np.logical_and(bx_peaks[0] > 99, bx_peaks[0] < 2100)]
by_peaks   = by_peaks[0][np.logical_and(by_peaks[0] > 99, by_peaks[0] < 2100)]
bx_valleys = bx_valleys[0][np.logical_and(bx_valleys[0] > 99, bx_valleys[0] < 2100)]
by_valleys = by_valleys[0][np.logical_and(by_valleys[0] > 99, by_valleys[0] < 2100)]

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_x.
# The region between -1100 and xborder1 goes up to the first valley and will be fitted with a polynomial.
# The region between xborder1 and xborder2 goes form the first valley to the first peak and will be fitted with another polynomial.
# The region between xborder2 and xborder3 is the sinusoidal area.
# The region between xborder3 and xborder4 goes from the last peak to the last valley and will be fitted with a polynomial.
# The region between xborder4 and 1100 goes from the last valley to the end and will be fitted with a polynomial.
xborder1 = bx_valleys[0]    # z = -992
xborder2 = bx_peaks[0]      # z = -974
xborder3 = bx_peaks[-1]     # z =  970
xborder4 = bx_valleys[-1]   # z =  990

# Splits the magnetic field into five regions. The regions are decided based on the peaks and valleys of B_y.
# The region between -1100 and yborder1 goes up to the first valley and will be fitted with a polynomial.
# The region between yborder1 and yborder2 goes form the first valley to the first peak and will be fitted with another polynomial.
# The region between yborder2 and yborder3 is the sinusoidal area.
# The region between yborder3 and yborder4 goes from the last peak to the last valley and will be fitted with a polynomial.
# The region between yborder4 and 1100 goes from the last valley to the end and will be fitted with a polynomial.
yborder1 = by_peaks[0]      # z = -1000
yborder2 = by_valleys[0]    # z = -984
yborder3 = by_valleys[-1]   # z =  961
yborder4 = by_peaks[-1]     # z =  978

# Assign the z-arrays for each region.
zx_region1 = z_values[:xborder1].copy()
zx_region2 = z_values[xborder1:xborder2].copy()
zx_region3 = z_values[xborder2:xborder3].copy()
zx_region4 = z_values[xborder3:xborder4].copy()
zx_region5 = z_values[xborder4:].copy()

# Same for y.
zy_region1 = z_values[:yborder1].copy()
zy_region2 = z_values[yborder1:yborder2].copy()
zy_region3 = z_values[yborder2:yborder3].copy()
zy_region4 = z_values[yborder3:yborder4].copy()
zy_region5 = z_values[yborder4:].copy()

# And the B_x arrays
bx_region1 = bx_values[:xborder1].copy()
bx_region2 = bx_values[xborder1:xborder2].copy()
bx_region3 = bx_values[xborder2:xborder3].copy()
bx_region4 = bx_values[xborder3:xborder4].copy()
bx_region5 = bx_values[xborder4:].copy()

# And for B_y
by_region1 = by_values[:yborder1].copy()
by_region2 = by_values[yborder1:yborder2].copy()
by_region3 = by_values[yborder2:yborder3].copy()
by_region4 = by_values[yborder3:yborder4].copy()
by_region5 = by_values[yborder4:].copy()

########################################################################################################################
# SINUSOID FITTING
########################################################################################################################

# Define the function to be fitted, a sinusoid superimposed on an enge function.
def sinusoid(x, *params):
    # Define the output array
    y = np.zeros_like(x, dtype=np.float64)

    # The range depends on the number of sinusoids there are present in the function.
    # B_x is best approximated with two, whereas B_y only needs one.
    # Note the integer division by 3, because each mode has three parameters.
    for i in range(len(params)//3):
        # Parameters are:
        # params[3i] = A1, the amplitude of the cosine
        # params[3i+1] = A2, the amplitude of the sine
        # params[3i+2] = A3, the frequency
        A1 = params[3 * i]
        A2 = params[3 * i + 1]
        freq = params[3 * i + 2]

        # General sinusoidal part is a linear combination of cosine and sine.
        # This is equivalent to a single function with a phase-offset, but more numerically stable.
        y += A1 * np.cos(2 * np.pi * freq * x) + A2 * np.sin(2 * np.pi * freq * x)

    return y

# These frequencies have been found before using a Fourier Transform.
# As stated before, B_x contains two modes and B_y only one.
x_freqs = [19, 37]#[0.019, 0.037]
y_freqs = [28]#[0.028]

# Initial parameter guesses [A1.1, A1.2, freq1, A2.1, A2.2, freq2]
x_initial_guess = np.array([0.29, 0.031, x_freqs[0], 0.083, -0.1, x_freqs[1]])
y_initial_guess = np.array([0.3, 0.8, y_freqs[0]])

# Fit the curve.
xpoptreg3, xpcovreg3 = curve_fit(sinusoid, zx_region3, bx_region3, p0=x_initial_guess)
ypoptreg3, ypcovreg3 = curve_fit(sinusoid, zy_region3, by_region3, p0=y_initial_guess)

print(xpoptreg3)
print(ypoptreg3)

# Calculates the output of the fit.
xfit_reg3 = sinusoid(zx_region3, *xpoptreg3)
yfit_reg3 = sinusoid(zy_region3, *ypoptreg3)
#bt_re3_fit = np.sqrt(bx_re3_fit ** 2 + by_re3_fit ** 2)

# Find the pole length from k_y, the wavenumber of B_y
# The frequency already has a 2pi factored out, so wavelength = 1/freq.
# The pole length is 1/4 of the wavelength.
# This pole length includes any spacing between the poles.
# Returns 9 mm.
polelength = 1 / 4 / ypoptreg3[2]
print(f"Pole length [mm] = {polelength}")

########################################################################################################################
# EDGE FITTING
########################################################################################################################


def fit_polynomial_matching(degree, z_region, b_region, b_match_region, left):
    """
    Fits a polynomial to any region that matches the middle sinusoidal region.

    Args:
        z_region (np.array): Array of z-coordinates for the region of interest (should be either region2 or region4).
        b_region (np.array): Array of B-values for the region of interest.
        sinparams (dict): Dictionary of boundary parameters.

    Returns:
        list: Polynomial coefficients for a polynomial of order N.
    """

    # Derivatives of region
    # First derivative (centered difference)
    db_dz_region = np.gradient(b_region, 0.001, edge_order=2)
    # Second derivative (applying gradient twice)
    d2b_dz2_region = np.gradient(db_dz_region, 0.001, edge_order=2)

    # Derivatives of match region
    # First derivative (centered difference)
    db_dz_match = np.gradient(b_match_region, 0.001, edge_order=2)
    # Second derivative (applying gradient twice)
    d2b_dz2_match = np.gradient(db_dz_match, 0.001, edge_order=2)

    if degree >= 5:
        # Fit polynomial with boundary conditions
        if left:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], b_match_region[0]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_region[0], db_dz_match[0]],
                xpp0=[z_region[0], z_region[-1]],
                ypp0=[d2b_dz2_region[0], d2b_dz2_match[0]]
            )

        else:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_match_region[-1], b_region[-1]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_match[-1], db_dz_region[-1]],
                xpp0=[z_region[0], z_region[-1]],
                ypp0=[d2b_dz2_match[-1], d2b_dz2_region[-1]]
            )
    elif degree >=3:
        # Fit polynomial with boundary conditions
        if left:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], b_match_region[0]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_region[0], db_dz_match[0]]
            )

        else:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_match_region[-1], b_region[-1]],
                xp0=[z_region[0], z_region[-1]],
                yp0=[db_dz_match[-1], db_dz_region[-1]]
            )

    elif degree >=1:
        # Fit polynomial with boundary conditions
        if left:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_region[0], b_match_region[0]]
            )

        else:
            bpol_reg = poly_fit.poly_fit(
                N=degree,
                xdata=z_region,
                ydata=b_region,
                x0=[z_region[0], z_region[-1]],
                y0=[b_match_region[-1], b_region[-1]]
            )


    return bpol_reg

degree = 5
# Polynomial fits for regions 2 and 4
# B_x, region 2
xpoptreg2 = fit_polynomial_matching(degree, zx_region2, bx_region2, xfit_reg3, left=True)
xpoly2 = numpy.polynomial.Polynomial(xpoptreg2)
xfit_reg2 = xpoly2(zx_region2)

# B_x, region 4
xpoptreg4 = fit_polynomial_matching(degree, zx_region4, bx_region4, xfit_reg3, left=False)
xpoly4 = numpy.polynomial.Polynomial(xpoptreg4)
xfit_reg4 = xpoly4(zx_region4)

# B_y, region 2
ypoptreg2 = fit_polynomial_matching(degree, zy_region2, by_region2, yfit_reg3, left=True)
ypoly2 = numpy.polynomial.Polynomial(ypoptreg2)
yfit_reg2 = ypoly2(zy_region2)

# B_y, region 4
ypoptreg4 = fit_polynomial_matching(degree, zy_region4, by_region4, yfit_reg3, left=False)
ypoly4 = numpy.polynomial.Polynomial(ypoptreg4)
yfit_reg4 = ypoly4(zy_region4)

degree = 2
# Polynomial fits for regions 1 and 5
# B_x, region 1
xpoptreg1 = fit_polynomial_matching(degree, zx_region1, bx_region1, xfit_reg2, left=True)
xpoly1 = numpy.polynomial.Polynomial(xpoptreg1)
xfit_reg1 = xpoly1(zx_region1)

# B_x, region 5
xpoptreg5 = fit_polynomial_matching(degree, zx_region5, bx_region5, xfit_reg4, left=False)
xpoly5 = numpy.polynomial.Polynomial(xpoptreg5)
xfit_reg5 = xpoly5(zx_region5)

# B_y, region 1
ypoptreg1 = fit_polynomial_matching(degree, zy_region1, by_region1, yfit_reg2, left=True)
ypoly1 = numpy.polynomial.Polynomial(ypoptreg1)
yfit_reg1 = ypoly1(zy_region1)

# B_y, region 5
ypoptreg5 = fit_polynomial_matching(degree, zy_region5, by_region5, yfit_reg4, left=False)
ypoly5 = numpy.polynomial.Polynomial(ypoptreg5)
yfit_reg5 = ypoly5(zy_region5)

########################################################################################################################
# MERGE IT ALL TOGETHER
########################################################################################################################

bx_fit = np.concatenate((xfit_reg1, xfit_reg2, xfit_reg3, xfit_reg4, xfit_reg5))
by_fit = np.concatenate((yfit_reg1, yfit_reg2, yfit_reg3, yfit_reg4, yfit_reg5))
bt_fit = np.sqrt(bx_fit**2 + by_fit**2)


########################################################################################################################
# PLOTS
########################################################################################################################
fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(z_values, bx_values)
ax1.plot(z_values, bx_fit)
#ax1.plot(zx_region1, xfit_reg1)
#ax1.plot(zx_region2, xfit_reg2)
#ax1.plot(zx_region3, xfit_reg3)
#ax1.plot(zx_region4, xfit_reg4)
#ax1.plot(zx_region5, xfit_reg5)
ax2.plot(z_values, by_values)
ax2.plot(z_values, by_fit)
#ax2.plot(zy_region1, yfit_reg1)
#ax2.plot(zy_region2, yfit_reg2)
#ax2.plot(zy_region3, yfit_reg3)
#ax2.plot(zy_region4, yfit_reg4)
#ax2.plot(zy_region5, yfit_reg5)
ax3.plot(z_values, bt_values)
ax3.plot(z_values, bt_fit)

ax1.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax3.set_xlabel("Longitudinal Position, $s$, [m]")
ax1.set_ylabel("Horizontal Field, $B_x$, [T]")
ax2.set_ylabel("Vertical Field, $B_y$, [T]")
ax3.set_ylabel("Magnitude, $|B|$, [T]")

ax1.legend(["$B_x$ Data", "$B_x$ Fit"], loc="upper right")
ax2.legend(["$B_y$ Data", "$B_y$ Fit"], loc="upper right")
ax3.legend(["$|B|$ Data", "$|B|$ Fit"], loc="upper right")

ax1.grid()
ax2.grid()
ax3.grid()
plt.show()