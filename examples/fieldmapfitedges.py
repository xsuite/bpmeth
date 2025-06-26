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
x_freqs = [0.019, 0.037]
y_freqs = [0.028]

# Initial parameter guesses [A1.1, A1.2, freq1, A2.1, A2.2, freq2, engepoly1, engepoly2, ...]
# engepoly1, engepoly2 etc. are the coefficients of the polynomial in the Enge function
x_initial_guess = np.array([0.63, -0.78, x_freqs[1], 1, 0, x_freqs[0]])
y_initial_guess = np.array([0.92, -0.38, y_freqs[0]])

# Fit the curve.
xpoptmid, xpcovmid = curve_fit(sinusoid, zx_region3, bx_region3, p0=x_initial_guess)
ypoptmid, ypcovmid = curve_fit(sinusoid, zy_region3, by_region3, p0=y_initial_guess)

print(xpoptmid)
print(ypoptmid)

# Calculates the output of the fit.
bx_re3_fit = sinusoid(zx_region3, *xpoptmid)
by_re3_fit = sinusoid(zy_region3, *ypoptmid)
#bt_re3_fit = np.sqrt(bx_re3_fit ** 2 + by_re3_fit ** 2)

# Find the pole length from k_y, the wavenumber of B_y
# The frequency already has a 2pi factored out, so wavelength = 1/freq.
# The pole length is 1/4 of the wavelength.
# This pole length includes any spacing between the poles.
# Returns 9 mm.
polelength = 1 / 4 / ypoptmid[2]
print(f"Pole length [mm] = {polelength}")

########################################################################################################################
# EDGE FITTING
########################################################################################################################

# Write a sympy script to evaluate the sinusoidal part at the edges of region 3.
# Define symbols.
z, A1, A2, ki = sp.symbols('z A1 A2 omega')

# Define the function b_i(x)
Bi = A1 * sp.cos(ki * z) + A2 * sp.sin(ki * z)

# Compute derivatives
Bi_prime = sp.diff(Bi, z)      # First derivative
Bi_double_prime = sp.diff(Bi_prime, z)  # Second derivative

# Points where we enforce matching conditions
zmatch = sp.symbols('zmatch')

# Evaluate Bi and its derivatives at z1.
Bi_zmatch = Bi.subs(z, zmatch)
Bi_prime_zmatch = Bi_prime.subs(z, zmatch)
Bi_double_prime_zmatch = Bi_double_prime.subs(z, zmatch)

# Convert symbolic expressions to numerical functions (for evaluation)
Bi_num = sp.lambdify((z, A1, A2, ki), Bi, 'numpy')
Bi_prime_num = sp.lambdify((z, A1, A2, ki), Bi_prime, 'numpy')
Bi_double_prime_num = sp.lambdify((z, A1, A2, ki), Bi_double_prime, 'numpy')

# Evaluate f, f', f'' at x1 and x2
Bi0 = Bi_num(zx_region2[-1], xpoptmid[0], xpoptmid[1], xpoptmid[2])
Bip0 = Bi_prime_num(zx_region2[-1], xpoptmid[0], xpoptmid[1], xpoptmid[2])
Bipp0 = Bi_double_prime_num(zx_region2[-1], xpoptmid[0], xpoptmid[1], xpoptmid[2])

xpol_reg2 = poly_fit.poly_fit(
    N=6,
    xdata=zx_region2,
    ydata=bx_region2,
    x0=[zx_region2[0], zx_region2[-1]],
    y0=[bx_region2[0], Bi0],
    xp0=[zx_region2[0], zx_region2[-1]],
    yp0=[0, Bip0]
)

xfit_reg2 = xpol_reg2[0] * zx_region2**2 + xpol_reg2[1] * zx_region2 + xpol_reg2[2]


########################################################################################################################
# PLOTS
########################################################################################################################
fig1, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(z_values, bx_values)
ax1.plot(zx_region2, xfit_reg2)
ax1.plot(zx_region3, bx_re3_fit)
ax2.plot(z_values, by_values)
ax2.plot(zy_region3, by_re3_fit)
ax3.plot(z_values, bt_values)
#ax3.plot(zx_region3, bt_re3_fit)

ax1.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax3.set_xlabel("Longitudinal Position, $s$, [m]")
ax1.set_ylabel("Horizontal Field, $B_x$, [T]")
ax2.set_ylabel("Vertical Field, $B_y$, [T]")
ax3.set_ylabel("Magnitude, $|B|$, [T]")

ax1.legend(["$B_x$"], loc="upper right")
ax2.legend(["$B_y$"], loc="upper right")
ax3.legend(["$|B|$"], loc="upper right")

ax1.grid()
ax2.grid()
ax3.grid()
plt.show()