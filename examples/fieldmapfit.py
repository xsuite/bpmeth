import bpmeth
import xtrack as xt
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

plt.close("all")

# Function to parse the data from an csv.
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

# Select at which transverse coordinates (x, y) we want to evaluate the field.
xy_point = (0, 0)
subset = df.xs(xy_point, level=['X', 'Y'])

# Extract the transverse fields and the longitudinal axis as numpy arrays.
z_values = subset.index.to_numpy()
bx_values = subset['Bx'].to_numpy()
by_values = subset['By'].to_numpy()

# For integration to check if the field evaluates to zero.
N = z_values.size
ds = z_values[1] - z_values[0]
bx_int = np.trapezoid(bx_values, dx=ds)
by_int = np.trapezoid(by_values, dx=ds)

#print(f"integral B_x ds = {bx_int}")
#print(f"integral B_y ds = {by_int}")

# Also perform the Fourier Transform to extract the modes that are present.
k_values = fftfreq(N, ds)
bx_fft = np.abs(fft(bx_values))
by_fft = np.abs(fft(by_values))
x_freqs = k_values[bx_fft > 92]
y_freqs = k_values[by_fft > 400]

#print(x_freqs)
#print(y_freqs)

def cosinelike(x, *params):
    """
    Combination of sinusoids with known frequencies.
    params should be arranged as [A1, phi1, A2, phi2, ...] for each frequency
    """
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x_freqs)):
        A = params[3 * i]
        phi = params[3 * i + 1]
        freq = params[3 * i + 2]
        y += A * np.cos(2 * np.pi * freq * x + phi)

    return y

def sinelike(x, *params):
    """
    Combination of sinusoids with known frequencies.
    params should be arranged as [A1, phi1, A2, phi2, ...] for each frequency
    """
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(len(y_freqs)):
        A = params[3 * i]
        phi = params[3 * i + 1]
        freq = params[3 * i + 2]
        y += A * np.sin(2 * np.pi * freq * x + phi)

    return y

# From the Fourier Transform, we can get good initial guesses for the frequencies present.
# Tried to include a higher mode for y, but it doesn't fit very well. This is a good first-order approximation.
x_freqs = [0.019, 0.037]
y_freqs = [0.028]

# Initial parameter guesses [A1, phi1, freq1, tanhslope1.1, tanhslope1.2, tanhoffs]
x_initial_guess = [0.4, 0, x_freqs[1], 0.2, 0, x_freqs[0]]
y_initial_guess = [1, 0.75, y_freqs[0]]#, 0.01, 0, 0.0091]

# Fit the curve. The border is to remove the contribution due to edge effects.
# Turns out the x is close to a cos and the y is close to a sin.
border = 117
xpopt, xpcov = curve_fit(cosinelike, z_values[border: 2200 - border], bx_values[border: 2200 - border], p0=x_initial_guess)
ypopt, ypcov = curve_fit(sinelike, z_values[border: 2200 - border], by_values[border: 2200 - border], p0=y_initial_guess)

print(xpopt)
print(ypopt)

# Evaluate the resulting functions with the parameters from the fit.
x_fit = cosinelike(z_values[border: 2200 - border], *xpopt)
y_fit = sinelike(z_values[border: 2200 - border], *ypopt)

# Extract the y-parameters for building the wiggler magnet from B_y at the axis.
y_amp = ypopt[0]
y_phi = ypopt[1]
y_frq = 2 * np.pi * ypopt[2] * 1000

# Extract the x-parameters for building the wiggler magnet from B_x at the axis.
x_amp1 = xpopt[0]
x_amp2 = xpopt[3]
x_phi1 = xpopt[1]
x_phi2 = xpopt[4]
x_frq1 = 2 * np.pi * xpopt[2] * 1000
x_frq2 = 2 * np.pi * xpopt[5] * 1000

# Particle.
p_sp = bpmeth.SympyParticle()
qp0 = [0,0,0,0,0,0]  # Initial conditions x, y, tau, px, py, ptau
p_np = bpmeth.NumpyParticle(qp0)

# Some parameters.
curv=0  # Curvature of the reference frame
length=2

# Write down b1 from the fitted parameters and build the wiggler from there.
b1 = f"{y_amp}*sin({y_frq}*s + {y_phi})"
a1 = f"{x_amp1}*cos({x_frq1}*s + {x_phi1}) + {x_amp2}*cos({x_frq2}*s + {x_phi2})"
wiggler = bpmeth.GeneralVectorPotential(hs=f"{curv}", a=(f"{a1}",) ,b=(f"{b1}",))
# wiggler = bpmeth.FringeVectorPotential(hs=f"{curv}", b1=b1) # Does the same as above

#A_wiggler = wiggler.get_Aval(p_sp)
H_wiggler = bpmeth.Hamiltonian(length, curv, wiggler)

ivp_opt={"rtol":1e-4, "atol":1e-7}
sol_wiggler = H_wiggler.solve(qp0, ivp_opt=ivp_opt)
H_wiggler.plotsol(qp0, ivp_opt=ivp_opt)

p = xt.Particles(x = np.linspace(0, 0, 1), energy0=2.7e9, mass0=xt.ELECTRON_MASS_EV)
sol = H_wiggler.track(p, return_sol=True)
t = sol[0].t
x = sol[0].y[0]
y = sol[0].y[1]

print(f"x = {x}")
print(f"y = {y}")

wiggler.plotfield_xz()
wiggler.plotfield_yz()

fig1, (ax1, ax2) = plt.subplots(2)
ax1.plot(t, x)
ax2.plot(t, y)
ax1.set_title("Particle Trajectory Through Wiggler")
ax2.set_xlabel("Time, $t$, [s]")
ax1.set_ylabel("Deflection, $x$, [m]")
ax2.set_ylabel("Deflection, $y$, [m]")
ax1.legend(["x"], loc="upper right")
ax2.legend(["y"], loc="upper right")
ax1.grid()
ax2.grid()
plt.show()

fig2, (ax3, ax4) = plt.subplots(2)
ax3.plot(z_values, bx_values)
ax3.plot(z_values[border: 2200 - border], x_fit)
ax4.plot(z_values, by_values)
ax4.plot(z_values[border: 2200 - border], y_fit)
ax3.set_title(f"Magnetic Field at (X, Y) = {xy_point}")
ax4.set_xlabel("Longitudinal Position, $s$, [m]")
ax3.set_ylabel("Horizontal Field, $B_x$, [T]")
ax4.set_ylabel("Vertical Field, $B_y$, [T]")
ax3.legend(["$B_x$"], loc="upper right")
ax4.legend(["$B_y$"], loc="upper right")
ax3.grid()
ax4.grid()
plt.show()

fig3, (ax5, ax6) = plt.subplots(2)
ax5.plot(k_values, bx_fft)
ax6.plot(k_values, by_fft)
ax5.set_title(f"Fourier Transform of Magnetic Field at (X, Y) = {xy_point}")
ax6.set_xlabel("Wave Number, $k$, [m⁻¹]")
ax5.set_ylabel("Fourier Transform, $F[B_x]$, [T]")
ax6.set_ylabel("Fourier Transform, $F[B_y]$, [T]")
ax5.legend(["$F[B_x]$"], loc="upper right")
ax6.legend(["$F[B_y]$"], loc="upper right")
ax5.grid()
ax6.grid()
plt.show()