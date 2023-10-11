# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/photon_calibration_in_readout_140531"
title = "Qubit Spectroscopy with Driven Resonator"
xlabel = "Frequency (GHz)"
scale_x = 1e9

ylabel = "Amplitude Scaling (%)"
scale_y = 1e-2

z_label = "Readout Signal Difference in I (a. u.)"
scale_z = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")
dispersive_shift = 745e3
dispersive_shift_err = 9e3

x_data = data.pulse_frequency
y_data = data.resonator_drive_amplitude_scaling
z_data = data.readout__final__I__avg[1] - data.readout__final__I__avg[0]
z_err = (
    data.readout__final__I__avg__error[1] ** 2
    + data.readout__final__I__avg__error[0] ** 2
)
z_err = np.sqrt(z_err)


# Fitting Loop
from iminuit import Minuit
from iminuit.cost import LeastSquares

fit_name = "Lorentzian"


def fit_func(x, f0, A, gamma, offset):
    return offset + A * gamma**2 / (gamma**2 + (x - f0) ** 2)


guesses = {"f0": 5.981e9, "A": 1e-3, "gamma": 0.01e9, "offset": 1e-4}


frequence_at_amplitude = np.zeros(len(y_data))
frequence_at_amplitude_err = np.zeros(len(y_data))
mask = np.ones(len(y_data), dtype=bool)

for i in range(len(y_data)):
    ls = LeastSquares(
        x_data.values.flatten(),
        z_data[:, i].values.flatten(),
        z_err[:, i].values.flatten(),
        model=fit_func,
    )
    if i == 0:
        minimizer = Minuit(ls, **guesses)
    else:
        minimizer = Minuit(ls, **minimizer.values.to_dict())
    minimizer.migrad()

    if minimizer.valid:
        frequence_at_amplitude[i] = minimizer.values["f0"]
        frequence_at_amplitude_err[i] = minimizer.errors["f0"]
    else:
        mask[i] = False

mask[frequence_at_amplitude_err < 1e3] = False

# Fit second order polynomial
func = lambda x, a, b, c: a * x**2 + b * x + c
ls = LeastSquares(
    y_data.values.flatten()[mask],
    frequence_at_amplitude[mask],
    frequence_at_amplitude_err[mask],
    model=func,
)
minimizer = Minuit(ls, a=-1e2, b=0, c=5.98e9)
minimizer.migrad()

# Setup the Image Plot
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("cmap", ["black", "C0", "C1", "white"])

fig, ax = plt.subplots()

img = ax.imshow(
    z_data / scale_z,
    aspect="auto",
    origin="lower",
    cmap=cmap,
    extent=[
        y_data.min() / scale_y,
        y_data.max() / scale_y,
        x_data.min() / scale_x,
        x_data.max() / scale_x,
    ],
)

ax.set_xlabel(ylabel)
ax.set_ylabel(xlabel)
ax.set_title(title)

cbar = fig.colorbar(img, ax=ax, label=z_label, pad=0.01)
cbar.ax.yaxis.set_label_position("right")

# cbar.ax.yaxis.tick_left()
cbar.ax.yaxis.label.set_rotation(-90)

# Plot the fitted frequencies
ax.plot(y_data / scale_y, frequence_at_amplitude / scale_x, "x", color="black")
ax.errorbar(
    y_data / scale_y,
    frequence_at_amplitude / scale_x,
    yerr=frequence_at_amplitude_err / scale_x,
    ls="none",
    color="black",
    capsize=2,
    elinewidth=1,
)
fig.tight_layout()

# Plot the fitted frequencies
ax.plot(
    y_data / scale_y,
    func(y_data, *minimizer.values) / scale_x,
    "-",
    color="black",
    label="Fit",
    alpha=0.75,
)

fig.savefig(f"../Figures/{title}.pdf")


# Figure with conversion from amplitude to photon number
fig, ax = plt.subplots()

distance_to_zero = frequence_at_amplitude - minimizer.values["c"]
distance_to_zero_err = np.sqrt(
    frequence_at_amplitude_err**2 + minimizer.errors["c"] ** 2
)


photons = -distance_to_zero / dispersive_shift
photons_err = (
    np.sqrt(
        (distance_to_zero_err / distance_to_zero) ** 2
        + (dispersive_shift_err / dispersive_shift) ** 2
    )
    * photons
)

ax.plot(y_data / scale_y, photons, "o")
ax.errorbar(
    y_data / scale_y,
    photons,
    yerr=photons_err,
    ls="none",
    color="black",
    capsize=2,
    elinewidth=1,
)

# Plot a scaled fit
photons_from_fit = (
    -(func(y_data, *minimizer.values) - minimizer.values["c"]) / dispersive_shift
)
ax.plot(y_data / scale_y, photons_from_fit, "-", color="black", alpha=0.75)

ax.set(
    title="Photon Number vs. Amplitude Scaling",
    xlabel=ylabel,
    ylabel="Photon Number",
)

value = func(1.0, *minimizer.values) - minimizer.values["c"]
photons_at_100_percent = -value / dispersive_shift

value_err = np.sqrt(
    minimizer.errors["a"] ** 2 + minimizer.errors["b"] ** 2 + minimizer.errors["c"] ** 2
)
photon_at_100_percent_err = photons_at_100_percent * np.sqrt(
    value_err**2 / value**2 + dispersive_shift_err**2 / dispersive_shift**2
)

print(
    f"Photons at 100%: {photons_at_100_percent:.2f} +- {photon_at_100_percent_err:.2f}"
)


ax.vlines(
    1.0e2,
    0,
    photons_at_100_percent,
    color="black",
    linestyle="dashed",
)

ax.hlines(
    photons_at_100_percent,
    0,
    1.0e2,
    color="black",
    linestyle="dashed",
)


ax.text(
    50,
    photons_at_100_percent + 1,
    f"Photon Number: ${int(np.round(photons_at_100_percent, 0))} \pm {int(np.round(photon_at_100_percent_err, 0))}$",
    va="bottom",
    ha="center",
)

fig.tight_layout()
fig.savefig(f"../Figures/photon_number.pdf")
