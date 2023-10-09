# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.figsize"] = (18, 6)

data_folder = "../Data/dephasing_by_measurement"
title = "Qubit Spectroscopy with Driven Resonator"
xlabel = "Readout Amplitude (mV)"
scale_x = 1e-3

ylabel = "Rotation Angle (rad)"
scale_y = 1

z_label = "Readout Signal I (a. u.)"
scale_z = 1e-6

data = xr.open_dataset(data_folder + "/dataset.nc")
dispersive_shift = 745e3
dispersive_shift_err = 9e3

x_data = data.resonator_drive_amplitude_scaling
y_data = data.rotation_angle_2pi
z_data = data.readout__final__I__avg
z_err = data.readout__final__I__avg__error

from matplotlib.gridspec import GridSpec

fig = plt.figure()
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 0.2, 1, 1])


# Imshow of Data
# Setup the Image Plot
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("cmap", ["black", "C0", "C1", "white"])

ax = fig.add_subplot(gs[:, :2])
img = ax.imshow(
    z_data / scale_z,
    aspect="auto",
    origin="lower",
    cmap=cmap,
    extent=[
        y_data.min() * 2 * np.pi / scale_y,
        y_data.max() * 2 * np.pi / scale_y,
        x_data.min() / scale_x,
        x_data.max() / scale_x,
    ],
)

ax.set_xlabel(ylabel)
ax.set_ylabel(xlabel)
ax.set_title(title)

ax.set_xticks(np.linspace(-2 * np.pi, 2 * np.pi, 5))
ax.set_xticklabels(["$-2\pi$", "$-\pi$", "$0$", "$\pi$", "$2\pi$"])

cbar = fig.colorbar(img, ax=ax, label=z_label, pad=0.10)
cbar.ax.yaxis.set_label_position("left")

fig.tight_layout()


# For first and last fit we want to visualize the plots
ax_first = fig.add_subplot(gs[0, 2])
ax_last = fig.add_subplot(gs[1, 2], sharex=ax_first)
ax_first.tick_params(labelbottom=False)

# Fitting Loop
from iminuit import Minuit
from iminuit.cost import LeastSquares

func = lambda x, A, f, phi, c: A * np.cos(2 * np.pi * f * x + phi) + c

amplitudes = np.zeros(x_data.shape[0])
amplitudes_err = np.zeros(x_data.shape[0])

for i in range(x_data.shape[0]):
    ls = LeastSquares(
        y_data.values.flatten(),
        z_data[i].values.flatten(),
        z_err[i].values.flatten(),
        model=func,
    )
    minimizer = Minuit(ls, A=2e-3, f=1, phi=0, c=0)
    minimizer.migrad()

    if i == 0:
        ax_first.plot(
            y_data.values.flatten() / scale_y,
            z_data[i].values.flatten() / scale_z,
            "o",
            color="C0",
        )
        ax_first.errorbar(
            y_data.values.flatten() / scale_y,
            z_data[i].values.flatten() / scale_z,
            yerr=z_err[i].values.flatten() / scale_z,
            ls="none",
            color="C0",
            capsize=2,
            elinewidth=1,
        )
        x_fit = np.linspace(-1, 1, 1000)
        ax_first.plot(
            x_fit / scale_y,
            func(x_fit, *minimizer.values) / scale_z,
            "-",
            color="black",
            label="Fit",
        )
        # break
        ax_first.text(
            1,
            1,
            "amplitude: {:.2f} mV".format(x_data[i].values / scale_x),
            transform=ax_first.transAxes,
            va="top",
            ha="right",
            fontsize=16,
        )

    elif i == x_data.shape[0] // 2:
        ax_last.plot(
            y_data.values.flatten() / scale_y,
            z_data[i].values.flatten() / scale_z,
            "o",
            color="C1",
        )
        ax_last.errorbar(
            y_data.values.flatten() / scale_y,
            z_data[i].values.flatten() / scale_z,
            yerr=z_err[i].values.flatten() / scale_z,
            ls="none",
            color="C1",
            capsize=2,
            elinewidth=1,
        )
        x_fit = np.linspace(-1, 1, 1000)
        ax_last.plot(
            x_fit / scale_y,
            func(x_fit, *minimizer.values) / scale_z,
            "-",
            color="k",
            label="Fit",
        )
        ax_last.text(
            1,
            1,
            "amplitude: {:.2f} mV".format(x_data[i].values / scale_x),
            transform=ax_last.transAxes,
            va="top",
            ha="right",
            fontsize=16,
        )
    if minimizer.valid:
        amplitudes[i] = abs(minimizer.values["A"])
        amplitudes_err[i] = minimizer.errors["A"]


mask = amplitudes > 0

ax_first.set(
    title="Coherence at given Amplitude",
    ylabel="Signal I (a.u.)",
)

ax_last.set(
    xlabel=ylabel,
    ylabel="Signal I (a.u.)",
)

ax_last.set_xticks(np.linspace(-1, 1, 5))
ax_last.set_xticklabels(["$-2\pi$", "$-\pi$", "$0$", "$\pi$", "$2\pi$"])

# Plot the fitted frequencies
ax = fig.add_subplot(gs[:, 3])
ax.plot(x_data[mask] / scale_x, amplitudes[mask] / scale_z, "o", color="black")
ax.errorbar(
    x_data[mask] / scale_x,
    amplitudes[mask] / scale_z,
    yerr=amplitudes_err[mask] / scale_z,
    ls="none",
    color="black",
    capsize=2,
    elinewidth=1,
)

# Fit Normal Distribution
from scipy.stats import norm

func = lambda x, A, mu, sigma: norm.pdf(x, loc=mu, scale=sigma) * A

ls = LeastSquares(
    x_data.values.flatten()[mask], amplitudes[mask], amplitudes_err[mask], model=func
)
minimizer = Minuit(ls, A=1e-5, mu=0, sigma=0.75)
minimizer.migrad()

x_fit = np.linspace(x_data.min(), x_data.max(), 1000)
ax.plot(
    x_fit / scale_x,
    func(x_fit, *minimizer.values) / scale_z,
    "-",
    color="black",
    label="Fit",
)

ax.set(
    title="Cosine Amplitudes",
    xlabel=xlabel,
    ylabel="Cosine Amplitude (a.u.)",
)

fig.tight_layout()
fig.savefig("../Figures/dephasing_by_measurement.pdf")
