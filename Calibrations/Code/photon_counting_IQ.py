# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")

data_folder = "../Data/photon_calibration_adc"


xlabel = "Amplitude Scaling (mV)"
scale_x = 1e-3


y_label = "Readout Signal"
scale_y = 1e-3


data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.resonator_drive_amplitude_scaling
time = data.adc_timestamp
timescale = 1e-9

# Intermediate frequency
import json

config = json.load(open(data_folder + "/state_after.json", "r"))
lo = config["readout_lines[]/lo_freq"][0]
fr = config["readout_resonators[]/f_opt"][0]
intermediate_frequency = abs(lo - fr)

I_g = data.readout__final__adc_I__avg[0]
Q_g = data.readout__final__adc_Q__avg[0]

I_g = I_g - I_g.mean()
Q_g = Q_g - Q_g.mean()

from numpy.fft import fft, fftfreq

freqs = fftfreq(time.size, (time[1] - time[0]).values)
amplitudes = abs(fft(I_g[-1, :].values))
intermediate_frequency_fft = freqs[np.argmax(amplitudes)]


# plt.plot(freqs, amplitudes)
# plt.xlim(0, 5e8)

# Demodulate
g = I_g + 1j * Q_g
g = g * np.exp(1j * 2 * np.pi * intermediate_frequency * time)
I_g = g.real
Q_g = g.imag
abs_g = I_g**2 + Q_g**2

# plt.plot(I_g[-1, :])


# Running Mean
def running_mean(x, N):
    cummean = np.zeros_like(x)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    cummean[N:-N] = (cumsum[N:] - cumsum[:-N]) / float(N)
    return cummean


def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode="same")


# Setup the Image Plot
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("cmap", ["black", "C0", "C1", "white"])
mean_dur = 50
window = 51
cap = window // 2
every = 10

from scipy.signal import savgol_filter

fig, ax = plt.subplots(nrows=3)
for i in range(0, I_g.shape[0], 10):
    filter_I = savgol_filter(I_g[i, :], window, 4)
    # filter_I = running_mean(filter_I, mean_dur)
    ax[0].plot(
        time[cap:-cap] / timescale,
        filter_I[cap:-cap] / scale_y,
        color=cmap(1 - i / I_g.shape[0]),
        label="I",
    )

    filter_Q = savgol_filter(I_g[i, :], window, 4)
    mean_Q = running_mean(Q_g[i, :], mean_dur)
    ax[1].plot(
        time[cap:-cap] / timescale,
        filter_Q[cap:-cap] / scale_y,
        color=cmap(1 - i / Q_g.shape[0]),
        label="Q",
    )
    filter_abs = np.sqrt(filter_I**2 + filter_Q**2)
    ax[2].plot(
        time[cap:-cap] / timescale,
        filter_abs[cap:-cap] / scale_y,
        color=cmap(1 - i / abs_g.shape[0]),
        label="abs",
    )


fig, ax = plt.subplots()

total = (I_g**2 + Q_g**2) ** (1 / 2)
total_sig = (I_g[:, 100:300].mean(axis=1) ** 2 + Q_g[:, 100:300].mean(axis=1) ** 2) ** (
    1 / 2
)
total_sig_err = (
    I_g[:, 100:300].std(axis=1) ** 2 + Q_g[:, 100:300].std(axis=1) ** 2
) ** 2
# abs_sig_error =


ax.scatter(
    x_data / scale_x,
    total_sig / scale_y,
    c=cmap(1 - np.arange(I_g.shape[0]) / I_g.shape[0]),
)

ax.errorbar(
    x_data / scale_x,
    total_sig / scale_y,
    yerr=total_sig_err / scale_y,
    ls="none",
    color="black",
    capsize=2,
    elinewidth=1,
)

# ax[1].scatter(
#     x_data / scale_x,
#     Q_g[:, 500:].mean(axis=1) / scale_y,
#     c=cmap(1 - np.arange(Q_g.shape[0]) / Q_g.shape[0]),
# )

# ax[2].scatter(
#     x_data / scale_x,
#     np.sqrt(I_g[:, 500:].mean(axis=1) ** 2 + Q_g[:, 500:].mean(axis=1) ** 2) / scale_y,
#     c=cmap(1 - np.arange(abs_g.shape[0]) / abs_g.shape[0]),
# )

# img = ax.imshow(
#     I_g.cumsum(axis=1) / scale_y,
#     aspect="auto",
#     origin="lower",
#     cmap=cmap,
#     extent=[
#         time.min() / timescale,
#         time.max() / timescale,
#         x_data.min() / scale_x,
#         x_data.max() / scale_x,
#     ],
# )


# img = ax.imshow(
#     z_data / scale_z,
#     aspect="auto",
#     origin="lower",
#     cmap=cmap,
#     extent=[
#         y_data.min() / scale_y,
#         y_data.max() / scale_y,
#         x_data.min() / scale_x,
#         x_data.max() / scale_x,
#     ],
# )

# ax.set_xlabel(ylabel)
# ax.set_ylabel(xlabel)
# ax.set_title(title)

# cbar = fig.colorbar(img, ax=ax, label=z_label, pad=0.01)
# cbar.ax.yaxis.set_label_position("right")

# # cbar.ax.yaxis.tick_left()
# cbar.ax.yaxis.label.set_rotation(-90)

# # Plot the fitted frequencies
# ax.plot(y_data / scale_y, frequence_at_amplitude / scale_x, "x", color="black")
# ax.errorbar(
#     y_data / scale_y,
#     frequence_at_amplitude / scale_x,
#     yerr=frequence_at_amplitude_err / scale_x,
#     ls="none",
#     color="black",
#     capsize=2,
#     elinewidth=1,
# )
# fig.tight_layout()

# # Plot the fitted frequencies
# ax.plot(
#     y_data / scale_y,
#     func(y_data, *minimizer.values) / scale_x,
#     "-",
#     color="black",
#     label="Fit",
#     alpha=0.75,
# )

# fig.savefig(f"../Figures/{title}.pdf")


# # Figure with conversion from amplitude to photon number
# fig, ax = plt.subplots()

# distance_to_zero = frequence_at_amplitude - minimizer.values["c"]
# distance_to_zero_err = np.sqrt(
#     frequence_at_amplitude_err**2 + minimizer.errors["c"] ** 2
# )


# photons = -distance_to_zero / dispersive_shift
# photons_err = (
#     np.sqrt(
#         (distance_to_zero_err / distance_to_zero) ** 2
#         + (dispersive_shift_err / dispersive_shift) ** 2
#     )
#     * photons
# )

# ax.plot(y_data / scale_y, photons, "o")
# ax.errorbar(
#     y_data / scale_y,
#     photons,
#     yerr=photons_err,
#     ls="none",
#     color="black",
#     capsize=2,
#     elinewidth=1,
# )

# # Plot a scaled fit
# photons_from_fit = (
#     -(func(y_data, *minimizer.values) - minimizer.values["c"]) / dispersive_shift
# )
# ax.plot(y_data / scale_y, photons_from_fit, "-", color="black", alpha=0.75)

# ax.set(
#     title="Photon Number vs. Amplitude Scaling",
#     xlabel=ylabel,
#     ylabel="Photon Number",
# )


# fig.tight_layout()
# fig.savefig(f"../Figures/photon_number.pdf")

# # Fit
