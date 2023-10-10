# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import iminuit

plt.style.use("../../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["axes.titlesize"] = 18

data_folder = "../Data/SNR_from_readout_amplitude"
title = "SNR vs Readout Amplitude Scaling"
xlabel = "Frequency (GHz)"
scale_x = 1e9

y_label = "Readout Signal Difference in I (a. u.)"
scale_y = 1e-3

data = xr.open_dataset(data_folder + "/dataset.nc")

x_data = data.resonator_drive_amplitude_scaling
I_g = data.readout__final__I__ss[:, 0, :]
I_e = data.readout__final__I__ss[:, 1, :]
Q_g = data.readout__final__Q__ss[:, 0, :]
Q_e = data.readout__final__Q__ss[:, 1, :]


# Fitting Loop
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm


gauss_func = lambda x, mu, sigma: norm.pdf(x, loc=mu, scale=sigma)

guesses = {"mu": 1e-4, "sigma": 2e-4}

SNR_I = np.zeros(len(x_data))
SNR_I_err = np.zeros(len(x_data))

for i in range(x_data.shape[0]):
    NLLH_ground = UnbinnedNLL(
        I_g[i],
        gauss_func,
    )
    minimizer_ground = Minuit(NLLH_ground, **guesses)
    minimizer_ground.migrad()

    NLLH_excited = UnbinnedNLL(
        I_e[i],
        gauss_func,
    )
    minimizer_excited = Minuit(NLLH_excited, **guesses)
    minimizer_excited.migrad()

    SNR_I[i] = abs(
        (minimizer_excited.values["mu"] - minimizer_ground.values["mu"])
        / np.sqrt(
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    numerator = minimizer_excited.values["mu"] - minimizer_ground.values["mu"]
    errors_numerator = np.sqrt(
        minimizer_excited.errors["mu"] ** 2 + minimizer_ground.errors["mu"] ** 2
    )
    denominator = np.sqrt(
        minimizer_excited.values["sigma"] ** 2 + minimizer_ground.values["sigma"] ** 2
    )
    errors_denominator = np.sqrt(
        (
            minimizer_excited.values["sigma"] ** 2
            * minimizer_excited.errors["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
            * minimizer_ground.errors["sigma"] ** 2
        )
        / (
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    SNR_I_err[i] = np.sqrt(
        SNR_I[i] * (errors_numerator / numerator) ** 2
        + (SNR_I[i] * errors_denominator / denominator) ** 2
    )

SNR_Q = np.zeros(len(x_data))
SNR_Q_err = np.zeros(len(x_data))

for i in range(x_data.shape[0]):
    NLLH_ground = UnbinnedNLL(
        Q_g[i],
        gauss_func,
    )
    minimizer_ground = Minuit(NLLH_ground, **guesses)
    minimizer_ground.migrad()

    NLLH_excited = UnbinnedNLL(
        Q_e[i],
        gauss_func,
    )
    minimizer_excited = Minuit(NLLH_excited, **guesses)
    minimizer_excited.migrad()

    SNR_Q[i] = abs(
        (minimizer_excited.values["mu"] - minimizer_ground.values["mu"])
        / np.sqrt(
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    numerator = minimizer_excited.values["mu"] - minimizer_ground.values["mu"]
    errors_numerator = np.sqrt(
        minimizer_excited.errors["mu"] ** 2 + minimizer_ground.errors["mu"] ** 2
    )
    denominator = np.sqrt(
        minimizer_excited.values["sigma"] ** 2 + minimizer_ground.values["sigma"] ** 2
    )
    errors_denominator = np.sqrt(
        (
            minimizer_excited.values["sigma"] ** 2
            * minimizer_excited.errors["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
            * minimizer_ground.errors["sigma"] ** 2
        )
        / (
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    SNR_Q_err[i] = np.sqrt(
        SNR_Q[i] * (errors_numerator / numerator) ** 2
        + (SNR_Q[i] * errors_denominator / denominator) ** 2
    )

# Optimal "Cut"

SNR_opt = np.zeros(len(x_data))
SNR_opt_err = np.zeros(len(x_data))

# two_gauss = lambda x, mu, mu1, sigma, p: (1 - p) * norm.pdf(
#     x, loc=mu, scale=sigma
# ) + p * norm.pdf(x, loc=mu1, scale=sigma)

# guesses = {"mu": -0.5, "mu1": 0.5, "sigma": 1, "p": 0.5}

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

for i in range(x_data.shape[0]):
    all_I = np.concatenate(
        [
            I_g[i].values.flatten(),
            I_e[i].values.flatten(),
        ]
    )
    all_Q = np.concatenate(
        [
            Q_g[i].values.flatten(),
            Q_e[i].values.flatten(),
        ]
    )
    all_data = np.stack([all_I, all_Q]).T
    labels = np.concatenate(
        [
            np.zeros_like(I_g[i].values.flatten()),
            np.ones_like(I_e[i].values.flatten()),
        ]
    )

    lda = LDA()
    lda.fit(all_data, labels)

    transformed = lda.transform(all_data)

    NLLH_ground = UnbinnedNLL(
        transformed.flatten()[labels == 0],
        gauss_func,
    )
    minimizer_ground = Minuit(NLLH_ground, **guesses)
    minimizer_ground.migrad()

    NLLH_excited = UnbinnedNLL(
        transformed.flatten()[labels == 1],
        gauss_func,
    )
    minimizer_excited = Minuit(NLLH_excited, **guesses)
    minimizer_excited.migrad()

    SNR_opt[i] = abs(
        (minimizer_excited.values["mu"] - minimizer_ground.values["mu"])
        / np.sqrt(
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    numerator = minimizer_excited.values["mu"] - minimizer_ground.values["mu"]
    errors_numerator = np.sqrt(
        minimizer_excited.errors["mu"] ** 2 + minimizer_ground.errors["mu"] ** 2
    )
    denominator = np.sqrt(
        minimizer_excited.values["sigma"] ** 2 + minimizer_ground.values["sigma"] ** 2
    )
    errors_denominator = np.sqrt(
        (
            minimizer_excited.values["sigma"] ** 2
            * minimizer_excited.errors["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
            * minimizer_ground.errors["sigma"] ** 2
        )
        / (
            minimizer_excited.values["sigma"] ** 2
            + minimizer_ground.values["sigma"] ** 2
        )
    )

    SNR_opt_err[i] = np.sqrt(
        SNR_opt[i] * (errors_numerator / numerator) ** 2
        + (SNR_opt[i] * errors_denominator / denominator) ** 2
    )


# Combine the two SNR
SNR = np.sqrt(SNR_I**2 + SNR_Q**2)
SNR_errors = np.sqrt(
    (SNR_I**2 * SNR_I_err**2 + SNR_Q**2 * SNR_Q_err**2)
    / (SNR_I**2 + SNR_Q**2)
)


fig, ax = plt.subplots(ncols=2)
mask = SNR_I_err < 0.5
ax[0].plot(x_data[mask] * 100, SNR, "o")
ax[0].errorbar(
    x_data[mask] * 100,
    SNR[mask],
    yerr=SNR_errors[mask],
    ls="none",
    color="black",
    capsize=2,
    elinewidth=1,
)

# Linear fit
from iminuit.cost import LeastSquares

func = lambda x, a, b: a * x + b

ls = LeastSquares(
    x_data,
    SNR,
    SNR_errors,
    model=func,
)
minimizer = Minuit(ls, a=1e-3, b=0)
minimizer.migrad()

ax[0].plot(
    x_data * 100, func(x_data, *minimizer.values), "-", color="black", alpha=0.75
)

ax[0].set(
    title="SNR vs. Readout Amplitude Scaling",
    xlabel="Readout Amplitude Scaling (%)",
    ylabel="SNR",
)

ax[1].hist(I_g[-1] / scale_y, bins=30, histtype="step", linewidth=3)
ax[1].hist(I_e[-1] / scale_y, bins=30, histtype="step", linewidth=3)

ax[1].set(
    title="Histogram of Readout Signal at Max. Amplitude",
    xlabel="Readout Signal in I (a. u.)",
    ylabel="Counts",
)

# fig.tight_layout()
fig.savefig(f"../Figures/SNR_vs_amplitude.pdf")

# x_data = data.pulse_frequency
# y_data = data.resonator_drive_amplitude_scaling
# z_data = data.readout__final__I__avg[1] - data.readout__final__I__avg[0]
# z_err = (
#     data.readout__final__I__avg__error[1] ** 2
#     + data.readout__final__I__avg__error[0] ** 2
# )
# z_err = np.sqrt(z_err)


# # Fitting Loop
# from iminuit import Minuit
# from iminuit.cost import LeastSquares

# fit_name = "Lorentzian"


# def fit_func(x, f0, A, gamma, offset):
#     return offset + A * gamma**2 / (gamma**2 + (x - f0) ** 2)


# guesses = {"f0": 5.983e9, "A": 1e-3, "gamma": 0.01e9, "offset": 1e-4}


# frequence_at_amplitude = np.zeros(len(y_data))
# frequence_at_amplitude_err = np.zeros(len(y_data))

# for i in range(len(y_data)):
#     ls = LeastSquares(
#         x_data.values.flatten(),
#         z_data[:, i].values.flatten(),
#         z_err[:, i].values.flatten(),
#         model=fit_func,
#     )
#     minimizer = Minuit(ls, **guesses)
#     minimizer.migrad()

#     frequence_at_amplitude[i] = minimizer.values["f0"]
#     frequence_at_amplitude_err[i] = minimizer.errors["f0"]

# # Fit second order polynomial
# func = lambda x, a, b, c: a * x**2 + b * x + c
# ls = LeastSquares(
#     y_data.values.flatten(),
#     frequence_at_amplitude,
#     frequence_at_amplitude_err,
#     model=func,
# )
# minimizer = Minuit(ls, a=-1e2, b=0, c=5.98e9)
# minimizer.migrad()

# # Setup the Image Plot
# from matplotlib.colors import LinearSegmentedColormap

# cmap = LinearSegmentedColormap.from_list("cmap", ["black", "C0", "C1", "white"])

# fig, ax = plt.subplots()

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
