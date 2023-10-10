# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/IQ_threshold_141420"

# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys, os

plt.style.use("../code/matplotlib_style/inline_figure.mplstyle")

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)

from matplotlib.gridspec import GridSpec

data = xr.open_dataset(os.path.join(path, "demodulated_dataset.nc")).sel(
    adc_timestamp=slice(0, 1e-6)
)

sample = 1

weights_I = np.abs(
    data.I_ground.mean("sample") - data.I_excited.mean("sample")
) / np.var(data.I_ground.mean("sample") - data.I_excited.mean("sample"))

weights_Q = np.abs(
    data.Q_ground.mean("sample") - data.Q_excited.mean("sample")
) / np.var(data.Q_ground.mean("sample") - data.Q_excited.mean("sample"))

# Normalize the integration to give same units
weights_I *= len(weights_I) / weights_I.sum()
weights_Q *= len(weights_Q) / weights_Q.sum()

data["I_ground"] *= weights_I
data["I_excited"] *= weights_I
data["Q_ground"] *= weights_Q
data["Q_excited"] *= weights_Q

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

all_I = np.concatenate(
    [
        data.I_ground.sum("adc_timestamp").values,
        data.I_excited.sum("adc_timestamp").values,
    ]
)
all_Q = np.concatenate(
    [
        data.Q_ground.sum("adc_timestamp").values,
        data.Q_excited.sum("adc_timestamp").values,
    ]
)

states = np.concatenate(
    [
        np.zeros_like(data.I_ground.sum("adc_timestamp").values),
        np.ones_like(data.I_excited.sum("adc_timestamp").values),
    ]
)

lda = LDA()
lda.fit(np.stack([all_I, all_Q]).T, states)

from scipy.special import logit

prob = lda.predict_proba(np.stack([all_I, all_Q]).T)[:, 1]
logit_score = logit(prob)


from sklearn.metrics import confusion_matrix

fidelities = []
fidelities_err = []
thresholds_to_include = np.linspace(0.10, 1.00, 20)
for include in thresholds_to_include:
    print(include)
    mask = np.abs(logit_score) > np.quantile(np.abs(logit_score), 1 - include)

    c_matrix = confusion_matrix(
        states[mask], lda.predict(np.stack([all_I, all_Q]).T)[mask]
    )

    fpr, fnr = c_matrix[0, 1], c_matrix[1, 0]

    fpr_sigma, fnr_sigma = np.sqrt(fpr), np.sqrt(fnr)

    fidelities += [1 - (fpr + fnr) / c_matrix.sum() * 2]
    fidelities_err += [np.sqrt(fpr_sigma**2 + fnr_sigma**2) / c_matrix.sum()]


fig, ax = plt.subplots()

ax.set(
    xlabel="Included Fraction of Data (%)",
    ylabel="Fidelity",
    title="Fidelity vs. Included Fraction of Data",
)

ax.errorbar(
    100 * thresholds_to_include,
    fidelities,
    fidelities_err,
    marker="o",
    linestyle="none",
    capsize=2,
    elinewidth=2,
    label="data",
)

# Fit second order polynomial
from iminuit import Minuit
from iminuit.cost import LeastSquares


def parabola(x, a, b, c):
    return a + b * x + c * x**2


cost = LeastSquares(thresholds_to_include, fidelities, fidelities_err, parabola)
minimizer = Minuit(cost, a=-1, b=0, c=1)

minimizer.migrad()

x = np.linspace(0, 1, 1000)
y = parabola(x, *minimizer.values)

ax.plot(x * 100, y, "k--", label="Parabolic Fit")

ax.legend()


fig.savefig("Figs/fidelity_vs_included_fraction.pdf", bbox_inches="tight")

open("Logs/fidelity_vs_included_fraction.txt", "w").write(
    f"Best fit: {minimizer.values}\n Best fit errors: {minimizer.errors}\n"
)
