import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../../code/matplotlib_style/fullwidth_figure.mplstyle")

import json, sys, os, pickle
import time

sys.path.append("../..")


name = "qubit_T1"

schroedinger_data = pickle.load(
    open("../data/" + name + "_schoedinger_dispersive.pkl", "rb")
)
monte_carlo_data = pickle.load(
    open("../data/" + name + "_monte_carlo_dispersive.pkl", "rb")
)
lindblad_data = pickle.load(open("../data/" + name + "_lindblad_dispersive.pkl", "rb"))
sme_data = pickle.load(open("../data/" + name + "_sme_dispersive.pkl", "rb"))

# Setup Figure
plt.rcParams["axes.titlesize"] = 16
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)


axes[1].set_yticks([0, 1])
axes[1].set_yticklabels([0, 1])
axes[2].set_yticklabels([])
axes[3].set_yticklabels([])




# Plotting Simulation Data
ax = axes[1]

times = schroedinger_data.times[::250] * 1e-3
y_data = schroedinger_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Schroedinger")

ax.set(
    ylabel=r"$\langle P_1 \rangle$",
    xlabel="Time (µs)",
    ylim=(-0.05, 1.05),
    title="SE - Dispersive",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[2]

times = monte_carlo_data.times[::250] * 1e-3
y_data = monte_carlo_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Monte Carlo")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(-0.05, 1.05),
    title="MC - Dispersive",
)
# ax.legend()

# Plotting Simulation Data
ax = axes[3]

times = lindblad_data.times[::250] * 1e-3
y_data = lindblad_data.exp_vals[::250]

ax.plot(times, y_data, "o", label="Lindblad")

ax.set(
    # ylabel=r"Expectation Value of $P_{1}$",
    xlabel="Time (µs)",
    ylim=(-0.05, 1.05),
    title="ME - Dispersive",
)
# ax.legend()


# SME Plot
measurements = np.array(sme_data.measurements)
I_measurements = measurements.mean(0)[:-1, 0, 0]

measurements = I_measurements.reshape(-1, 20).mean(1).real
measurements_err = (I_measurements.reshape(-1, 20).real.std(1)) / np.sqrt(20)

ax = axes[0]
ax.plot(sme_data.times[:-1:20] * 1e-3, measurements, "o", label="SME")
ax.errorbar(
    sme_data.times[:-1:20] * 1e-3,
    measurements,
    yerr=measurements_err,
    ls="none",
    color="C0",
    capsize=2,
    elinewidth=1,
)

ax.set(
    title="SME - Dispersive",
    xlabel="Time (µs)",
    ylabel="Readout Signal I (a. u.)",
)

fig.tight_layout()

# fig.align_ylabels(axes)
fig.savefig("../Figs/" + name + "_dispersive.pdf")
