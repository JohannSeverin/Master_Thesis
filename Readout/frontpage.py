# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/IQ_threshold_141420"

# Imports
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys, os

plt.style.use("../code/matplotlib_style/fullwidth_figure.mplstyle")
plt.rcParams["axes.titlesize"] = 18  # because of three subplots this size is better
plt.rcParams["figure.figsize"] = (18, 12)  # Make them more square

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("mycmap", ["C0", "C1"], N=2)

from matplotlib.gridspec import GridSpec

data = xr.open_dataset(os.path.join(path, "demodulated_dataset.nc")).sel(
    adc_timestamp=slice(0, 1e-6)
)


# Check with simulated data
import pickle

transformed_sim = pickle.load(
    open(
        "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Simulations/budgets/transformed.pkl",
        "rb",
    )
)

plt.style.use("../code/matplotlib_style/inline_figure.mplstyle")


fig, ax = plt.subplots(figsize=(8.268, 11.693))

I_ground = data["I_ground"].coarsen({"adc_timestamp": 2}, boundary="pad").sum().values
Q_ground = data["Q_ground"].coarsen({"adc_timestamp": 2}, boundary="pad").sum().values
I_excited = data["I_excited"].coarsen({"adc_timestamp": 2}, boundary="pad").sum().values
Q_excited = data["Q_excited"].coarsen({"adc_timestamp": 2}, boundary="pad").sum().values

for s in range(100):
    ax.plot(I_ground[s].cumsum(), Q_ground[s].cumsum(), color="C0", alpha=0.15)
    ax.plot(I_excited[s].cumsum(), Q_excited[s].cumsum(), color="C1", alpha=0.20)

ax.scatter(
    I_ground.sum(axis=1),
    Q_ground.sum(axis=1),
    color="C0",
    alpha=0.35,
    s=40,
    zorder=10,
    edgecolors="k",
    linewidths=0.5,
)
ax.scatter(
    I_excited.sum(axis=1),
    Q_excited.sum(axis=1),
    color="C1",
    alpha=0.35,
    s=40,
    zorder=10,
    edgecolors="k",
    linewidths=0.5,
)


# transformed_sim["tra"]
