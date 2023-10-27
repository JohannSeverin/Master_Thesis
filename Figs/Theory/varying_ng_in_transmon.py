import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../code/matplotlib_style/margin_figure.mplstyle")
plt.rcParams["figure.figsize"] = (6, 7)

n_cutoff = 30

from devices.device import Transmon

ngs_to_try = np.linspace(-3, 3, 61)
EJs_to_try = (1, 50)

fig, ax = plt.subplots(
    nrows=2,
    sharex=True,
)


for i, EJ in enumerate(EJs_to_try):
    energy_levels = []
    for ng in ngs_to_try:
        transmon = Transmon(EJ=EJ, EC=1, levels=4, ng=ng)
        energy_levels.append(np.diag(transmon.hamiltonian.full()))

    energy_levels = np.array(energy_levels).real

    for j in range(3):
        ax[i].plot(ngs_to_try, energy_levels[:, j] / 2 / np.pi, label=f"k = {j}")


ax[0].set(
    title="$E_J = E_C$",
    # xlabel="$n_g$",
    ylabel="Energy ($E_C$)",
)

ax[1].set(
    title="$E_J = 50E_C$",
    xlabel="$n_g$",
    ylim=(-8, 1),
    ylabel="Energy ($E_C$)",
)

ax[0].legend(fontsize=16)

fig.align_ylabels(ax)

# fig.suptitle("Energy Spacing of Transmon Levels", y=0.95, va="top")
fig.tight_layout()

fig.savefig("Transmon_energy_vs_ng.pdf", bbox_inches="tight")
