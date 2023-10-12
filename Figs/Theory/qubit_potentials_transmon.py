import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../../code/matplotlib_style/inline_figure.mplstyle")


n_cutoff = 15

# Cooper Pair Box Parameters
EJ = 50
EC = 1
phi_ext = 0
ng = 0

phi = np.linspace(-np.pi, np.pi, 2 * n_cutoff + 1)
cos_phi_mat = np.diag(np.cos(phi))

dphi = phi[1] - phi[0]

n = np.diag(np.ones(2 * n_cutoff), 1) - np.diag(np.ones(2 * n_cutoff), -1) / (2 * dphi)

n_squared = -(
    np.diag(np.ones(2 * n_cutoff), 1)
    - 2 * np.diag(np.ones(2 * n_cutoff + 1), 0)
    + np.diag(np.ones(2 * n_cutoff), -1)
) / (dphi)

H = 4 * EC * n_squared - EJ * cos_phi_mat

eigvals, eigvecs = np.linalg.eigh(H)
eigvals, eigvecs = eigvals[np.argsort(eigvals)], eigvecs[:, np.argsort(eigvals)]


def potential(phi):
    return -EJ * np.cos(phi + phi_ext)


fig, ax = plt.subplots()
ax.plot(phi, potential(phi), color="black")


ax.set(
    xlabel="phase, $\\varphi \;(\phi_0)$",
    ylabel="Energy $E \;(E_J)$",
    title="Potential and Energy Levels of CPB",
)

ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"])

ax.set_yticks([-1, 0, 1])

for i in range(4):
    ax.fill_between(
        phi,
        np.ones_like(phi) * eigvals[i],
        eigvals[i] + 1.5 * eigvecs[:, i],
        color=f"C{i}",
    )
    ax.text(
        -np.pi,
        eigvals[i] + 0.05,
        f"$E_{i}$",
        color=f"C{i}",
        fontsize=18,
        va="bottom",
        ha="left",
    )


E_01 = eigvals[1] - eigvals[0]
ax.hlines(eigvals[0] + 2 * E_01, -np.pi, np.pi, color="C4", linestyle="--")

ax.text(
    -np.pi / 2,
    eigvals[0] + 2 * E_01 + 0.05,
    f"$E_0 + 2 E_{{01}}$",
    color="C4",
    fontsize=18,
    va="bottom",
    ha="center",
)

# fig.savefig("CPB_potential_flux.pdf", bbox_inches="tight")


# Repeat in charge basis
charge_basis = np.diag(np.arange(-n_cutoff, n_cutoff + 1))
charge_basis_squared = np.diag(np.arange(-n_cutoff, n_cutoff + 1) ** 2)


def charge_potential(n):
    return 4 * EC * (n - ng) ** 2


# Cos phi matrix in charge basis
cos_phi_mat = np.diag(np.ones(2 * n_cutoff), -1) + np.diag(np.ones(2 * n_cutoff), 1)
cos_phi_mat /= 2

H = 4 * EC * charge_basis_squared - EJ * cos_phi_mat

eigvals, eigvecs = np.linalg.eigh(H)
eigvals, eigvecs = eigvals[np.argsort(eigvals)], eigvecs[:, np.argsort(eigvals)]


fig, ax = plt.subplots()
ns = np.arange(-n_cutoff, n_cutoff + 1)
ns_to_look_at = 2


ax.plot(ns, charge_potential(ns), color="black")
ax.set(
    xlabel="Charge, $n$",
    ylabel="Energy $E \;(E_J)$",
    title="Potential and Energy Levels of CPB",
    xlim=(-ns_to_look_at, ns_to_look_at),
    ylim=(0, charge_potential(-ns_to_look_at)),
)


for i in range(4):
    ax.fill_between(
        ns,
        np.ones_like(ns) * eigvals[i] / EJ,
        eigvals[i] / EJ + 1.5 * eigvecs[:, i],
        color=f"C{i}",
    )
    # ax.text(
    #     -n_cutoff,
    #     eigvals[i] / EJ + 0.05,
    #     f"$E_{i}$",
    #     color=f"C{i}",
    #     fontsize=18,
    #     va="bottom",
    #     ha="left",
    # )
