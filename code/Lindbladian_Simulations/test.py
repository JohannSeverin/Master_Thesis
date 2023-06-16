# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use("../matplotlib_style/standard_plot_style.mplstyle")

import qutip

# %matplotlib

save_figures = False
save_path    = "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/Figs/Dispersive_Simulations"

sys.path.append("../utils")


# %%
simulation_steps = 2001

# Setup Qubit, Resonator and Coupling
n_cutoff    = 15
EJ          = 15     * 2 * np.pi # h GHz
EC          = EJ / 25

resonator_states        = 20
resonator_frequency     = 6.02 * 2 * np.pi    

coupling_strength       = 0.250 * 2 * np.pi


# Setup drive
drive_amplitude        = 0.020
driving_time           = (0, 250)
drive_function         = lambda t, args: drive_amplitude * np.cos(args["driving_frequency"] * t)

# driving_frequencies_to_scan = 2 * np.pi * np.linspace(*scan_range, resolution)

# Define Qubit object
import components as comp
qubit = comp.Transmon(
    EC = EC,
    EJ = EJ,
    basis = "charge",
    n_cutoff = n_cutoff
)
# Define resonator operators
from qutip import destroy
a       = destroy(resonator_states)
a_dag   = a.dag()

# %%
qubit_states = 3

# Get Qubit operators for the two level qubit
H_qubit, jump_matrix = qubit.lowest_k_eigenstates(k = qubit_states)

# %%
# Calculate all types of constants, which is useful for further analysis:
omega_01 = H_qubit.diag()[1] - H_qubit.diag()[0]
omega_02 = H_qubit.diag()[2] - H_qubit.diag()[0]
omega_12 = H_qubit.diag()[2] - H_qubit.diag()[1]

# Calculate the dispersive shifts
# Multi qubit shifts
g_squared_matrix = coupling_strength ** 2 * abs(jump_matrix.full()) ** 2
omega_ij_matrix = np.expand_dims(H_qubit.diag(), 1) - np.expand_dims(H_qubit.diag(), 0)
omega_r = resonator_frequency

# The Chi-matrix
chi_matrix = g_squared_matrix * (1 / (omega_ij_matrix - omega_r) + 1 / (omega_ij_matrix + omega_r)) 

# The dis
dispersive_shifts = chi_matrix.sum(axis = 1) / 2 / np.pi


# Omega resonator for qubits
omega_resonator_qubit = resonator_frequency + dispersive_shifts



# Information to print:
printing_dict = {
    "omega_01" : f"{omega_01 / 2 / np.pi:.3f} GHz",
    "omega_12" : f"{omega_12 / 2 / np.pi:.3f} GHz",
    "omega_02" : f"{omega_02 / 2 / np.pi:.3f} GHz",
    "omega_r"  : f"{resonator_frequency / 2 / np.pi:.3f} GHz",
    "omega_r_tilde": f"{np.mean(omega_resonator_qubit[:2]) / 2 / np.pi:.3f} GHz",
    "coupling_strength": f"{coupling_strength / 2 / np.pi:.3f} GHz",
    "dispersive_shift": f"{(dispersive_shifts[0] - dispersive_shifts[1]) * 1000 / 2 / 2 / np.pi:.3f} MHz",
    "drive_amplitude": f"{drive_amplitude * 1000:.3f} MHz",
}

printing_dict_symbolic = {
    r"\omega_{01}" : f"{omega_01 / 2 / np.pi:.3f} GHz",
    r"\omega_{12}" : f"{omega_12 / 2 / np.pi:.3f} GHz",
    r"\omega_{02}" : f"{omega_02 / 2 / np.pi:.3f} GHz",
    r"\omega_r"  : f"{resonator_frequency / 2 / np.pi:.3f} GHz",
    r"\tilde{\omega_r}": f"{np.mean(omega_resonator_qubit[:2]) / 2 / np.pi:.3f} GHz",
    r"g" : f"{coupling_strength / 2 / np.pi:.3f} GHz",
    r"\chi" : f"{(dispersive_shifts[0] - dispersive_shifts[1]) * 1000 / 2 / 2 / np.pi:.3f} MHz",
}

variable_dict = {
    "omega_01" : omega_01,
    "omega_12" : omega_12,
    "omega_02" : omega_02,
    "omega_r"  : resonator_frequency,
    "omega_r_tilde": np.mean(omega_resonator_qubit[:2]),
    "coupling_strength": coupling_strength,
    "dispersive_shift": (dispersive_shifts[0] - dispersive_shifts[1]),
}

for key, value in printing_dict.items():
    print(f"{key} \t= {value}")

# %% [markdown]
# For now, we will just consider the dispersive approximation since the Hamiltonian is not time dependent in this basis:

# %% [markdown]
# ### Simulation for the Dispersive Limit Hamiltonian
# Non interacting:  
# 
# $H_{eff} = (\omega_r - \omega_d  + \sum_k \chi_k \ket{k}\ket{k} ) a^\dagger a$
# 
# Driving
# 
# $H_{d, eff} = \epsilon (a^\dagger + a)$

# %%
from qutip import tensor, basis, ket2dm
from tqdm import tqdm

# We can now define the Hamiltonian:
def get_Hamiltonian(drive_frequency):
    H_res   = (omega_r - drive_frequency) * tensor(qutip.qeye(qubit_states), a_dag * a)

    H_disp  = tensor(qutip.Qobj(np.diag(dispersive_shifts)), a_dag * a)

    H_drive = drive_amplitude * tensor(qutip.qeye(qubit_states), a_dag + a)

    return H_res + H_disp + H_drive


# %% [markdown]
# Now we setup the different parameters to watch in the simulation:

# %%
# We want expectation values of the following operators throughout the simulation:
I = a + a_dag
Q = 1j * (a_dag - a)
n = a_dag * a

from qutip import ptrace, expect
exp_value_operators = [I, Q, n]
# exp_value_operators = [lambda t, state: expect(op, ptrace(state, 1)) for op in exp_value_operators]
def exp_values(t, state):
    reso_state = ptrace(state, 1)
    exp_vals   = [expect(op, reso_state) for op in exp_value_operators]
    return exp_vals

# We choose to drive it resonant to the qubit_0 frequency
drive_frequency = omega_resonator_qubit[0] / 2 + omega_resonator_qubit[1]  / 2 # variable_dict["omega_r_tilde"]
hamiltonian = get_Hamiltonian(drive_frequency)



# %% [markdown]
# ## Setup collapse operators and decay rates

# %% [markdown]
# First we just setup the qubit decay. $\ket{1} \to \ket{0}$. This happens with a rate $\Gamma_1 = \frac{1}{T_1}$.
# 
# The corresponding Lindblad operator is $\ket{0}\bra{1}$. And of course nothing happens on the resonator:

# %%
# Setup the qubit decay
rate_qubit = 1 / 250

qubit_decay_matrix = np.zeros((qubit_states, qubit_states))
qubit_decay_matrix[0, 1] = 1

qubit_decay = np.sqrt(rate_qubit) * tensor(qutip.Qobj(qubit_decay_matrix), qutip.qeye(resonator_states))


# Setup the resonator decay
rate_reso = 1 / 50

reso_decay = np.sqrt(rate_reso) * tensor(qutip.qeye(qubit_states), a)


decays = [qubit_decay, reso_decay]

# %% [markdown]
# ## Run the Simulation

# %%
# And simply start in the intial state
initial_qubit = basis(qubit_states, 0)
initial_resonator = basis(resonator_states, 0)
initial_state = ket2dm(tensor(initial_qubit, initial_resonator))

# We can now simulate the system
from qutip import mesolve, Options
result_0 = mesolve(
    hamiltonian,
    initial_state,
    np.linspace(*driving_time, simulation_steps),
    c_ops = decays,
    e_ops = exp_values,
    options = Options(store_states = True),
    progress_bar = True
)

exp_values_from_simulation_0 = np.array(result_0.expect)

# Do the same for first excited state
initial_qubit = basis(qubit_states, 1)
initial_resonator = basis(resonator_states, 0)
initial_state = ket2dm(tensor(initial_qubit, initial_resonator))

result_1 = mesolve(
    hamiltonian,
    initial_state,
    np.linspace(*driving_time, simulation_steps),
    c_ops = decays,
    e_ops = exp_values,
    options = Options(store_states = True),
    progress_bar = True
)

exp_values_from_simulation_1 = np.array(result_1.expect)

# %% [markdown]
# ## Plotting the Results

# %%
xvec = yvec = np.linspace(-5, 5, 100)

from qutip import QFunc
qfunc = QFunc(xvec, yvec)

# %%
from matplotlib.colors import LinearSegmentedColormap
cmap_0 = LinearSegmentedColormap.from_list("mycmap", ["white", "C0"])
cmap_1 = LinearSegmentedColormap.from_list("mycmap", ["white", "C1"])

printing_dict_symbolic[r"\epsilon"] = f"{drive_amplitude * 1000:.3f} MHz"
printing_dict_symbolic[r"\omega_{drive}"] = f"{drive_frequency / 2 / np.pi:.3f} GHz"

written_out = [f"${key} \t= {value}$" for key, value in printing_dict_symbolic.items()]
text_to_print = "\n".join(written_out)

def segment_sum(data, segment_ids, reso = 100):
    data = np.asarray(data)
    s = np.zeros((reso), dtype=data.dtype)
    np.add.at(s, segment_ids, data)
    return s


# %%
from matplotlib.gridspec import GridSpec

def inspect_results(time):
    time_idx = int(time / (driving_time[1] - driving_time[0]) * simulation_steps)

    fig = plt.figure(figsize = (12, 12))
    gs  = GridSpec(3, 3, figure = fig)

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(result_0.times, exp_values_from_simulation_0[:, 2])
    ax0.plot(result_1.times, exp_values_from_simulation_1[:, 2])
    ax0.set(
        xlabel = "Time [ns]",
        ylabel = "Occupation number",
        title = "Occupation of the Resonator"
    )
    ax0.vlines(time, *ax0.get_ylim(), color = "gray", ls = "--")

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(exp_values_from_simulation_0[:, 0], exp_values_from_simulation_0[:, 1], ls = "--", alpha = 0.5, label = "I(t), Q(t)" )
    ax1.plot(exp_values_from_simulation_1[:, 0], exp_values_from_simulation_1[:, 1], ls = "--", alpha = 0.5)

    mark_every = 50 # ns
    mark_every_idx = int(mark_every / (driving_time[1] - driving_time[0]) * simulation_steps)

    ax1.scatter(exp_values_from_simulation_0[time_idx, 0], exp_values_from_simulation_0[time_idx, 1], marker = "x", color = "gray", s = 200)
    ax1.scatter(exp_values_from_simulation_1[time_idx, 0], exp_values_from_simulation_1[time_idx, 1], marker = "x", color = "gray", s = 200)

    ax1.set(
        xlabel = "<I>",
        ylabel = "<Q>",
        title = "I-Q plot of resonator",
        xlim = (xvec[0], xvec[-1]),
        ylim = (yvec[0], yvec[-1])

    )


    ax2 = fig.add_subplot(gs[1:, :2])

    ax2.contour(qfunc(ptrace(result_0.states[time_idx], 1)), extent = [xvec[0], xvec[-1], yvec[0], yvec[-1]], origin = "lower", cmap = cmap_0, alpha = 0.7)
    ax2.contour(qfunc(ptrace(result_1.states[time_idx], 1)), extent = [xvec[0], xvec[-1], yvec[0], yvec[-1]], origin = "lower", cmap = cmap_1, alpha = 0.7)
    ax2.set(
        xlabel = r"$Re(\alpha)$",
        ylabel = r"$Im(\alpha)$",
        title = "Q-function of Resonator"
    )

    ax3 = fig.add_subplot(gs[1:, 2])

    X, Y = np.meshgrid(xvec, yvec)
    q_0, q_1 = qfunc(ptrace(result_0.states[time_idx], 1)), qfunc(ptrace(result_1.states[time_idx], 1))
    
    means_0 = np.array((np.sum(X * q_0) / np.sum(q_0), np.sum(Y * q_0) / np.sum(q_0)))
    means_1 = np.array((np.sum(X * q_1) / np.sum(q_1), np.sum(Y * q_1) / np.sum(q_1)))

    vects   = means_0 - means_1
    if vects[0] != 0 and vects[1] != 0:
        vects  /= np.linalg.norm(vects)

        coordinates_on_primary_axis = vects[0] * X + vects[1] * Y
        
        x_primes = np.linspace(coordinates_on_primary_axis.min(), coordinates_on_primary_axis.max(), 100)
        binwidth = x_primes[1] - x_primes[0]

        idxs = coordinates_on_primary_axis.flatten() / binwidth
        idxs = (idxs - idxs.min()).astype(int)

        density_on_primary_axis_0 = segment_sum(q_0.flatten(), idxs) / binwidth
        density_on_primary_axis_1 = segment_sum(q_1.flatten(), idxs) / binwidth

        ax3.plot(density_on_primary_axis_0, x_primes)
        ax3.plot(density_on_primary_axis_1, x_primes)

        ax3.set(
            xlabel = "Projected Q-func",
            ylabel = "Coordinate on primary axis",
            title  = "Q-Function Projection"
        )

    
        line_xs = np.linspace(*ax2.get_xlim(), 100)
        line_ys = (line_xs -  means_0[0]) * vects[1] / vects[0] +  means_0[1]

        ax2.plot(line_xs, line_ys, color = "gray", ls = "-.")
        ax2.scatter(means_0[0], means_0[1], marker = "x", color = "C0", s = 200)
        ax2.scatter(means_1[0], means_1[1], marker = "x", color = "C1", s = 200)

    # fig.text(0.75, 0.4, text_to_print)


    fig.tight_layout()

from ipywidgets import interact
interact(inspect_results, time = (0, driving_time[1] - 1, 1))

# %%



