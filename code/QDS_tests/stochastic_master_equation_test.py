## Setup
import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle

sys.path.append("..")

# Paths and imports
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/test"


# Load devices/system
from devices.device import SimpleQubit, Resonator
from devices.system import QubitResonatorSystem, NewDispersiveQubitResonatorSystem
from devices.pulses import SquareCosinePulse
from simulation.experiment import StochasticMasterEquationExperiment


from devices.system import dispersive_shift
from analysis.auto import automatic_analysis


## Define devices
qubit = SimpleQubit(frequency=4.0, anharmonicity=0.5)

resonator = Resonator(
    frequency=6.0,
    kappa=0.1,
    levels=10,
)

resonator_pulse = SquareCosinePulse(
    amplitude=0.1,
    frequency=6.0 + 0.025 * np.linspace(0, 1, 10),
    phase=0.0,
    duration=1000,
    start_time=0.0,
)

# System
system = QubitResonatorSystem(
    qubit,
    resonator,
    coupling_strength=0.25 * 2 * np.pi,
    resonator_pulse=resonator_pulse,
    readout_efficiency=1,
).dispersive_approximation()

# system = system.dispersive_approximation()

# print(dispersive_shift(system) / 2 / np.pi)

##### SIMULATIONS #####
name = "stochastic_tests"
experiment_name = os.path.join(experiment_path, name)
times = np.linspace(0, 500, 500)
nsubsteps = 10

# Experiment
experiment = StochasticMasterEquationExperiment(
    system,
    [system.get_states(0, 0), system.get_states(1, 0)],
    times,
    store_states=True,
    only_store_final=False,
    expectation_operators=[
        system.qubit_state_occupation_operator(1),
        system.photon_number_operator(),
    ],
    save_path=experiment_name,
    ntraj=10,
    method="heterodyne",
    nsubsteps=nsubsteps,
    store_measurements=True,
)

results = experiment.run()

automatic_analysis(results)
