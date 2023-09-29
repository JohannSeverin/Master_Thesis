## Setup
import json, sys, os, pickle, gc
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, Boltzmann
from qutip import ket2dm

sys.path.append("..")
from qubit_builder import build_qubit, build_resonator
from analysis.auto import automatic_analysis
from devices.system import QubitResonatorSystem
from devices.pulses import SquareCosinePulse
from simulation.experiment import (
    LindbladExperiment,
    StochasticMasterEquationExperiment,
)

config = json.load(open("../qubit_calibration.json", "r"))
config["g"] *= 2  # Looks like a small error in the coupling strength.
config["eta"] *= 9
# This is to make up for the fact that the experiment has a steady state photon count of 30

ntraj = 100
save_path = "data/"
overwrite = False
show_plots = False

timescale = 1e-9  # ns


def scale(temperature, factor):
    return temperature * factor


### Run Following options of config files
config_dicts = {
    r"temperature_10_increase": {"temperature": scale(config["temperature"], 1.1)},
    r"temperature": {"temperature": scale(config["temperature"], 1.0)},
    r"temperature_10_reduction": {"temperature": scale(config["temperature"], 0.9)},
    r"temperature_25_reduction": {"temperature": scale(config["temperature"], 0.75)},
    r"temperature_50_reduction": {"temperature": scale(config["temperature"], 0.50)},
    r"temperature_100_reduction": {"temperature": scale(config["temperature"], 0.00)},
}

for name, config_dict in config_dicts.items():
    print(f"Setting up experiments for {name}")
    # Update parameters from config dict
    config.update(config_dict)

    # Save path
    save_path = os.path.join("data", name)

    # Build Devices from config file
    qubit = build_qubit(config, timescale)
    resonator = build_resonator(config, timescale, levels=20)

    resonator_pulse = SquareCosinePulse(
        amplitude=25e-3, frequency=config["fr"] * timescale
    )

    # Combine to System
    system_lindblad = QubitResonatorSystem(
        qubit,
        resonator,
        resonator_pulse=resonator_pulse,
        coupling_strength=config["g"] * timescale,
        readout_efficiency=None,
    ).dispersive_approximation(config["chi"] * timescale)
    system_sme = QubitResonatorSystem(
        qubit,
        resonator,
        resonator_pulse=resonator_pulse,
        coupling_strength=config["g"] * timescale,
        readout_efficiency=config["eta"],
    ).dispersive_approximation(config["chi"] * timescale)

    # Build initial states
    if qubit.temperature > 0:
        unit_factor = 2 * np.pi * hbar * 1e9 / Boltzmann
        correctly_initilized_fraction = 1 / (
            1 + np.exp(-unit_factor * qubit.frequency / qubit.temperature)
        )
        falsely_initilized_fraction = 1 - correctly_initilized_fraction
    else:
        correctly_initilized_fraction = 1
        falsely_initilized_fraction = 0

    initial_ground = correctly_initilized_fraction * ket2dm(
        system_lindblad.get_states(0)
    ) + falsely_initilized_fraction * ket2dm(system_lindblad.get_states(1))
    initial_excited = correctly_initilized_fraction * ket2dm(
        system_lindblad.get_states(1)
    ) + falsely_initilized_fraction * ket2dm(system_lindblad.get_states(0))

    initial_states = [initial_ground, initial_excited]

    # Build Experiment
    times = np.arange(0, 1010, 10, dtype=np.float64)

    # Lindblad
    if not os.path.exists(save_path + "_lindblad.pkl") or overwrite:
        print("Running Lindblad Experiment")
        experiment = LindbladExperiment(
            system_lindblad,
            initial_states,
            times,
            expectation_operators=[
                system_lindblad.resonator_I(),
                system_lindblad.resonator_Q(),
                system_lindblad.photon_number_operator(),
            ],
            only_store_final=False,
            store_states=True,
            save_path=save_path + "_lindblad.pkl",
        )

        start_time = time.time()
        results = experiment.run()

        print(f"Time for Lindblad: {time.time() - start_time}")

        if show_plots:
            automatic_analysis(results)
            plt.show()

    # Stochastic Master Equation
    if not os.path.exists(save_path + "_sme.pkl") or overwrite:
        print("Running Stochastic Master Equation Experiment")
        experiment = StochasticMasterEquationExperiment(
            system_sme,
            initial_states,
            times,
            only_store_final=False,
            store_states=False,
            ntraj=ntraj,
            nsubsteps=5,
            save_path=save_path + "_sme.pkl",
        )
        start_time = time.time()
        results = experiment.run()

        print(f"Time for SME Experiment: {time.time() - start_time}")

        del results
        gc.collect()