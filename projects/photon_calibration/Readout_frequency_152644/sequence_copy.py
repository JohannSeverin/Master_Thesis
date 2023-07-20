""" Single pulse sequence

This file contains the single pulse sequence used for multiple experiments.

Made by: Malthe A. M. Nielsen (vpq602, malthe.asmus.nielsen@nbi.ku.dk)
Last updated: 2023-05-19
"""
from opx_control._clean_version.sequences._sequence_abc import SequenceABC
from qm.qua import *


class Sequence(SequenceABC):
    def __init__(self, machine, **kwargs):
        super().__init__(machine, **kwargs)

    def play(self, qua_readout):
        new_pulse_amplitude = declare(float)
        assign(
            new_pulse_amplitude,
            self.parameters["pulse_amplitude_scaling"] * self.parameters["pulse_amplitude_scaling_shift"],
        )
        # Set DC offset --------------------------------------------------------- #
        self.set_dc_offset()

        # Reset qubit ----------------------------------------------------------- #
        align()
        update_frequency(self.qubit.name, self.get_qubit_IF(self.qubit))
        update_frequency(self.resonator.name, self.get_resonator_IF(self.qubit))
        self.reset_qubit()

        # Play pulse ------------------------------------------------------------ #
        align()
        update_frequency(self.qubit.name, self.parameters["pulse_frequency_if"])

        play(
            self.parameters["gate"] * amp(new_pulse_amplitude),
            self.qubit.name,
            duration=self.parameters["pulse_duration_cycles"],
        )

        # Readout --------------------------------------------------------------- #
        self.delay(self.parameters["readout_delay_cycles"])
        update_frequency(self.resonator.name, self.parameters["readout_frequency_if"])
        self.tomography_axis()

        align()
        self.measurment(qua_readout)
