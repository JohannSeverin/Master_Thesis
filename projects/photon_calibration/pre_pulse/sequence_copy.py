""" Single pulse sequence

This file contains the single pulse sequence used for multiple experiments.

Made by: Malthe A. M. Nielsen (vpq602, malthe.asmus.nielsen@nbi.ku.dk)
Last updated: 2023-05-19
"""
from opx_control._clean_version.sequences._sequence_abc import SequenceABC
from qm.qua import *

from dataanalyzer import Valueclass


class Sequence(SequenceABC):
    def __init__(self, machine, **kwargs):
        super().__init__(machine, **kwargs)

        # Set default parameters ------------------------------------------------ #
        self._parameters.update({"pulse_duration_cycles": int(self.qubit.driving.drag_cosine.length / 4e-9)})

    def transform_parameter(self, parameter: Valueclass) -> Valueclass:
        parameterCopy = parameter.copy()

        if parameter.tag == "resonator_drive_amplitude":  # Readout amplitude
            parameterCopy.tag = "resonator_drive_amplitude_scaling"

            readout_amplitude_copy = np.copy(self.resonator.readout_amplitude)
            self.resonator.readout_amplitude = max(self.resonator.readout_amplitude, parameter.abs.max())
            self.readout_scaling = self.resonator.readout_amplitude / readout_amplitude_copy

            parameterCopy.value = parameter.value / self.resonator.readout_amplitude

            return parameterCopy

        if parameter.tag == "resonator_drive_duration":  # Pulse duration
            parameterCopy = self.seconds_to_clock_cycles(parameterCopy)
            return parameterCopy

        return super().transform_parameter(parameter)

    def play(self, qua_readout):
        qubit_wait = self.parameters["resonator_drive_duration_cycles"] - self.parameters["pulse_duration_cycles"]

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
            "const" * amp(self.parameters["resonator_drive_amplitude_scaling"]),
            self.resonator.name,
            duration=self.parameters["resonator_drive_duration_cycles"],
        )

        wait(qubit_wait, self.qubit.name)
        play(
            self.parameters["gate"] * amp(self.parameters["pulse_amplitude_scaling"]),
            self.qubit.name,
            duration=self.parameters["pulse_duration_cycles"],
        )

        # Readout --------------------------------------------------------------- #
        self.delay(self.parameters["readout_delay_cycles"])
        update_frequency(self.resonator.name, self.parameters["readout_frequency_if"])
        self.tomography_axis()

        align()
        self.measurment(qua_readout, amplitude_scaling=self.readout_scaling)
