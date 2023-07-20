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

    def play(self, qua_readout):
        qubit_wait = int(self.machine.readout_lines[0].length / 2 / 4e-9)

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
        align()

        self.measurment(qua_readout)

        wait(qubit_wait, self.qubit.name)
        play(
            self.parameters["gate"] * amp(self.parameters["pulse_amplitude_scaling"]),
            self.qubit.name,
            duration=self.parameters["pulse_duration_cycles"],
        )

        align()
