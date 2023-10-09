""" Single pulse sequence

This file contains the single pulse sequence used for multiple experiments.

Made by: Malthe A. M. Nielsen (vpq602, malthe.asmus.nielsen@nbi.ku.dk)
Last updated: 2023-05-19
"""
from opx_control.sequences._sequence_abc import SequenceABC
from qm.qua import *

from dataanalyzer import Valueclass


class Sequence(SequenceABC):
    def __init__(self, machine, **kwargs):
        super().__init__(machine, **kwargs)

        # Set default parameters ------------------------------------------------ #
        self._parameters.update({"pulse_duration_cycles": int(self.qubit.driving.drag_cosine.length / 4e-9)})
        self.readout_only = kwargs.get("readout_only", False)

    def play(self, qua_readout):
        self.qubit_wait = (
            int(self.machine.readout_lines[0].flattop_plateau / 4e-9) - self._parameters["pulse_duration_cycles"]
        )

        # Set DC offset --------------------------------------------------------- #
        self.set_dc_offset()

        # Reset qubit ----------------------------------------------------------- #
        align()
        update_frequency(self.qubit.name, self.get_qubit_IF(self.qubit))
        update_frequency(self.resonator.name, self.get_resonator_IF(self.qubit))
        self.reset_qubit()

        # Play pulse ------------------------------------------------------------ #
        if not self.readout_only:
            align()
            update_frequency(self.qubit.name, self.parameters["pulse_frequency_if"])
            align()
            align()
            play("readout_flattop" * amp(self.parameters["resonator_drive_amplitude_scaling"]), self.resonator.name)

            wait(self.qubit_wait, self.qubit.name)
            play(
                self.parameters["gate"] * amp(self.parameters["pulse_amplitude_scaling"]),
                self.qubit.name,
                duration=self.parameters["pulse_duration_cycles"],
            )

            align()
            self.measurement(qua_readout)
        else:
            align()
            play(
                self.parameters["gate"] * amp(self.parameters["pulse_amplitude_scaling"]),
                self.qubit.name,
                duration=self.parameters["pulse_duration_cycles"],
            )
            align()
            self.measurement(
                qua_readout,
                pulse="readout_flattop" * amp(self.parameters["resonator_drive_amplitude_scaling"]),
            )


# class Sequence(SequenceABC):
#     def __init__(self, machine, **kwargs):
#         super().__init__(machine, **kwargs)

#         # Set default parameters ------------------------------------------------ #
#         self._parameters.update(
#             {
#                 # "pulse_duration_cycles": int(self.qubit.driving.drag_cosine.length / 4e-9),
#                 "readout_lines[0].length": self.machine.readout_lines[0].length,
#             }
#         )

#     # def transform_parameter(self, parameter: Valueclass) -> Valueclass:
#     #     parameterCopy = parameter.copy()

#     #     if parameter.tag == "readout_lines[0].length":
#     #         print("Transforming readout_lines[0].length")
#     #         self.qubit_wait = int(parameterCopy.value / 2 / 4e-9)

#     #     return super().transform_parameter(parameter)

#     def play(self, qua_readout):
#         self.qubit_wait = int(self.parameters["readout_lines[0].length"] / 2 / 4e-9)

#         # Set DC offset --------------------------------------------------------- #
#         self.set_dc_offset()

#         # Reset qubit ----------------------------------------------------------- #
#         align()
#         update_frequency(self.qubit.name, self.get_qubit_IF(self.qubit))
#         update_frequency(self.resonator.name, self.get_resonator_IF(self.qubit))
#         self.reset_qubit()

#         # Play pulse ------------------------------------------------------------ #
#         align()
#         update_frequency(self.qubit.name, self.parameters["pulse_frequency_if"])
#         align()

#         self.measurment(qua_readout)

#         wait(self.qubit_wait, self.qubit.name)
#         play(
#             self.parameters["gate"] * amp(self.parameters["pulse_amplitude_scaling"]),
#             self.qubit.name,
#             # duration=self.parameters["pulse_duration_cycles"],
#         )

#         align()
