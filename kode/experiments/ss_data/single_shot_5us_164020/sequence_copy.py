from qm.qua import *
from opx_control.sequences import routines
from opx_control.utilities import qua_support


class Sequence:
    def __init__(self, machine, params={}, **kwargs):
        self.machine = machine
        self.params = {
            "amp_scaling": params.get("amp_scaling", 1),
            "pulse_freq_if": params.get("pulse_int_freq", machine.get_qubit_IF(0)),
            "readout_delay_cycles": params.get("readout_delay_cycles", 0),
            "readout_freq_if": params.get("readout_int_freq", machine.get_readout_IF(0)),
            "readout_amp_scaling": params.get("readout_amp_scaling", 1),
        }
        self.qubit_name = machine.qubits[0].name
        self.reset_method = kwargs.get("reset_method", "cooldown")

    def transform_param(self, param):
        param = param.copy()
        if param.name == "amplitude":
            param.name = "amp_scaling"
            max_amp = param.abs.max()
            if max_amp > self.machine.qubits[0].driving.drag_cosine.angle2volt.deg180:
                self.machine.qubits[0].driving.drag_cosine.angle2volt.deg180 = max_amp
            param.value = param.value / self.machine.qubits[0].driving.drag_cosine.angle2volt.deg180

        if param.name == "readout_delay":
            param = qua_support.seconds_to_clock_cycles(param)

        if param.name == "pulse_freq":
            param.name = "pulse_freq_if"
            param = (param - self.machine.drive_lines[0].lo_freq).astype(int)

        if param.name == "readout_freq":
            param.name = "readout_freq_if"
            param = (param - self.machine.readout_lines[0].lo_freq).astype(int)

        if param.name == "amp_scaling":
            param.astype("float")

        if param.name == "readout_duration":
            self.machine.readout_lines[0].length = param.abs.max()

        if param.name == "readout_amplitude":
            param.name == "readout_amp_scaling"
            max_amp = param.abs.max()
            if max_amp > self.machine.readout_resonators[0].readout_amplitude:
                self.machine.readout_resonators[0].readout_amplitude = max_amp
            param.value = param.value / self.machine.readout_resonators[0].readout_amplitude

        return param

    def play(self, qua_readout):
        align()
        routines.reset_qubit(self.machine, self.reset_method, qubit_index=0)

        align()
        update_frequency(self.qubit_name, self.params["pulse_freq_if"])
        play("x180" * amp(self.params["amp_scaling"]), self.qubit_name)

        if (
            qua_support.is_qua(self.params["readout_delay_cycles"]) or self.params["readout_delay_cycles"] > 0
        ):  # This check should be built into wait function
            wait(self.params["readout_delay_cycles"], self.qubit_name)

        align()
        update_frequency(self.machine.readout_resonators[0].name, self.params["readout_freq_if"])
        routines.measurement(
            self.machine,
            name="final",
            qubit_index=0,
            qua_readout=qua_readout,
            amp_scaling=self.params["readout_amp_scaling"],
        )
