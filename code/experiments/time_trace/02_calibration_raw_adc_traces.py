"""
raw_adc_traces.py: template for acquiring raw ADC traces from inputs 1 and 2
"""

from quam import QuAM
from rich import print
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from qualang_tools.units import unit

import matplotlib

matplotlib.use("Qt5Agg")

##################
# State and QuAM #
##################
u = unit()
experiment = "raw_adc_traces"
debug = True
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

# machine.readout_resonators[0].f_opt = 6.145e9
# machine.readout_resonators[0].readout_amplitude =0.01
machine.readout_resonators[0].wiring.time_of_flight = 380
machine.readout_resonators[0].wiring.smearing = 100
machine.readout_lines[0].length = 0.4e-6

config = machine.build_config(digital_out=[], qubits=qubit_list, shape=gate_shape)


qmm = QuantumMachinesManager(machine.network.qop_ip)

##############################
# Program-specific variables #
##############################
n_avg = 1000  # Number of averaging loops
cooldown_time = 2000 // 4  # Resonator cooldown time in clock cycles (4ns)
q = 0
###################
# The QUA program #
###################
with program() as raw_trace_prog:
    n = declare(int)
    adc_st = [declare_stream(adc_trace=True) for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):
        with for_(n, 0, n < n_avg, n + 1):
            reset_phase(machine.readout_resonators[q].name)
            measure("readout", machine.readout_resonators[q].name, adc_st[i])
            wait(cooldown_time, machine.readout_resonators[q].name)

    with stream_processing():
        for i, q in enumerate(qubit_list):
            # Will save average:
            adc_st[i].input1().average().save(f"adc1_{q}")
            adc_st[i].input2().average().save(f"adc2_{q}")
            # Will save only last run:
            adc_st[i].input1().save(f"adc1_single_run_{q}")
            adc_st[i].input2().save(f"adc2_single_run_{q}")


qm = qmm.open_qm(config)
job = qm.execute(raw_trace_prog)
res_handles = job.result_handles
res_handles.wait_for_all_values()
figures = []
for i, q in enumerate(qubit_list):
    adc1 = u.raw2volts(res_handles.get(f"adc1_{q}").fetch_all())
    adc2 = u.raw2volts(res_handles.get(f"adc2_{q}").fetch_all())
    adc1_single_run = u.raw2volts(res_handles.get(f"adc1_single_run_{q}").fetch_all())
    adc2_single_run = u.raw2volts(res_handles.get(f"adc2_single_run_{q}").fetch_all())

    fig = plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    print(adc1_single_run.shape)
    plt.plot(adc1_single_run, label="Input 1")
    plt.plot(adc2_single_run, label="Input 2")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()

    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc1, label="Input 1")
    plt.plot(adc2, label="Input 2")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.suptitle(f"Qubit {q}")
    plt.tight_layout()
    figures.append(fig)
    print(f"Qubit {q}:")
    print(f"\nInput1 mean: {np.mean(adc1)} V\n" f"Input2 mean: {np.mean(adc2)} V")
    # plt.xlim([0, 2000])
    plt.show()
machine.save_results(experiment, figures)


# machine.readout_resonators[0].wiring.time_of_flight += 100
# machine.save("latest_quam.json")