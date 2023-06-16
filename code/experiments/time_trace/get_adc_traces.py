"""
raw_adc_traces.py: template for acquiring raw ADC traces from inputs 1 and 2
"""

from quam import QuAM
from rich import print
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from qualang_tools.units import unit
from macros import wait_cooldown_time, reset_qubit


import matplotlib

matplotlib.use("Qt5Agg")

##################
# State and QuAM #
##################
u = unit()
experiment = "raw_adc_traces_ground_excited"
debug = True
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

reset_method = "cooldown"  # "cooldown" or "active" or "mixed"


# machine.readout_resonators[0].f_opt = 6.145e9
# machine.readout_resonators[0].readout_amplitude =0.01
machine.readout_resonators[0].wiring.time_of_flight = 380
machine.readout_resonators[0].wiring.smearing = 100
machine.readout_lines[0].length = 5e-6 # 0.4e-6

config = machine.build_config(digital_out=[], qubits=qubit_list, shape=gate_shape)

qmm = QuantumMachinesManager(machine.network.qop_ip)

##############################
# Program-specific variables #
##############################
n_experiments = 1000
cooldown_time = 2000 // 4  # Resonator cooldown time in clock cycles (4ns)
q = 0

###################
# The QUA program #
###################
with program() as raw_trace_prog:
    n = declare(int)
    adc_stream_ground   = [declare_stream(adc_trace = True) for _ in range(len(qubit_list))]
    adc_stream_excited  = [declare_stream(adc_trace = True) for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):

        # Focus only on qubit in focus
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux",
            "single",
            machine.get_flux_bias_point(q, "working_point").value,
        )

        # Loop experiment
        with for_(n, 0, n < n_experiments, n + 1):
            ### Ground State Experiment ### 
            # Reset global phase ? 
            reset_phase(machine.readout_resonators[q].name)

            # Reset qubit by keyword sat above
            reset_qubit(
                machine,
                method= reset_method,
                qubit_index=q,
                threshold=machine.readout_resonators[q].ge_threshold,
                # threshold_q=-0.002,
                max_tries=100,
                cooldown_time=int(10 * machine.qubits[q].t1 * 1e9) // 4,
            )

            align()

            measure(
                "readout", 
                machine.readout_resonators[q].name, 
                adc_stream_ground[i]
            )

            align()


            ### Excited State Experiment ###
            reset_phase(machine.readout_resonators[q].name)

            reset_qubit(
                machine,
                method= reset_method,
                qubit_index=q,
                threshold=machine.readout_resonators[q].ge_threshold,
                # threshold_q=-0.002,
                max_tries=100,
                cooldown_time=int(10 * machine.qubits[q].t1 * 1e9) // 4,
            )

            align()
            
            # X gate to flip 0 to 1
            play("x180", machine.qubits[q].name)
            align()
          
            measure(
                "readout",
                machine.readout_resonators[q].name,
                adc_stream_excited[i],
            )

            align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            adc_stream_ground[i].input1().save_all(f"adc_ground_I_{q}")
            adc_stream_ground[i].input2().save_all(f"adc_ground_Q_{q}")
            adc_stream_excited[i].input1().save_all(f"adc_excited_I_{q}")
            adc_stream_excited[i].input2().save_all(f"adc_excited_Q_{q}")


# Run the program
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
