"""
Measure the IQ blobs of the qubit in the ground and excited states to estimate the readout fidelity
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
# from quam import QuAM
from qm import SimulationConfig
import matplotlib.pyplot as plt
from macros import wait_cooldown_time, reset_qubit
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.analysis.discriminator import two_state_discriminator


import matplotlib

matplotlib.use("Qt5Agg")


##################
# State and QuAM #
##################
experiment  = "Time_Trace"
debug       = True
simulate    = False
qubit_list  = [0]
digital     = []
machine     = QuAM("latest_quam.json")

gate_shape = "drag_cosine"

save_to_latest = False

# Set params for experiment
machine.readout_resonators[0].readout_amplitude = 0.03      # Amplitude of Readout Pulse
machine.readout_lines[0].length                 = 0.1e-6    # Duration of Readout Pulse (s)

# Readout freq - Drive in valley
machine.readout_resonators[0].f_opt             = machine.readout_resonators[0].f_res  -  machine.readout_resonators[0].chi

# The reset method will be cooldown, we want it in equilibrium
reset_method = "cooldown"  # "cooldown" or "active" or "mixed"


# Possible keyword to fix:
# machine.readout_resonators[0].wiring.time_of_flight       = 300
# machine.readout_resonators[0].readout_amplitude           = 0.03
# machine.readout_resonators[0].wiring.time_of_flight       = 500
# machine.readout_resonators[0].wiring.smearing             = 100
# machine.readout_resonators[0].wiring.time_of_flight       = 600
# machine.readout_resonators[0].wiring.smearing             = 127
# machine.readout_lines[0].length                           = 0.1e-6


# Integration weights? 
# time_of_wegihts = 250 # given in 4 ns ? right ? 
# machine.readout_resonators[0].integration_weights[0].cosine   = [1] * time_of_weights
# machine.readout_resonators[0].integration_weights[0].sine     = [0] * time_of_weights
# machine.readout_resonators[0].integration_weights[1].cosine   = [0] * time_of_weights
# machine.readout_resonators[0].integration_weights[1].sine     = [1] * time_of_weights
# machine.readout_resonators[0].integration_weights[2].cosine   = [0] * time_of_weights
# machine.readout_resonators[0].integration_weights[2].sine     = [-1] * time_of_weights


# Build machine
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_runs = 40e3

with program() as iq_blobs:
    n       = [declare(int)     for _ in range(len(qubit_list))]
    n_st    = [declare_stream() for _ in range(len(qubit_list))]
    a       = declare(fixed)
    I_g     = [declare(fixed)   for _ in range(len(qubit_list))]
    Q_g     = [declare(fixed)   for _ in range(len(qubit_list))]
    I_g_st  = [declare_stream() for _ in range(len(qubit_list))]
    Q_g_st  = [declare_stream() for _ in range(len(qubit_list))]
    I_e     = [declare(fixed)   for _ in range(len(qubit_list))]
    Q_e     = [declare(fixed)   for _ in range(len(qubit_list))]
    I_e_st  = [declare_stream() for _ in range(len(qubit_list))]
    Q_e_st  = [declare_stream() for _ in range(len(qubit_list))]

    trace = declare_stream(adc_trace=True)
    # counter = [declare(int) for _ in range(len(qubit_list))]
    # counter_st = [declare_stream() for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        
        set_dc_offset(
            machine.qubits[q].name + "_flux",
            "single",
            machine.get_flux_bias_point(q, "working_point").value,
        )

        with for_(n[i], 0, n[i] < n_runs, n[i] + 1):

            align()
            reset_qubit(
                machine,
                method=reset_method,
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
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_minus_sin", "out2", I_g[i]),
                dual_demod.full("rotated_sin", "out1", "rotated_cos", "out2", Q_g[i]),
            )
            # measure(
            #     "readout",
            #     machine.readout_resonators[q].name,
            #     None,
            #     dual_demod.full(
            #         "cos", "out1", "minus_sin", "out2", I_g[i]
            #     ),
            #     dual_demod.full("sin", "out1", "cos", "out2", Q_g[i]),
            # )

            # measure(
            #     "readout",
            #     machine.readout_resonators[q].name,
            #     trace,
            #     dual_demod.full("optimal_cos", "out1", "optimal_minus_sin", "out2", I_g[i]),
            #     dual_demod.full("optimal_sin", "out1", "optimal_cos", "out2", Q_g[i]),
            # )

            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])
            align()
            reset_qubit(
                machine,
                method=reset_method,
                qubit_index=q,
                threshold=machine.readout_resonators[q].ge_threshold,
                # threshold_q=0.002,
                max_tries=100,
                cooldown_time=int(10 * machine.qubits[q].t1 * 1e9) // 4,
            )

            align()
            play("x180", machine.qubits[q].name)
            align()
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_minus_sin", "out2", I_e[i]),
                dual_demod.full("rotated_sin", "out1", "rotated_cos", "out2", Q_e[i]),
            )
            # measure(
            #     "readout",
            #     machine.readout_resonators[q].name,
            #     None,
            #     dual_demod.full(
            #         "cos", "out1", "minus_sin", "out2", I_e[i]
            #     ),
            #     dual_demod.full("sin", "out1", "cos", "out2", Q_e[i]),
            # )

            # measure(
            #     "readout",
            #     machine.readout_resonators[q].name,
            #     trace,
            #     dual_demod.full("optimal_cos", "out1", "optimal_minus_sin", "out2", I_e[i]),
            #     dual_demod.full("optimal_sin", "out1", "optimal_cos", "out2", Q_e[i]),
            # )

            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])

            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):

            I_g_st[i].save_all(f"Ig{q}")
            Q_g_st[i].save_all(f"Qg{q}")
            I_e_st[i].save_all(f"Ie{q}")
            Q_e_st[i].save_all(f"Qe{q}")
            n_st[i].save(f"iteration{q}")
            # counter_st[i].save_all(f"counter{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iq_blobs)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i, q in enumerate(qubit_list):
        # Live plotting
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(
            job,
            [f"Ig{q}", f"Qg{q}", f"Ie{q}", f"Qe{q}", f"iteration{q}"],
            mode="live",
        )
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_runs:

            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["Ig"] = data[0]
            qubit_data[i]["Qg"] = data[1]
            qubit_data[i]["Ie"] = data[2]
            qubit_data[i]["Qe"] = data[3]
            qubit_data[i]["iteration"] = data[4]

            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_runs, start_time=my_results.start_time)
        # PLot the IQ blobs end derive the readout fidelity
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            qubit_data[i]["Ig"],
            qubit_data[i]["Qg"],
            qubit_data[i]["Ie"],
            qubit_data[i]["Qe"],
            b_print=True,
            b_plot=True,
        )

        kb = 1.380649e-23  # Boltzmann constant
        hbar = 1.054571817e-34  # Reduced Planck constant
        omega = 2 * np.pi * machine.qubits[q].f_01
        temperature = omega * hbar / (np.log(gg / (1 - gg)) * kb)
        print(f"Temperature: {temperature*1e3:.1f} mK")

        machine.readout_resonators[q].readout_fidelity = fidelity
        machine.readout_resonators[q].ge_threshold = threshold
        machine.readout_resonators[q].rotation_angle += angle * 180 / np.pi
        plt.suptitle(f"Qubit {q}")
        figures.append(plt.gcf())
        # plt.show()

    plt.pause(0.1)
    machine.save_results(experiment, figures)
    # if save_to_latest:
    #     machine.save("latest_quam.json")
