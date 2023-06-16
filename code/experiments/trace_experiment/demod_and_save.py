import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, json, tqdm

# plt.style.use("../../matplotlib_style/standard_plot_style.mplstyle")

data_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/kode/experiments/trace_data/experiment_04_20.json"
save_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/kode/experiments/trace_data/trace_2k_10us.csv"

# Set frequencies for experiment # Could propably be gathered from the json file
driven_frequency = 7555991319.0
local_oscillator_frequency = 7510000000


# Set frequency to work in nano seconds
intermediate_frequency = 1e-9 * (driven_frequency - local_oscillator_frequency)


# Course grained timesteps
course_grain = False
if course_grain:
    course_grain_dt = 10  # in nano seconds

with open(data_path) as file:
    data = json.load(file)

# Load data
ground_I_raw = np.array(
    data["experiment_results"]["Readout signal (ground_state I) raw ss"]["value"]
)
ground_Q_raw = np.array(
    data["experiment_results"]["Readout signal (ground_state Q) raw ss"]["value"]
)
excited_I_raw = np.array(
    data["experiment_results"]["Readout signal (excited_state I) raw ss"]["value"]
)
excited_Q_raw = np.array(
    data["experiment_results"]["Readout signal (excited_state Q) raw ss"]["value"]
)

del data

# Subtract mean
mean_val_I = np.mean([ground_I_raw, excited_I_raw])
mean_val_Q = np.mean([ground_Q_raw, excited_Q_raw])
ground_I_raw = ground_I_raw - mean_val_I
ground_Q_raw = ground_Q_raw - mean_val_Q
excited_I_raw = excited_I_raw - mean_val_I
excited_Q_raw = excited_Q_raw - mean_val_Q

# Convert to complex numbers
z_ground_if = ground_I_raw + 1j * ground_Q_raw
z_excited_if = excited_I_raw + 1j * excited_Q_raw

# Demodulate signal
demodulation = np.exp(
    1j * 2 * np.pi * intermediate_frequency * np.arange(ground_Q_raw.shape[1])
)
z_ground_demod = z_ground_if * demodulation
z_excited_demod = z_excited_if * demodulation

# Convert back to I and Q components
I_ground = z_ground_demod.real
Q_ground = z_ground_demod.imag
I_excited = z_excited_demod.real
Q_excited = z_excited_demod.imag

# Convert to DataFrame with trajectory id, initial_state, time and I and Q components
ground_dataframes = []
for i in tqdm.tqdm(range(I_ground.shape[0])):
    df_at_i = pd.DataFrame(
        {
            "trajectory": i,
            "initial_state": 0,
            "t": np.arange(I_ground.shape[1]),
            "I": I_ground[i],
            "Q": Q_ground[i],
        }
    )
    ground_dataframes.append(df_at_i)

excited_dataframes = []
for i in tqdm.tqdm(range(I_excited.shape[0])):
    df_at_i = pd.DataFrame(
        {
            "trajectory": i + I_ground.shape[0],
            "initial_state": 1,
            "t": np.arange(I_excited.shape[1]),
            "I": I_excited[i],
            "Q": Q_excited[i],
        }
    )
    excited_dataframes.append(df_at_i)


df_ground = pd.concat(ground_dataframes)
df_excited = pd.concat(excited_dataframes)

# Concatenate the two dataframes
df = pd.concat([df_ground, df_excited])

# Course grain the data
if course_grain:

    def course_grain(df, dt):
        df_grouped_by_dt = df.groupby(df["t"] // dt)
        df_summed = df_grouped_by_dt.aggregate(
            {
                "trajectory": "first",
                "initial_state": "first",
                "I": "sum",
                "Q": "sum",
            }
        )
        df_summed["t"] = df_summed.index * dt
        return df_summed

    df = df.groupby(["trajectory"]).apply(course_grain, course_grain_dt)

    # Fix index
    df.index = df.index.droplevel(1)
    df.reset_index(inplace=True, drop=True)


# Save to csv
df.to_csv(save_path, index=False)
