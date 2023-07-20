# Point to the xarrays containing the excited and ground state data
path = "/mnt/d/Data/readout test_160751/dataset.nc"


# Demodulation parameters
driven_frequency = 7555025786.082029e-9
local_oscillator_frequency = 7610000000e-9

intermediate_frequency = local_oscillator_frequency - driven_frequency

course_grain = 10  # in nano seconds

# Imports
import numpy as np
import xarray as xr

# Open the data
data = xr.open_dataset(path)

# rescale the times to ns
times = data.adc_timestamp.values * 1e9

I_offset_ground = data.readout__ground_state__adc_I__ss.mean().compute().values
Q_offset_ground = data.readout__ground_state__adc_Q__ss.mean().compute().values

I_offset_excited = data.readout__excited_state__adc_I__ss.mean().compute().values
Q_offset_excited = data.readout__excited_state__adc_Q__ss.mean().compute().values

I_offset = (
    I_offset_ground + I_offset_excited
) / 2  # Meaned ground state / excited state
Q_offset = (Q_offset_ground + Q_offset_excited) / 2  #


# Define the demodulation function
def demodulate_array(array_I, array_Q, intermediate_frequency, times):
    array_I = array_I - I_offset
    array_Q = array_Q - Q_offset
    zs = array_I + 1j * array_Q
    zs_demodulated = zs * np.exp(-1j * 2 * np.pi * intermediate_frequency * times)
    return zs_demodulated.real, zs_demodulated.imag


demod_ground_I, demod_ground_Q = xr.apply_ufunc(
    lambda I, Q: demodulate_array(I, Q, intermediate_frequency, times),
    data.readout__ground_state__adc_I__ss,
    data.readout__ground_state__adc_Q__ss,
    dask="parallelized",
    output_dtypes=(np.float32, np.float32),
    input_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    output_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    keep_attrs=True,
)

demod_excited_I, demod_excited_Q = xr.apply_ufunc(
    lambda I, Q: demodulate_array(I, Q, intermediate_frequency, times),
    data.readout__excited_state__adc_I__ss,
    data.readout__excited_state__adc_Q__ss,
    dask="parallelized",
    output_dtypes=(np.float32, np.float32),
    input_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    output_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    keep_attrs=True,
)

## CHECK
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

ax = ax.flatten()

ax[0].set(title="Full trajectories", xlabel="I", ylabel="Q")

duration = 5000
for i in range(demod_ground_I.sample.size):
    ax[0].plot(
        demod_ground_I.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        demod_ground_Q.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        color="C0",
        ls="-",
        alpha=0.25,
    )
    ax[0].plot(
        demod_excited_I.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        demod_excited_Q.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        color="C1",
        ls="-",
        alpha=0.25,
    )


ax[1].set(title="First µs Trajectories", xlabel="I", ylabel="Q")
duration = 1000
for i in range(demod_ground_I.sample.size):
    ax[1].plot(
        demod_ground_I.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        demod_ground_Q.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        color="C0",
        ls="-",
        alpha=0.25,
    )
    ax[1].plot(
        demod_excited_I.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        demod_excited_Q.isel({"sample": i})[:duration].cumsum("adc_timestamp"),
        color="C1",
        ls="-",
        alpha=0.25,
    )


ax[2].set(title="Cumulative Meaned Trajectories", xlabel="I", ylabel="Q")
duration = 5000
warm_up = 100
for i in range(demod_ground_I.sample.size):
    ax[2].plot(
        (demod_ground_I.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        (demod_ground_Q.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        color="C0",
        ls="-",
        alpha=0.25,
    )
    ax[2].plot(
        (demod_excited_I.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        (demod_excited_Q.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        color="C1",
        ls="-",
        alpha=0.25,
    )

ax[3].set(title="Cumulative Meaned Trajectories First µs", xlabel="I", ylabel="Q")
duration = 1000
warm_up = 100
for i in range(demod_ground_I.sample.size):
    ax[3].plot(
        (demod_ground_I.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        (demod_ground_Q.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        color="C0",
        ls="-",
        alpha=0.25,
    )
    ax[3].plot(
        (demod_excited_I.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        (demod_excited_Q.isel({"sample": i}).cumsum("adc_timestamp") / (times + 1))[
            warm_up:duration
        ],
        color="C1",
        ls="-",
        alpha=0.25,
    )

ax[1].plot([], [], color="C0", label="Ground", ls="-")
ax[1].plot([], [], color="C1", label="Excited", ls="-")

ax[1].legend(loc="upper right")
