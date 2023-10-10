# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/IQ_threshold_141420"

import json, os

state = json.load(open(os.path.join(path, "state_after.json")))

# Demodulation parameters
driven_frequency = state["readout_resonators[]/f_opt"][0]
local_oscillator_frequency = state["readout_lines[]/lo_freq"][0]

intermediate_frequency = local_oscillator_frequency - driven_frequency

course_grain = 10  # in nano seconds

# Load the data
chunk_size = 100  # amount of samples to consider at a time (to avoid memory overload)


# Imports
import numpy as np
import xarray as xr

# Open the data
data = xr.open_dataset(os.path.join(path, "dataset.nc"), chunks={"sample": chunk_size})

# rescale the times to ns
times = data.adc_timestamp.values


#
I_offset = data.readout__final__adc_I__ss.mean().compute().values
Q_offset = data.readout__final__adc_Q__ss.mean().compute().values


# Define the demodulation function
def demodulate_array(array_I, array_Q, intermediate_frequency, times):
    array_I = array_I - I_offset
    array_Q = array_Q - Q_offset
    zs = array_I + 1j * array_Q
    zs_demodulated = zs * np.exp(1j * 2 * np.pi * intermediate_frequency * times)
    return zs_demodulated.real, zs_demodulated.imag


demod_I, demod_Q = xr.apply_ufunc(
    lambda I, Q: demodulate_array(I, Q, intermediate_frequency, times),
    data.readout__final__adc_I__ss[0].drop(
        ["pulse_amplitude_scaling", "pulse_amplitude_scaling_trans"]
    ),
    data.readout__final__adc_Q__ss[0].drop(
        ["pulse_amplitude_scaling", "pulse_amplitude_scaling_trans"]
    ),
    dask="parallelized",
    output_dtypes=(np.float32, np.float32),
    input_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    output_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    keep_attrs=True,
)

if course_grain:
    demod_I = demod_I.coarsen(adc_timestamp=course_grain).mean()
    demod_Q = demod_Q.coarsen(adc_timestamp=course_grain).mean()

output_data = {"I_ground": demod_I, "Q_ground": demod_Q}

demod_I, demod_Q = xr.apply_ufunc(
    lambda I, Q: demodulate_array(I, Q, intermediate_frequency, times),
    data.readout__final__adc_I__ss.sel(sweep_0=1).drop(
        ["pulse_amplitude_scaling", "pulse_amplitude_scaling_trans"]
    ),
    data.readout__final__adc_Q__ss.sel(sweep_0=1).drop(
        ["pulse_amplitude_scaling", "pulse_amplitude_scaling_trans"]
    ),
    dask="parallelized",
    output_dtypes=(np.float32, np.float32),
    input_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    output_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    keep_attrs=True,
)

if course_grain:
    demod_I = demod_I.coarsen(adc_timestamp=course_grain).mean()
    demod_Q = demod_Q.coarsen(adc_timestamp=course_grain).mean()

output_data.update({"I_excited": demod_I, "Q_excited": demod_Q})
output_dataset = xr.Dataset(output_data)

from IPython.display import display

display(output_dataset)

write_job = output_dataset.to_netcdf(
    os.path.join(path, "demodulated_dataset.nc"), compute=False
)

from dask.diagnostics import ProgressBar

with ProgressBar():
    write_job.compute()
