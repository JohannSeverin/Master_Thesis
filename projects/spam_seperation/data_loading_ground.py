# Point to the xarrays containing the excited and ground state data
path = "/mnt/c/Users/johan/Downloads/dataset_ground.nc"

# Where to save
save_path = (
    "/mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/data/spam_seperation/"
    + "demodulated_ground.nc"
)


# Demodulation parameters
driven_frequency = 7555299739.0e-9
local_oscillator_frequency = 7610000000.0e-9

intermediate_frequency = local_oscillator_frequency - driven_frequency

course_grain = 10  # in nano seconds

# Load the data
chunk_size = 100  # amount of samples to consider at a time (to avoid memory overload)


# Imports
import numpy as np
import xarray as xr

# Open the data
data = xr.open_dataset(path, chunks={"sample": chunk_size})


# rescale the times to ns
times = data.adc_timestamp.values * 1e9


# I_offset = data.readout__final__adc_I__ss.mean().compute().values
# Q_offset = data.readout__final__adc_Q__ss.mean().compute().values


I_offset = (-0.00581589 - 0.00580621) / 2  # Meaned ground state / excited state
Q_offset = (-0.00616023 - 0.00615719) / 2  #


# Define the demodulation function
def demodulate_array(array_I, array_Q, intermediate_frequency, times):
    array_I = array_I - I_offset
    array_Q = array_Q - Q_offset
    zs = array_I + 1j * array_Q
    zs_demodulated = zs * np.exp(-1j * 2 * np.pi * intermediate_frequency * times)
    return zs_demodulated.real, zs_demodulated.imag


demod_I, demod_Q = xr.apply_ufunc(
    lambda I, Q: demodulate_array(I, Q, intermediate_frequency, times),
    data.readout__final__adc_I__ss,
    data.readout__final__adc_Q__ss,
    dask="parallelized",
    output_dtypes=(np.float32, np.float32),
    input_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    output_core_dims=[["adc_timestamp"], ["adc_timestamp"]],
    keep_attrs=True,
)

if course_grain:
    demod_I = demod_I.coarsen(adc_timestamp=course_grain).mean()
    demod_Q = demod_Q.coarsen(adc_timestamp=course_grain).mean()

output_data = xr.Dataset({"I": demod_I, "Q": demod_Q})

from IPython.display import display

display(output_data)

write_job = output_data.to_netcdf(save_path, compute=False)

from dask.diagnostics import ProgressBar

with ProgressBar():
    write_job.compute()
