# Import necessary libraries
import xarray as xr  # For working with labeled multi-dimensional arrays
import numpy as np  # For numerical operations
from dask.distributed import Client  # For parallel computing
from pathlib import Path  # For filesystem path operations

# Define the path to the input data. Uncomment the path that you need.
datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
#datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

# Define the case study. Uncomment the case that you're working with.
#case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new05_mix1_taupos5"
#case = "BLOM_channel_new05_mix1_tauneg10"
#case = "BLOM_channel_new02_mix1"
case = "BLOM_channel_new02_mix1_tauneg10"

# Define the output path where the processed data will be saved
outpath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
# Ensure the output directory exists, create it if it doesn't
Path(outpath).mkdir(parents=True, exist_ok=True)

# Define chunk sizes for dask. Set to -1 for automatic chunking by dask.
xchunk = -1
ychunk = -1
sigmachunk = -1
timechunk = 30  # Load 30 time steps at a time for processing

# Load the dataset using xarray's open_mfdataset function. This function allows for
# reading multiple files as a single dataset.
# The `chunks` parameter enables lazy loading with dask, improving performance.
# The `data_vars` parameter specifies which variables to load. In this case, only 'sealv'.
ds = xr.open_mfdataset(
    datapath + case + "/*hd_*.nc",
    chunks={"x": xchunk, "y": ychunk, "sigma": sigmachunk, "time": timechunk},
    data_vars=["sealv"],
)

# Extract the 'sealv' variable from the dataset
da = ds.sealv

# Save the extracted data to a NetCDF file in the specified output path.
# The filename includes the case name for easy identification.
da.to_netcdf(outpath + case + "_eta.nc")
