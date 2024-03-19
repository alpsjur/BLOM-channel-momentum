# Import necessary libraries
import xarray as xr
import numpy as np
import postprocessing_functions as f  # Custom functions for postprocessing

# Define the data path and case study
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"
case = "BLOM_channel_new02_mix1"
outpath = "/nird/home/annals/BLOM_analysis/data/" + case + "/"

# Physical and grid parameters
dx = 2e3  # Grid spacing in the x direction [m]
dy = 2e3  # Grid spacing in the y direction [m]
rho = 1e3  # Density of seawater [kg m^-3]
dt = 1 * 24 * 60 * 60  # Time step [s]

# Dask chunking parameters for optimizing performance
xchunk, ychunk, sigmachunk, depthchunk, timechunk = -1, -1, -1, -1, 30

# Read daily data and select depth range
ds = xr.open_mfdataset(datapath + case + "/*hd_*.nc", chunks={"x": xchunk, "y": ychunk, "sigma": sigmachunk, "time": timechunk})

# Initialize an empty dataset for processed variables
pp = xr.Dataset()

# Process horizontal velocities: converting face velocities to center velocities
u, v = ds.uvel, ds.vvel
uc, vc = f.xface2center(u), f.yface2center(v)
pp["uc"], pp["vc"] = uc, vc
pp.uc.attrs = {"long_name": "Horizontal x-velocity at grid center", "units": "m s^-1"}
pp.vc.attrs = {"long_name": "Horizontal y-velocity at grid center", "units": "m s^-1"}

# Pad u in the reentrance direction and calculate its derivative
upad = f.pad_reentrance(uc)
dudx = upad.differentiate("x").isel(x=slice(1, -1)) / dx
dvdy = pp.vc.differentiate("y") / dy

# Use continuity equation to calculate vertical velocity's derivative
dwdz = -(dudx + dvdy)

# Calculate vertical distance (dz) between depth bounds
dz = ds.depth_bnds.isel(bounds=1) - ds.depth_bnds.isel(bounds=0)

# Calculate surface vertical velocity from sea level changes
ws = ds.sealv.differentiate("time", datetime_unit="s")

# Integrate vertical velocity's derivative from the surface downward
wc1 = ws + (-dwdz * dz).cumsum("depth")
pp["wc1"] = wc1
pp.wc1.attrs = {"long_name": "Vertical velocity calculated from continuity and change in sea surface elevation", "units": "m s^-1"}

# Calculate vertical velocities from vertical mass fluxes
A = dx * dy
wflx = ds.wflxlvl
wc2 = wflx / (rho * A)
pp["wc2"] = wc2
pp.wc2.attrs = {"long_name": "Vertical velocity calculated from vertical mass flux", "units": "m s^-1"}

# Copy sea level to processed dataset
pp["eta"] = ds.sealv

# Calculate time-mean values for all variables
mean = pp.mean("time")
attrs = ["Time-mean sea level", "Horizontal time-mean x-velocity at grid center", "Horizontal time-mean y-velocity at grid center", 
         "Vertical time-mean velocity at grid center calculated from continuity and change in sea surface elevation", 
         "Vertical time-mean velocity at grid center calculated from vertical mass flux"]
units = ["m", "m s^-1", "m s^-1", "m s^-1", "m s^-1"]
for var, attr, unit in zip(["eta", "uc", "vc", "wc1", "wc2"], attrs, units):
    mean[var].attrs = {"long_name": attr, "units": unit}

# Save the mean values to a NetCDF file
mean.to_netcdf(outpath + case + "_mean_velocities_lvl.nc")

# Calculate standard deviation for all variables
std = pp.std("time")
attrs = ["Sea level std", "Horizontal x-velocity std at grid center", "Horizontal y-velocity std at grid center", 
         "Vertical velocity std at grid center calculated from continuity and change in sea surface elevation", 
         "Vertical velocity std at grid center calculated from vertical mass flux"]
for var, attr, unit in zip(["eta", "uc", "vc", "wc1", "wc2"], attrs, units):
    std[var].attrs = {"long_name": attr, "units": unit}
    
# Save the standard deviation to a NetCDF file
std.to_netcdf(outpath + case + "_std_velocities_lvl.nc")