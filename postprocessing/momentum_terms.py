# Import necessary libraries
import xarray as xr
import numpy as np
from pathlib import Path
import postprocessing_functions as f  # Custom postprocessing functions

# Define data paths and case names
#datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
#case = "BLOM_channel_new05_mix1"
case = "BLOM_channel_new02_mix1"

# Option to save bathymetry data
save_bath = False

# Output path configuration
outpath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
Path(outpath).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# Physical and simulation parameters
dx = 2e3  # Grid spacing in the x direction [m]
dy = 2e3  # Grid spacing in the y direction [m]
rho = 1e3  # Density of seawater [kg/m^3]
f0 = 1e-4  # Coriolis parameter [s^-1]
tauxs = 0.05/rho  # Normalized surface stress [N/m^2 / kg/m^3]

# Chunk sizes for dask array operations (x, y, sigma, time)
xchunk, ychunk, sigmachunk, timechunk = -1, -1, -1, 30

# Variables to be read from the dataset
data_vars = ["uvel", "vvel", "dz", "pbot", "sealv", "ubaro", "uflx", "vflx"]

# Reading daily data with specified chunks for efficient computation
ds = xr.open_mfdataset(datapath + case + "/*hd_*.nc", chunks={"x": xchunk, "y": ychunk, "sigma": sigmachunk, "time": timechunk}, data_vars=data_vars)

# Preparation for data processing
# Initialize empty datasets for storing processed data
pds, dsvel, dsflx = xr.Dataset(), xr.Dataset(), xr.Dataset()

# Calculate dz on x and y faces
dzx, dzy = f.center2xface(ds.dz), f.center2yface(ds.dz)
ds["dzx"], ds["dzy"], dsvel["dz"], dsvel["dzx"], dsvel["dzy"], dsflx["dz"], dsflx["dzx"], dsflx["dzy"] = dzx, dzy, ds.dz, dzx, dzy, ds.dz, dzx, dzy

# Process velocities 
uvel, vvel = ds.uvel, ds.vvel                                              # Velocities on z levels
uflx, vflx = ds.uflx / (rho * dx * ds.dzx), ds.vflx / (rho * dy * ds.dzy)  # Compute velocities from mass fluxes

# Convert face values to center values
uvelc, vvelc = f.xface2center(uvel), f.yface2center(vvel)
uflxc, vflxc = f.xface2center(uflx), f.yface2center(vflx)

# Store processed velocity data
dsvel["u"], dsvel["v"], dsvel["uc"], dsvel["vc"] = [v.chunk({"x": xchunk, "y": ychunk, "sigma": sigmachunk, "time": timechunk}) for v in [uvel, vvel, uvelc, vvelc]]
dsflx["u"], dsflx["v"], dsflx["uc"], dsflx["vc"] = [v.chunk({"x": xchunk, "y": ychunk, "sigma": sigmachunk, "time": timechunk}) for v in [uflx, vflx, uflxc, vflxc]]

# Calculate additional momentum terms
pds["dUdt"] = f.dUdt(dsvel, method="center last")  # Time derivative of depth integrated zonal velocity
pds["fV_flx"], pds["fV_vel"] = f.fV(dsflx, method="center last"), f.fV(dsvel, method="center last")  # Advection of planetary vorticity
pds["phidhdx"] = f.phidhdx(ds, rho, dx)  # Topographic form stress term
pds["dUVdy"] = f.dUVdy(dsvel, dy)  # Momentum flux divergence

# Bottom drag calculations with two different alpha parameters
pds["tauxb_1"], pds["tauxb_2"] = f.tauxb(dsvel, alpha=1, method="center first"), f.tauxb(dsvel, alpha=0.5, method="center first")

# Add mean zonal velocity
pds["ubar"] = f.ubar(ds)

# Add coordinate values for easier interpretation
f.add_coordinate_values(pds, dx, dy)

# Optional: Save bottom bathymetry data
if save_bath:
    bath = -f.total_depth(ds)
    f.add_coordinate_values(bath, dx, dy)
    bath.to_netcdf(outpath + case + "_bathymetry.nc")

# Calculate and save zonal mean of the processed data
results = pds.mean("x")
results["tauxs"] = (["time", "y"], np.ones_like(results.ubar)*tauxs)
results.attrs = {"Naming convention": "tailing 1: calculations done on grid faces, then interpolated to grid center\n trailing 2: variables interpolated to grid center before calculations"}

# Save monthly averaged data
ntime, n = len(results.time), 30
nmonths = ntime // n
for t in np.arange(nmonths):
    result = results.isel(time=slice(t * n, (t + 1) * n))
    print(t)
    result.to_netcdf(outpath + f"{case}_momentumterms_{t:03}.nc")
