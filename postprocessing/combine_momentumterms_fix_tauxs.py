import xarray as xr
import numpy as np

case = "BLOM_channel_new05_mix1"

# Define data and figure paths
datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"

ds = xr.open_mfdataset(datapath+f"{case}_momentumterms_*.nc")
#ds["tauxs"] = -ds["tauxs"]

ds.to_netcdf(datapath+f"{case}_momentumterms.nc")
