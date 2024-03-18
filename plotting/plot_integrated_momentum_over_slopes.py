from functions import find_slopeloc
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"

datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"

ds_flux = xr.open_mfdataset(datapath+f"from_flux/{case}_from_flux_momentumterms_*.nc")
ds_vel = xr.open_mfdataset(datapath+f"from_vel/{case}_from_vel_momentumterms_*.nc")

figurepath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

#ds = xr.open_mfdataset(datapath+f"{case}_momentumterms_*.nc")#.isel(y=slice(1,-1))
bath = xr.open_dataarray(datapath+f"{case}_bathymetry.nc")#.isel(y=slice(1,-1))


ds_vel["h"] = bath.mean(dim="x")
lx0, lx1, rx0, rx1 = find_slopeloc(ds_vel.h, epsilon=0.01)

