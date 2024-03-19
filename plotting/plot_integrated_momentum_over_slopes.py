from plotting_functions import find_slopeidx, hanning_filter
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"
case = "BLOM_channel_new05_mix1_taupos5"

datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"

ds = xr.open_mfdataset(datapath+f"/{case}_momentumterms_*.nc")

figurepath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

#ds = xr.open_mfdataset(datapath+f"{case}_momentumterms_*.nc")#.isel(y=slice(1,-1))
bath = xr.open_dataarray(datapath+f"{case}_bathymetry.nc")#.isel(y=slice(1,-1))


ds["h"] = bath.mean(dim="x")
lx0, lx1, rx0, rx1 = find_slopeidx(ds.h, epsilon=0.01)


ds_west = ds.isel(y=slice(lx0,lx1))
ds_east = ds.isel(y=slice(rx0,rx1))


dsi_west = ds_west.mean("y")
dsi_east = ds_east.mean("y")


time = dsi_west.time

varname = "ubar"

west = hanning_filter(dsi_west[varname], window_length=180)
east = hanning_filter(dsi_east[varname], window_length=180)

sns.set_theme()
fig, ax = plt.subplots()
ax.plot(west, label=f"{varname} west")
ax.plot(east, label=f"{varname} east")

ax.legend()

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(figurepath+f"{case}_timevariable_integrated_momentumterms.png")
