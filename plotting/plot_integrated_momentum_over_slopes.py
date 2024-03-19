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


bath = xr.open_dataarray(datapath+f"{case}_bathymetry.nc")#.isel(y=slice(1,-1))


ds["h"] = bath.mean(dim="x")
lx0, lx1, rx0, rx1 = find_slopeidx(ds.h, epsilon=0.01)


ds_west = ds.isel(y=slice(lx0,lx1))
ds_east = ds.isel(y=slice(rx0,rx1))


dsm_west = ds_west.mean("y")
dsm_east = ds_east.mean("y")

dsstd_west = ds_west.std("y")
dsstd_east = ds_east.std("y")


time = dsm_west.time
window = 180

sns.set_theme()
fig, [axw, axe] = plt.subplots(2,1)

vars = ["dUdt", "tauxs", "dUVdy", "phidhdx", "fV_flx", "tauxb_1"]
signs = ["+", "+", "-", "-", "+", "-"]

east_res = 0
west_res = 0
for varname, sign in zip(vars,signs):
    east = dsm_east[varname]
    west = dsm_west[varname]
    if sign == "-":
        east *= -1
        west *= -1
        
    east_res = east_res + east
    west_res = west_res + west

    
    east_filter = hanning_filter(east, window_length=window)
    west_filter = hanning_filter(west, window_length=window)


    axw.plot(west_filter, label=f"{sign}{varname}")
    axe.plot(east_filter, label=f"{sign}{varname}")

east_filter = hanning_filter(east_res, window_length=window)
west_filter = hanning_filter(west_res, window_length=window)

axw.plot(west_filter, label=f"residual")
axe.plot(east_filter, label=f"residual")


axw.legend()

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(figurepath+f"{case}_timevariable_integrated_momentumterms.png")
