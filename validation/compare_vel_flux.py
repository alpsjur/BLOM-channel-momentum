import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"
figpath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

path = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"

ds_flux = xr.open_mfdataset(path+f"from_flux/{case}_from_flux_momentumterms_*.nc").mean("time")
ds_vel = xr.open_mfdataset(path+f"from_vel/{case}_from_vel_momentumterms_*.nc").mean("time")


fig, axes = plt.subplots(3,3, figsize=(18,14), sharex=True)
for varname, ax in zip(ds_flux.data_vars, axes.flatten()):
    da_flux = ds_flux[varname]
    da_vel = ds_vel[varname]
    
    ax.plot(da_flux, label="flux")
    ax.plot(da_vel, label="velocity")
    ax.set_title(varname)
    
axes[0,0].legend()
fig.savefig(figpath+case+"_compare_vel_flux.png")