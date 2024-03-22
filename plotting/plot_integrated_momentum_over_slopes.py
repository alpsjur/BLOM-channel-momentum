from plotting_functions import find_slopeidx, hanning_filter
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"
#case = "BLOM_channel_new05_mix1_taupos5"

datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"

ds = xr.open_mfdataset(datapath+f"/{case}_momentumterms.nc")

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
day = np.arange(len(time))

window= 100

sns.set_theme()
fig, ax = plt.subplots(1,1, sharex=True, figsize=(12,8))

vars = ["dUdt", "tauxs", "dUVdy", "phidhdx", "fV_flx", "tauxb_1"]
signs = ["+", "+", "-", "-", "+", "-"]
colors = mpl.colormaps['tab10'].resampled(len(vars)+1).colors

east_res = 0
west_res = 0
for varname, sign, color in zip(vars,signs, colors):
    east = dsm_east[varname].copy()
    west = dsm_west[varname].copy()
    if sign == "-":
        east *= -1
        west *= -1
        
    east_res = east_res + east
    west_res = west_res + west

    
    east_filter = hanning_filter(east, window_length=window)
    west_filter = hanning_filter(west, window_length=window)


    ax.plot(day, west_filter, label=f"{sign}{varname}", color=color)
    ax.plot(day, east_filter, ls="--", color=color)


dsm_east["residual"] = east_res
dsm_west["residual"] = west_res

east_filter = hanning_filter(east_res, window_length=window)
west_filter = hanning_filter(west_res, window_length=window)

ax.plot(day, west_filter, label=f"residual", color=colors[-1])
ax.plot(day, east_filter, ls="--", color=colors[-1])


ax.plot([None], [None], label="west", color="gray")
ax.plot([None], [None], ls="--", label="east", color="gray")

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(figurepath+f"integrated_slope_momentum/{case}_timevariable_integrated_momentumterms.png")

# Convert the xarray Dataset to a pandas DataFrame
# Note: This step may require adjustments based on the structure of your dataset
dfw = dsm_west.to_dataframe().reset_index().drop(['time', 'h', "tauxs"], axis=1)
dfe = dsm_east.to_dataframe().reset_index().drop(['time', 'h', "tauxs"], axis=1)

# Compute the correlation matrix
corrw = dfw.corr().astype(float)
corre = dfe.corr().astype(float)

# Generate a mask for the upper triangle
maskw = np.triu(np.ones_like(corrw, dtype=bool))
maske = np.triu(np.ones_like(corre, dtype=bool))

# Set up the matplotlib figure
sns.set_style("white")
fig, [axw, axe] = plt.subplots(1,2, figsize=(16, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrw, 
            mask=maskw, 
            vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar=False,
            ax=axw, cmap="vlag", annot=True, fmt=".2f"
            )
sns.heatmap(corre, 
            mask=maske, 
            vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar=False,
            ax=axe, cmap="vlag", annot=True, fmt=".2f"
            )

axw.set_title("West slope")
axe.set_title("East slope")

fig.savefig(figurepath+f"validation/{case}_ts_correlation_matrix.png")