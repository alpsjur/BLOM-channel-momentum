# %%
import xarray as xr
import matplotlib as plt
# %%

datapath = "/home/alsjur/PhD/Data/BLOM_channel/test/"
filename = "BLOM_channel_new30_weakerwind_hd_2033.09.nc"

ds = xr.open_dataset(datapath+filename)
