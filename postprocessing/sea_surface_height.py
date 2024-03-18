import xarray as xr
import numpy as np
from dask.distributed import Client
from pathlib import Path

#datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
#case = "BLOM_channel_new05_mix1_tauneg10"
case = "BLOM_channel_new05_mix1"

outpath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
Path(outpath).mkdir(parents=True, exist_ok=True)

xchunk = -1
ychunk = -1
sigmachunk = -1
timechunk = 30

# read daily data 
ds = xr.open_mfdataset(datapath+case+"/*hd_2034*.nc", 
                       #parallel=True,
                       chunks = {"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk},
                       data_vars = ["sealv"],
                      )

da = ds.sealv

da.to_netcdf(outpath+case+"_eta.nc")