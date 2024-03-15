import xarray as xr
import numpy as np
from dask.distributed import Client
from pathlib import Path
import functions as f

#datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"

method = "from_vel"
#method = "from_flux"

outpath = f"/nird/home/annals/BLOM_analysis/data/{case}/{method}/"
Path(outpath).mkdir(parents=True, exist_ok=True)

#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')

dx = 2e3             # [m]
dy = 2e3             # [m]
rho = 1e3            # stemmer dette?
f0 = 1e-4            # [s-1]

#print(sorted(os.listdir(datapath+case+"/")))

xchunk = -1
ychunk = -1
sigmachunk = -1
timechunk = 30

data_vars = ["uvel", "vvel", "dz", "pbot", "sealv", "ubaro", "uflx", "vflx"]


# read daily data 
ds = xr.open_mfdataset(datapath+case+"/*hd_*.nc", 
                       #parallel=True,
                       chunks = {"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk},
                       data_vars = data_vars,
                      )

# empty data set for storing processed data
pds = xr.Dataset()

dzx = f.center2xface(ds.dz)
dzy = f.center2yface(ds.dz)

if method == "from_vel":
    u = ds.uvel
    v = ds.vvel
elif method == "from_flux":
    uflx = ds.uflx 
    Ax = dx*ds.dzx
    u = uflx/(rho*Ax)
    
    vflx = ds.vflx 
    Ay = dy*ds.dzy
    v = vflx/(rho*Ay)
    

uc = f.xface2center(u)
vc = f.yface2center(v)


# store values 
ds["u"] = u.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
ds["v"] = v.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
ds["uc"] = uc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
ds["vc"] = vc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
ds["dzx"] = dzx
ds["dzy"] = dzy


# time derivative of depth integrated zonal velocity
pds["dUdt1"] = f.dUdt(ds, method="center last")
pds["dUdt2"] = f.dUdt(ds, method="center first")


# advection of planetary vorticity
pds["fV1"] = f.fV(ds, method="center last") 
pds["fV2"] = f.fV(ds, method="center first") 

# topographic form stress term
pds["phidhdx"] = f.phidhdx(ds, rho, dx)

# momentum flux divergence
pds["dUVdy"] =  f.dUVdy(ds, dy)

# bottom drag
pds["tauxb1"] = f.tauxb(ds, method="center last")
pds["tauxb2"] = f.tauxb(ds, method="center last")

# add mean zonal velocity
pds["ubar"] = f.ubar(ds)

# set values to coordinates. 
f.add_coordinate_values(pds, dx, dy)

# save bottom bathymetry
bath = -f.total_depth(ds)
f.add_coordinate_values(bath, dx, dy)
bath.to_netcdf(outpath+case+"_bathymetry.nc")

# zonal mean
results = pds.mean("x")#.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})

results.attrs = {"Naming convention" : "tailing 1: calculations done on grid faces, then interpolated to grid center \ntailing 2: variables interpolated to grid center before calculations"}


ntime = len(results.time)
n = 30
nmonths = ntime//n
for t in np.arange(nmonths):
    result = results.isel(time=slice(t*n,(t+1)*n))
    print(t)
    result.to_netcdf(outpath+f"{case}_{method}_momentumterms_{t:03}.nc")

#results.to_netcdf(outpath+case+"_momentumterms.nc")
