import xarray as xr
import numpy as np
from dask.distributed import Client
from pathlib import Path
import postprocessing_functions as f

#datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"

save_bath = False

outpath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
Path(outpath).mkdir(parents=True, exist_ok=True)

#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')

dx = 2e3             # [m]
dy = 2e3             # [m]
rho = 1e3            # stemmer dette?
f0 = 1e-4            # [s-1]
tauxs = 0.05/rho

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


# empty data sets for storing processed data
pds = xr.Dataset()
dsvel = xr.Dataset()
dsflx = xr.Dataset()

dzx = f.center2xface(ds.dz)
dzy = f.center2yface(ds.dz)
ds["dzx"] = dzx
ds["dzy"] = dzy

dsvel["dz"] = ds.dz
dsvel["dzx"] = dzx
dsvel["dzy"] = dzy

dsflx["dz"] = ds.dz
dsflx["dzx"] = dzx
dsflx["dzy"] = dzy


uvel = ds.uvel
vvel = ds.vvel

u_flx = ds.uflx 
Ax = dx*ds.dzx
uflx = u_flx/(rho*Ax)
    
v_flx = ds.vflx 
Ay = dy*ds.dzy
vflx = v_flx/(rho*Ay)
    

uvelc = f.xface2center(uvel)
vvelc = f.yface2center(vvel)

uflxc = f.xface2center(uflx)
vflxc = f.yface2center(vflx)


# store values 
dsvel["u"] = uvel.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsvel["v"] = vvel.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsvel["uc"] = uvelc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsvel["vc"] = vvelc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})

dsflx["u"] = uflx.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsflx["v"] = vflx.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsflx["uc"] = uflxc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
dsflx["vc"] = vflxc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})


# time derivative of depth integrated zonal velocity
pds["dUdt"] = f.dUdt(dsvel, method="center last")
pds.dUdt.attrs = {"Method":"Calculated from uvel variable."}

# advection of planetary vorticity
pds["fV_flx"] = f.fV(dsflx, method="center last") 
pds["fV_vel"] = f.fV(dsvel, method="center last") 
pds.fV_flx.attrs = {"Method":"Calculated from vflx variable."}
pds.fV_vel.attrs = {"Method":"Calculated from vvel variable."}

# topographic form stress term
pds["phidhdx"] = f.phidhdx(ds, rho, dx)

# momentum flux divergence
pds["dUVdy"] =  f.dUVdy(dsvel, dy)
pds.dUVdy.attrs = {"Method":"Calculated from uvel and vvel variables."}

# bottom drag
#pds["tauxb1"] = f.tauxb(dsvel, method="center last")
pds["tauxb_1"] = f.tauxb(dsvel, alpha=1, method="center first")
pds["tauxb_2"] = f.tauxb(dsvel, alpha=0.5, method="center first")
pds.tauxb_1.attrs = {"Method":"Calculated from uvel and vvel variables. Argument alpha=1. Centered first."}
pds.tauxb_2.attrs = {"Method":"Calculated from uvel and vvel variables. Argument alpha=0.5. Centered first."}

# add mean zonal velocity
pds["ubar"] = f.ubar(ds)

# set values to coordinates. 
f.add_coordinate_values(pds, dx, dy)

if save_bath:
    # save bottom bathymetry
    bath = -f.total_depth(ds)
    f.add_coordinate_values(bath, dx, dy)
    bath.to_netcdf(outpath+case+"_bathymetry.nc")


# zonal mean
results = pds.mean("x")#.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
results["tauxs"] = np.array([tauxs]*len(results.y))

results.attrs = {"Naming convention" : "tailing 1: calculations done on grid faces, then interpolated to grid center \ntailing 2: variables interpolated to grid center before calculations"}


ntime = len(results.time)
n = 30
nmonths = ntime//n
for t in np.arange(nmonths):
    result = results.isel(time=slice(t*n,(t+1)*n))
    print(t)
    result.to_netcdf(outpath+f"{case}_momentumterms_{t:03}.nc")

#results.to_netcdf(outpath+case+"_momentumterms.nc")
