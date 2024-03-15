import xarray as xr
import numpy as np
import dask
import functions as f

datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
#case = "BLOM_channel_new05_mix1"
case = "BLOM_channel_new02_mix1"
outpath = "/nird/home/annals/BLOM_analysis/data/"+case+"/"

dx = 2e3             # [m]
dy = 2e3             # [m]
rho = 1e3            # [kg m-3] stemmer dette?
dt = 1*24*60*60      # [s]

xchunk = -1
ychunk = -1
sigmachunk = -1
depthchunk= -1
timechunk = 30

data_vars = ["uvellvl", "vvellvl", "sealv", "depth_bnds"]

# read daily data 
ds = xr.open_mfdataset(datapath+case+"/*hd_*.nc", 
                       #parallel=True,
                       chunks = {"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk},
                       data_vars = data_vars,
                      ).sel(depth=slice(0, 2300))
pp = xr.Dataset()

u = ds.uvellvl
v = ds.vvellvl

uc = f.xface2center(u)
pp["uc"] = uc
pp.uc.attrs = {"long_name" : "Horizontal x-velocity at grid center", "units" : "m s-1"}

vc = f.yface2center(v)
pp["vc"] = vc
pp.vc.attrs = {"long_name" : "Horizontal y-velocity at grid center", "units" : "m s-1"}

# spesify coordinate values, usefull for differentiating
pp["x"] = dx*np.arange(0,ni)
pp["y"] = dy*np.arange(0,nj)


# pad u in reentranse direction
upad = xr.concat([pp.uc.isel(x=-1), pp.uc, pp.uc.isel(x=0)], dim="x").chunk({"x":-1, "y":-1})
upad["x"] = dx*np.arange(-1,len(pp.x)+1)

# Calculate derivative of u and v using central difference    
dudx = upad.differentiate("x").isel(x=slice(1,-1))
dvdy = pp.vc.differentiate("y")


# use continuity to calculate derivative of vertical velocity w
dwdz = -(dudx+dvdy)

# calculate dz
dz = (ds.depth_bnds.isel(bounds=1)-ds.depth_bnds.isel(bounds=0))

# calculate vertical velocities at surface due to changes in sea surface elevation
ws = ds.sealv.differentiate("time", datetime_unit="s")


# integrate dwdz from surface and down
wc1 = ws+(-dwdz*dz).cumsum("depth")
pp["wc1"] = wc1
pp.wc1.attrs = {"long_name" : "Vertical velocity calculated from continuity and change in sea surface elevation", "units" : "m s-1"}

# calculate vertical velocities from mass fluxes
A = dx*dy
wflx = ds.wflxlvl
wc2 = wflx/(rho*A)
pp["wc2"] = wc2
pp.wc2.attrs = {"long_name" : "Vertical velocity calculated from vertical mass flux", "units" : "m s-1"}

pp["eta"] = ds.sealv


mean = pp.mean("time")
mean.eta.attrs = {"long_name" : "Time-mean sea level", "units" : "m"}
mean.uc.attrs = {"long_name" : "Horizontal time-mean x-velocity at grid center", "units" : "m s-1"}
mean.vc.attrs = {"long_name" : "Horizontal time-mean y-velocity at grid center", "units" : "m s-1"}
mean.wc1.attrs = {"long_name" : "Vertical time-mean velocity at grid center calculated from continuity and change in sea surface elevation", "units" : "m s-1"}
mean.wc2.attrs = {"long_name" : "Vertical time-mean velocity at grid center calculated from vertical mass flux", "units" : "m s-1"}

mean.to_netcdf(outpath+case+f"_mean_velocities_lvl.nc")

std = pp.mean("time")
std.eta.attrs = {"long_name" : "Sea level std", "units" : "m"}
std.uc.attrs = {"long_name" : "Horizontal x-velocity std at grid center", "units" : "m s-1"}
std.vc.attrs = {"long_name" : "Horizontal y-velocity std at grid center", "units" : "m s-1"}
std.wc1.attrs = {"long_name" : "Vertical velocity std at grid center calculated from continuity and change in sea surface elevation", "units" : "m s-1"}
std.wc2.attrs = {"long_name" : "Vertical velocity std at grid center calculated from vertical mass flux", "units" : "m s-1"}

mean.to_netcdf(outpath+case+f"_std_velocities_lvl.nc")

"""
results = pp.coarsen(time=timechunk, boundary="trim").mean()

ntime = len(results.time)
nmonths = ntime//30
for t in np.arange(nmonths):
    result = results.isel(time=slice(t*30,(t+1)*30))
    print(t)
    result.to_netcdf(outpath+case+f"_velocities_{t:03}.nc")
"""