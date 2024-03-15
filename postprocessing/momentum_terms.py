import xarray as xr
import numpy as np
from dask.distributed import Client
import os
import functions as f

#datapath = "/projects/NS9869K/noresm/cases/BLOM_channel/"
datapath = "/projects/NS9252K/noresm/cases/BLOM_channel/"

#case = "BLOM_channel_new05_mix1_taupos5"
case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"
outpath = "/nird/home/annals/BLOM_analysis/data/"+case+"/"

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


u = ds.uvel
v = ds.vvel

uc = f.xface2center(u)
vc = f.yface2center(v)

# store values 
ds["uc"] = uc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})
ds["vc"] = vc.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})



# calculate momentum advection
ds["uv"] = f.momentumAdvection(u,v)


# calculate depth integral of momentum advection and u velocity
pds["UV"] = f.centerDepthIntegral(ds.uv, ds.dz)
pds["U1"] = f.centerDepthIntegral(ds.uc, ds.dz)
pds["U2"] = f.xfaceDepthIntegral(ds.u, ds.dz)

# calculate time derivative of velocity, second order difference
dUdt1 = f.timeDerivative(pds.U1)
dUdt2 = f.timeDerivative(pds.U2)

pds["dUdt1"] = dUdt1
pds["dUdt2"] = xface2center(dUdt2)


# advection of planetary vorticity
pds["fV1"] = f.centerDepthIntegral(f0*vc, dz) 
pds["fV2"] = yface2center(f.yfaceDepthIntegral(f0*v,dz))


# calculate depth H. Total water height - sea surface elevation
H = (ds.dz.sum(dim = 'sigma') - ds.sealv).isel(time=0)#.mean("time")            # [m]

# pad H in reentranse direction
Hpad = xr.concat([H.isel(x=-1), H, H.isel(x=0)], dim="x").chunk({"x":-1, "y":-1})
#Hpad["x"] = dx*np.arange(-1,len(ds.x)+1)

# Calculate derivative of bottom height using central difference    
dhdx = -Hpad.differentiate("x").isel(x=slice(1,-1))/dx
ds["dhdx"] = dhdx
pds["phidhdx"] = ds.pbot*ds.dhdx/rho

# calculate momentum flux divergence, second order difference
dUVdy = pds.UV.differentiate("y")/dy
pds["dUVdy"] =  dUVdy

# calculate bottom drag
cbar = 0.05 # is RMS flow speed for linear bottom friction law in [m s-1].
cb=0.002  # is Coefficient of quadratic bottom friction [unitless].


# from ubbl.py
# dette forstår jeg ikke helt. Hva gjør np.where i dette tilfellet? Hvorfor gir den en liste med 34 arrays, der vi bare bruker første?
def bottom_vel(u,dz):
    bi=np.where(dz>1)[0]
    if len(bi>0):
        return u[bi[-1]]
    else:
        return np.nan


ub = xr.apply_ufunc(bottom_vel, u, dzx,
                                      input_core_dims=[['sigma'],['sigma']],
                                      output_core_dims=[[]],
                                      vectorize=True,
                                      dask='parallelized',
                                      output_dtypes=[u.dtype])
    
vb = xr.apply_ufunc(bottom_vel, v, dzx,
                                  input_core_dims=[['sigma'],['sigma']],
                                  output_core_dims=[[]],
                                  vectorize=True,
                                  dask='parallelized',
                                  output_dtypes=[v.dtype])

ubc1 = xface2center(ub)
vbc1 = yface2center(vb)


ubc2 = xr.apply_ufunc(bottom_vel, uc, dz,
                                      input_core_dims=[['sigma'],['sigma']],
                                      output_core_dims=[[]],
                                      vectorize=True,
                                      dask='parallelized',
                                      output_dtypes=[uc.dtype])
    
vbc2 = xr.apply_ufunc(bottom_vel, vc, dz,
                                  input_core_dims=[['sigma'],['sigma']],
                                  output_core_dims=[[]],
                                  vectorize=True,
                                  dask='parallelized',
                                  output_dtypes=[vc.dtype])




q1 = cb*(np.sqrt(ubc1**2+vbc1**2)+cbar)
tauxb1 = ubc1*q1

pds["tauxb1"] = tauxb1

q2 = cb*(np.sqrt(ubc2**2+vbc2**2)+cbar)
tauxb2 = ubc2*q2

pds["tauxb2"] = tauxb2

# add mean zonal velocity
pds["ubar"] = xface2center(ds.ubaro)

# set values to coordinates. 
pds["x"] = dx*np.arange(0,len(pds.x))
pds["y"] = dy*np.arange(0,len(pds.y))


# save bottom bathymetry
bath = -H
bath["x"] = dx*np.arange(0,len(bath.x))
bath["y"] = dy*np.arange(0,len(bath.y))
print(bath)
bath.to_netcdf(outpath+case+"_bathymetry.nc")

# zonal mean
results = pds.mean("x")#.chunk({"x":xchunk, "y":ychunk, "sigma":sigmachunk, "time":timechunk})

results.attrs = {"Naming convention" : "tailing 1: calculations done on grid faces, then interpolated to grid center \ntailing 2: variables interpolated to grid center before calculations"}


ntime = len(results.time)
n = 15
nmonths = ntime//n
for t in np.arange(nmonths):
    result = results.isel(time=slice(t*n,(t+1)*n))
    print(t)
    result.to_netcdf(outpath+case+f"_momentumterms_{t:03}.nc")

#results.to_netcdf(outpath+case+"_momentumterms.nc")
