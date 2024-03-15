import xarray as xr
import numpy as np
import dask

# functions for moving variables from face to center of cells
def xface2center(da):
    ni = len(da.x)

    dac = xr.concat([da.isel(x=slice(0,ni-1)), da.isel(x=slice(1,ni))], dim="temp").mean(dim="temp")
    dacend = xr.concat([da.isel(x=0), da.isel(x=-1)], dim="temp").mean(dim="temp")
    dac = xr.concat([dac, dacend], dim="x")

    return dac

def yface2center(da):
    nj = len(da.y)

    dac = xr.concat([da.isel(y=slice(0,nj-1)), da.isel(y=slice(1,nj))], dim="temp").mean(dim="temp")
    dacend = da.isel(y=-1)*np.nan
    dac = xr.concat([dac, dacend], dim="y")

    return dac

def center2xface(da):
    ni = len(da.x)

    dax = xr.concat([da.isel(x=slice(0,ni-1)), da.isel(x=slice(1,ni))], dim="temp").mean(dim="temp")
    daxfirst = xr.concat([da.isel(x=0), da.isel(x=-1)], dim="temp").mean(dim="temp")
    dax = xr.concat([daxfirst, dax], dim="x")

    return dax

def center2yface(da):
    nj = len(da.y)

    day = xr.concat([da.isel(y=slice(0,nj-1)), da.isel(y=slice(1,nj))], dim="temp").mean(dim="temp")
    dayfirst = da.isel(y=0)*np.nan
    day = xr.concat([dayfirst, day], dim="y")

    return day

def momentumAdvection(u,v):
    return u*v

def centerDepthIntegral(var, dz):
    return (var*dz).sum("sigma")

def xfaceDepthIntegral(var, dz):
    dzx = center2xface(dz)
    return (var*dzx).sum("sigma")

def yfaceDepthIntegral(var, dz):
    dzy = center2yface(dz)
    return (var*dzy).sum("sigma")

def timeDerivative(var):
    return var.differentiate("time", datetime_unit="s")