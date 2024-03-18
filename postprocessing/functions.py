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

def momentum_advection(u,v):
    return u*v

def depth_integral(var, dz, zcoord="sigma"):
    return (var*dz).sum(zcoord)

def time_derivative(var):
    return var.differentiate("time", datetime_unit="s")

def total_depth(ds):
    return (depth_integral(1,ds.dz) - ds.sealv).isel(time=0)

def pad_reentrance(var):
    return xr.concat([var.isel(x=-1), var, var.isel(x=0)], dim="x")

# from ubbl.py
# dette forstår jeg ikke helt. Hva gjør np.where i dette tilfellet? Hvorfor gir den en liste med 34 arrays, der vi bare bruker første?
def _bottom_value(u,dz):
    bi=np.where(dz>1)[0]
    if len(bi)>0:
        return u[bi[-1]]
    else:
        return np.nan
    
def calculate_dhdx(ds, dx):
    # calculate depth H. Total water height - sea surface elevation
    H = total_depth(ds)

    # pad H in reentranse direction
    Hpad = pad_reentrance(H).chunk({"x":-1, "y":-1})

    # Calculate derivative of bottom height using central difference    
    dhdx = -Hpad.differentiate("x").isel(x=slice(1,-1))/dx
    
    return dhdx

def dUVdy(ds, dy):
    # calculate momentum flux divergence, second order difference
    uv = momentum_advection(ds.uc,ds.vc)
    UV = depth_integral(uv, ds.dz)
    dUVdy = UV.differentiate("y")/dy
    
    return dUVdy


def get_bottom_value(var, dz, zcoord="sigma"):
    varb = xr.apply_ufunc(_bottom_value, var, dz,
                                      input_core_dims=[[zcoord],[zcoord]],
                                      output_core_dims=[[]],
                                      vectorize=True,
                                      dask='parallelized',
                                      output_dtypes=[var.dtype])
    
    return varb

def _calculate_bottomdrag(ub, vb):
    # bottom drag coefficintes
    cbar = 0.05 # is RMS flow speed for linear bottom friction law in [m s-1].
    cb=0.002  # is Coefficient of quadratic bottom friction [unitless].
    q = cb*(np.sqrt(ub**2+vb**2)+cbar)
    tauxb = ub*q
    return tauxb


def add_coordinate_values(ds, dx, dy):
    ds["x"] = dx*np.arange(0,len(ds.x))
    ds["y"] = dy*np.arange(0,len(ds.y))
    
def _dUdt_center_first(ds):
    uc = ds.uc
    dz = ds.dz
    U = depth_integral(uc, dz)
    dUdt = time_derivative(U)
    return dUdt

def _dUdt_center_last(ds):
    uc = ds.uc
    dzx = ds.dzx
    U = depth_integral(uc, dzx)
    dUdt = time_derivative(U)
    return xface2center(dUdt)
    
def dUdt(ds, method="center first"):
    if method == "center first":
        return _dUdt_center_first(ds)
    elif method == "center last":
        return _dUdt_center_last(ds)
    else:
        raise ValueError("Method must be either ""center first"" or ""center last""")
    
def _fV_center_first(ds, f0):
    vc = ds.vc
    dz = ds.dz
    return depth_integral(f0*vc, dz)

def _fV_center_last(ds, f0):
    v = ds.v
    dzy = ds.dzy
    return yface2center(depth_integral(f0*v,dzy))
    
def fV(ds, f0=1e-4, method="center first"):
    if method == "center first":
        return _fV_center_first(ds, f0)
    elif method == "center last":
        return _fV_center_last(ds, f0)
    else:
        raise ValueError("Method must be either ""center first"" or ""center last""")


def phidhdx(ds, rho=1e3, dx=2e3):
    dhdx = calculate_dhdx(ds, dx)
    return ds.pbot*dhdx/rho

def ubar(ds):
    return xface2center(ds.ubaro)

def _tauxb_center_first(ds):
    ucb = get_bottom_value(ds.uc,ds.dz)
    vcb = get_bottom_value(ds.vc,ds.dz)
    return _calculate_bottomdrag(ucb, vcb)

def _tauxb_center_last(ds):
    ub = get_bottom_value(ds.u,ds.dzx)
    vb = get_bottom_value(ds.v,ds.dzy)
    ucb = xface2center(ub)
    vcb = yface2center(vb)
    return _calculate_bottomdrag(ucb, vcb)

def tauxb(ds, method="center first"):
    if method == "center first":
        return _tauxb_center_first(ds)
    elif method == "center last":
        return _tauxb_center_last(ds)
    else:
        raise ValueError("Method must be either ""center first"" or ""center last""")
