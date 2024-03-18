import xarray as xr
import numpy as np

def find_slopeloc(bath, epsilon=0.01):
    shelf_depth = np.min(bath)
    central_basin_depth = np.max(bath)
    upper_treshold = shelf_depth+epsilon*(central_basin_depth-shelf_depth)
    lower_treshold = central_basin_depth-epsilon*(central_basin_depth-shelf_depth)
    
    lx1 = next(x for x in bath if x < upper_treshold).y
    lx0 = next(x for x in bath if x < lower_treshold).y
    
    bath_r = bath.isel(y=slice(-1,0,-1))
    
    rx0 = next(x for x in bath_r if x < upper_treshold).y
    rx1 = next(x for x in bath_r if x < lower_treshold).y
    
    return lx0, lx1, rx0, rx1


def find_slopeidx(bath, epsilon=0.01, dy=2e3):
    shelf_depth = np.min(bath)
    central_basin_depth = np.max(bath)
    upper_treshold = shelf_depth+epsilon*(central_basin_depth-shelf_depth)
    lower_treshold = central_basin_depth-epsilon*(central_basin_depth-shelf_depth)
    
    lx1 = int(next(x for x in bath if x < upper_treshold).y/dy)
    lx0 = int(next(x for x in bath if x < lower_treshold).y/dy)
    
    bath_r = bath.isel(y=slice(-1,0,-1))
    
    rx0 = int(next(x for x in bath_r if x < upper_treshold).y/dy)
    rx1 = int(next(x for x in bath_r if x < lower_treshold).y/dy)
    
    return lx0, lx1, rx0, rx1

def hanning_filter(da, window_length = 20):
    # following http://xarray.pydata.org/en/stable/computation.html#rolling-window-operations
    hamming = np.hanning(window_length)
    window = xr.DataArray(hamming/np.sum(hamming), dims=['window'])
    rolling = da.rolling(time=window_length, center=True)
    filtered = rolling.construct(time='window').dot(window)
    
    return filtered