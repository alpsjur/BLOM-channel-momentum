import xarray as xr
import numpy as np

def find_slopeloc(bath, epsilon=0.01):
    shelf_depth = np.min(bath)
    central_basin_depth = np.max(bath)
    upper_treshold = shelf_depth+epsilon*(central_basin_depth-shelf_depth)
    lower_treshold = central_basin_depth-epsilon*(central_basin_depth-shelf_depth)
    
    lx0 = next(x for x in bath if x < upper_treshold).y
    lx1 = next(x for x in bath if x < lower_treshold).y
    
    bath_r = bath.isel(y=slice(-1,0,-1))
    
    rx1 = next(x for x in bath_r if x < upper_treshold).y
    rx0 = next(x for x in bath_r if x < lower_treshold).y
    
    return lx0, lx1, rx0, rx1