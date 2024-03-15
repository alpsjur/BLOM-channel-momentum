import xarray as xr
import numpy as np
import dask
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

case = "BLOM_channel_new02_mix1"
#case = "BLOM_channel_new05_mix1"
datapath = f"/home/alsjur/PhD/Kode/BLOM-channel-momentum/data/{case}/"

figurepath = f"/home/alsjur/PhD/Kode/BLOM-channel-momentum/figures/"

# open datasets. ds contains velocites ans sea surface height, bathy contains the bathymetry

ds = xr.open_dataset(datapath+f"{case}_velocities.nc").transpose("y", "x", "depth")
bath = xr.open_dataarray(datapath+f"{case}_bathymetry.nc").transpose("y", "x")

# extract grid coordinates and transfor to kilometers
x = bath.x*1e-3
y = bath.y*1e-3
z = -ds.depth*1e-3

# grid distances
dx = 2
dy = 2

### Bathymetry

# prepare pathymetry for plotting
X, Y = np.meshgrid(x, y)
bath_z = bath.values*1e-3


# plot bathymetry
fig = go.Figure(data=[
    go.Surface(
        x=X, 
        y=Y, 
        z=bath_z,
        # Set to a single color
        colorscale=[(0, 'gray'), (1, 'gray')],
        showscale=False,  # Hide the colorbar
        opacity=0.5,
        name = "Bathymetry"
    )
])

### Sea surface height


# remove outlier
eta = ds.eta.where(np.abs(ds.eta)<2)

"""
# scale eta
etas = eta*1e-1

# shift eta for improved plotting
eta_shift = -np.nanmin(etas.values.flatten())*2

# Add the sea surface height anomaly on top
fig.add_trace(
    go.Surface(
        x=X, 
        y=Y, 
        z=etas+eta_shift,
        cmid = eta_shift, 
        colorscale='Tropic',  
        showscale=False,      # Optionally hide or show the colorbar for eta
        opacity=0.9,          # Adjust opacity for visual clarity
        name = "Sea level"
    )
)
"""

### Sea surface height anomaly


# calculate zonal anomaly. Scale with 2 for improved plotting
# NB eta is not plottet n correct scale (would be invisible)
eta_anomaly = (eta-eta.mean("x")).values*2

# shift eta for improved plotting
#eta_anomaly_shift = np.nanmax(etas)+eta_shift-np.nanmin(eta_anomaly.flatten())*2
eta_anomaly_shift = -np.nanmin(eta_anomaly.flatten())*2

# Add the sea surface height anomaly on top
fig.add_trace(
    go.Surface(
        x=X, 
        y=Y, 
        z=eta_anomaly+eta_anomaly_shift,
        cmid = eta_anomaly_shift, 
        colorscale='Tropic',  
        showscale=False,      # Optionally hide or show the colorbar for eta
        opacity=0.9,          # Adjust opacity for visual clarity
        name = "Zonal sea\nlevel anomaly"
    )
)

# option ot include contours for sea surface anommaly
#fig.update_traces(contours_z=dict(show=True))

# update plot apparence
fig.update_layout(title=case, 
                  #template="plotly_dark",
                  template="plotly",
                  scene=dict(
        xaxis=dict(range=[min(x), max(x)], title='x [km]'),  
        yaxis=dict(range=[min(y), max(y)], title='y [km]'),  
        zaxis=dict(range=[np.nanmin(bath_z.flatten())*1.001, np.nanmax(eta_anomaly.flatten())*1.1+eta_anomaly_shift], title='Depth [km]'),
        aspectmode='manual',  
        aspectratio=dict(x=len(x)/np.max([len(y), len(x)]), y=len(y)/np.max([len(y), len(x)]), z=0.5)
    )
)

### Streamtubes

# coarsen dataset (or else file will be too large)
xc = 7
yc = 7
zc = 1
dsc = ds.coarsen(x=xc, y=yc, depth=zc,boundary="trim").mean()
dsc["x"] = dsc["x"]+(xc+1)/2*dx
dsc["y"] = dsc["y"]+(yc+1)/2*dy
#dsc = ds.isel(x=slice(0,None,xc), y=slice(0,None,yc), depth=slice(0,None,zc))

# extract variables to plot
u = dsc.uc
v = dsc.vc
w = dsc.wc2

mask = ~np.isnan(u)

w = w.where(mask)

# convert to kilometers
x = u.x.values*1e-3
y = u.y.values*1e-3
z = -u.depth.values*1e-3


X, Y, Z = np.meshgrid(x, y, z)

Xs = np.copy(X)
Ys = np.copy(Y)
Zs = np.copy(Z)

Xs[~mask] = np.nan
Ys[~mask] = np.nan
Zs[~mask] = np.nan

# Flatten the arrays for streamtube plotting
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
u_flat = u.values.flatten()
v_flat = v.values.flatten()
w_flat = w.values.flatten()

# spesify indexes of start seeds
sz_surf = 7
sz_mid = 5
sz_bottom = 1
zi = np.concatenate([np.arange(2, 22, sz_surf), np.arange(22, 46, sz_mid), np.arange(46,52,sz_bottom)])
#szi = np.concatenate([np.array([0, 1, 8, 12]), np.arange(16, 52, 2)])
sy = 4
#sx = 8
xi = np.array([10, 25])

# extract start positions of seeds
xstart = Xs[2::sy, xi][::,::, zi].flatten()
ystart = Ys[2::sy, xi][::,::, zi].flatten()
zstart = Zs[2::sy, xi][::,::, zi].flatten()

# remove values beneath topography
xstart = xstart[~np.isnan(xstart)]
ystart = ystart[~np.isnan(ystart)]
zstart = zstart[~np.isnan(zstart)]


# Create a streamtube trace 
stream_trace = go.Streamtube(x=X_flat, y=Y_flat, z=Z_flat, 
                            u=u_flat, v=v_flat, w=w_flat, 
                            starts=dict(x=xstart, y=ystart, z=zstart), 
                            #cmin=cmin, cmax=cmax, 
                            colorscale='Plasma', 
                            #showscale=False, 
                            sizeref = 2,
                            opacity=0.8,
                            name = "Speed",
                            colorbar = dict(title=dict(text="Speed [m s-1]")),
                            )

# Add streamtube trace to the figure
fig.add_trace(stream_trace)

# Save as html
fig.write_html(figurepath+f"{case}_streamtubes.html",  
               #include_plotlyjs="cdn", 
               #full_html=False
               )


#fig.show()