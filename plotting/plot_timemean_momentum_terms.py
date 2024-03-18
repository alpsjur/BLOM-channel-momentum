# Import necessary libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from functions import find_slopeloc

# Constants
rho = 1e3  # Density of water (kg/m^3)

# Select the case study
case = "BLOM_channel_new05_mix1"
#case = "BLOM_channel_new02_mix1"

# Define data and figure paths
datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
figurepath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

# Load datasets: mean over time
ds_flux = xr.open_mfdataset(datapath+f"from_flux/{case}_from_flux_momentumterms_*.nc").mean("time")
ds_vel = xr.open_mfdataset(datapath+f"from_vel/{case}_from_vel_momentumterms_*.nc").mean("time")

# Load bathymetry and calculate mean depth
bath = xr.open_dataarray(datapath+f"{case}_bathymetry.nc")#.isel(y=slice(1,-1))
ds_vel["h"] = bath.mean(dim="x")

# Seaborn theme for plotting
sns.set_theme()

# Initialize the plot
fig, ax = plt.subplots(figsize=(12,8))
y = ds_vel.y/1e3 # Convert y coordinates to km

# Calculate various momentum terms
surface_forcing = [-0.05/rho]*len(y.values) 
MFD = -ds_vel.dUVdy             # Momentum advection divergence
TFS = -ds_vel.phidhdx           # Topographic form stress
dUdt = ds_flux.dUdt2            # Rate of change of velocity
fV = ds_flux.fV1                # Coriolis acceleration term
bottom_drag = -ds_flux.tauxb1   # Bottom drag
res = -dUdt+MFD+TFS+fV+bottom_drag+surface_forcing   # Residual term


# Plot settings and labels for each momentum term
vars = [dUdt, MFD, fV, TFS, surface_forcing, bottom_drag, res]
labels = [r"$\frac{\partial}{\partial t}\overline{U}$", 
          r"$-\overline{\frac{\partial}{\partial y}\int_{-H}^0(uv)\,dz}$", 
          r"$+\overline{fV}$",
          r"$-\overline{\Phi_b\frac{\partial}{\partial x}h_b}$",
          r"$+\overline{\tau^x_s}$",
          r"$-\overline{\tau^x_b}$",
          "residual"
          ]

# Define colors for the plot lines
colors = mpl.colormaps['tab10'].resampled(len(vars)).colors

# Horizontal line of 0 momentum
ax.axhline(0, color="gray")

# Plot each variable
maxx = 0
for var, label, color in zip(vars, labels, colors):
    ax.plot(y, var, 
            label=label, 
            lw = 2,
            color=color
            )
    maxx = np.max([maxx, np.abs(var).max()])
    
# Scale and plot the mean velocity profile
scale = maxx/np.max(np.abs(ds_vel.ubar))
ax.plot(y, ds_vel.ubar*scale, 
         color="gray", 
         alpha=0.5,
         lw = 3,
         ls="--", 
         zorder=100,
         label=r"$\propto\overline{u}$"
         )    

# Highlight the slope regions
lx0, lx1, rx0, rx1 = find_slopeloc(ds_vel.h, epsilon=0.01)
ax.axvspan(lx0/1e3, lx1/1e3, alpha=0.2, color="gray", label="slope")
ax.axvspan(rx0/1e3, rx1/1e3, alpha=0.2, color="gray")

# Configure legend, labels, and title
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
ax.set_ylabel("depth-integrated momentum [m²/s²]")
ax.set_xlabel("cross-channel position [km]")
ax.set_xlim(np.min(y), np.max(y))

# Construct and set the equation as title
eq = labels[0]+"="
for term in labels[1:-1]:
    eq+=term
ax.set_title(eq+"\n", fontsize=18)

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(figurepath+f"{case}_timemean_momentum_terms.png")