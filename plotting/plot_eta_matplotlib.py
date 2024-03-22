import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmocean

#case = "BLOM_channel_new05_mix1_taupos5"
#case = "BLOM_channel_new05_mix1_tauneg10"
#case = "BLOM_channel_new05_mix1"
case = "BLOM_channel_new02_mix1"

datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
figurepath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

# Assuming `eta` is your xarray DataArray loaded with data
eta = xr.open_dataarray(datapath+case+"_eta.nc").transpose("time", "x", "y")
# find limits
vlim = np.max(np.abs(eta))
cmap = "magma"

# axis values
x = eta.x*2
y = eta.y*2

# Prepare the figure and axis for plotting
fig, ax = plt.subplots(figsize=(15,5))
ax.set_aspect("equal")


# Initial plot
time_index = 0
eta_slice = eta.isel(time=time_index)
cax = ax.pcolormesh(y, x, eta_slice, 
                    shading='auto', 
                    #vmin=-vlim, 
                    #vmax=vlim,
                    cmap=cmap
                    )
fig.colorbar(cax, ax=ax, label="Sea surface elevation [m]")
ax.set_xlabel("Cross-channel distance [km]")
ax.set_ylabel("Along-channel distance [km]")

def animate(i):
    ax.clear()  # Clear previous frame
    eta_slice = eta.isel(time=i)
    cax = ax.pcolormesh(y, x, eta_slice, 
                        shading='auto', 
                        #vmin=-vlim, 
                        #vmax=vlim,
                        cmap=cmap
                        )
    ax.set_title(f"Time: {eta.time.values[i]}")
    ax.set_xlabel("Cross-channel distance [km]")
    ax.set_ylabel("Along-channel distance [km]")
    return cax,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(eta.time), interval=100, blit=False)

#plt.show()
ani.save(figurepath+f'eta/{case}_eta_animation.mp4', 
         writer='ffmpeg'
         )

