import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

case = "BLOM_channel_new05_mix1_taupos5"
datapath = f"/nird/home/annals/BLOM-channel-momentum/data/{case}/"
figurepath = f"/nird/home/annals/BLOM-channel-momentum/figures/"

# Assuming `eta` is your xarray DataArray loaded with data
eta = xr.open_dataarray(datapath+case+"_eta.nc")  # Example for loading from a NetCDF file

# Create a figure with Plotly
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

# Extract the time, x, and y coordinates
times = eta.time.values
x = eta.x.values
y = eta.y.values

# Initialize animation frames
frames = []

for i, time in enumerate(times):
    # Extract data for the current time step
    z = eta.sel(time=time).values
    
    frame = go.Frame(data=[go.Surface(z=z, x=x, y=y)],
                     name=str(time))
    frames.append(frame)

# Add data for the first frame to start with
fig.add_trace(go.Surface(z=eta.sel(time=times[0]).values, x=x, y=y))

# Update layout to make it suitable for animations
fig.update_layout(
    title='eta over time',
    updatemenus=[dict(type='buttons',
                      showactive=False,
                      y=1,
                      x=0.8,
                      xanchor='left',
                      yanchor='bottom',
                      buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=500, redraw=True), 
                                                     fromcurrent=True, 
                                                     mode='immediate')]),
                               dict(label='Pause',
                                    method='animate',
                                    args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                                       mode='immediate')])])])

# Animation settings
fig.frames = frames

# Save the figure
fig.to_html(figurepath+f'{case}_eta_animation.html')
