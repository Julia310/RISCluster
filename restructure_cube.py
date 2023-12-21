import xarray as xr

# Load the Zarr file
ds = xr.open_zarr('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr')

# Suppose your data cube has 3 dimensions and you want to name them 'time', 'lat', 'lon'
# You can rename the dimensions like this:
ds = ds.rename({'dim_0': 'time', 'dim_1': 'channel', 'dim_2': 'freq'})

ds_chunked = ds.chunk({'time': 60, 'channel': 5, 'freq': 101})

# Save the chunked and modified dataset back to Zarr
ds_chunked.to_zarr('/work/users/jp348bcyy/rhoneCubeNeu/modified_Cube.zarr')

