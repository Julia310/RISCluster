import xarray as xr

# Load the Zarr file
ds = xr.open_zarr('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr')

# Suppose your data cube has 3 dimensions and you want to name them 'time', 'lat', 'lon'
# You can rename the dimensions like this:
ds = ds.rename({'dim_0': 'time', 'dim_1': 'channel', 'dim_2': 'freq'})

ds_chunked = ds.chunk({'time': 60, 'channel': 5, 'freq': 101})

# Save the chunked and modified dataset back to Zarr
ds_chunked.to_zarr('/work/users/jp348bcyy/rhoneCubeNeu/modified_Cube.zarr')


# Load the chunked dataset
ds = xr.open_zarr('/work/users/jp348bcyy/rhoneCubeNeu/modified_Cube.zarr')

# Define a function to calculate chunk boundaries
def get_chunk_boundaries(dim_size, chunk_size):
    return [(i, min(i + chunk_size, dim_size)) for i in range(0, dim_size, chunk_size)]

# Get chunk boundaries for each dimension
time_chunks = get_chunk_boundaries(ds.dims['time'], 60)
channel_chunks = get_chunk_boundaries(ds.dims['channel'], 5)
freq_chunks = get_chunk_boundaries(ds.dims['freq'], 101)  # Assuming one chunk for 'freq'

# Iterate over chunks
for t_start, t_end in time_chunks:
    for c_start, c_end in channel_chunks:
        for f_start, f_end in freq_chunks:
            chunk = ds.isel(time=slice(t_start, t_end), channel=slice(c_start, c_end), freq=slice(f_start, f_end))
            print(chunk)
