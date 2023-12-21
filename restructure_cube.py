import xarray as xr
import dask.array as da
import zarr

# Open the Zarr array with Zarr and wrap it with Dask
zarr_array = zarr.open('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr', mode='r')
dask_array = da.from_zarr(zarr_array)

# Convert the Dask-backed array to an Xarray DataArray, applying chunking in the process
ds = xr.DataArray(dask_array, dims=['time', 'channel', 'freq'])

# Since the array is already chunked by Dask, you don't need to re-chunk it in Xarray
# However, if you wish to change the chunk sizes, you can do so:
ds = ds.chunk({'time': 60, 'channel': 5, 'freq': 101})

# Save the chunked and modified dataset back to Zarr
# Save the chunked dataset back to a new Zarr file
ds.to_zarr("/work/users/jp348bcyy/rhoneDataCube/chunked_Cube.zarr", mode='w')

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
