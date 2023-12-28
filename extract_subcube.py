import dask.array as da
import zarr

x_start = 0  # Starting index in the first dimension
x_end = 7197  # Ending index in the first dimension

# Open the Zarr array with Zarr and wrap it with Dask
zarr_array = zarr.open('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr', mode='r')
dask_array = da.from_zarr(zarr_array)

# Select the subcube
# This operation is lazy and does not load data into memory
subcube = dask_array[x_start:x_end]

# Save the subcube to a new Zarr file
# This will process the data in chunks, without loading the entire subcube into memory
subcube.to_zarr('/work/users/jp348bcyy/rhoneDataCube/subcube.zarr')
