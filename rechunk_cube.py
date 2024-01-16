import xarray as xr
import rechunker

# Path to the original Zarr dataset
source_path = '/work/users/jp348bcyy/rhoneDataCube/Cube_chunked.zarr'

# Path to store the rechunked Zarr dataset
target_path = '/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_60.zarr'

# Open the dataset using xarray
source_ds = xr.open_zarr(source_path)

# Assigning dimension names
source_ds = source_ds.rename({'dim_0': 'time', 'dim_1': 'channel', 'dim_2': 'freq'})

# Define the target chunk shape
target_chunks = {'time': 60, 'channel': 1, 'freq': 101}

# Create a rechunking plan
rechunk_plan = rechunker.rechunk(source_ds, target_chunks, target_path, temp_store=None)

# Execute the rechunking
rechunk_plan.execute()

print("Rechunking completed successfully.")
