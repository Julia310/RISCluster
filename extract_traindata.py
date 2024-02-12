import torch
import numpy as np
import dask.array as da
import zarr

def read_and_write_zarr(source_zarr_path, target_zarr_path, sample_size):
    # Open the source Zarr dataset
    source_group = zarr.open_group(source_zarr_path, mode='r')
    source_zarr_array = source_group[list(source_group.keys())[0]]
    source_ds = da.from_zarr(source_zarr_array)

    # Calculate the number of samples based on the dataset shape and sample size
    num_samples = source_ds.shape[0] // sample_size * source_ds.shape[1]

    # Create the target Zarr dataset
    target_group = zarr.open_group(target_zarr_path, mode='w')
    target_zarr_array = target_group.create_dataset('data', shape=source_ds.shape, chunks=True, dtype='float64')

    prev_channel = -1  # Initialize with an invalid channel index to ensure target_channel starts with 0
    target_channel = -1  # Initialize target_channel

    # Process and write each chunk to the new Zarr dataset
    for idx in range(num_samples):
        start_time = (idx * sample_size) % source_ds.shape[0]
        end_time = start_time + sample_size
        init_channel = (idx * sample_size) // source_ds.shape[0]

        # Increment target_channel when init_channel changes
        if init_channel != prev_channel:
            target_channel += 1
            prev_channel = init_channel

        # Read chunk from source
        chunk = source_ds[start_time:end_time, init_channel, :].compute()

        # Write chunk to target
        target_zarr_array[start_time:end_time, target_channel, :] = chunk

    print(f'Data from {source_zarr_path} has been processed and stored in {target_zarr_path}.')


source_zarr_path = '/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr'
target_zarr_path = '/work/users/jp348bcyy/rhoneDataCube/subcube_chunked_5758.zarr'
read_and_write_zarr(source_zarr_path, target_zarr_path, 4)