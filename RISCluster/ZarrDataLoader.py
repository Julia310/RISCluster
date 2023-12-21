import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split, Subset
from torch.utils.data import Dataset
import os
from zarr.storage import KVStore
import numpy as np



class ZarrDataset(Dataset):
    class SpecgramNormalizer(object):
        def __init__(self, transform=None):
            self.transform = transform

        def __call__(self, X):
            n, o = X.shape
            if self.transform == "sample_normalization":
                X /= np.abs(X).max(axis=(0,1))
            elif self.transform == "sample_norm_cent":
                X = (X - X.mean()) / (np.abs(X).max() + 1e-8)
            elif self.transform == "vec_norm":
                X = np.reshape(X, (1,-1))
                norm = np.linalg.norm(X) + 1e-8
                X /= norm
                X = np.reshape(X, (n, o))
            return X

    class SpecgramCrop(object):
        def __call__(self, X):
            return X[:-1, 1:]

    class SpecgramToTensor(object):
        def __call__(self, X):
            X = np.expand_dims(X, axis=0)
            return torch.from_numpy(X)

    def __init__(self, zarr_path, chunk_size, transform=None):
        old_zarr_array = zarr.open(zarr_path, mode='r')

        # Check if the array is already chunked
        if old_zarr_array.chunks is None:
            # Create a new chunked Zarr array
            new_zarr_path = zarr_path + '_chunked'
            chunks = (64, 1, 101)  # Adjust as needed
            new_zarr_array = zarr.open(new_zarr_path, mode='w', shape=old_zarr_array.shape, dtype=old_zarr_array.dtype, chunks=chunks)
            new_zarr_array[:] = old_zarr_array[:]
            zarr_path = new_zarr_path  # Use the new chunked path

        # Open the (possibly new) Zarr file with xarray
        self.ds = xr.open_zarr(zarr_path, chunks={'dim_0': 'auto', 'dim_1': 'auto', 'dim_2': 'auto'})

        self.chunk_size = chunk_size
        self.transform = transform
        self.num_sequences = 50
        self.samples_per_sequence = old_zarr_array.shape[0] * 2 // (11 * 24 * chunk_size)

    def __len__(self):
        return self.samples_per_sequence * self.num_sequences

    def __getitem__(self, idx):
        sequence_idx = idx // self.samples_per_sequence
        chunk_idx = idx % self.samples_per_sequence
        start_idx = chunk_idx * self.chunk_size // 2
        end_idx = start_idx + self.chunk_size

        # Adjust the slicing to get the subcube of size (4, 1, 101)
        subcube = self.ds.isel(dim_0=slice(start_idx, start_idx + 4), dim_1=sequence_idx,
                               dim_2=slice(0, 101)).variable.load()

        # Convert to float64 and apply transformation if any
        sample = subcube.astype(np.float64)
        if self.transform:
            sample = self.transform(sample)

        return sample



def get_zarr_data(split_dataset=True):
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        ZarrDataset.SpecgramToTensor(),
        lambda x: x.double(),
    ])

    chunk_size = 4
    #full_dataset = ZarrDataset('./1907_NEW_1Hz_TRUNC.zarr', chunk_size, transform=transform_pipeline)
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr', chunk_size, transform=transform_pipeline)
    print('full dataset length: ', len(full_dataset))

    if split_dataset:

        # Determine the size of the training and test sets
        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Split the dataset into training and test sets
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


        return train_dataset, test_dataset
    else:
        return full_dataset



#dataset = ZarrDataset('1907_NEW_1Hz_TRUNC.zarr', transform=transform_to_tensor)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of iterating over batches
#for batch in dataloader:
#        print(batch.shape)


#train_dataloader, test_dataloader = get_zarr_data(batch_size=32, workers=5)



# Training loop
#for batch in train_dataloader:
#    # Your training code here
#    print("traindata")
#    print(batch.shape)


# Evaluation loop
#for batch in test_dataloader:
#    # Your evaluation code here
#    print("testdata")
#    print(batch.shape)
