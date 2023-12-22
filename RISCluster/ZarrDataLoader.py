import torch
import xarray as xr
import zarr
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
import dask.array as da


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
            #print('initial shape: ', X.shape)
            X = np.expand_dims(X, axis=0)
            return torch.from_numpy(X)

    def __init__(self, zarr_path, sample_size, transform=None, chunk_size=400):
        # Open the Zarr array with Zarr and wrap it with Dask
        zarr_array = zarr.open(zarr_path, mode='r')
        dask_array = da.from_zarr(zarr_array)

        # Convert the Dask-backed array to an Xarray DataArray, applying chunking
        self.ds = xr.DataArray(dask_array, dims=['time', 'channel', 'freq']).chunk(
            {'time': chunk_size, 'channel': 1, 'freq': 101})

        self.chunk_size_time = chunk_size  # Total size of each chunk in the 'time' dimension
        self.chunk_size_channel = 1  # Total size of each chunk in the 'channel' dimension
        self.sample_size = sample_size  # Size of each individual sample in the 'time' dimension
        self.sample_shape = (sample_size, 1, 101)
        self.transform = transform

        self.num_chunks = (self.ds.shape[0] // chunk_size) * self.ds.shape[1]
        self.samples_per_chunk = (self.chunk_size_time // self.sample_size * 2 - 1) * self.chunk_size_channel  # Number of samples per chunk, considering overlap
        self.total_samples = self.samples_per_chunk * self.num_chunks

        # Initialize variables to manage chunk loading
        self.preloaded_chunk = None
        self.current_chunk_index = -1
        self.current_channel = 0

    def __len__(self):
        return self.total_samples

    def load_chunk(self, chunk_index):
        start_idx_time = chunk_index * self.chunk_size_time
        end_idx_time = start_idx_time + self.chunk_size_time

        start_idx_channel = self.current_channel
        end_idx_channel = self.current_channel + 1

        # Extract the larger chunk from the dataset
        self.preloaded_chunk = self.ds.isel(
            time=slice(start_idx_time, end_idx_time),
            channel=slice(start_idx_channel, end_idx_channel)
        ).compute()
        self.current_chunk_index = chunk_index

        if end_idx_time % self.ds.shape[0] == 0 and end_idx_time != 0:
            self.current_channel += 1

    def __getitem__(self, idx):
        sequence_idx = idx // self.total_samples
        sample_idx_within_sequence = idx % self.total_samples
        chunk_index = (sample_idx_within_sequence // self.samples_per_chunk) % (self.ds.shape[0] // self.chunk_size_time)
        sample_idx_in_chunk = sample_idx_within_sequence % self.samples_per_chunk

        # Load the chunk if it's not already loaded
        if chunk_index != self.current_chunk_index:
            self.load_chunk(chunk_index)

        # Calculate the indices to extract the small sample from the preloaded chunk
        sample_start = sample_idx_in_chunk * self.sample_shape[0] // 2
        sample_end = sample_start + self.sample_shape[0]

        # Extract the small sample
        sample = self.preloaded_chunk[sample_start:sample_end, sequence_idx, :].astype(np.float64)

        if self.transform:
            sample = self.transform(sample)

        return sample



def get_zarr_data(split_dataset=True):
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        ZarrDataset.SpecgramToTensor(),
        lambda x: x.double(),
    ])

    sample_size = 4
    #full_dataset = ZarrDataset('./1907_NEW_1Hz_TRUNC.zarr', sample_size, transform=transform_pipeline)
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr', sample_size, transform=transform_pipeline)
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


