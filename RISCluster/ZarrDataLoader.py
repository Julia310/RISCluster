import torch
import xarray as xr
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
import dask.array as da
import zarr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ZarrDataset(Dataset):
    class SpecgramNormalizer(object):
        def __init__(self, transform=None):
            self.transform = transform

        def __call__(self, X):
            #logging.info(f'shape of sample: {X.shape}')
            if self.transform == "sample_normalization":
                X /= np.abs(X).max(axis=(0,1))
            elif self.transform == "sample_norm_cent":
                X = (X - X.mean()) / (np.abs(X).max() + 1e-8)
            elif self.transform == "vec_norm":
                n, o = X.shape
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


    def __init__(self, zarr_path, sample_size, transform=None):
        # Open the Zarr dataset as an xarray Dataset, this will handle lazy loading
        #self.ds = xr.open_zarr(zarr_path, consolidated=True)
        zarr_array = zarr.open(zarr_path, mode='r')
        self.ds = da.from_zarr(zarr_array)

        self.sample_size = sample_size  # Size of each individual sample in the 'time' dimension
        self.transform = transform

        # Assuming each sample is non-overlapping for simplicity
        #self.num_samples = self.ds.dims['time'] // sample_size * self.ds.dims['channel']
        self.num_samples = self.ds.shape[0] //sample_size * self.ds.shape[1]
        self.current_channel = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate the start and end indices for the time dimension
        #start_time = (idx % self.ds.dims['time']) * self.sample_size
        start_time = (idx % self.ds.shape[0]) * self.sample_size
        logging.info(f'idx {idx}')
        end_time = start_time + self.sample_size

        #if end_time >= self.ds.dims['time']:
        if end_time >= self.ds.shape[0]:
            # Increment channel index and reset time index
            self.current_channel = (self.current_channel + 1)
            #logging.info(f"current channel: {self.current_channel}, time: {self.ds.shape[0]}, end_time: {end_time}")
            start_time = 0
            end_time = self.sample_size

        # Use xarray's indexing to lazily load the data slice
        #sample = self.ds.isel(time=slice(start_time, end_time), channel=self.current_channel)
        #sample = sample.compute()

        sample = self.ds[start_time:end_time, self.current_channel, :].compute()

        # Convert the xarray DataArray to a numpy array
        # Since xarray uses dask under the hood for lazy loading, the actual computation happens here
        #sample = sample.to_array()
        #sample = sample.values

        #sample = torch.from_numpy(sample)

        # Apply transformations if any
        if self.transform is not None:
            sample = self.transform(sample)

        # Add a channel dimension to the numpy array to be compatible with PyTorch
        #sample = np.expand_dims(sample, axis=0)

        #sample = sample.compute()

        # Convert the numpy array to a PyTorch tensor
        return sample


def get_zarr_data(split_dataset=True):
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        ZarrDataset.SpecgramToTensor(),
        lambda x: x.double(),
    ])

    sample_size = 4
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/subcube_current.zarr', sample_size, transform=transform_pipeline)
    #full_dataset = ZarrDataset("/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_60.zarr", sample_size, transform=transform_pipeline)
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


