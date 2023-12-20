import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split, Subset
from torch.utils.data import Dataset
import os
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
        self.zarr_array = zarr.open(zarr_path, mode='r')
        self.chunk_size = chunk_size  # Size of each chunk in the first dimension
        self.transform = transform
        #self.num_sequences = self.zarr_array.shape[1]  # Number of sequences in the second dimension
        self.num_sequences = 500  # Number of sequences in the second dimension
        #self.samples_per_sequence = self.zarr_array.shape[0] // chunk_size  * 2# Number of chunks per sequence
        self.samples_per_sequence = 172740 // chunk_size * 2# Number of chunks per sequence
        print('samples per sequence', self.samples_per_sequence)
        print('number of sequences', self.num_sequences)

    def __len__(self):
        return self.samples_per_sequence * self.num_sequences

    def __getitem__(self, idx):
        sequence_idx = idx // self.samples_per_sequence
        chunk_idx = idx % self.samples_per_sequence

        # Calculate the start and end indices for the chunk
        start_idx = chunk_idx * self.chunk_size // 2
        end_idx = start_idx + self.chunk_size

        # Extract the chunk from the Zarr array
        sample = self.zarr_array[start_idx:end_idx, sequence_idx, :].astype(np.float32)

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
        train_size = int(0.8 * len(full_dataset))
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
