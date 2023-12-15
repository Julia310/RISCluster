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

    def __init__(self, zarr_path, transform=None):
        #1907_NEW_1Hz_TRUNC.zarr'
        self.data = load_data_from_zarr(zarr_path)
        self.transform = transform





    def __len__(self):
        return self.data.shape[1]  # Adjust according to the dimension representing the number of samples

    def __getitem__(self, idx):
        sample = self.data[:, idx, :].values
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data_from_zarr(zarr_path):
    zarr_array = zarr.open(zarr_path, mode='r')
    return xr.DataArray(zarr_array)


def transform_to_tensor(data):
    return torch.from_numpy(data).float().unsqueeze(0)


#def get_zarr_data(batch_size, workers, split_dataset = True):
def get_zarr_data(split_dataset=True):
    print(os.path.abspath('./1907_NEW_1Hz_TRUNC.zarr'))


    # Example usage
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        ZarrDataset.SpecgramToTensor()
    ])

    full_dataset = ZarrDataset('./1907_NEW_1Hz_TRUNC.zarr', transform=transform_pipeline)

    #full_dataset = ZarrDataset('./1907_NEW_1Hz_TRUNC.zarr', transform=transform_to_tensor)
    if split_dataset:

        # Determine the size of the training and test sets
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Split the dataset into training and test sets
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Create dataloaders for the training and test sets
        #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


        #return train_dataloader, test_dataloader
        return train_dataset, test_dataset
    else:
        #return DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
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
