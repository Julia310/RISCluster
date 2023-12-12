#!/usr/bin/env python3

"""This script contains neural network architectures used in the RIS package.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
"""

import torch
import torch.nn as nn


# ======== This network is for data of dimension 100x87 (4 s) =================
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # Output: (4, 600, 51)
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 300, 26)
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 150, 13)
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 75, 7)
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 38, 4)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 19, 2)
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 10, 1)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(320, 64),  # Adjust the number of input features
            nn.ReLU(True),
            nn.Linear(64, 9),  # Further reduction
            nn.ReLU(True)

        )

    def forward(self, x):
        return self.encoder(x)




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent2dec = nn.Sequential(
            nn.Linear(9, 64),  # Further reduction
            nn.ReLU(True),
            nn.Linear(64, 320),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # [Initial layers to expand the latent vector]
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=(0,1), padding=1), # Output: (32, 19, 2)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1), # Output: (16, 38, 4)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1), # Output: (16, 75, 7)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=(1,0), padding=1),  # Output: (8, 150, 13)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, output_padding=1, padding=1),   # Output: (32, 38, 72)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, output_padding=(1,0), padding=1),   # Output: (4, 600, 51)
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, output_padding=(1, 0), padding=1),    # Output: (16, 150, 288)
            nn.ReLU(True),

        )


    def forward(self, x):
        x = self.latent2dec(x)
        x = x.view(-1, 32, 10, 1)  # Reshape to match the output of the last encoder convolutional layer
        x = self.decoder(x)
        return x


class AEC(nn.Module):
    """
    Description: Autoencoder model; combines encoder and decoder layers.
    Inputs:
        - Input data (spectrograms)
    Outputs:
        - Reconstructed data
        - Latent space data
    """
    def __init__(self):
        super(AEC, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


def init_weights(m):
    """
    Description: Initializes weights with the Glorot Uniform distribution.
    Inputs:
        - Latent space data
    Outputs:
        - Reconstructed data
    """
    if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ClusteringLayer(nn.Module):
    """
    Description: Generates soft cluster assignments using latent features as
    input.
    Arguments:
        - n_clusters: User-defined
        - n_features: Must match output dimension of encoder.
        - alpha: Exponential factor (default: 1.0)
        - weights: Initial values for the cluster centroids
    Inputs:
        Encoded data (output of encoder)
    Outputs:
        Soft cluster assignments
    """
    def __init__(self, n_clusters, n_features=9, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_features = int(n_features)
        self.n_clusters = int(n_clusters)
        self.alpha = alpha
        if weights is None:
            initial_weights = torch.zeros(
                self.n_clusters, self.n_features, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_weights)
        else:
            initial_weights = weights
        self.weights = nn.Parameter(initial_weights)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weights
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x


class DEC(nn.Module):
    """Description: Deep Embedded Clustering Model; combines autoencoder with
    clustering layer to generate end-to-end deep embedded clustering neural
    network model.

    Parameters
    ----------
    n_clusters : int
        Number of clusters

    Returns
    -------
    q : array
        Soft cluster assignments

    x : array
        Reconstructed data

    z : array
        Latent space data
    """
    def __init__(self, n_clusters):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.clustering = ClusteringLayer(self.n_clusters)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        q = self.clustering(z)
        return q, x, z
