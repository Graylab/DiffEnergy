import numpy as np
import torch
from torch.utils.data import Dataset

def sample_trimodal_gaussian(n, mu1, sigma1, w1, mu2, sigma2, w2, mu3, sigma3, w3, dimension=2):
    """
    Inputs:
        n: Number of points
        mu1, sigma1: Mean and standard deviation of first Gaussian
        mu2, sigma2: Mean and standard deviation of second Gaussian
        mu3, sigma3: Mean and standard deviation of third Gaussian
        weight: w1, w2, w3
    Outputs:
        x: A vector of n points, torch.tensor
    """
    w1, w2, w3 = 0.4, 0.3, 0.3

    # Sample mixture component indices based on weights
    choices = torch.multinomial(torch.tensor([w1, w2, w3]), num_samples=n, replacement=True)

    # Sample from each Gaussian
    x = torch.zeros((n,dimension))
    x[choices == 0] = torch.tensor(np.random.normal(mu1, sigma1, (torch.sum(choices == 0),dimension)),dtype=torch.float32)
    x[choices == 1] = torch.tensor(np.random.normal(mu2, sigma2, (torch.sum(choices == 1),dimension)),dtype=torch.float32)
    x[choices == 2] = torch.tensor(np.random.normal(mu3, sigma3, (torch.sum(choices == 2),dimension)),dtype=torch.float32)

    return x

# Add noise to the data
def add_noise(x, sigma=0.1):
    """
    Inputs:
        x: A vector of n points, torch.tensor
        sigma: Standard deviation of the noise
    Outputs:
        x: A vector of n points, torch.tensor
    """
    noise = torch.randn_like(x) * sigma
    return x + noise

class TrimodalGaussianSampler(object):
    def __init__(self, mu1, sigma1, w1, mu2, sigma2, w2, mu3, sigma3, w3, dimension=2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.w1 = w1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.w2 = w2
        self.mu3 = mu3
        self.sigma3 = sigma3
        self.w3 = w3
        self.dimension = 2

    def __call__(self, n):
        return sample_trimodal_gaussian(n, self.mu1, self.sigma1, self.w1, self.mu2, self.sigma2, self.w2, self.mu3, self.sigma3, self.w3, dimension=self.dimension)

class TrimodalGaussianDataset(Dataset):
    def __init__(self, sampler:TrimodalGaussianSampler, noise_std, num_samples=1000):
        """
        Args: 
            sampler: Instance of TrimodalGaussianSampler
            noise_std: Standard deviation of the noise
            num_samples: Number of samples to generate
        """
        self.sampler = sampler
        self.noise_std = noise_std
        self.num_samples = num_samples

        self.data = self.sampler(num_samples) #no unsqueeze cause nd
        self.noisy_data = add_noise(self.data, self.noise_std)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample.
        Returns:
            Tuple of (noisy data, clean target).
        """
        return self.noisy_data[idx], self.data[idx]
