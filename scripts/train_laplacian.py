# --------------------------------------------------------------------------
# author Sudeep Sarma
# minimal diffusion model using MLP for a laplacian distribution training
# --------------------------------------------------------------------------
import torch
import functools
import numpy as np
from torch.optim import Adam
import torchvision.transforms as transforms
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from laplacian_network import ScoreNetMLP, NegativeGradientMLP
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
import hydra

from diffenergy.laplacian.dataset import TrimodalGaussianSampler, TrimodalGaussianDataset
from diffenergy.laplacian.loss import loss_fn
from diffenergy.laplacian.helper_gpu import marginal_prob_std, diffusion_coeff


# --------------------------------------------------------------------------------
# Sample data points from trimodal gaussian distribution
# P(x) = w1 * N(mu1, sigma1**2) + w2 * N(mu2, sigma2**2)+ w3 * N(mu3, sigma3**2)
# --------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="train_1d")
def main(config: DictConfig):
  
  # Print the entire configuration
  print(OmegaConf.to_yaml(config))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # set seed
  np.random.seed(1)
  torch.manual_seed(1)

  sigma_min = config.sigma_min
  sigma_max = config.sigma_max
  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max)

  # noise
  sigma_noise = config.sigma_noise
  # number of epochs
  n_epochs = config.n_epochs
  # size of a mini-batch
  batch_size = config.batch_size
  # percentage of data to use as the test set
  test_size = config.test_size
  # percentage of data to use as the validation set
  val_size = config.val_size
  # type of training data
  tr_data = config.tr_data

  outpath = Path(config.out_dir)
  if not outpath.exists():
    outpath.mkdir()

  # import dataset and split into train and validation sets
  if tr_data == 'laplace':
    # # Trimodal laplace sampler
    # sampler = TrimodalLaplaceSampler(mu1=-30, mu2=0, mu3=40, b1=8.0, b2=3.0, b3=10.0, w1=0.4, w2=0.3, w3=0.3)
    # dataset = LaplaceDataset(sampler, noise_std = sigma_noise, num_samples = 20000)
    # train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    raise NotImplementedError("Laplace dataset is not implemented yet.")
  elif tr_data == 'trimodal_gaussian':
    sampler = TrimodalGaussianSampler(mu1=-30, sigma1=8.0, w1=0.4 , mu2=0, sigma2=5.0, w2=0.3, mu3=40, sigma3=10.0, w3=0.3)
    dataset = TrimodalGaussianDataset(sampler, noise_std=0.1, num_samples=20000)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

  tr_type = config.tr_type

  # create a model
  input_dim = sampler(1).unsqueeze(1).size(1)

  if tr_type == 'non_conservative':
    score_model = torch.nn.parallel.DataParallel(ScoreNetMLP(input_dim, embed_dim = 512, layers = (512, 512, 512), marginal_prob_std = marginal_prob_std_fn))
    score_model = score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr = 1e-4, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
  elif tr_type == 'conservative':
    score_model = torch.nn.parallel.DataParallel(NegativeGradientMLP(input_dim, embed_dim = 512, layers = (512, 512, 512), marginal_prob_std = marginal_prob_std_fn))
    score_model = score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr = 1e-4, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

  tqdm_epoch = tqdm.trange(n_epochs)
  loss_train_values = []

  for epoch in tqdm_epoch:
    # Training loop
    score_model.train()
    avg_train_loss = 0.
    num_train_items = 0

    for noisy_data, clean_data in train_loader:
      noisy_data = noisy_data.to(device)
      loss = loss_fn(score_model, noisy_data, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()
      # Add gradient clipping
      torch.nn.utils.clip_grad_norm_(score_model.parameters(), max_norm=1.0)
      optimizer.step()
      avg_train_loss += loss.item() * noisy_data.shape[0]
      num_train_items += noisy_data.shape[0]

    avg_train_loss /= num_train_items
    scheduler.step(avg_train_loss)
    loss_train_values.append(avg_train_loss)

    # Print the averaged training loss so far.
    tqdm_epoch.set_description(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}')
    # Update the checkpoint after each epoch of training.
    wt_file = outpath / config.wt_file
    torch.save(score_model.state_dict(), wt_file)

  # Plot the loss curve of training and validation

  window_size = 10
  smoothed_loss_train_values = np.convolve(loss_train_values, np.ones(window_size)/window_size, mode='valid')
  
  # Training loss
  plt.figure()
  plt.plot(range(window_size, n_epochs + 1), smoothed_loss_train_values, label=f'Smoothed Training Loss (window={window_size})', linewidth=2)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Curve')
  plt.legend()
  plt.savefig(outpath / 'loss_curve.png')

if __name__ == '__main__':
    main()
