from pathlib import Path
import csv
import functools
from diffenergy.perturbation import FlowPerturbationIntegral
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import hydra

from diffenergy.likelihood import DiffTotalIntegral, FlowSpaceIntegral, FlowTimeIntegral, DiffSpaceIntegral, DiffTimeIntegral, FlowTotalIntegral#, UnmovingIntegral
from diffenergy.helper import marginal_prob_std, diffusion_coeff, prior_gaussian_1d
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.gaussian_1d.divergence import score_eval_wrapper, divergence_eval_wrapper

def del_sample_fn(sample, prev_sample):
    if prev_sample is not None:
        del_sample = sample - prev_sample
    else:
        del_sample = torch.zeros_like(sample)

    del_sample = del_sample.reshape(-1)

    return del_sample

def batch_process_fn(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch

class gaussian_1d_dataset(Dataset):
    def __init__(self, ids, samples):
        self.ids = ids
        self.samples = samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'id': self.ids[idx], 'sample': self.samples[idx]}
    
class interpolated_gaussian_1d_dataset(Dataset):
    def __init__(self,ids,samples,num_interp):
        self.ids = ids
        self.samples = samples
        self.num_interp = num_interp

    def __len__(self):
        return len(self.ids)*(self.num_interp - 1) + 1

    def __getitem__(self, idx):
        idx1 = idx // self.num_interp
        idx2 = idx1 + 1

        if idx2 >= len(self.ids): #final point
            return {'id': self.ids[idx1], 'sample': self.samples[idx1]}

        frac1 = (idx % self.num_interp) / self.num_interp
        frac2 = 1 - frac1

        id = self.ids[idx1]*frac1 + self.ids[idx2]*frac2 #we love numeric ids
        samp = self.samples[idx1]*frac1 + self.samples[idx2]*frac2
        return {'id': id, 'sample': samp}


def load_test_data(data_path, batch_size, num_workers):
    """Loads dataset from a CSV file and returns a DataLoader."""

    df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
    ids = df.iloc[:, 0].values  # Extract the first column as ids
    samples = df.iloc[:, 1].values  # Extract the second column as samples

    # Convert to tensors
    ids = torch.tensor(ids, dtype=torch.int64)  # ids as integers
    samples = torch.tensor(samples, dtype=torch.float32)  # samples as integers 

    dataset = gaussian_1d_dataset(ids, samples)  # Create a dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader

def load_interpolated_test_data(data_path, batch_size, num_workers, num_interpolants:int=1):
    """Loads dataset from a CSV file and returns a DataLoader. Uses the interpolated gaussian_1d_dataset to interpolate the dataset"""

    df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
    ids = df.iloc[:, 0].values  # Extract the first column as ids
    samples = df.iloc[:, 1].values  # Extract the second column as samples

    # Convert to tensors
    ids = torch.tensor(ids, dtype=torch.int64)  # ids as integers
    samples = torch.tensor(samples, dtype=torch.float32)  # samples as integers 

    dataset = interpolated_gaussian_1d_dataset(ids, samples, num_interpolants)  # Create a dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader

@hydra.main(version_base=None, config_path="../configs", config_name="likelihood_gaussian_1d")
def main(config: DictConfig):

    # Print the entire configuration
    print(OmegaConf.to_yaml(config))
    file_name = Path(config.out_file)

    parent_folder = file_name.parent
    if not parent_folder.exists():
        parent_folder.mkdir()
    # Check if the file exists to determine whether to write the header
    file_exists = file_name.exists()

    inference_type = config.inference_type

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    # set marginal probability distribution and diffusion coefficient distribution
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = False)
    if inference_type == 'FlowTimeIntegral':
        diffusion_coeff_fn = functools.partial(
            diffusion_coeff_fn, clamp = True)

    # set models
    weights_path = config.checkpoint
    ckpt = torch.load(weights_path, map_location = device)

    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

    tr_type = config.tr_type

    # Initialize score model
    if tr_type == 'non_conservative':
        score_model = ScoreNetMLP(
            input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
    elif tr_type == 'conservative':
        score_model = NegativeGradientMLP(
            input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
    else:
        raise ValueError(tr_type)

    # Load the checkpoint weights into the model    
    score_model.load_state_dict(ckpt)


    dataloader:DataLoader = None
    dataloaders:dict[str,DataLoader] = None

    # load dataset
    if "Flow" in inference_type:
        dataloader = load_test_data(config.data_samples, batch_size=1, num_workers=config.num_workers)
    else:
        load_data_fn = load_test_data
        if config.interpolate_trajectories:
            load_data_fn = functools.partial(load_interpolated_test_data,num_interpolants=config.num_interpolants)
        with open(config.trajectory_index_file, 'r') as f:
            data_lists = f.read().splitlines()
        dataloaders = {data_list: load_data_fn(data_list, batch_size=1, num_workers=config.num_workers) for data_list in data_lists}
    
    prior_likelihood_fn = functools.partial(prior_gaussian_1d, sigma = sigma_max)

    if inference_type == 'FlowTimeIntegral':
        likelihood = FlowTimeIntegral(dataloader=dataloader,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper,
                                      divergence_eval_wrapper=divergence_eval_wrapper,
                                      ode_steps=config.ode_steps,
                                      odeint_rtol=config.odeint_rtol,
                                      odeint_atol=config.odeint_atol,
                                      odeint_method=config.odeint_method,
                                      device=device)
    elif inference_type == 'FlowSpaceIntegral':
        likelihood = FlowSpaceIntegral(dataloader=dataloader,
                                       batch_process_fn=batch_process_fn,
                                       score_model=score_model,
                                       diffusion_coeff_fn=diffusion_coeff_fn,
                                       prior_likelihood_fn=prior_likelihood_fn,
                                       score_eval_wrapper=score_eval_wrapper,
                                       ode_steps=config.ode_steps,
                                       odeint_rtol=config.odeint_rtol,
                                       odeint_atol=config.odeint_atol,
                                       odeint_method=config.odeint_method,
                                       device=device)
    elif inference_type == 'FlowTotalIntegral':
        likelihood = FlowTotalIntegral(dataloader=dataloader,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper,
                                      divergence_eval_wrapper=divergence_eval_wrapper,
                                      ode_steps=config.ode_steps,
                                      odeint_rtol=config.odeint_rtol,
                                      odeint_atol=config.odeint_atol,
                                      odeint_method=config.odeint_method,
                                      device=device)
    elif inference_type == 'UnmovingIntegral':
        likelihood = UnmovingIntegral(dataloader=dataloader,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper,
                                      divergence_eval_wrapper=divergence_eval_wrapper,
                                      ode_steps=config.ode_steps,
                                      odeint_rtol=config.odeint_rtol,
                                      odeint_atol=config.odeint_atol,
                                      odeint_method=config.odeint_method,
                                      device=device);
    elif inference_type == 'DiffSpaceIntegral':
        likelihood = DiffSpaceIntegral(dataloaders=dataloaders,
                                       batch_process_fn=batch_process_fn,
                                       score_model=score_model,
                                       diffusion_coeff_fn=diffusion_coeff_fn,
                                       prior_likelihood_fn=prior_likelihood_fn,
                                       score_eval_wrapper=score_eval_wrapper,
                                       del_sample_fn=del_sample_fn,
                                       diffusion_steps=config.diffusion_steps,
                                       device=device)
    elif inference_type == 'DiffTimeIntegral':
        likelihood = DiffTimeIntegral(dataloaders=dataloaders,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper,
                                      divergence_eval_wrapper=divergence_eval_wrapper,
                                      diffusion_steps=config.diffusion_steps,
                                      device=device)
    elif inference_type == 'DiffTotalIntegral':
        likelihood = DiffTotalIntegral(dataloaders=dataloaders,
                                       batch_process_fn=batch_process_fn,
                                       score_model=score_model,
                                       diffusion_coeff_fn=diffusion_coeff_fn,
                                       prior_likelihood_fn=prior_likelihood_fn,
                                       divergence_eval_wrapper=divergence_eval_wrapper,
                                       score_eval_wrapper=score_eval_wrapper,
                                       del_sample_fn=del_sample_fn,
                                       diffusion_steps=config.diffusion_steps,
                                       device=device)
    elif inference_type == 'FlowPerturbationIntegral':
        likelihood = FlowPerturbationIntegral(dataloader=dataloader,
                                               batch_process_fn=batch_process_fn,
                                               score_model=score_model,
                                               diffusion_coeff_fn=diffusion_coeff_fn,
                                               prior_likelihood_fn=prior_likelihood_fn,
                                               score_eval_wrapper=score_eval_wrapper,
                                               divergence_eval_wrapper=divergence_eval_wrapper,
                                               ode_steps=config.ode_steps,
                                               num_perturbations=config.num_perturbations,
                                               perturbation_sigma=config.perturbation_sigma,
                                               odeint_rtol=config.odeint_rtol,
                                               odeint_atol=config.odeint_atol,
                                               odeint_method=config.odeint_method,
                                               device=device)
    elif inference_type == 'FlowPerturbationIntegral':
        likelihood = FlowPerturbationIntegral(dataloader=dataloader,
                                               batch_process_fn=batch_process_fn,
                                               score_model=score_model,
                                               diffusion_coeff_fn=diffusion_coeff_fn,
                                               prior_likelihood_fn=prior_likelihood_fn,
                                               score_eval_wrapper=score_eval_wrapper,
                                               divergence_eval_wrapper=divergence_eval_wrapper,
                                               ode_steps=config.ode_steps,
                                               num_perturbations=config.num_perturbations,
                                               perturbation_sigma=config.perturbation_sigma,
                                               odeint_rtol=config.odeint_rtol,
                                               odeint_atol=config.odeint_atol,
                                               odeint_method=config.odeint_method,
                                               device=device)

    else:
        raise ValueError(f"Unknown inference type: {inference_type}")
    
    data_list = likelihood.run_likelihood()

    # Write the data_list to a CSV file
    if data_list:

        with open(file_name, 'a', newline='') as f:

            writer = csv.DictWriter(f, fieldnames=data_list[0].keys())

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write each dictionary in the list as a row
            for data in data_list:
                writer.writerow(data)

if __name__ == '__main__':
    main()
