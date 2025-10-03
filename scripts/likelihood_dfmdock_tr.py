from pathlib import Path
import csv
import functools
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
import hydra

from diffenergy.likelihood import FlowTimeIntegral, DiffSpaceIntegral, DiffTimeIntegral
from diffenergy.dfmdock_tr.docked_dataset import DockingDataset
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel 
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.dfmdock_tr.divergence import (
    score_eval_wrapper_tr_ode, 
    score_eval_wrapper_tr_diffspace, 
    divergence_eval_wrapper_tr
)
from diffenergy.helper import diffusion_coeff, prior_dfmdock_tr

def del_sample_fn(sample, prev_sample):
    # Compute del_position as the difference between current and previous lig_pos
    if prev_sample is not None:
        center_lig_pos = sample[...,1,:].mean(dim=0)
        center_prev_lig_pos = prev_sample[...,1,:].mean(dim=0)
        center_del_pos = center_lig_pos - center_prev_lig_pos
    else:
        center_del_pos = torch.zeros_like(sample)
        center_del_pos = center_del_pos[...,1,:].mean(dim=0)
    
    return center_del_pos

def batch_process_fn(batch, device):
    # Get batch from testset loader
    _id = batch['id'][0]
    rec_x = batch['rec_x'].squeeze(0)
    lig_x = batch['lig_x'].squeeze(0)
    rec_pos = batch['rec_pos'].squeeze(0)
    lig_pos = batch['lig_pos'].squeeze(0)
    position_matrix = batch['position_matrix'].squeeze(0)

    # Wrap to a batch
    batch = {
        "id": _id,
        "rec_x": rec_x.to(device),
        "lig_x": lig_x.to(device),
        "rec_pos": rec_pos.to(device),
        "sample": lig_pos.to(device),
        "position_matrix": position_matrix.to(device),
    }
    return batch

@hydra.main(version_base=None, config_path="../configs", config_name="likelihood_dfmdock_tr")
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

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = False)

    # set models
    score_model = Score_Model.load_from_checkpoint(config.checkpoint)
    score_model.freeze()
    score_model.to(device)

    data_dir = config.data_directory
    
    esm_model = ESMLanguageModel()

    # load dataset
    if inference_type == 'FlowTimeIntegral':
        testset = DockingDataset(data_dir=data_dir, data_list=config.data_listpdb, esm_model=esm_model, esm_alphabet=esm_model.alphabet)
        dataloader = DataLoader(testset, batch_size=1, num_workers=config.num_workers, shuffle=False)
    else:
        with open(config.data_listpdb, 'r') as f:
            data_lists = f.read().splitlines()
        testsets = [DockingDataset(data_dir=data_dir, data_list=data_list, esm_model=esm_model, esm_alphabet=esm_model.alphabet) 
                    for data_list in data_lists]
        dataloaders = {data_list: DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers) 
                       for data_list, testset in zip(data_lists, testsets)}
    
    prior_likelihood_fn = functools.partial(prior_dfmdock_tr, sigma = sigma_max)

    if inference_type == 'FlowTimeIntegral':
        likelihood = FlowTimeIntegral(dataloader=dataloader,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper_tr_ode,
                                      divergence_eval_wrapper=divergence_eval_wrapper_tr,
                                      ode_steps=config.ode_steps,
                                      odeint_rtol=config.odeint_rtol,
                                      odeint_atol=config.odeint_atol,
                                      odeint_method=config.odeint_method,
                                      reset_seed_each_sample=config.reset_seed_each_sample,
                                      seed=config.seed,
                                      device=device)
        data_list = likelihood.run_likelihood()
    elif inference_type == 'DiffSpaceIntegral':
        likelihood = DiffSpaceIntegral(dataloaders=dataloaders,
                                       batch_process_fn=batch_process_fn,
                                       score_model=score_model,
                                       diffusion_coeff_fn=diffusion_coeff_fn,
                                       prior_likelihood_fn=prior_likelihood_fn,
                                       score_eval_wrapper=score_eval_wrapper_tr_diffspace,
                                       del_sample_fn=del_sample_fn,
                                       diffusion_steps=config.diffusion_steps,
                                       reset_seed_each_sample=config.reset_seed_each_sample,
                                       seed=config.seed,
                                       device=device)
        data_list = likelihood.run_likelihood()
    elif inference_type == 'DiffTimeIntegral':
        likelihood = DiffTimeIntegral(dataloaders=dataloaders,
                                      batch_process_fn=batch_process_fn,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      divergence_eval_wrapper=divergence_eval_wrapper_tr,
                                      diffusion_steps=config.diffusion_steps,
                                      reset_seed_each_sample=config.reset_seed_each_sample,
                                      seed=config.seed,
                                      device=device)
        data_list = likelihood.run_likelihood()
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

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
