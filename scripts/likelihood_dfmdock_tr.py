from pathlib import Path
import csv
import functools
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
import hydra

from diffenergy.likelihood import FlowTimeIntegral, DiffSpaceIntegral, DiffTimeIntegral
from diffenergy.dfmdock_tr.docked_dataset import DockingDataset
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.dfmdock_tr.divergence import score_eval_wrapper_tr, divergence_eval_wrapper_tr
from diffenergy.helper_gpu import diffusion_coeff, prior_likelihood

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

class DockingDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['sample'] = item['ligand_pos']
        
        return item

@hydra.main(version_base=None, config_path="../configs", config_name="likelihood_dfmdock_tr")
def main(config: DictConfig):

    # Print the entire configuration
    print(OmegaConf.to_yaml(config))
    file_name = Path(config.out_file)

    parent_folder = file_name.parent()
    if not parent_folder.exists():
        parent_folder.mkdir()
    # Check if the file exists to determine whether to write the header
    file_exists = file_name.exists()

    inference_type = config.inference

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = False)

    # set models
    score_model = Score_Model.load_from_checkpoint(config.ckpt)
    score_model.freeze()
    score_model.to(device)

    # batch size
    batch_size = config.batch_size

    data_dir = config.data_directory

    # load dataset
    if inference_type == 'FlowTimeIntegral':
        testset = DockingDataset(data_dir=data_dir, data_list=config.data_listpdb)
        testset = DockingDatasetWrapper(testset)
        dataloader = DataLoader(testset, batch_size=batch_size, num_workers=6)
    else:
        with open(config.data_listpdb, 'r') as f:
            data_lists = f.read().split('\n')
        testsets = [DockingDataset(data_dir=data_dir, data_list=data_list) for data_list in data_lists]
        testsets = [DockingDatasetWrapper(testset) for testset in testsets]
        dataloaders = [DataLoader(testset, batch_size=batch_size, num_workers=6) for testset in testsets]
    
    prior_likelihood_fn = functools.partial(prior_likelihood, sigma = sigma_max)

    if inference_type == 'FlowTimeIntegral':
        likelihood = FlowTimeIntegral(dataloader=dataloader,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      score_eval_wrapper=score_eval_wrapper_tr,
                                      divergence_eval_wrapper=divergence_eval_wrapper_tr,
                                      diffusion_steps=config.diffusion_steps,
                                      device=device)
        data_list = likelihood.run_likelihood()

    elif inference_type == 'DiffSpaceIntegral':
        likelihood = DiffSpaceIntegral(dataloaders=dataloaders,
                                       score_model=score_model,
                                       diffusion_coeff_fn=diffusion_coeff_fn,
                                       prior_likelihood_fn=prior_likelihood_fn,
                                       score_eval_wrapper=score_eval_wrapper_tr,
                                       del_sample_fn=del_sample_fn,
                                       diffusion_steps=config.diffusion_steps,
                                       device=device)
        data_list = likelihood.run_likelihood()

    elif inference_type == 'DiffTimeIntegral':
        likelihood = DiffTimeIntegral(dataloaders=dataloaders,
                                      score_model=score_model,
                                      diffusion_coeff_fn=diffusion_coeff_fn,
                                      prior_likelihood_fn=prior_likelihood_fn,
                                      divergence_eval_wrapper=divergence_eval_wrapper_tr,
                                      diffusion_steps=config.diffusion_steps,
                                      device=device)
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
