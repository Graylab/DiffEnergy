

#Sampling is done using the same mechanism as normal likelihood computation - specifically, using ReverseSDEPath and saving the output
import functools
import math
from typing import Sequence
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from diffenergy.gaussian_1d.likelihood_helpers import from_array_batch, to_array as to_array_nobatch, from_array as from_array_nobatch, to_array_batch
from diffenergy.helper import diffusion_coeff
from scripts.likelihood import SizedIter, get_likelihoods, get_paths
from scripts.likelihood import MapDataset
from scripts.likelihood_gaussian_1d import load_model, write_likelihood_outputs


@hydra.main(version_base=None, config_path="../configs", config_name="sample_gaussian_1d")
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))


    with open_dict(config):
        ## Set various sampling/trajectory output config parameters based on sampling parameters:
        # out_dir -> out_dir [unchanged]
        # wt_file | checkpoint -> checkpoint [take either for backwards compatibility]
        # sample_file: Raise warning if specified, all samples now have filename "samples.csv"
        # sample_num: Discard for dfmdock, dfmdock needs samples_per_pdb
        # num_steps -> sde_steps
        # save_trajectory -> save_trajectories: Also sets write_trajectory_index to True

        ## Other relevant parameters:
        # trajectory_extra_indices: list of index cutoffs (e.g. [25, 1000]), default=[]. Will always include a full index
        
        config.path_type = "reverse_sde"
        config.integral_type = "diff"
        config.sde_timeschedule = "reverse_uniform" #make sure to go from t=1 to t=0!

        if config.get("wt_file",None):
            config.checkpoint = config.wt_file
        
        if config.get("sample_file",None):
            warnings.warn("sample_file included, but will be ignored! Sample files are now always saved to {out_dir}/samples.csv!")
        
        if config.get("num_steps",None):
            config.sde_steps = config.num_steps

        if config.get("save_trajectory",False):
            config.save_trajectories = True

        if config.get("save_trajectories",False):
            config.write_trajectory_index = True
        
        config.write_samples=True
        config.write_likelihoods=False
        
    # set device
    device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    batched = config.get("batched",False)
    batch_size = int(config.batch_size) if batched else None

    if batch_size == -1: 
        batch_size = config.sample_num #Do it in one **massive** batch because why not

    to_array = to_array_nobatch if not batched else to_array_batch
    from_array = functools.partial(from_array_nobatch if not batched else from_array_batch,device=device)

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    # set models
    model_eval = load_model(config,sigma_min,sigma_max,device);
    
    scorefn = model_eval.score if not batched else model_eval.batch_score
    divergencefn = model_eval.divergence if not batched else model_eval.batch_divergence

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))


    ## SAMPLE DIFFUSION TRAJECTORIES
    def load_noised_samples()->SizedIter[tuple[int|Sequence[int],torch.Tensor,None]]:
        bsize = batch_size or 1
        indices = [(i,) for i in range(math.ceil(config.sample_num//bsize))] #make into tuples grumble
        def getchunk(i:int):
            ids = range(config.sample_num)[i*bsize:(i+1)*bsize]
            chunksize = len(ids)
            x = torch.randn(chunksize,1,device=device)*sigma_max
            if not batched:
                ids = ids[0]
            return (ids,x,None)
        return MapDataset(indices,getchunk)

    def err(*args): raise ValueError()
    load_trajectories_fn = err
    get_trajectory_fn = err

    
    paths = get_paths(config,
                      from_array,
                      to_array,
                      scorefn,
                      divergencefn,
                      diffusion_coeff_fn,
                      load_noised_samples,
                      load_trajectories_fn,
                      get_trajectory_fn,
                      device)


    ## COMPUTE "LIKELIHOODS"
    likelihoods = get_likelihoods(config,
                                 paths,
                                 [],
                                 device)
    
    write_likelihood_outputs(config,
                             likelihoods,
                             [],
                             [],
                             batch_size)
    

if __name__ == '__main__':
    main()
