

#Sampling is done using the same mechanism as normal likelihood computation - specifically, using ReverseSDEPath and saving the output
import csv
import functools
from io import TextIOWrapper
import itertools
import math
import os
from pathlib import Path
import shutil
from typing import Iterable, Optional, Sequence, TypeVar
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pandas as pd
import torch

from diffenergy.gaussian_1d.likelihood_helpers import from_array_batch, to_array as to_array_nobatch, from_array as from_array_nobatch, to_array_batch
from diffenergy.helper import diffusion_coeff
from scripts.likelihood import SizedIter, get_likelihoods, get_paths
from scripts.likelihood import MapDataset
from scripts.likelihood_gaussian_1d import load_model, write_likelihood_outputs

X = TypeVar('X')
Y = TypeVar('Y')
def unzip(it:Iterable[tuple[X,Y]])->tuple[list[X],list[Y]]:
    x,y = zip(*it)
    return list(x),list(y)

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
    
    ### WRITE OUTPUT
    out_dir = Path(config.out_dir)
    
    if out_dir.exists():
        if not config.get("overwrite_output",False):
            raise FileExistsError(out_dir,"Pass '++overwrite_output=True' in the command line (recommended over config) or use config.overwrite_output to overwrite existing output.")
        else:
            backup_out = out_dir.with_stem(out_dir.stem + "_backup")
            warnings.warn(f"Moving dir {out_dir} to backup directory {backup_out}. Subsequent calls will DELETE THIS BACKUP, so be careful!!")
            if backup_out.exists():
                shutil.rmtree(backup_out)
            os.rename(out_dir,backup_out)
    out_dir.mkdir(parents=True,exist_ok=True)

    ## WRITE CONFIG
    config_copy = out_dir/"config.yaml"
    with open(config_copy,"w") as f:
        f.write(OmegaConf.to_yaml(config))
    
    ## WRITE SAMPLES PREP
    samples_file = out_dir/"samples.csv"
    write_samples = config.get("write_samples",False)
    samples_handle: Optional[TextIOWrapper] = None
    samples_writer: Optional[csv.DictWriter] = None

    if write_samples:
        samples_handle = open(samples_file,"w")
        fieldnames = ["index","Samples"]
        samples_writer = csv.DictWriter(samples_handle,fieldnames=fieldnames) #TODO: regularize capitalization aaaa
        samples_writer.writeheader()



    ## WRITE TRAJECTORIES PREP
    trajectory_folder = out_dir/"trajectories"
    trajectory_indices: list[tuple[int|None,TextIOWrapper,csv.DictWriter]] = [] #a little silly lol
    acc_trajnum = 0
    save_trajectories = config.get("save_trajectories",False)
    save_trajectory_index = config.get("write_trajectory_index",True) and save_trajectories
    if save_trajectories:
        try:
            next(trajectory_folder.glob("*")) #if any files in directory, clear directory
            shutil.rmtree(trajectory_folder)
        except StopIteration:
            pass
        trajectory_folder.mkdir(exist_ok=True)
    if save_trajectory_index:
        indices: list[tuple[int|None,TextIOWrapper]] = [(None,open(trajectory_folder/"trajectory_index.csv","w"))]
        if (extra := config.get("trajectory_extra_indices",None)):
            indices += [(ind,open(trajectory_folder/f"trajectory_index_{ind}.csv","w")) for ind in extra]
        trajectory_indices = [(ind,f,csv.DictWriter(f,fieldnames=["index","filename"])) for (ind,f) in indices]
        [index[2].writeheader() for index in trajectory_indices]
    

    ## WRITE OUTPUT
    try:
        for (ids,path) in paths:
            
            #potentially batched
            trajectories, times = unzip(path)

            ## Unbatch results for writing
            #since trajectories are in 'array' form they always have a batch dimension
            trajectories = torch.stack(trajectories,dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
            assert trajectories.ndim == 3 #BxNxD
            if not batched:
                ids:Iterable[str|int] = [ids]
                times = [times]
            else:
                ids: Iterable[str|int] = ids
                times = itertools.repeat(times)

            for id, trajectory, time in zip(ids, trajectories, times):
                if write_samples:
                    #TODO: CONFIGURATION FOR WHICH POINT TO SAVE
                    sample = {"index":id,"Samples":trajectory[-1].item()}
                    samples_writer.writerow(sample)

                if save_trajectories: #save trajectory to folder
                    trajectory_name = f"trajectory_{id}.csv"
                    trajectory_file = trajectory_folder/f"trajectory_{id}.csv"
                    xtraj = torch.as_tensor(trajectory).numpy(force=True)
                    assert xtraj.ndim == 2
                    assert xtraj.shape[1] == 1
                    xtraj = xtraj[:,0] #1d position into scalar
                    ttraj = torch.as_tensor(time).numpy(force=True)
                    trajectory_df = pd.DataFrame({"Timestep":ttraj,"Sample":xtraj})
                    trajectory_df.to_csv(trajectory_file,index_label="index")
                    

                    for cutoff,file,writer in trajectory_indices:
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow({"index":id,"filename":trajectory_name})
                    acc_trajnum += 1
    finally:
        #make sure the files are closed!!!
        if samples_handle is not None:
            samples_handle.close()
        for cutoff,file,writer in trajectory_indices:
            file.close()

if __name__ == '__main__':
    main()
