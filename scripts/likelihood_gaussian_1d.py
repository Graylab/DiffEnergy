from fileinput import filename
from io import TextIOWrapper
import itertools
from logging import warning
import math
import os
from pathlib import Path
import csv
import functools
import shutil
from typing import Callable, Iterable, Mapping, Optional, Sequence, overload

import numpy as np

from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel, batched_normpdf_matrix, batched_normpdf_scalar

from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import hydra

from diffenergy.helper import diffusion_coeff, int_diffusion_coeff_sq, marginal_prob_std, prior_gaussian_nd
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, from_array_batch, to_array as to_array_nobatch, from_array as from_array_nobatch, to_array_batch
from diffenergy.likelihood import LikelihoodIntegrand
from scripts.likelihood import get_integrands, get_likelihoods, SizedIter, get_paths, ArrayLike

@overload
def load_samples(data_path, batch_size:None,device:str|torch.device='cuda')->SizedIter[tuple[str,torch.Tensor,None]]: ...
@overload
def load_samples(data_path, batch_size:int,device:str|torch.device='cuda')->SizedIter[tuple[Sequence[str],torch.Tensor,None]]: ...
def load_samples(data_path, batch_size:int|None=None, device:str|torch.device='cuda')->SizedIter[tuple[str,torch.Tensor,None]]|SizedIter[tuple[Sequence[str],torch.Tensor,None]]:
    """Loads dataset from a CSV file and returns an iterable of tuples ('id',x).
    if batch_size is None, each x will be an array of shape D. Otherwise, x has shape batch_sizexD
    """

    df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
    ids = df.loc[:, "index"].values  # Extract the first column as ids
    samples = df.loc[:, "Samples"].values  # Extract the second column as samples

    # Convert to tensors
    ids = torch.tensor(ids, dtype=torch.int64)  # ids as integers
    samples = torch.tensor(samples, dtype=torch.float32, device=device)[...,None]  # samples as floats, add a dimension to make them vectors

    if batch_size is None:
        return [(str(int(id.item())),x,None) for id,x in zip(ids,samples)]
    else:
        assert batch_size > 0
        return [([str(int(id)) for id in ids[i*batch_size:(i+1)*batch_size].tolist()],samples[i*batch_size:(i+1)*batch_size],None) 
                for i in range(math.ceil(len(samples)/batch_size))]


@overload
def load_trajectories(trajectory_index_file:str|Path,batch_size:int)->list[tuple[tuple[str,...],tuple[Path,...],None]]: ...
@overload
def load_trajectories(trajectory_index_file:str|Path,batch_size:None)->list[tuple[str,Path,None]]: ...
def load_trajectories(trajectory_index_file:str|Path,batch_size:int|None=None)->list[tuple[str,Path,None]]|list[tuple[tuple[str,...],tuple[Path,...],None]]:
    trajectory_index_file = Path(trajectory_index_file)
    assert trajectory_index_file.suffix == '.csv'
    df = pd.read_csv(trajectory_index_file)
    dlist:dict[str,str] = df.set_index('index').T.to_dict(orient='records')[0] #make ids columnames, then convert to a dict of [{colname:value,colname:value}] and get the first result
    res = [(str(id),trajectory_index_file.parent/p,None) for id,p in dlist.items()]
    if batch_size is None:
        return res
    else:
        return [(tuple(b[0] for b in batch),tuple(b[1] for b in batch),None) for batch in itertools.batched(res,batch_size)]


def load_trajectory(data_path:str|Path|tuple[str|Path,...],device:str|torch.device='cuda')->tuple[torch.Tensor,torch.Tensor]:
    batched = isinstance(data_path,tuple)
    paths = data_path if batched else [data_path]
    alltimes:Optional[torch.Tensor] = None
    sampleslist:list[torch.Tensor] = []
    for path in paths:
        df = pd.read_csv(path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'

        # we reverse the trajectory so it matches flow, with time going from 0 to 1. 
        # have to do this and numpy AND make a copy, cause tensors don't support negative stride -_-
        df = df.iloc[::-1].copy()

        if "Timestep" in df.columns:
            times = torch.as_tensor(df.loc[:, "Timestep"].values,dtype=torch.float32) #Extract Timestep column as times.
        else:
            steps = torch.as_tensor(df.loc[:,"Index"].values,dtype=torch.float32) # Extract the Index column, convert to timesteps
            times = 1 - steps/steps.max() #steps go from 0 to N, so divide by N and subtract from 1 to get time from 1 to 0

        if alltimes is not None:
            assert torch.allclose(alltimes,times)
        else:
            alltimes = times
        
        sampleslist.append(torch.as_tensor(df.loc[:, "Sample"].values,dtype=torch.float32,device=device))  # Extract the Sample column
    assert alltimes is not None
    samples = torch.stack(sampleslist,dim=1) if batched else sampleslist[0] #needs to be NxB 
    return samples[...,None],alltimes  #add dimension to samples so iteration of 1d vectors


@hydra.main(version_base=None, config_path="../configs/likelihoodv3", config_name="likelihood_gaussian_1d")
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    # set device
    device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    batched = config.get("batched",False)
    batch_size = int(config.batch_size) if batched else None

    to_array = to_array_nobatch if not batched else to_array_batch
    from_array = functools.partial(from_array_nobatch if not batched else from_array_batch,device=device)

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    model_eval = load_model(config,sigma_min,sigma_max,device)
    
    scorefn = model_eval.score if not batched else model_eval.batch_score
    divergencefn = model_eval.divergence if not batched else model_eval.batch_divergence

    load_samples_fn = lambda: load_samples(config.data_samples, batch_size=batch_size, device=device)
    load_trajectories_fn = lambda: load_trajectories(config.trajectory_index_file,batch_size=batch_size)
    
    def get_trajectory(path,cond:None)->list[tuple[torch.Tensor,float]]:
        samples,times = load_trajectory(path,device=device)
        return list(zip(map(from_array,samples),times))


    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))
    
    
    priors = load_priors(config,
                         sigma_min,
                         sigma_max,
                         device,
                         batch_size)

    integrands = get_integrands(config,
                                from_array,
                                to_array,
                                scorefn,
                                divergencefn,
                                diffusion_coeff_fn)

    paths = get_paths(config,
                      from_array,
                      to_array,
                      scorefn,
                      divergencefn,
                      diffusion_coeff_fn,
                      load_samples_fn,
                      load_trajectories_fn,
                      get_trajectory,
                      device)

    likelihoods = get_likelihoods(config,
                                 paths,
                                 integrands,
                                 device)
    
    write_likelihood_outputs(config,
                             likelihoods,
                             integrands,
                             priors,
                             batch_size)
    
def load_model(config:DictConfig,
               sigma_min:float,
               sigma_max:float,
               device:torch.device):
    # set marginal probability distribution and diffusion coefficient distribution
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)

    # set models
    weights_path = config.checkpoint
    ckpt = torch.load(weights_path, map_location = device)

    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

    # Initialize score model, load the checkpoint weights into the model    
    tr_type = config.tr_type
    if tr_type == 'non_conservative':
        score_model = ScoreNetMLP(
            input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
        print("wow")
        score_model.load_state_dict(ckpt)
        model_eval = ModelEval(score_model)
    elif tr_type == 'conservative':
        score_model = NegativeGradientMLP(
            input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
        score_model.load_state_dict(ckpt)
        model_eval = ModelEval(score_model)
    elif tr_type == 'ground_truth':
        means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float)
        variances = torch.tensor([8.0,5.0,10.0])**2
        weights = torch.tensor([0.4,0.3,0.3])

        model_eval = MultimodalGaussianGroundTruthScoreModel(means,variances,weights,sigma_min,sigma_max)
        model_eval.to(device)
    else:
        raise ValueError(tr_type)
    
    return model_eval
    
def load_priors(config:DictConfig,
                sigma_min:float,
                sigma_max:float,
                device:torch.device,
                batch_size:int|None):
    
    batched = batch_size is not None
    
    ### LOAD PRIORS
    priors:list[tuple[str,Callable[[torch.Tensor,float,None],float]]] = []
    functions:DictConfig = config.get("prior_fns","smax_gaussian")
    if isinstance(functions,str):
        functions = functions.split(" ")
    if not isinstance(functions,Mapping):
        functions = DictConfig({f:{} for f in functions}) #ensure types is a DictConfig of Dicts
    
    for prior_fn,params in functions.items():
        match prior_fn:
            case "convolved_data":
                means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float, device=device)
                variances = torch.tensor([8.0,5.0,10.0], device=device)**2
                weights = torch.tensor([0.4,0.3,0.3], device=device)

                gt_time = params.get("time",1)
                if gt_time == "Any":
                    gt_time = None
                else:
                    try:
                        gt_time = float(gt_time)
                    except:
                        pass
                    if not isinstance(gt_time,float) or not (0 <= gt_time and gt_time <= 1):
                        raise ValueError(f"'time' parameter to the convolved_data prior can only be a float in [0,1] or \"Any\", received {gt_time}")
                gt_time = torch.as_tensor(gt_time, dtype=torch.float, device=device)

                def gt_prior_fn(x:torch.Tensor,t:float,condition:None):
                    tt = torch.as_tensor(t)
                    if gt_time is not None: assert torch.allclose(tt, gt_time), t
                    currvar = variances + int_diffusion_coeff_sq(tt,sigma_min=sigma_min,sigma_max=sigma_max)

                    assert x.ndim == 2,x.shape #BxD
                    assert means.ndim == 2,x.shape #NxD
                    dx = x[:,None,...] - means[None,...]
                    if variances.ndim == 1:
                        probs = batched_normpdf_scalar(dx,currvar) #shape:BxN
                    else:
                        probs = batched_normpdf_matrix(dx,currvar) #shape:BxN
                    logprob = torch.log(torch.sum(weights[None,...]*probs,dim=-1)) #shape:B
                    
                    return logprob.squeeze().item() if not batched else logprob.numpy(force=True)
                    
                priors.append((prior_fn,gt_prior_fn))

                print("using ground truth prior fn")

    
            case "smax_gaussian":
                def prior_likelihood_fn(x:torch.Tensor,t:float,condition:None):
                    tt = torch.as_tensor(t)
                    assert torch.allclose(tt, torch.ones_like(tt)), t #diffeq errors might mean it's not *quite* 1 but that's fine
                    res = (prior_gaussian_nd(x,sigma_max)[0])
                    return res.numpy(force=True) if batched else res.item()
                priors.append((prior_fn,prior_likelihood_fn))
    
    return priors


def write_likelihood_outputs(
        config:DictConfig,
        likelihoods:Iterable[tuple[int|str|Sequence[str|int],Sequence[torch.Tensor],Sequence[float],None,dict[str,Sequence[float|ArrayLike]]]],
        integrands:list[LikelihoodIntegrand[torch.Tensor,None]],
        priors:Sequence[tuple[str,Callable[[torch.Tensor,float,None],float]]],
        batch_size:int|None,
        ):
    batched = batch_size is not None

    ### WRITE OUTPUT
    out_dir = Path(config.out_dir)
    
    if out_dir.exists():
        if not config.get("overwrite_output",False):
            raise FileExistsError(out_dir,"Pass '++overwrite_output=True' in the command line (recommended over config) or use config.overwrite_output to overwrite existing output.")
        else:
            backup_out = out_dir.with_stem(out_dir.stem + "_backup")
            warning(f"Moving dir {out_dir} to backup directory {backup_out}. Subsequent calls will DELETE THIS BACKUP, so be careful!!")
            if backup_out.exists():
                shutil.rmtree(backup_out)
            os.rename(out_dir,backup_out)
    out_dir.mkdir(parents=True,exist_ok=True)

    ## WRITE CONFIG
    config_copy = out_dir/"config.yaml"
    with open(config_copy,"w") as f:
        f.write(OmegaConf.to_yaml(config))

    ## WRITE LIKELIHOODS PREP
    likelihoods_file = out_dir/"likelihood.csv"
    write_likelihoods = config.get("write_likelihoods",True)
    likelihoods_handle: Optional[TextIOWrapper] = None
    likelihoods_writer: Optional[csv.DictWriter] = None
    if write_likelihoods:
        likelihoods_handle = open(likelihoods_file,"w")
        fieldnames = ['id',"prior_position","prior_time"] + [f"prior:{name}" for name,prior in priors] + [f"integrand:{integrand.name()}" for integrand in integrands]
        likelihoods_writer = csv.DictWriter(likelihoods_handle,fieldnames=fieldnames)
        likelihoods_writer.writeheader()

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
    save_trajectory_index = config.get("write_trajectory_index",False) and save_trajectories
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
        for ids,trajectories,times,conditions,integrand_resultss in likelihoods:

            ##Calculate priors
            prior_eval_endpoint:tuple[torch.Tensor,float,None] = (trajectories[-1], times[-1], conditions)

            prior_result:dict[str,float|list[float]] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_eval_endpoint))) for name,prior_fn in priors}

            ## Unbatch results for writing
            #since trajectories are in 'array' form they always have a batch dimension
            trajectories = torch.stack(trajectories,dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
            assert trajectories.ndim == 3 #BxNxD
            if not batched:
                ids:Iterable[str|int] = [ids]
                times = [times]
                integrand_resultss = [integrand_resultss]
                prior_resultss = [prior_result]
            else:
                batch = len(prior_eval_endpoint[0]) #size of this batch, could be smaller than batch_size if last batch
                ids: Iterable[str|int] = ids
                times = itertools.repeat(times)
                integrand_resultss = [
                    {name:np.array(result)[...,i] for name,result in integrand_resultss.items()}
                    for i in range(batch)
                ]
                prior_resultss = [
                    {name:np.array(result)[...,i] for name,result in prior_result.items()}
                    for i in range(batch)
                ]

            for id, trajectory, time, integrand_results, prior_results \
                in zip(ids, trajectories, times, integrand_resultss, prior_resultss):
                prior_endpoint:tuple[torch.Tensor,float] = (trajectory[-1], time[-1])
                assert isinstance(id,str|int),id


                if write_likelihoods:
                    row = {"id":int(id),
                        "prior_position":prior_endpoint[0].item(),  #since here we're in 1d, let's make it a scalar w/ item()
                            "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                            **{f"prior:{name}":val for name,val in prior_results.items()},
                            **{f"integrand:{name}":val[-1] for name,val in integrand_results.items()}} #save the last accumulated likelihood
                    likelihoods_writer.writerow(row)

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
                    if integrand_results is not None:
                        for name,result in integrand_results.items():
                            trajectory_df[f"accumulated_integrand:{name}"] = result
                    trajectory_df.to_csv(trajectory_file,index_label="index")
                    

                    for cutoff,file,writer in trajectory_indices:
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow({"index":id,"filename":trajectory_name})
                    acc_trajnum += 1


    finally:
        #make sure the files are closed!!!
        if likelihoods_handle is not None:
            likelihoods_handle.close()
        if samples_handle is not None:
            samples_handle.close()
        for cutoff,file,writer in trajectory_indices:
            file.close()


if __name__ == '__main__':
    main()
