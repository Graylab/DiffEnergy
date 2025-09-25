from fileinput import filename
from logging import warning
import os
from pathlib import Path
import csv
import functools
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, TypedDict

import numpy as np
from tqdm import tqdm
from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel, batched_normpdf_matrix, batched_normpdf_scalar
from diffenergy.likelihoodv3 import FlowEquivalentODEPath, IntegrablePath, IntegrableSequence, InterpolatedIntegrableSequence, InterpolatedUniformIntegrableSequence, LikelihoodIntegrand, LinearPath, LinearizedFlowPath, SpaceIntegrand, TimeIntegrand, TotalIntegrand, UniformIntegrableSequence, run_diff_likelihoods, run_ode_likelihoods
from diffenergy.perturbation import FlowPerturbationIntegral
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import hydra

from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std, diffusion_coeff, prior_gaussian_1d
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, to_array, from_array

class datapoint(TypedDict):
    id:torch.Tensor
    sample:torch.Tensor

class gaussian_1d_dataset(Dataset[datapoint]):
    def __init__(self, ids:torch.Tensor, samples:torch.Tensor):
        self.ids = ids
        self.samples = samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx)->datapoint:
        return {'id': self.ids[idx], 'sample': self.samples[idx]}
    
class interpolated_gaussian_1d_dataset(Dataset[datapoint]):
    def __init__(self,ids,samples,num_interp):
        self.ids = ids
        self.samples = samples
        self.num_interp = num_interp

    def __len__(self):
        return len(self.ids)*(self.num_interp - 1) + 1

    def __getitem__(self, idx)->datapoint:
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

def load_trajectories(trajectory_index_file:str|Path)->dict[str,str]:
    trajectory_index_file = Path(trajectory_index_file)
    assert trajectory_index_file.suffix == '.csv'
    df = pd.read_csv(trajectory_index_file)
    d = df.set_index('index').T.to_dict(orient='records')[0] #make ids columnames, then convert to a dict of [{colname:value,colname:value}] and get the first result
    d = {id:trajectory_index_file.parent/p for id,p in d.items()}
    return d
    

def load_endpoints(data_path:str):
    samples,steps = load_trajectory(data_path)
    assert steps[0] == 0 and steps[-1] == 1
    return samples[0], samples[1]

def load_trajectory(data_path:str)->tuple[torch.Tensor,torch.Tensor]:
    df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'

    # we reverse the trajectory so it matches flow, with time going from 0 to 1. 
    # have to do this and numpy AND make a copy, cause tensors don't support negative stride -_-
    df = df.iloc[::-1].copy()

    if "Timestep" in df.columns:
        times = torch.as_tensor(df.loc[:, "Timestep"].values,dtype=torch.float32) #Extract Timestep column as times.
    else:
        steps = torch.as_tensor(df.loc[:,"Index"].values,dtype=torch.float32) # Extract the Index column, convert to timesteps
        times = 1 - steps/steps.max() #steps go from 0 to N, so divide by N and subtract from 1 to get time from 1 to 0
    
    samples = torch.as_tensor(df.loc[:, "Sample"].values,dtype=torch.float32)  # Extract the Sample column
    
    return samples[:,None], times  #add dimension to samples so iteration of 1d vectors



T = TypeVar("T")
def flatten_dict(d:Mapping[str,Mapping[str,T]],formatstr="{}_{}"):
    return {
        formatstr.format(outname,inname): val
        for outname, inner in d.items()
        for inname,val in inner.items()
    }

def flatten_keys(d:Mapping[str,Iterable[str]],formatstr="{}_{}"):
    return [
        formatstr.format(outname,inname)
        for outname, inner in d.items()
        for inname in inner
    ]



@hydra.main(version_base=None, config_path="../configs/likelihoodv3", config_name="likelihood_gaussian_1d")
def main(config: DictConfig):
    global from_array

    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    # set device
    
    device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    from_array = functools.partial(from_array,device=device)

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    # set marginal probability distribution and diffusion coefficient distribution
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.clamp_diffusion_coefficient)
    # if config.path_type == 'FlowTimeIntegral':
    #     diffusion_coeff_fn = functools.partial(
    #         diffusion_coeff_fn, clamp = True)

    # set models
    weights_path = config.checkpoint
    ckpt = torch.load(weights_path, map_location = device)

    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

    tr_type = config.tr_type

    # Initialize score model, load the checkpoint weights into the model    
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
    
    

    

    ### LOAD INTEGRANDS
    integrands:list[LikelihoodIntegrand] = []
    types:DictConfig = config.integrand_types
    if isinstance(types,str):
        types = types.split(" ")
    if not isinstance(types,Mapping):
        types = DictConfig({t:{} for t in types}) #ensure types is a DictConfig of Dicts

    def prior_likelihood_fn(x:torch.Tensor,t:float):
        assert np.allclose(t, 1), t #diffeq errors might mean it's not *quite* 1 but that's fine
        return torch.squeeze(prior_gaussian_1d({"sample":x},sigma_max)[0]).item()

    for integrand_type,params in types.items():
        def param(p:str,*args):
            if params is not None:
                if params.get(p,None) is not None:
                    return p
            return config.get(p,*args)
            
        match integrand_type:
            case TotalIntegrand.__name__:
                integrand = TotalIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array)
            case TimeIntegrand.__name__:
                integrand = TimeIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array)
            case SpaceIntegrand.__name__:
                integrand = SpaceIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array)
            case _:
                raise ValueError("Unknown integrand type:",integrand_type)
        integrands.append(integrand)

    
    if len(integrands) == 0:
        raise ValueError("""Must specify at least one integrand type!\n
                         `integrand_types` can either be a list of strings (classnames) or a list of mappings from classnames to
                         integrand-specific parameters. If an integrand parameter is not provided in the integrand-specific section,
                         it will look for that parameter in the global scope instead (hence defaults/shared parameters can be provided globally).
                         To specify an empty mapping, simply include the `ClassName:` line without any subitems below it.""")


    ### LOAD PRIORS
    priors:list[tuple[str,Callable[[torch.Tensor,float],float]]] = []
    functions:DictConfig = config.get("prior_fns","smax_gaussian")
    if isinstance(functions,str):
        functions = functions.split(" ")
    if not isinstance(functions,Mapping):
        functions = DictConfig({f:{} for f in functions}) #ensure types is a DictConfig of Dicts
    
    for prior_fn,params in functions.items():
        match prior_fn:
            case "convolved_data":
                means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float)
                variances = torch.tensor([8.0,5.0,10.0])**2
                weights = torch.tensor([0.4,0.3,0.3])

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

                def gt_prior_fn(x:torch.Tensor,t:float):
                    if gt_time is not None: assert np.allclose(t, gt_time), t
                    currvar = variances + int_diffusion_coeff_sq(t,sigma_min=sigma_min,sigma_max=sigma_max)
                    dx = x[:,None,...] - means[None,...]
                    if variances.ndim == 1:
                        probs = batched_normpdf_scalar(dx,currvar) #shape:BxN
                    else:
                        probs = batched_normpdf_matrix(dx,currvar) #shape:BxN
                    logprob = torch.log(torch.sum(weights[None,...]*probs,dim=-1)) #shape:N
                    return logprob.squeeze().item()
                    
                priors.append((prior_fn,gt_prior_fn))

                print("using ground truth prior fn")

    
            case "smax_gaussian":
                priors.append((prior_fn,prior_likelihood_fn))

    ### LOAD PATHS
    #ode integration: needs a timeschedule. Unused if diffusion / other trajectory used
    match config.get("ode_timeschedule","uniform"):
        case "uniform":
            ode_times = torch.linspace(0,1,config.ode_steps+1,device=device)
        case default:
            raise ValueError("Unknown timeschedule method:",default)


    # load path and associated dataset
    paths:Iterable[tuple[str,IntegrablePath[torch.Tensor]]]
    match config.path_type:
        case "flow_ode":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_test_data(config.data_samples, batch_size=1, num_workers=config.num_workers)

            paths = ( #maybe this should be a dataloader or something idk
                (initial["id"],FlowEquivalentODEPath[torch.Tensor](
                    model_eval.score,
                    diffusion_coeff_fn,
                    ode_times,
                    (from_array(initial["sample"]),0),
                    config.odeint_rtol,
                    config.odeint_atol,
                    config.odeint_method,
                    to_array,
                    from_array))
                    for initial in tqdm(dataloader)
                )
        case "sde_trajectories":
            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories(config.trajectory_index_file)
            
            pathclass = IntegrableSequence[torch.Tensor]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[torch.Tensor],n_interp=config.num_interpolants)

            def get_trajectory(path):
                samples,times = load_trajectory(path)
                return zip(map(from_array,samples),times)
            
            paths = (
                (id,pathclass(list(get_trajectory(path)),
                              to_arr=to_array,
                              from_arr=from_array,))
                for id,path in tqdm(trajectories.items())
            )
        case "linear_trajectories":
            #linear: take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories(config.trajectory_index_file)
            
            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory)) for id,trajectory in tqdm(trajectories.items())) 

            paths = (
                (id,LinearPath((from_array(start),0),(from_array(end),1),ode_times,
                            config.odeint_rtol,
                            config.odeint_atol,
                            config.odeint_method,
                            to_array,
                            from_array))
                for (id,(start,end)) in endpoints
            )
        case "linearized_flow":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_test_data(config.data_samples, batch_size=1, num_workers=config.num_workers)

            paths = ( #maybe this should be a dataloader or something idk
                (initial["id"],LinearizedFlowPath[torch.Tensor](
                    model_eval.score,
                    diffusion_coeff_fn,
                    ode_times,
                    (from_array(initial["sample"]),0),
                    config.odeint_rtol,
                    config.odeint_atol,
                    config.odeint_method,
                    to_array,
                    from_array))
                    for initial in tqdm(dataloader)
                )
        case "data_translation":
            # Translation in data space: like linear_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling

            # take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories(config.trajectory_index_file)
            
            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory)) for id,trajectory in tqdm(trajectories.items())) 

            paths = (
                (id,LinearPath((from_array(start),0),(from_array(end),0),ode_times, #start and end both 0!
                            config.odeint_rtol,
                            config.odeint_atol,
                            config.odeint_method,
                            to_array,
                            from_array))
                for (id,(start,end)) in endpoints
            )

        case _:
            raise ValueError("Unknown path type:",config.path_type)


    ### RUN LIKELIHOOD COMPUTATION
    int_type = config.get("integral_type")
    if int_type == "ode":
        #assume paths are ode integrable. error will be thrown otherwise
        likelihoods = run_ode_likelihoods(paths,integrands,priors)
    elif int_type == "diff":
        #use standard integration
        likelihoods = run_diff_likelihoods(paths,integrands,priors)
    else:
        raise ValueError(f"Unknown integral type: {int_type}. For standard (non-ode solver) numerical integration, use integral_type: \"diff\" (the default).")


    ### WRITE OUTPUT
    out_dir = Path(config.out_dir)
    likelihoods_name = out_dir/"likelihood.csv"
    
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


    config_copy = out_dir/"config.yaml"
    with open(config_copy,"w") as f:
        f.write(OmegaConf.to_yaml(config))


    with open(likelihoods_name, 'w', newline='') as f:
        fieldnames = ['id',"prior_position","prior_time"] + [f"prior:{name}" for name,prior in priors] + [f"integrand:{integrand.name()}" for integrand in integrands]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        # Write each dictionary in the list as a row
        # with torch.autograd.profiler.profile(record_shapes=True,with_stack=True,use_device='cuda',profile_memory=True) as p:
        for id,prior_pt,prior_results,ingtegrand_results in likelihoods:
            row = {"id":int(id),
                   "prior_position":prior_pt[0].item(),  #since here we're in 1d, let's make it a scalar w/ item()
                    "prior_time":torch.as_tensor(prior_pt[1]).item(), 
                    **{f"prior:{name}":val for name,val in prior_results.items()},
                    **{f"integrand:{name}":val for name,val in ingtegrand_results.items()}}
            writer.writerow(row)

if __name__ == '__main__':
    main()
