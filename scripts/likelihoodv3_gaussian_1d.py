from fileinput import filename
import itertools
from logging import warning
import math
import os
from pathlib import Path
import csv
import functools
import shutil
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, TypeVarTuple, TypedDict, overload

import numpy as np
from tqdm import tqdm
from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel, batched_normpdf_matrix, batched_normpdf_scalar
from diffenergy.likelihoodv3 import FlowEquivalentODEPath, IntegrablePath, IntegrableSequence, InterpolatedIntegrableSequence, InterpolatedUniformIntegrableSequence, LikelihoodIntegrand, LinearPath, LinearizedFlowPath, PerturbedPath, ScoreDivDiffIntegrand, SpaceIntegrand, TimeIntegrand, TotalIntegrand, UniformIntegrableSequence, run_diff_likelihoods, run_ode_likelihoods
from diffenergy.perturbation import FlowPerturbationIntegral
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import hydra

from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std, diffusion_coeff, prior_gaussian_1d, prior_gaussian_nd
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, from_array_batch, to_array as to_array_nobatch, from_array as from_array_nobatch, to_array_batch

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


@overload
def load_test_data(data_path, batch_size:None,device:str|torch.device='cuda')->list[tuple[str,torch.Tensor]]: ...
@overload
def load_test_data(data_path, batch_size:int,device:str|torch.device='cuda')->list[tuple[Sequence[str],torch.Tensor]]: ...
def load_test_data(data_path, batch_size:int|None=None, device:str|torch.device='cuda')->list[tuple[str,torch.Tensor]]|list[tuple[Sequence[str],torch.Tensor]]:
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
        return [(str(int(id.item())),x) for id,x in zip(ids,samples)]
    else:
        assert batch_size > 0
        return [([str(int(id)) for id in ids[i*batch_size:(i+1)*batch_size].tolist()],samples[i*batch_size:(i+1)*batch_size]) 
                for i in range(math.ceil(len(samples)/batch_size))]


@overload
def load_trajectories(trajectory_index_file:str|Path,batch_size:int)->list[tuple[tuple[str,...],tuple[Path,...]]]: ...
@overload
def load_trajectories(trajectory_index_file:str|Path,batch_size:None)->list[tuple[str,Path]]: ...
def load_trajectories(trajectory_index_file:str|Path,batch_size:int|None=None)->list[tuple[str,Path]]|list[tuple[tuple[str,...],tuple[Path,...]]]:
    trajectory_index_file = Path(trajectory_index_file)
    assert trajectory_index_file.suffix == '.csv'
    df = pd.read_csv(trajectory_index_file)
    dlist:dict[str,str] = df.set_index('index').T.to_dict(orient='records')[0] #make ids columnames, then convert to a dict of [{colname:value,colname:value}] and get the first result
    res = [(str(id),trajectory_index_file.parent/p) for id,p in dlist.items()]
    if batch_size is None:
        return res
    else:
        return [(tuple(b[0] for b in batch),tuple(b[1] for b in batch)) for batch in itertools.batched(res,batch_size)]



def load_endpoints(data_path:str|Path|tuple[str|Path,...],device:str|torch.device='cuda'):
    samples,steps = load_trajectory(data_path,device=device)
    assert steps[0] == 0 and steps[-1] == 1
    return samples[0], samples[-1] #Time dimension is always first, even in batching

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

    batched = config.get("batched",False)
    batch_size = int(config.batch_size) if batched else None

    to_array = to_array_nobatch if not batched else to_array_batch
    from_array = functools.partial(from_array_nobatch if not batched else from_array_batch,device=device)

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
    
    scorefn = model_eval.score if not batched else model_eval.batch_score
    divergencefn = model_eval.divergence if not batched else model_eval.batch_divergence

    ### LOAD INTEGRANDS
    integrands:list[LikelihoodIntegrand] = []
    types:DictConfig = config.integrand_types
    if types is None:
        types = []
    if not isinstance(types,Mapping):
        types = DictConfig({t:{} for t in types}) #ensure types is a DictConfig of Dicts

    for integrand_type,params in types.items():
        def param(p:str,*args):
            if params is not None:
                if params.get(p,None) is not None:
                    return p
            return config.get(p,*args)
            
        intclasses:dict[str,type[ScoreDivDiffIntegrand]] = {cls.__name__:cls for cls in [TotalIntegrand,TimeIntegrand,SpaceIntegrand]}

        if integrand_type in intclasses:
            intcls = intclasses[integrand_type]
            integrand = intcls(scorefn,divergencefn,diffusion_coeff_fn,to_array,from_array)
        else:
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

                def gt_prior_fn(x:torch.Tensor,t:float):
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
                def prior_likelihood_fn(x:torch.Tensor,t:float):
                    tt = torch.as_tensor(t)
                    assert torch.allclose(tt, torch.ones_like(tt)), t #diffeq errors might mean it's not *quite* 1 but that's fine
                    res = (prior_gaussian_nd(x,sigma_max)[0])
                    return res.numpy(force=True) if batched else res.item()
                priors.append((prior_fn,prior_likelihood_fn))

    ### LOAD PATHS
    #ode integration: needs a timeschedule. Unused if diffusion / other trajectory used
    match config.get("ode_timeschedule","uniform"):
        case "uniform":
            ode_times = torch.linspace(0,1,config.ode_steps+1,device=device)
        case default:
            raise ValueError("Unknown timeschedule method:",default)


    # load path and associated dataset
    paths:Iterable[tuple[str|Sequence[str],IntegrablePath[torch.Tensor]]]
    match config.path_type:
        case "flow_ode":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_test_data(config.data_samples,batch_size=batch_size, device=device)

            paths = ( #maybe this should be a dataloader or something idk
                (id,FlowEquivalentODEPath[torch.Tensor](
                    scorefn,
                    diffusion_coeff_fn,
                    ode_times,
                    (from_array(initial),0),
                    config.odeint_rtol,
                    config.odeint_atol,
                    config.odeint_method,
                    to_array,
                    from_array))
                    for (id,initial) in tqdm(dataloader)
                )
        case "sde_trajectories":
            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories(config.trajectory_index_file,batch_size=batch_size)
            
            pathclass = IntegrableSequence[torch.Tensor]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[torch.Tensor],n_interp=config.num_interpolants)

            def get_trajectory(path):
                samples,times = load_trajectory(path,device=device)
                return zip(map(from_array,samples),times)
            
            paths = (
                (id,pathclass(list(get_trajectory(path)),
                              to_arr=to_array,
                              from_arr=from_array,))
                for id,path in tqdm(trajectories)
            )
        case "linear_trajectories":
            #linear: take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories(config.trajectory_index_file, batch_size=batch_size)
            
            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,device=device)) for id,trajectory in trajectories) 

            paths = (
                (id,LinearPath[torch.Tensor]((from_array(start),0),(from_array(end),1),ode_times,
                            config.odeint_rtol,
                            config.odeint_atol,
                            config.odeint_method,
                            to_array,
                            from_array))
                for (id,(start,end)) in tqdm(endpoints)
            )
        case "linearized_flow":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_test_data(config.data_samples, batch_size=batch_size, device=device)

            paths = ( #maybe this should be a dataloader or something idk
                (id,LinearizedFlowPath[torch.Tensor](
                    scorefn,
                    diffusion_coeff_fn,
                    ode_times,
                    (from_array(sample),0),
                    config.odeint_rtol,
                    config.odeint_atol,
                    config.odeint_method,
                    to_array,
                    from_array))
                    for id,sample in tqdm(dataloader)
                )
        case "diff_data_translation":
            # Diffusion trajectory solely in data space: like sde_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling
            
            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories(config.trajectory_index_file, batch_size=batch_size)
            
            pathclass = IntegrableSequence[torch.Tensor]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[torch.Tensor],n_interp=config.num_interpolants)

            def get_trajectory(path):
                samples,times = load_trajectory(path,device=device)
                return zip(map(from_array,samples),times)
            
            paths = (
                (id,pathclass([(x,0) for x,t in (get_trajectory(path))],
                              to_arr=to_array,
                              from_arr=from_array,))
                for id,path in tqdm(trajectories)
            )

        case "data_translation":
            # Linear translation in data space: like linear_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling

            # take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories(config.trajectory_index_file, batch_size=batch_size)
            
            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,device=device)) for id,trajectory in tqdm(trajectories)) 

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

    if config.get("perturb_path",False):
        if config.integral_type == "ode":
            raise ValueError("Can't stochastically perturb an ODE! However, ODEIntegrablePaths can be used in discrete integral mode. Please set integral_type to 'diff' or disable perturbation")
        sigma:float = config.perturbation_sigma
        schedule:str = config.get("perturbation_schedule","data")
        paths = ((id,PerturbedPath(path,schedule,sigma)) for id,path in paths) #god I love generators


    ### RUN LIKELIHOOD COMPUTATION
    int_type = config.get("integral_type")
    parallel = config.get("parallel",False)
    cluster_kwargs = config.get("cluster_kwargs",{})
    if parallel:
        import ray
        ray.init(**cluster_kwargs)
    actor_kwargs = config.get("actor_kwargs",{})
    if device.type == 'cuda' and 'num_gpus' not in actor_kwargs:
        actor_kwargs['num_gpus'] = 1 #assume each actor will consume an entire gpu
    if int_type == "ode":
        #assume paths are ode integrable. error will be thrown otherwise
        likelihoods = run_ode_likelihoods(paths,integrands,parallel=parallel,remote_kwargs=actor_kwargs)
    elif int_type == "diff":
        #use standard integration
        likelihoods = run_diff_likelihoods(paths,integrands,parallel=parallel,remote_kwargs=actor_kwargs)
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

    ## WRITE CONFIG
    config_copy = out_dir/"config.yaml"
    with open(config_copy,"w") as f:
        f.write(OmegaConf.to_yaml(config))

    ## WRITE TRAJECTORIES PREP
    if config.get("save_trajectories",False):
        trajectory_folder = out_dir/"trajectories"
        try:
            next(trajectory_folder.glob("*")) #if any files in directory, clear directory
            shutil.rmtree(trajectory_folder)
        except StopIteration:
            pass
        trajectory_folder.mkdir(exist_ok=True)
    else:
        trajectory_folder = None

    ## WRITE LIKELIHOODS
    with open(likelihoods_name, 'w', newline='') as f:
        fieldnames = ['id',"prior_position","prior_time"] + [f"prior:{name}" for name,prior in priors] + [f"integrand:{integrand.name()}" for integrand in integrands]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ids,trajectories,times,integrand_resultss in likelihoods:

            ##Calculate priors
            prior_endpoint:tuple[torch.Tensor,float] = (trajectories[-1], times[-1])

            prior_result:dict[str,float|list[float]] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_endpoint))) for name,prior_fn in priors}




            ## Unbatch results for writing
            if not batched:
                ids:Iterable[str] = [ids]
                trajectories = [trajectories]
                times = [times]
                integrand_resultss = [integrand_resultss]
                prior_resultss = [prior_result]
                prior_endpoints = [prior_endpoint]
            else:
                batch = len(prior_endpoint[0]) #size of this batch, could be smaller than batch_size if last batch
                ids: Iterable[str] = ids
                trajectories = torch.stack(trajectories,dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
                assert trajectories.ndim == 3 #BxNxD
                times = itertools.repeat(times)
                integrand_resultss = [
                    {name:result[i] for name,result in integrand_resultss.items()}
                    for i in range(batch)
                ]
                prior_resultss = [
                    {name:result[i] for name,result in prior_result.items()}
                    for i in range(batch)
                ]
                prior_endpoints = [(prior_endpoint[0][i],prior_endpoint[1]) for i in range(batch)]

                
            
            for id, trajectory, time, integrand_results, prior_results, prior_endpoint \
                in zip(ids,trajectories,times, integrand_resultss, prior_resultss, prior_endpoints):
                assert isinstance(id,str),id
                if trajectory_folder: #save trajectory to folder
                    trajectory_file = trajectory_folder/f"trajectory_{id}.csv"
                    xtraj = torch.as_tensor(trajectory).numpy(force=True)
                    assert xtraj.ndim == 2
                    assert xtraj.shape[1] == 1
                    xtraj = xtraj[:,0] #1d position into scalar
                    ttraj = torch.as_tensor(time).numpy(force=True)
                    trajectory_df = pd.DataFrame({"Timestep":ttraj,"Sample":xtraj})
                    trajectory_df.to_csv(trajectory_file,index_label="Index")

                row = {"id":int(id),
                    "prior_position":prior_endpoint[0].item(),  #since here we're in 1d, let's make it a scalar w/ item()
                        "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                        **{f"prior:{name}":val for name,val in prior_results.items()},
                        **{f"integrand:{name}":val for name,val in integrand_results.items()}}
                writer.writerow(row)

                f.flush()

if __name__ == '__main__':
    main()
