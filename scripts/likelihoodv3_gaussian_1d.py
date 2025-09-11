from fileinput import filename
from logging import warning
from pathlib import Path
import csv
import functools
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, TypedDict

import numpy as np
from tqdm import tqdm
from diffenergy.likelihoodv3 import FlowEquivalentODEPath, IntegrablePath, InterpolatedUniformIntegrableSequence, LikelihoodIntegrand, LikelihoodResult, SpaceIntegrand, TimeIntegrand, TotalIntegrand, UniformIntegrableSequence, run_diff_likelihoods, run_ode_likelihoods
from diffenergy.perturbation import FlowPerturbationIntegral
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import hydra

from diffenergy.helper import marginal_prob_std, diffusion_coeff, prior_gaussian_1d
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

    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, to_array, from_array

    model_eval = ModelEval(score_model)

    # load integrands
    integrands:list[LikelihoodIntegrand] = []
    priors:list[Callable[[torch.Tensor,float],float]] = []
    types:DictConfig = config.integrand_types
    if not isinstance(types,Mapping):
        types = DictConfig({t:{} for t in types}) #ensure types is a DictConfig of Dicts

    for integrand_type,params in config.integrand_types.items():
        def param(p:str):
            if params is not None:
                if params.get(p,None) is not None:
                    return p
                return config.get(p)

        match integrand_type:
            case TotalIntegrand.__name__:
                integrand = TotalIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array,device=device)
            case TimeIntegrand.__name__:
                integrand = TimeIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array,device=device)
            case SpaceIntegrand.__name__:
                integrand = SpaceIntegrand(model_eval.score,model_eval.divergence,diffusion_coeff_fn,to_array,from_array,device=device)
            case _:
                raise ValueError("Unknown integrand type:",integrand_type)
            
        def prior_likelihood_fn(x:torch.Tensor,t:float):
            assert np.allclose(t, 1) #diffeq errors might mean it's not *quite* 1 but that's fine
            return torch.squeeze(prior_gaussian_1d({"sample":x},sigma_max)[0]).item()
        priors.append(prior_likelihood_fn)
        integrands.append(integrand)
        

    if len(integrands) == 0:
        raise ValueError("""Must specify at least one integrand type!\n
                         `integrand_types` can either be a list of strings (classnames) or a list of mappings from classnames to
                         integrand-specific parameters. If an integrand parameter is not provided in the integrand-specific section,
                         it will look for that parameter in the global scope instead (hence defaults/shared parameters can be provided globally).
                         To specify an empty mapping, simply include the `ClassName:` line without any subitems below it.""")
    
    

    # load path and associated dataset
    paths:Iterable[IntegrablePath[torch.Tensor]]
    match config.path_type:
        case "flow_ode":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_test_data(config.data_samples, batch_size=1, num_workers=config.num_workers)

            #ode integration: needs a timeschedule
            schedule = config.get("ode_timeschedule","uniform")
            match schedule:
                case "uniform":
                    times = list(np.linspace(0,1,config.ode_steps+1,endpoint=True));
                case _:
                    raise ValueError("Unknown timeschedule method:",schedule)

            paths = ( #maybe this should be a dataloader or something idk
                FlowEquivalentODEPath[torch.Tensor](
                    model_eval.score,
                    diffusion_coeff_fn,
                    times,
                    (from_array(initial["sample"]),0),
                    config.odeint_rtol,
                    config.odeint_atol,
                    config.odeint_method,
                    to_array,
                    from_array,
                    device=device)
                    for initial in tqdm(dataloader)
                )
        case "sde_trajectories":
            #sde: get paths from diffusion tajectories
            with open(config.trajectory_index, 'r') as f:
                data_lists = f.read().splitlines()
            dataloaders = {data_list: load_test_data(data_list, batch_size=1, num_workers=config.num_workers) for data_list in data_lists}

            
            pathclass = UniformIntegrableSequence[torch.Tensor]
            if config.interpolate_trajectories:
                pathclass = functools.partial(InterpolatedUniformIntegrableSequence[torch.Tensor],n_interp=config.num_interpolants,to_arr=to_array,from_arr=from_array)
            
            paths = (
                pathclass(map(lambda x: from_array(x["sample"]), loader),tmin=0,tmax=0)
                for name,loader in tqdm(dataloaders.items())
            )
        case _:
            raise ValueError("Unknown path type:",config.path_type)

    #run integral
    int_type = config.get("integral_type")
    if int_type == "ode":
        #assume paths are ode integrable. error will be thrown otherwise
        likelihoods = run_ode_likelihoods(paths,integrands,priors)
    elif int_type == "diff":
        #use standard integration
        likelihoods = run_diff_likelihoods(paths,integrands,priors)
    else:
        raise ValueError(f"Unknown integral type: {int_type}. For standard (non-ode solver) numerical integration, use integral_type: \"diff\" (the default).")


    # Write the data_list to a CSV file
    if likelihoods:
        out_dir = Path(config.out_dir)
        file_name = out_dir/"likelihood.csv"
        
        if out_dir.exists() and not config.overwrite_output:
            raise FileExistsError(out_dir,"Pass '++overwrite_output=True' in the command line (recommended over config) or use config.overwrite_output to overwrite existing output.")

        out_dir.mkdir(parents=True,exist_ok=True)


        config_copy = out_dir/"config.yaml"
        with open(config_copy,"w") as f:
            f.write(OmegaConf.to_yaml(config))


        with open(file_name, 'w', newline='') as f:
            fieldnames = flatten_keys({integrand.name():(LikelihoodResult.__required_keys__) for integrand in integrands})

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            # Write each dictionary in the list as a row
            for result in likelihoods:
                writer.writerow(flatten_dict(result))

if __name__ == '__main__':
    main()
