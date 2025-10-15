

from csv import DictWriter
import functools
from pathlib import Path
from typing import Literal
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from scripts.likelihoodv3 import get_paths
from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std, diffusion_coeff, prior_gaussian_nd
from diffenergy.likelihoodv3 import Array, ArrayLike, LikelihoodIntegrand
from scripts.likelihoodv3 import MapDataset, SizeWrappedIter, SizedIter, get_integrands, get_likelihoods, get_paths

from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, from_array_batch, to_array as to_array_nobatch, from_array as from_array_nobatch, to_array_batch
from scripts.likelihoodv3_gaussian_1d import load_model, load_samples, load_trajectories, load_trajectory


@hydra.main(version_base=None, config_path='../configs/likelihoodv3')
def main(config:DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    # set device
    device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    batched = config.get("batched",False)
    batch_size = int(config.batch_size) if batched else None
    assert batched == False

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
    
    def get_trajectory(path)->list[tuple[torch.Tensor,float]]:
        samples,times = load_trajectory(path,device=device)
        return list(zip(map(from_array,samples),times))


    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))
    
    
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
    
    out_folder = Path(config.out_dir)
    forces_folder = out_folder/'forces'
    forces_folder.mkdir(exist_ok=True,parents=True)
    index_file = out_folder/'force_index.csv'
    scorecols = [f'score']
    poscols = [f'pos']
    with open(index_file,'w',newline='') as f:
        index_writer = DictWriter(f,fieldnames=['id','Forces_CSV'])
        index_writer.writeheader()
        for (id,p) in paths:
            c = p.condition
            forces_csv_file = forces_folder/f'{id}.csv'
            index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
            with open(forces_csv_file,'w',newline='') as f2:
                forces_writer = DictWriter(f2,fieldnames=['Index','Timestep','Diffusion_Coeff','Divergence'] + scorecols + poscols)
                forces_writer.writeheader()
                for i,(x,t) in enumerate(p):
                    force = scorefn(x,t,c)
                    div = divergencefn(x,t,c)

                    forcedict = {"Index":i,"Timestep":torch.as_tensor(t).item(),'Diffusion_Coeff':diffusion_coeff_fn(t).item(),'Divergence':torch.as_tensor(div).item(),**dict(zip(scorecols,[force.item()])),**dict(zip(poscols,[x.item()]))}
                    forces_writer.writerow(forcedict)
            


if __name__ == "__main__":
    main()