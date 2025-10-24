

from csv import DictWriter
import functools
from pathlib import Path
from typing import Literal
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from diffenergy.dfmdock_tr.docked_dataset import PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.score_model import Score_Model
from scripts.likelihoodv3 import get_paths

from diffenergy.dfmdock_tr.utils.geometry import axis_angle_to_matrix
from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std, diffusion_coeff, prior_gaussian_nd
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDict, LigDict, ModelEval, to_array as to_array_nobatch, from_array as from_array_nobatch
from diffenergy.likelihoodv3 import Array, ArrayLike, LikelihoodIntegrand
from scripts.likelihoodv3 import MapDataset, SizeWrappedIter, SizedIter, get_integrands, get_likelihoods, get_paths
from scripts.likelihoodv3_dfmdock import load_samples, load_trajectories, load_trajectory, offset_trajectory_columns


@hydra.main(version_base=None, config_path='../configs/likelihoodv3')
def main(config:DictConfig):
    
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    torch.cuda.memory._record_memory_history(
       max_entries=100000
    )

    # set device
    device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    batched = config.get("batched",False)
    batch_size = int(config.batch_size) if batched else None
    if batched:
        raise ValueError("Batched DFMDock evaluation not supported!")

    to_array = to_array_nobatch# if not batched else to_array_batch
    from_array = functools.partial(from_array_nobatch,device=device)# if not batched else from_array_batch,device=device)

    # set sigma_values
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max

    # set models
    score_model = Score_Model.load_from_checkpoint(config.checkpoint)
    score_model.freeze()
    score_model.to(device)

    offset_type:Literal["Translation","Rotation","Translation+Rotation"] = config.offset_type
    valid_offsets = ["Translation", "Rotation", "Translation+Rotation"]
    if offset_type not in valid_offsets:
        raise ValueError("offset_type must be one of",valid_offsets)

    model_eval = ModelEval(score_model,offset_type=offset_type,reset_seed_each_eval=config.get("reset_seed_each_sample",False),manual_seed=config.get("seed",0))
    
    scorefn = model_eval.score# if not batched else model_eval.batch_score
    divergencefn = model_eval.divergence# if not batched else model_eval.batch_divergence

    esm_model = ESMLanguageModel()
    pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

    assert batch_size is None
    load_samples_fn = lambda: load_samples(config.data_samples, config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
    load_trajectories_fn = lambda: load_trajectories(config.trajectory_index_file,config.pdb_dir,config.trajectory_dir,pdb_importer,batch_size=batch_size)
    get_trajectory_fn = lambda trajectory_file: load_trajectory(trajectory_file, offset_type, device=device) #TODO: add pdb trajectory support, add pdb_dir as parameter [filenames are relative to pdb_dir]

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
                    get_trajectory_fn,
                    device)
    
    out_folder = Path(config.out_dir)
    forces_folder = out_folder/'forces'
    forces_folder.mkdir(exist_ok=True,parents=True)
    config_copy = out_folder/"config.yaml"
    with open(config_copy,"w") as f:
        f.write(OmegaConf.to_yaml(config))
    index_file = out_folder/'force_index.csv'
    cols = offset_trajectory_columns(offset_type)
    scorecols = [f'score:{col}' for col in cols]
    poscols = [f'pos:{col}' for col in cols]
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

                    forcedict = {"Index":i,"Timestep":torch.as_tensor(t).item(),'Diffusion_Coeff':diffusion_coeff_fn(t).item(),'Divergence':div.item(),**dict(zip(scorecols,force.tolist())),**dict(zip(poscols,x["offset"].tolist()))}
                    forces_writer.writerow(forcedict)
            


if __name__ == "__main__":
    main()