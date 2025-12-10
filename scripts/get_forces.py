import torch
from diffenergy.dfmdock_tr.docked_dataset import PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDockModelEval
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.helper import diffusion_coeff
from omegaconf import OmegaConf
from typing import Literal
from scripts.likelihood_class import DiffEnergyLikelihood


import functools
from contextlib import contextmanager
from csv import DictWriter

from scripts.dfmdock_class import DFMDockLikelihood
from scripts.gaussian_class import GaussianLikelihood
from scripts.likelihood import get_paths
from scripts.likelihood_dfmdock import (
    offset_trajectory_columns
)

# It's bad practice to inherit from the base class as a mixin but oh well. 
# I don't really feel like these belong in DiffEnergyLikelihood though
class ForcesMixin(DiffEnergyLikelihood):
    @functools.cached_property
    def forces_folder(self):
        forces_folder = self.out_dir/"forces"
        forces_folder.mkdir(exist_ok=True,parents=True)
        return forces_folder

    @property
    def forces_index_file(self):
        return self.out_dir/"force_index.csv"

    @contextmanager
    def forces_index_writer(self):
        with open(self.forces_index_file,'w',newline='') as f:
            index_writer = DictWriter(f,fieldnames=['id','Forces_CSV'])
            index_writer.writeheader()
            yield index_writer


class DFMDockForces(ForcesMixin, DFMDockLikelihood): #TODO: put this in the main class maybe? depends on how hydra cli should work
    def get_forces(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        if batched:
            raise ValueError("Batched DFMDock evaluation not supported!")

        to_array = self.to_array
        from_array = functools.partial(self.from_array,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        score_model = Score_Model.load_from_checkpoint(self.config.checkpoint,deterministic=self.config.get("deterministic_score",False))
        score_model.freeze()
        score_model.to(device)

        model_eval = DFMDockModelEval(score_model,offset_type=self.offset_type,reset_seed_each_eval=self.config.get("reset_seed_each_eval",False),manual_seed=self.config.get("seed",0))

        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        esm_model = ESMLanguageModel()
        pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

        assert batch_size is None
        load_samples_fn = lambda: self.load_samples(self.config.data_samples, self.config.pdb_dir, pdb_importer, device=device)  # noqa: E731
        load_trajectories_fn = lambda: self.load_trajectories(self.config.trajectory_index_file, self.config.pdb_dir, self.config.trajectory_dir, pdb_importer)  # noqa: E731
        get_trajectory_fn = lambda trajectory_file,condition: self.load_trajectory(trajectory_file, self.config.pdb_dir, condition, device=device)  # noqa: E731

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_samples_fn,
                        load_trajectories_fn,
                        get_trajectory_fn,
                        device)


        ### RUN FORCES

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        cols = self.offset_trajectory_columns
        scorecols = [f'score:{col}' for col in cols]
        poscols = [f'pos:{col}' for col in cols]

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        with self.forces_index_writer() as index_writer:
            for (id,P) in paths:
                if reset_seed_each_path:
                    torch.manual_seed(seed)

                c = P.condition
                forces_csv_file = self.forces_folder/f'{id}.csv'
                index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
                with open(forces_csv_file,'w',newline='') as f2:
                    forces_writer = DictWriter(f2,fieldnames=['Index','Timestep','Diffusion_Coeff','Divergence'] + scorecols + poscols)
                    forces_writer.writeheader()
                    for i,(x,t) in enumerate(P):
                        force = scorefn(x,t,c)
                        div = divergencefn(x,t,c)

                        forcedict = {
                            "Index":i,
                            "Timestep":torch.as_tensor(t).item(),
                            'Diffusion_Coeff':diffusion_coeff_fn(t).item(),
                            'Divergence':div.item(),
                            **dict(zip(scorecols,force.tolist())),
                            **dict(zip(poscols,x["offset"].tolist()))
                        }
                        forces_writer.writerow(forcedict)


class GaussianForces(ForcesMixin, GaussianLikelihood):
    def get_forces(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        assert not batched

        to_array = self.to_array if not batched else self.to_array_batch
        from_array = functools.partial(self.from_array if not batched else self.from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        model_eval = self.load_model(sigma_min,sigma_max,batched,device)
        
        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        load_samples_fn = (lambda: self.load_samples(self.config.data_samples, device=device)) if batch_size is None else (lambda: self.load_samples_batched(self.config.data_samples, batch_size, device=device))
        load_trajectories_fn = (lambda: self.load_trajectories(self.config.trajectory_index_file)) if batch_size is None else (lambda: self.load_trajectories_batched(self.config.trajectory_index_file,batch_size))
        
        def get_trajectory(path,c:None)->list[tuple[torch.Tensor,float]]:
            samples,times = self.load_trajectory(path,device=device)
            return list(zip(map(from_array,samples),times))


        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))
        
        
        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_samples_fn,
                        load_trajectories_fn,
                        get_trajectory,
                        device)
        
        
        ### RUN FORCES

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        scorecols = ['score']
        poscols = ['pos']

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        with self.forces_index_writer() as index_writer:
            for (id,P) in paths:
                if reset_seed_each_path:
                    torch.manual_seed(seed)

                c = P.condition
                forces_csv_file = self.forces_folder/f'{id}.csv'
                index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
                with open(forces_csv_file,'w',newline='') as f2:
                    forces_writer = DictWriter(f2,fieldnames=['Index','Timestep','Diffusion_Coeff','Divergence'] + scorecols + poscols)
                    forces_writer.writeheader()
                    for i,(x,t) in enumerate(P):
                        force = scorefn(x,t,c)
                        div = divergencefn(x,t,c)

                        forcedict = {
                            "Index":i,
                            "Timestep":torch.as_tensor(t).item(),
                            'Diffusion_Coeff':diffusion_coeff_fn(t).item(),
                            'Divergence':torch.as_tensor(div).item(),
                            **dict(zip(scorecols,[force.item()])),
                            **dict(zip(poscols,[x.item()]))
                        }
                        forces_writer.writerow(forcedict)