import functools
import itertools
import math
from typing import Iterable, Sequence
import warnings
from omegaconf import OmegaConf, open_dict
import pandas as pd
import torch
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval, from_array_batch, to_array_batch, from_array as from_array_nobatch, to_array as to_array_nobatch
from diffenergy.gaussian_1d.network import NegativeGradientMLP, ScoreNetMLP
from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel
from diffenergy.helper import diffusion_coeff, marginal_prob_std
from scripts.likelihood import MapDataset, SizedIter, get_paths
from scripts.likelihood_class import DiffEnergyLikelihood
from scripts.sample_gaussian_1d import unzip


class GaussianLikelihood(DiffEnergyLikelihood):
    
    def load_model(self,
               sigma_min:float,
               sigma_max:float,
               batched:bool,
               device:torch.device):
        # set marginal probability distribution and diffusion coefficient distribution
        marginal_prob_std_fn = functools.partial(
            marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)

        # set models
        weights_path = self.config.checkpoint
        ckpt = torch.load(weights_path, map_location = device)

        # Remove "module." prefix if necessary
        if any(key.startswith("module.") for key in ckpt.keys()):
            ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

        # Initialize score model, load the checkpoint weights into the model    
        tr_type = self.config.tr_type
        if tr_type == 'non_conservative':
            score_model = ScoreNetMLP(
                input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
            score_model.load_state_dict(ckpt)
            
            model_eval = ModelEval(score_model,batched=batched)
        elif tr_type == 'conservative':
            score_model = NegativeGradientMLP(
                input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
            score_model.load_state_dict(ckpt)

            model_eval = ModelEval(score_model,batched=batched)
        elif tr_type == 'ground_truth':
            means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float)
            variances = torch.tensor([8.0,5.0,10.0])**2
            weights = torch.tensor([0.4,0.3,0.3])

            model_eval = MultimodalGaussianGroundTruthScoreModel(means,variances,weights,sigma_min,sigma_max,batched=batched)
            model_eval.to(device)
        else:
            raise ValueError(tr_type)
        
        return model_eval
    
    
    def sample(self):
        # Print the entire self.configuration
        print(OmegaConf.to_yaml(self.config))


        with open_dict(self.config):
            ## Set various sampling/trajectory output self.config parameters based on sampling parameters:
            # out_dir -> out_dir [unchanged]
            # wt_file | checkpoint -> checkpoint [take either for backwards compatibility]
            # sample_file: Raise warning if specified, all samples now have filename "samples.csv"
            # sample_num: Discard for dfmdock, dfmdock needs samples_per_pdb
            # num_steps -> sde_steps
            # save_trajectory -> save_trajectories: Also sets write_trajectory_index to True

            ## Other relevant parameters:
            # trajectory_extra_indices: list of index cutoffs (e.g. [25, 1000]), default=[]. Will always include a full index
            
            self.config.path_type = "reverse_sde"
            self.config.integral_type = "diff"
            self.config.sde_timeschedule = "reverse_uniform" #make sure to go from t=1 to t=0!

            if self.config.get("wt_file",None):
                self.config.checkpoint = self.config.wt_file
            
            if self.config.get("sample_file",None):
                warnings.warn("sample_file included, but will be ignored! Sample files are now always saved to {out_dir}/samples.csv!")
            
            if self.config.get("num_steps",None):
                self.config.sde_steps = self.config.num_steps

            if self.config.get("save_trajectory",False):
                self.config.save_trajectories = True

            if self.config.get("save_trajectories",False):
                self.config.write_trajectory_index = True
            
            self.config.write_samples=True
            self.config.write_likelihoods=False
            
        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None

        if batch_size == -1: 
            batch_size = self.config.sample_num #Do it in one **massive** batch because why not

        to_array = to_array_nobatch if not batched else to_array_batch
        from_array = functools.partial(from_array_nobatch if not batched else from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        model_eval = self.load_model(sigma_min,sigma_max,batched,device);
        
        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        ## SAMPLE DIFFUSION TRAJECTORIES
        def load_noised_samples()->SizedIter[tuple[int|Sequence[int],torch.Tensor,None]]:
            bsize = batch_size or 1
            indices = [(i,) for i in range(math.ceil(self.config.sample_num//bsize))] #make into tuples grumble
            def getchunk(i:int):
                ids = range(self.config.sample_num)[i*bsize:(i+1)*bsize]
                chunksize = len(ids)
                x = torch.randn(chunksize,1,device=device)*sigma_max
                if not batched:
                    ids = ids[0]
                return (ids,x,None)
            return MapDataset(indices,getchunk)

        def err(*args): raise ValueError()
        load_trajectories_fn = err
        get_trajectory_fn = err

        
        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_noised_samples,
                        load_trajectories_fn,
                        get_trajectory_fn,
                        device)
        
        write_samples = self.config.get("write_samples",True)
        
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories
        
        ## WRITE OUTPUT
        acc_trajnum = 0
        with self.sample_index_writer(write_samples) as samples_writer, self.trajectory_index_writers(write_trajectory_index) as trajectory_indices:
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
                    if samples_writer:
                        #TODO: CONFIGURATION FOR WHICH POINT TO SAVE
                        sample = {"index":id,"Samples":trajectory[-1].item()}
                        samples_writer.writerow(sample)

                    if save_trajectories: #save trajectory to folder
                        trajectory_name = f"trajectory_{id}.csv"
                        trajectory_file = self.trajectory_folder/f"trajectory_{id}.csv"
                        xtraj = torch.as_tensor(trajectory).numpy(force=True)
                        assert xtraj.ndim == 2
                        assert xtraj.shape[1] == 1
                        xtraj = xtraj[:,0] #1d position into scalar
                        ttraj = torch.as_tensor(time).numpy(force=True)
                        trajectory_df = pd.DataFrame({"Timestep":ttraj,"Sample":xtraj})
                        trajectory_df.to_csv(trajectory_file,index_label="index")
                        
                        
                        for cutoff,writer in trajectory_indices:
                            if cutoff is None or acc_trajnum < cutoff:
                                writer.writerow({"index":id,"filename":trajectory_name})
                        acc_trajnum += 1