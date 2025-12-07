import functools
from functools import cached_property
from typing import Iterable, Literal
import warnings

from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from tqdm import tqdm
from diffenergy.dfmdock_tr.docked_dataset import PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDict, LigDict, ModelEval, to_array as to_array_nobatch, from_array as from_array_nobatch
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.helper import diffusion_coeff
from diffenergy.likelihood import run_diff_likelihood, run_ode_likelihood
from likelihood_class import DiffEnergyLikelihood
from scripts.likelihood import SizeWrappedIter, SizedIter, get_integrands, get_paths
from scripts.likelihood_dfmdock import load_priors, load_samples, load_trajectories, load_trajectory, offset_trajectory_columns, write_dfmdock_samples
from scripts.sample_dfmdock import sample_random_offset, unzip


class DFMDockLikelihood(DiffEnergyLikelihood[LigDict,DFMDict]):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
    
    
    @cached_property
    def out_pdb_folder(self):
        pdb_folder = self.out_dir/"pdb"
        pdb_folder.mkdir(parents=True,exist_ok=True)
        return pdb_folder
    
    @property
    def offset_type(self)->Literal["Translation","Rotation","Translation+Rotation"]:
        offset_type:str = self.config.offset_type
        if not offset_type in ("Translation","Rotation","Translation+Rotation"):
            raise ValueError("DFMDock offset_type must be one of 'Translation', 'Rotation', or 'Translation+Rotation'")
        return offset_type

    def sample_index_writer(self,write_samples:bool,extra_fieldnames:Iterable[str]=[]):
        return super().sample_index_writer(write_samples,extra_fieldnames=[*extra_fieldnames,*offset_trajectory_columns(self.offset_type)])
    
    
    def compute_likelihoods(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        if batched:
            raise ValueError("Batched DFMDock evaluation not supported!")

        to_array = to_array_nobatch
        from_array = functools.partial(from_array_nobatch,device=device)# if not batched else from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        score_model = Score_Model.load_from_checkpoint(self.config.checkpoint,deterministic=self.config.get("deterministic_score",False))
        score_model.freeze()
        score_model.to(device)

        offset_type:Literal["Translation","Rotation","Translation+Rotation"] = self.config.offset_type
        valid_offsets = ["Translation", "Rotation", "Translation+Rotation"]
        if offset_type not in valid_offsets:
            raise ValueError("offset_type must be one of",valid_offsets)
        
        if self.config.get("reset_seed_each_sample",False):
            raise ValueError("reset_seed_each_sample deprecated. Either use reset_seed_each_eval (original behavior of reset_seed_each_sample) or reset_seed_each_path.") 

        model_eval = ModelEval(score_model,offset_type=offset_type,reset_seed_each_eval=self.config.get("reset_seed_each_eval",False),manual_seed=self.config.get("seed",0),
                            divide_div_by_N=self.config.get("small_divergence",False)) #test bug in old code
        
        scorefn = model_eval.score# if not batched else model_eval.batch_score
        divergencefn = model_eval.divergence# if not batched else model_eval.batch_divergence

        esm_model = ESMLanguageModel()
        pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

        assert batch_size is None
        load_samples_fn = lambda: load_samples(self.config.data_samples, self.config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
        load_trajectories_fn = lambda: load_trajectories(self.config.trajectory_index_file,self.config.pdb_dir,self.config.trajectory_dir,pdb_importer,batch_size=batch_size)
        get_trajectory_fn = lambda trajectory_file, condition: load_trajectory(trajectory_file, self.config.pdb_dir, offset_type, condition, device=device)

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))

        priors = load_priors(self.config,
                            to_array,
                            offset_type,
                            sigma_min,
                            sigma_max,
                            batched)
        
        integrands = get_integrands(self.config,
                                    from_array,
                                    to_array,
                                    scorefn,
                                    divergencefn,
                                    diffusion_coeff_fn)

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

        ### RUN LIKELIHOOD COMPUTATION
        
        int_type = self.config.integral_type

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        write_samples = self.config.get("write_samples",True)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories
        
        ## WRITE OUTPUT
        acc_trajnum = 0
        with (  #open the various global output writers
                self.likelihoods_writer(True, 
                    prior_names=[name for (name,_) in priors],
                    integrand_names=[integrand.name() for integrand in integrands])  as likelihoods_writer,
                self.sample_index_writer(write_samples)                 as samples_writer, 
                self.trajectory_index_writers(write_trajectory_index)   as trajectory_indices
            ):
            for (id,path) in tqdm(paths):
                if reset_seed_each_path:
                    torch.manual_seed(seed)
                if int_type == "ode":
                    #just assume paths are ode integrable. error will be thrown otherwise
                    trajectory, time, likelihood_result = run_ode_likelihood(path,integrands,accumulate=save_trajectories)
                elif int_type == "diff":
                    #use standard integration
                    trajectory, time, likelihood_result = run_diff_likelihood(path,integrands,accumulate=save_trajectories)
                else:
                    raise ValueError(f"Unknown integral type: {int_type}. For standard (non-ode solver) numerical integration, use integral_type: \"diff\" (the default).")
                
                condition = path.condition
                
                # where to evaluate the prior. should be at time = 1, which is checked in the prior function; here's where the assumption
                # that likelihood paths always *end* at the prior matters, since we have to assume that that's where the prior point is
                # TODO: configurable property of the path, perhaps?
                prior_endpoint:tuple[LigDict,float,DFMDict] = (trajectory[-1], time[-1], condition)
                prior_result:dict[str,float|list[float]] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_endpoint))) for name,prior_fn in priors}

                if likelihoods_writer:
                    row = {"id":id,
                        "prior_position":to_array_nobatch(prior_endpoint[0]).tolist(),
                            "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                            **{f"prior:{name}":val for name,val in prior_result.items()},
                            **{f"integrand:{name}":val[-1] for name,val in likelihood_result.items()}} #write last accumulated likelihood
                    likelihoods_writer.writerow(row)
                    
                    
                sample_out, traj_out = write_dfmdock_samples(
                    self.out_trajectory_folder,
                    self.out_pdb_folder,
                    id,
                    trajectory,
                    time,
                    condition,
                    offset_type,
                    write_samples,
                    save_trajectories,
                    integrand_results=likelihood_result if self.config.get("save_trajectory_likelihoods") else None,
                    save_pdb_references=self.config.get("save_pdb_references",False),
                    pdb_reference_point=self.config.get("pdb_reference_point",None),
                    sample_save_point=self.config.get("sample_save_point","end"),
                    sample_save_type=self.config.get("sample_save_type","offset"),
                    force_copy_duplicate_sample=self.config.get("force_copy_duplicate_sample",False),
                    trajectory_save_type=self.config.get("trajectory_save_type","offset"),
                )

                if sample_out:
                    samples_writer.writerow(sample_out)

                if traj_out:
                    for cutoff,writer in trajectory_indices.items():
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow(traj_out)
                    acc_trajnum += 1

    
    def sample(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))


        with open_dict(self.config):
            ## Set various sampling/trajectory output config parameters based on sampling parameters:
            # out_dir -> out_dir [unchanged]
            # wt_file | checkpoint -> checkpoint [take either for backwards compatibility]
            # sample_file: Raise warning if specified, all samples now have filename "samples.csv"
            # sample_num: Discard for dfmdock, dfmdock needs samples_per_pdb
            # num_steps -> sde_steps
            # save_trajectories: Also sets write_trajectory_index to True

            ## Other relevant parameters: (see write_dfmdock_samples in the dfmdock likelihood script file for details)

            # save_pdb_references: bool, default False. whether to save reference pdbs to the output folder. Default False, meaning offsets will be relative to the pdbs provided during training
            # pdb_reference_point: Literal[null,'start','end'], default null. what point on the trajectory path to save the reference, or null to just copy the original pdb
            # sample_save_point: Literal['reference','start','end'], default end. Save as above, but for the sampled point. 'reference' saves the reference point as the sample 
            # ^IF YOU WANT TRAJECTORIES RELATIVE TO A SAMPLE PDB, THIS IS HOW TO DO IT! SET SAVE PDB REFERENCE TO TRUE AND SAMPLE SAVE POINT TO REFERENCE
            # sample_save_type: Literal['offset','pdb'], default offset. Whether to save the samples as pdbs, or as offsets relative to the reference pdb.
            # force_copy_duplicate_sample: bool, default False. Whether to force duplicate the pdb if the sample matches the pdb (See write_dfmdock_samples)
            # trajectory_save_type: Literal['offset','pdb'], default offset. Whether to save trajectory points as pdbs or as offsets relative to the reference pdb.
            # trajectory_extra_indices: list of index cutoffs (e.g. [25, 1000]), default=[]. Will always include a full index

            self.config.path_type = "reverse_sde"
            self.config.integral_type = "diff"
            self.config.sde_timeschedule = "reverse_uniform" #make sure to go from t=1 to t=0!

            if self.config.get("wt_file",None):
                self.config.checkpoint = self.config.wt_file
            
            if self.config.get("sample_file",None):
                warnings.warn("sample_file included, but will be ignored! Sample files are now always saved to {out_dir}/samples.csv!")
            if self.config.get("sample_num",None):
                warnings.warn("sample_num is unused for dfmdock sampling; data is generated from noising pdbs, so please specify samples_per_pdb instead!")
            
            if self.config.get("num_steps",None):
                self.config.sde_steps = self.config.num_steps

            self.config.write_samples=True
            self.config.write_likelihoods=False
            if self.config.get("save_trajectory",False):
                self.config.save_trajectories = True
            if self.config.get("save_trajectories",False):
                self.config.write_trajectory_index = True
        
            
        
        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        if batched:
            raise ValueError("Batched DFMDock evaluation not supported!")

        to_array = to_array_nobatch
        from_array = functools.partial(from_array_nobatch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        score_model = Score_Model.load_from_checkpoint(self.config.checkpoint)
        score_model.freeze()
        score_model.to(device)

        offset_type:Literal["Translation","Rotation","Translation+Rotation"] = self.config.offset_type
        valid_offsets = ["Translation", "Rotation", "Translation+Rotation"]
        if offset_type not in valid_offsets:
            raise ValueError("offset_type must be one of",valid_offsets)

        model_eval = ModelEval(score_model,offset_type=offset_type)
        
        scorefn = model_eval.score# if not batched else model_eval.batch_score
        divergencefn = model_eval.divergence# if not batched else model_eval.batch_divergence

        esm_model = ESMLanguageModel()
        pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        assert batch_size is None
        def load_noised_samples()->SizedIter[tuple[str,LigDict,DFMDict]]:
            dataset = load_samples(self.config.data_samples, self.config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
            
            def get_noised_sample(id:str,lig:LigDict,cond:DFMDict):
                assert torch.all(lig["offset"] == 0)
                rand_offset = sample_random_offset(cond["rec_pos"],cond["lig_pos_orig"],sigma_max,offset_type)
                lig = {"offset":rand_offset}
                return (id,lig,cond)
            
            samples_each = self.config.samples_per_pdb
            return SizeWrappedIter((get_noised_sample(f"{id}_{i}",lig,cond) for (id,lig,cond) in dataset for i in range(samples_each)),samples_each*len(dataset))

        # Reverse SDE path shouldn't need to load trajectories!
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
        
        

        ## WRITE OUTPUT
        write_samples = self.config.get("write_samples",True)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories
        
        acc_trajnum = 0
        with (  #open the various global output writers
                self.sample_index_writer(write_samples)                 as samples_writer, 
                self.trajectory_index_writers(write_trajectory_index)   as trajectory_indices
            ):
            for (id,path) in paths:
                trajectory,time = unzip(path)
                condition = path.condition
                
                sample_out, traj_out = write_dfmdock_samples( #TODO: move to this class as well
                    self.out_trajectory_folder,
                    self.out_pdb_folder,
                    id,
                    trajectory,
                    time,
                    condition,
                    offset_type,
                    write_samples,
                    save_trajectories,
                    integrand_results=None,
                    save_pdb_references=self.config.get("save_pdb_references",False),
                    pdb_reference_point=self.config.get("pdb_reference_point",None),
                    sample_save_point=self.config.get("sample_save_point","end"),
                    sample_save_type=self.config.get("sample_save_type","offset"),
                    force_copy_duplicate_sample=self.config.get("force_copy_duplicate_sample",False),
                    trajectory_save_type=self.config.get("trajectory_save_type","offset"),
                )

                if sample_out:
                    samples_writer.writerow(sample_out)

                if traj_out:
                    for cutoff,writer in trajectory_indices.items():
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow(traj_out)
                    acc_trajnum += 1

