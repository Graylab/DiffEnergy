

#Sampling is done using the same mechanism as normal likelihood computation - specifically, using ReverseSDEPath and saving the output
import functools
from typing import Literal
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from diffenergy.dfmdock_tr.docked_dataset import PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDict, LigDict, ModelEval, to_array as to_array_nobatch, from_array as from_array_nobatch
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.helper import diffusion_coeff
from scripts.likelihood import SizeWrappedIter, SizedIter, get_likelihoods, get_paths, ArrayLike
from scripts.likelihood_dfmdock import load_samples, write_likelihood_outputs


def sample_random_offset(rec_pos, lig_pos, sigma:float, offset_type:Literal["Translation", "Rotation", "Translation+Rotation"])->torch.Tensor:
    device=rec_pos.device

    # get center of mass
    rec_cen = torch.mean(rec_pos, dim=(0, 1))
    lig_cen = torch.mean(lig_pos, dim=(0, 1))

    # get rotat update: random rotation vector
    restensors = []
    
    if "Translation" in offset_type:
        # get trans update: random noise + translate x2 to x1
        restensors.append(torch.normal(0.0, sigma, size=(3,), device=device) + (rec_cen - lig_cen))
    if "Rotation" in offset_type:
        raise NotImplemented #uhhhhhh
        restensors.append(matrix_to_axis_angle(torch.from_numpy(Rotation.random().as_matrix()).float().to(device)))
    
    return torch.cat(restensors,dim=0);


@hydra.main(version_base=None, config_path="../configs", config_name="sample_dfmdock_tr")
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

        config.path_type = "reverse_sde"
        config.integral_type = "diff"
        config.sde_timeschedule = "reverse_uniform" #make sure to go from t=1 to t=0!

        if config.get("wt_file",None):
            config.checkpoint = config.wt_file
        
        if config.get("sample_file",None):
            warnings.warn("sample_file included, but will be ignored! Sample files are now always saved to {out_dir}/samples.csv!")
        if config.get("sample_num",None):
            warnings.warn("sample_num is unused for dfmdock sampling; data is generated from noising pdbs, so please specify samples_per_pdb instead!")
        
        if config.get("num_steps",None):
            config.sde_steps = config.num_steps

        config.write_samples=True
        config.write_likelihoods=False
        if config.get("save_trajectory",False):
            config.save_trajectories = True
        if config.get("save_trajectories",False):
            config.write_trajectory_index = True
    
        
    

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

    model_eval = ModelEval(score_model,offset_type=offset_type)
    
    scorefn = model_eval.score# if not batched else model_eval.batch_score
    divergencefn = model_eval.divergence# if not batched else model_eval.batch_divergence

    esm_model = ESMLanguageModel()
    pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))



    assert batch_size is None
    def load_noised_samples()->SizedIter[tuple[str,LigDict,DFMDict]]:
        dataset = load_samples(config.data_samples, config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
        
        def get_noised_sample(id:str,lig:LigDict,cond:DFMDict):
            assert torch.all(lig["offset"] == 0)
            rand_offset = sample_random_offset(cond["rec_pos"],cond["lig_pos_orig"],sigma_max,offset_type)
            lig = {"offset":rand_offset}
            return (id,lig,cond)
        
        samples_each = config.samples_per_pdb
        return SizeWrappedIter((get_noised_sample(f"{id}_{i}",lig,cond) for (id,lig,cond) in dataset for i in range(samples_each)),samples_each*len(dataset))


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

    likelihoods = get_likelihoods(config,
                                 paths,
                                 [],
                                 device)
    
    
    write_likelihood_outputs(config,
                             likelihoods,
                             [],
                             [],
                             offset_type,
                             batch_size)
    

if __name__ == '__main__':
    main()
