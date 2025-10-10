from io import TextIOWrapper
import itertools
from logging import warning
import math
import os
from pathlib import Path
import csv
import functools
import shutil
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, Optional, ParamSpec, Protocol, Sequence, Sized, TypeVar, TypeVarTuple, overload, override
import warnings

import numpy as np
from tqdm import tqdm
from diffenergy.dfmdock_tr.docked_dataset import DockedDatum, PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.score_model import Score_Model

from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import hydra

from diffenergy.dfmdock_tr.utils.geometry import axis_angle_to_matrix
from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std, diffusion_coeff, prior_gaussian_nd
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDict, LigDict, ModelEval, to_array as to_array_nobatch, from_array as from_array_nobatch
from diffenergy.likelihoodv3 import Array, ArrayLike, LikelihoodIntegrand
from scripts.likelihoodv3 import MapDataset, SizedIter, get_integrands, get_likelihoods, get_paths


def dockeddatum_to_condition(datum:DockedDatum,device:str|torch.device)->DFMDict:
    return {
        "orig_pdb": datum["pdb_file"],
        "lig_pos_orig": datum["lig_pos"].to(device),
        "rec_pos": datum["rec_pos"].to(device),
        "lig_x": datum["lig_x"].to(device),
        "rec_x": datum["rec_x"].to(device),
        "position_matrix": datum["position_matrix"].to(device),
    }



@overload
def load_samples(data_file, data_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:None,device:str|torch.device='cuda')->MapDataset[tuple[str,LigDict,DFMDict],str,Path]: ...
@overload
def load_samples(data_file, data_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:int,device:str|torch.device='cuda')->MapDataset[tuple[Sequence[str],LigDict,DFMDict],str,Path]: ...
def load_samples(data_file, data_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:int|None=None, device:str|torch.device='cuda')->MapDataset[tuple[str,LigDict,DFMDict],str,Path]|MapDataset[tuple[Sequence[str],LigDict,DFMDict],str,Path]:
    """Loads pdbs from a CSV file containing filenames. Returns tuples of (id, LigDict, DFMDict),
    where LigDict contains the offset vector for the ligand (zero by default!) and DFMDict contains the
    (NOT 0-CENTERED!) ligand and receptor coordinates as well as the other conditioning information (ESM embeddings, etc)
    required for model evaluation.
    """
    if batch_size is not None: raise ValueError("Batching not supported")

    df = pd.read_csv(data_file, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
    ids = df.loc[:, "index"].values  # Extract the first column as ids
    paths = df.loc[:, "Filename"].values  # Extract the second column as filenames

    def getpdb(id:str,path:Path)->tuple[str,LigDict,DFMDict]:
        dfmdict = dockeddatum_to_condition(importer.get_pdb(str(path),id),device)
        return (id,
                from_array_nobatch(torch.zeros((6 if offset_type == "Translation+Rotation" else 3,),device=device,dtype=dfmdict["lig_pos_orig"].dtype),device=device),
                dfmdict)
    
    data_dir = Path(data_dir)
    
    return MapDataset([(str(id),data_dir/path) for id,path in zip(ids,paths)],getpdb)



    # Convert to tensors
    samples = torch.tensor(samples, dtype=torch.float32, device=device)[...,None]  # samples as floats, add a dimension to make them vectors

    if batch_size is None:
        return [(str(int(id.item())),x) for id,x in zip(ids,samples)]
    else:
        assert batch_size > 0
        return [([str(int(id)) for id in ids[i*batch_size:(i+1)*batch_size].tolist()],samples[i*batch_size:(i+1)*batch_size]) 
                for i in range(math.ceil(len(samples)/batch_size))]


@overload
def load_trajectories(trajectory_index_file:str|Path,pdb_dir:str|Path,trajectory_dir:str|Path,pdb_importer:PDBImporter,batch_size:int)->SizedIter[tuple[tuple[str,...],tuple[Path,...],tuple[DFMDict,...]]]: ...
@overload
def load_trajectories(trajectory_index_file:str|Path,pdb_dir:str|Path,trajectory_dir:str|Path,pdb_importer:PDBImporter,batch_size:None)->SizedIter[tuple[str,Path,DFMDict]]: ...
def load_trajectories(trajectory_index_file:str|Path,pdb_dir:str|Path,trajectory_dir:str|Path,pdb_importer:PDBImporter,batch_size:int|None=None,device:str|torch.device='cuda')->SizedIter[tuple[str,Path,DFMDict]]|SizedIter[tuple[tuple[str,...],tuple[Path,...],tuple[DFMDict,...]]]:
    trajectory_index_file = Path(trajectory_index_file)
    pdb_dir = Path(pdb_dir)
    trajectory_dir = Path(trajectory_dir)

    assert trajectory_index_file.suffix == '.csv'
    df = pd.read_csv(trajectory_index_file)
    res:list[tuple[str,Path,DFMDict]] = [(id,trajectory_dir/trajectory_filename,dockeddatum_to_condition(pdb_importer.get_pdb(str(pdb_dir/pdb_filename),id),device=device))
              for id,pdb_filename,trajectory_filename in zip(df["index"],df["PDB_File"],df["Trajectory_File"])]
    if batch_size is None:
        return res
    else:
        return [(tuple(b[0] for b in batch),tuple(b[1] for b in batch), tuple(b[2] for b in batch)) for batch in itertools.batched(res,batch_size)]

@overload
def load_trajectory(data_path:str|Path, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], device:str|torch.device='cuda')->SizedIter[tuple[LigDict,torch.Tensor]]: ...
@overload
def load_trajectory(data_path:tuple[str|Path,...], offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], device:str|torch.device='cuda')->SizedIter[tuple[tuple[LigDict,...],torch.Tensor]]: ...
def load_trajectory(data_path:str|Path|tuple[str|Path,...], offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], device:str|torch.device='cuda')->SizedIter[tuple[LigDict|tuple[LigDict,...],torch.Tensor]]:
    paths = data_path if isinstance(data_path,tuple) else [data_path]
    alltimes:Optional[torch.Tensor] = None
    sampleslist:list[list[LigDict]] = []
    
    columns = offset_trajectory_columns(offset_type)
    
    for path in paths:
        assert str(path).endswith(".csv")
        df = pd.read_csv(path, header=0)  # Load CSV

        # we reverse the trajectory so it matches flow, with time going from 0 to 1. 
        # have to do this AND make a copy, cause tensors don't support negative stride -_-
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

        data = torch.as_tensor(df[columns].values,dtype=torch.float32,device=device) #this just works yesssss
        #make list of LigDict
        sampleslist.append([from_array_nobatch(data[i],device=device) for i in range(data.shape[0])])
    assert alltimes is not None
    samples:list[LigDict]|list[tuple[LigDict,...]] = [batch for batch in zip(*sampleslist)] if isinstance(data_path,tuple) else sampleslist[0] #needs to be NxB if batched so make sure to stack at dimension 1
    return list(zip(samples,alltimes))

def offset_trajectory_columns(offset_type:str):
    tr_columns = ["Offset_Tr_X", "Offset_Tr_Y", "Offset_Tr_Z"]
    rot_columns = ["Offset_Rot_X", "Offset_Rot_Y", "Offset_Rot_Z"]

    match offset_type:
        case "Translation": columns = tr_columns
        case "Rotation": columns = rot_columns
        case "Translation+Rotation": columns = tr_columns + rot_columns
        case _: raise ValueError
    return columns


@hydra.main(version_base=None, config_path="../configs/likelihoodv3", config_name="likelihood_gaussian_1d")
def main(config: DictConfig):
    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

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

    assert batch_size is None
    load_samples_fn = lambda: load_samples(config.data_samples, config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
    load_trajectories_fn = lambda: load_trajectories(config.trajectory_index_file,config.pdb_dir,config.trajectory_dir,pdb_importer,batch_size=batch_size)
    get_trajectory_fn = lambda trajectory_file: load_trajectory(trajectory_file, offset_type, device=device)

    diffusion_coeff_fn = functools.partial(
        diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))

    priors = load_priors(config,
                        to_array,
                        offset_type,
                        sigma_min,
                        sigma_max,
                        batched)
    
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
                      get_trajectory_fn,
                      device)

    likelihoods = get_likelihoods(config,
                                 paths,
                                 integrands,
                                 device)
    
    write_likelihood_outputs(config,
                             likelihoods,
                             integrands,
                             priors,
                             offset_type,
                             batch_size)
    
from biotite.structure.io.general import load_structure, save_structure
from biotite.structure import AtomArray, get_chains
from biotite.sequence.seqtypes import ProteinSequence
def modify_aa_coords(x, rot, tr):
    center = x.mean(axis=0)
    rot = axis_angle_to_matrix(rot).squeeze().cpu().numpy()
    x = (x - center) @ rot.T + center 
    x = x + tr.cpu().numpy()
    return x

# load pdb with **all** atoms, not just backbone atoms, and offset specified chain. Defaults to B cause that's the default ligand chain
def get_offset_pdb(
        orig:str|Path,
        offset_tr:None|torch.Tensor,
        offset_rot:None|torch.Tensor,
        offset_chain="B"
        ):
    orig_structure:AtomArray = load_structure(orig)
    all_chains = get_chains(orig_structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    elif offset_chain not in all_chains:
        raise ValueError(f"Cannot offset chain {offset_chain}; not in file!")

    #we can extract just the ligands, modify the structure, then assign it back using this boolean mask
    lig_filter = orig_structure.chain_id == offset_chain #get boolean mask
    lig_structure = orig_structure[lig_filter]

    offset_rot = offset_rot if offset_rot is not None else torch.zeros((3,),dtype=lig_structure.coord.dtype)
    offset_tr = offset_tr if offset_tr is not None else torch.zeros((3,),dtype=lig_structure.coord.dtype)

    #actually since we're modifying coord in-place I don't know if we need to re-assign but it probably makes a copy so safer than sorry
    lig_structure.coord = modify_aa_coords(lig_structure.coord,offset_rot,offset_tr) 
    
    orig_structure[lig_filter] = lig_structure
    
    return orig_structure

def split_offset(offset:torch.Tensor,offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],):
    tr_update = offset[:3].cpu() if 'Translation' in offset_type else None
    rot_update = offset[-3:].cpu() if 'Rotation' in offset_type else None
    
    return (tr_update,rot_update)

def write_dfmdock_trajectory(trajectory_dir:Path,
                             id:str,
                             trajectory:Sequence[LigDict],
                             times:Sequence[float],
                             condition:DFMDict,
                             offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
                             save_type:Literal['offset','pdb','offset+anchor_pdb']='offset',
                             offset_anchor:Literal['start','end','condition']='condition',
                             warn_modify_endpoint_unsave:bool=True,
                             )->tuple[Path,Path]:
    """Write a DFMDock trajectory to disk. Recall that for likelihood computation, the fundamental data representation
    is that of a 3d or 6d offset vector representing [translation or rotation] or [translation + rotation] respectively, 
    and is thus described relative to some conditioned pdb file's receptor and ligand coordinates (the data passed in 'condition').

    Depending on the value of 'offset_anchor', the offsets will be recalculated in terms of a particular receptor/ligand 
    configuration 'anchor'. For offset_anchor='condition' (default), the anchor is the default configuration specfied in
    `condition`, so no recalculation is necessary. For offset_anchor = 'end', the last point in the path is used; for 
    offset_anchor = 'start', the first point is used. UNUSED IF save_type = 'pdb' (since no anchor is necessary for raw pdbs).
    Additionally, if save_type = 'offset' and offset_anchor != 'condition', offsets will be recalculated, but the anchor
    WILL NOT BE SAVED! A warning will be displayed unless warn_modify_endpoint_unsave is set to False.
    
    The `save_type` parameter describes how you want the data to be saved:
    
    - With save_type = 'offset' (the default), timepoints ('Timestep') and offset vectors (see offset_trajectory_columns()) 
    will be saved to the csv file 'trajectory_{id}.csv', and no pdb data will be saved. Returns the path to the csv file
    
    - With save_type = 'pdb', the trajectory will instead be saved as a sequence of pdb files in a subfolder of the given 
    trajectory directory. These pdbs are generated by offsetting the original pdb file ligand by the offset vector.
    NOTE THAT THIS WILL SIGNIFICANTLY SLOW DOWN THE PROCESS OF WRITING TRAJECTORIES! Instead of offsets, the csv file 
    will be populated with timepoints ('Timestep') and filenames ('PDB_File') which point to the pdb files. Pdbs will 
    be saved with the naming convention {id}_{index}.pdb. Returns the path to the csv file.

    - With save_type = 'offset+anchor_pdb', the trajectory will be saved as with save_type = 'offset' along with a pdb file
    containing the "anchor" as specified by the value of `offset_anchor`. This allows the offsets and anchor file to be used
    as the trajectory and condition respectively for likelihood computation. Returns a tuple of the paths to the trajectory 
    and anchor files.
    
    """


        

    trajectory_csv = trajectory_dir/f'trajectory_{id}.csv'
    pdb_folder = trajectory_dir/'pdb'
    if 'pdb' in save_type:
        pdb_folder.mkdir(exist_ok=True)
    

    xtraj = torch.stack([to_array_nobatch(x) for x in trajectory]) #Nx3 or Nx6
    ttraj = torch.as_tensor(times)
    
    if 'offset' in save_type:
        if offset_anchor != 'condition' and save_type != 'offset+anchor_pdb' and warn_modify_endpoint_unsave:
                warnings.warn("Recalculating offsets without saving new relative pdb when writing dfmdock trajectory!")
                    
        if offset_anchor == 'end':
            anchor_offset = xtraj[-1] #yes I'm great at names
        elif offset_anchor == 'start':
            anchor_offset = xtraj[0]
        elif offset_anchor == 'condition':
            anchor_offset = torch.zeros((xtraj.shape[-1]))
        else:
            raise ValueError(anchor_offset)
        
        xtraj -= anchor_offset #ummm this probably just, like, *works* for rotation haha :nervous:
            
        assert xtraj.ndim == 2
        assert xtraj.shape[1] == (6 if offset_type == "Translation+Rotation" else 3)
        
        columns = ["Timestep"] + offset_trajectory_columns(offset_type)
        data = torch.cat([ttraj[...,None],xtraj],dim=1)                    
        trajectory_df = pd.DataFrame(columns=columns,data=data.numpy(force=True))
        trajectory_df.to_csv(trajectory_csv,index_label="Index")
        
        
        if save_type == 'offset+anchor_pdb':
            offset_pdb = get_offset_pdb(condition["orig_pdb"],*split_offset(anchor_offset,offset_type))
            pdb_file = pdb_folder/f'{id}.pdb'
            save_structure(pdb_file,offset_pdb)

            return (trajectory_csv,pdb_file)
        else:
            return trajectory_csv,Path(condition["orig_pdb"])
    else:
        filenames = [Path('pdb')/f'{id}_{i}.pdb' for i in range(len(xtraj))]
        for name,offset in zip(filenames,xtraj):
            offset_pdb = get_offset_pdb(condition["orig_pdb"],*split_offset(offset,offset_type))
            save_structure(trajectory_dir/name,offset_pdb)
        
        trajectory_df = pd.DataFrame({"Timestep":ttraj.numpy(force=True),"PDB_File":map(str,filenames)})
        trajectory_df.to_csv(trajectory_csv,index_label="Index")
        return trajectory_csv,Path(condition["orig_pdb"])

def make_dfmdock_sample(id:str,
                        trajectory:Sequence[LigDict],
                        times:Sequence[float],
                        condition:DFMDict,
                        offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
                        #TODO: UNIFY THIS POINT SELECTION AND ANCHOR CALCULATION WITH WRITE_DFMDOCK_TRAJECTORY!
                        save_point:Literal["start","end"]="start", 
                        save_new_pdb:bool=False,
                        new_pdb_file:Optional[str|Path]=None,#samples_pdb_folder/f"{id}.pdb",
                        include_offset:bool=False):#=config.get("write_sample_with_offset",False)):

    if save_point == 'end':
        point = trajectory[-1] 
    elif save_point == 'start':
        point = trajectory[0]
    else:
        raise ValueError(save_point)
    
    offset = to_array_nobatch(point)

    if not (save_new_pdb or include_offset or torch.allclose(offset,torch.zeros_like(offset))):
        raise ValueError("Can't faithfully save sample with nonzero offset without either writing the offset or saving the pdb!")

    if save_new_pdb:
        ##if we're going to save the offset, don't *also* offset the pdb!
        updates = split_offset(offset,offset_type) if not include_offset else (None,None)
        new_pdb = get_offset_pdb(condition["orig_pdb"],*updates)
        save_structure(new_pdb_file,new_pdb)
        pdb_file = new_pdb_file
    else:
        pdb_file = condition["orig_pdb"]

    result = {"index":id,"Filename":pdb_file}
    if include_offset:
        result.update(zip(offset_trajectory_columns(offset_type),offset.tolist()))

    return result

    

def write_likelihood_outputs(config:DictConfig,
                            likelihoods:Iterable[tuple[str,Sequence[LigDict],Sequence[float],DFMDict,dict[str,float|ArrayLike]]],
                            integrands:list[LikelihoodIntegrand[LigDict,DFMDict]],
                            priors:Sequence[tuple[str,Callable[[LigDict,float,DFMDict],float|Array]]],
                            offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
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

    samples_pdb_folder = out_dir/"pdb"
    samples_save_pdb = config.get("write_sample_save_pdb",False)

    if write_samples:
        samples_handle = open(samples_file,"w")
        fieldnames = ["index","Filename"]
        if config.get("write_sample_with_offset",False):
            fieldnames += offset_trajectory_columns(offset_type)
        samples_writer = csv.DictWriter(samples_handle,fieldnames=fieldnames) #TODO: regularize capitalization aaaa
        samples_writer.writeheader()

        if samples_save_pdb:
            samples_pdb_folder.mkdir(exist_ok=True)


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
        trajectory_indices = [(ind,f,csv.DictWriter(f,fieldnames=["index","PDB_File","Trajectory_File"])) for (ind,f) in indices]
        [index[2].writeheader() for index in trajectory_indices]
    

    ## WRITE OUTPUT
    try:
        for ids,trajectories,times,conditions,integrand_resultss in likelihoods:

            ##Calculate priors
            prior_eval_endpoint:tuple[LigDict,float,DFMDict] = (trajectories[-1], times[-1], conditions)

            prior_result:dict[str,float|list[float]] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_eval_endpoint))) for name,prior_fn in priors}

            ## Unbatch results for writing
            if not batched:
                ids:Iterable[str] = [ids]
                trajectories = [trajectories]
                times = [times]
                integrand_resultss = [integrand_resultss]
                prior_resultss = [prior_result]
            else:
                raise ValueError()
                batch = len(prior_eval_endpoint[0]) #size of this batch, could be smaller than batch_size if last batch
                ids: Iterable[str] = ids
                trajectories = torch.stack(trajectories,dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
                assert trajectories.ndim == 3 #BxNxD
                times = itertools.repeat(times)
                integrand_resultss = [
                    {name:np.array(result)[i] for name,result in integrand_resultss.items()}
                    for i in range(batch)
                ]
                prior_resultss = [
                    {name:np.array(result)[i] for name,result in prior_result.items()}
                    for i in range(batch)
                ]

            for id, trajectory, time, integrand_results, prior_results \
                in zip(ids, trajectories, times, integrand_resultss, prior_resultss):
                prior_endpoint:tuple[LigDict,float] = (trajectory[-1], time[-1])
                assert isinstance(id,str),id

                if write_likelihoods:
                    row = {"id":int(id),
                        "prior_position":to_array_nobatch(prior_endpoint[0]).tolist(),
                            "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                            **{f"prior:{name}":val for name,val in prior_results.items()},
                            **{f"integrand:{name}":val for name,val in integrand_results.items()}}
                    likelihoods_writer.writerow(row)

                if write_samples:
                    sample = make_dfmdock_sample(
                        id,
                        trajectory,
                        time,
                        conditions,
                        offset_type,
                        save_point="end", #TODO: CONFIGURATION FOR WHICH POINT TO SAVE
                        save_new_pdb=samples_save_pdb,
                        new_pdb_file=samples_pdb_folder/f"{id}.pdb",
                        include_offset=config.get("write_sample_with_offset",False))

                    samples_writer.writerow(sample)

                if save_trajectories: #save trajectory to folder                    
                    traj_path,pdb_path = write_dfmdock_trajectory(
                        trajectory_folder,
                        id,
                        trajectory,
                        time,
                        conditions,
                        offset_type,
                        config.get("trajectory_save_type","offset"),
                        config.get("trajectory_offset_anchor","condition"))
                    
                    for cutoff,file,writer in trajectory_indices:
                        if cutoff is None or acc_trajnum <= cutoff:
                            writer.writerow({"index":id,"PDB_File":pdb_path,"Trajectory_File":traj_path})
                    acc_trajnum += 1


                f.flush()
    finally:
        #make sure the files are closed!!!
        if likelihoods_handle is not None:
            likelihoods_handle.close()
        if samples_handle is not None:
            samples_handle.close()
        for cutoff,file,writer in trajectory_indices:
            file.close()


def load_priors(config:DictConfig,
                to_array:Callable[[LigDict],Array]|Callable[[Sequence[LigDict]],Array],
                offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
                sigma_min:float,
                sigma_max:float,
                batch_size:int|None):
    batched = batch_size is not None

    ### LOAD PRIORS
    priors:list[tuple[str,Callable[[LigDict|Sequence[LigDict],float,DFMDict],float]]] = []
    functions:DictConfig = config.get("prior_fns","smax_gaussian")
    if isinstance(functions,str):
        functions = functions.split(" ")
    if not isinstance(functions,Mapping):
        functions = DictConfig({f:{} for f in functions}) #ensure types is a DictConfig of Dicts
    
    for prior_fn,params in functions.items():
        match prior_fn:
            case "convolved_data":
                raise ValueError("Data Distribution Unknown for DFMDock!")
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
                def prior_likelihood_fn(x:LigDict|Sequence[LigDict],t:float, condition:DFMDict):
                    offset = to_array(x) #assume x matches batchness and has been dealt with
                    tt = torch.as_tensor(t)
                    assert torch.allclose(tt, torch.ones_like(tt)), t #diffeq errors might mean it's not *quite* 1 but that's fine
                    if offset_type == "Rotation": #Assume rotational prior is just 0!
                        res = torch.zeros(offset.shape[:-1],dtype=offset.dtype)
                    else:
                        res = (prior_gaussian_nd(offset[...,:3],sigma_max)[0]) #first three components are always x,y,z. Assume rotational prior is just 0!
                    return res.numpy(force=True) if batched else res.item()
                priors.append((prior_fn,prior_likelihood_fn))
    
    return priors


if __name__ == '__main__':
    main()
