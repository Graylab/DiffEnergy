from io import TextIOWrapper
import itertools
from logging import warning
import logging
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
from scripts.likelihoodv3 import MapDataset, SizeWrappedIter, SizedIter, get_integrands, get_likelihoods, get_paths


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
def load_samples(data_file, pdb_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:None,device:str|torch.device='cuda')->Sequence[tuple[str,LigDict,DFMDict]]: ...
@overload
def load_samples(data_file, pdb_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:int,device:str|torch.device='cuda')->Sequence[tuple[Sequence[str],LigDict,DFMDict]]: ...
def load_samples(data_file, pdb_dir, offset_type: Literal["Translation", "Rotation", "Translation+Rotation"], importer:PDBImporter, batch_size:int|None=None, device:str|torch.device='cuda')->Sequence[tuple[str,LigDict,DFMDict]]|Sequence[tuple[Sequence[str],LigDict,DFMDict]]:
    """Loads pdbs from a CSV file containing filenames. Returns tuples of (id, LigDict, DFMDict),
    where LigDict contains the offset vector for the ligand (zero by default!) and DFMDict contains the
    (NOT 0-CENTERED!) ligand and receptor coordinates as well as the other conditioning information (ESM embeddings, etc)
    required for model evaluation.
    """
    if batch_size is not None: raise ValueError("Batching not supported")

    df = pd.read_csv(data_file, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
    ids = df.loc[:, "index"].values  # Extract the first column as ids
    paths = df.loc[:, "Filename"].values  # Extract the second column as filenames
    if any([col.startswith("Offset") for col in df.columns]):
        offsets = df.loc[:,offset_trajectory_columns(offset_type)]
    else:
        offsets = None

    def getpdb(idx:int)->tuple[str,LigDict,DFMDict]:
        id = str(ids[idx])
        pdb_path = pdb_dir/paths[idx]
        dfmdict = dockeddatum_to_condition(importer.get_pdb(str(pdb_path),id,suppress_warnings=True),device)
        if offsets is None:
            offset = torch.zeros((6 if offset_type == "Translation+Rotation" else 3,),device=device,dtype=dfmdict["lig_pos_orig"].dtype)
        else:
            offset = torch.as_tensor(offsets[idx],device=device,dtype=dfmdict["lig_pos_orig"].dtype)
        return (id,
                from_array_nobatch(offset,device=device),
                dfmdict)
    
    pdb_dir = Path(pdb_dir)
    
    return MapDataset([(i,) for i in range(len(ids))],getpdb)



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
    #since we're reading the condition pdbs as we load the trajectories, needs to be a generator so we don't load them all at once!
    res:Iterable[tuple[str,Path,DFMDict]] = ((id,trajectory_dir/trajectory_filename,dockeddatum_to_condition(pdb_importer.get_pdb(str(pdb_dir/pdb_filename),id,suppress_warnings=True),device=device))
              for id,pdb_filename,trajectory_filename in zip(df["index"],df["PDB_File"],df["Trajectory_File"]))
    if batch_size is None:
        return SizeWrappedIter(res,len(df['index']))
    else:
        num_batches = math.ceil(len(df['index'])//batch_size)
        return SizeWrappedIter(((tuple(b[0] for b in batch),tuple(b[1] for b in batch), tuple(b[2] for b in batch)) for batch in itertools.batched(res,batch_size)),num_batches)

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

    offset_rot = offset_rot if offset_rot is not None else torch.zeros((3,),dtype=float)
    offset_tr = offset_tr if offset_tr is not None else torch.zeros((3,),dtype=float)

    #actually since we're modifying coord in-place I don't know if we need to re-assign but it probably makes a copy so safer than sorry
    lig_structure.coord = modify_aa_coords(lig_structure.coord,offset_rot.detach(),offset_tr.detach()) 
    
    orig_structure[lig_filter] = lig_structure
    
    return orig_structure

def split_offset(offset:torch.Tensor,offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],):
    tr_update = offset[:3].cpu() if 'Translation' in offset_type else None
    rot_update = offset[-3:].cpu() if 'Rotation' in offset_type else None
    
    return (tr_update,rot_update)

##handles both trajectory and sample writing!

def write_dfmdock_samples(trajectory_dir:str|Path,
                          pdb_dir:str|Path,
                          id:str,
                          trajectory:Sequence[LigDict],
                          times:Sequence[float],
                          condition:DFMDict,
                          offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
                          save_samples:bool,
                          save_trajectories:bool,
                          integrand_results:Optional[dict[str,Sequence[float|ArrayLike]]]=None,
                          save_pdb_references:bool=False,
                          pdb_reference_point:Literal[None,'start','end']=None,
                          sample_save_point:Literal['reference','start','end']='end', #None here means match the original pdb!! set to the same value as pdb_reference_point or explicitly 'reference' to save the reference as the sample
                          sample_save_type:Literal['pdb','offset']='offset',
                          force_copy_duplicate_sample:bool=False,
                          trajectory_save_type:Literal['pdb','offset']='offset',
                          )->tuple[Optional[dict[str,Any]],Optional[dict[str,Any]]]:
    """
    Save trajectories/samples (and, optionally, reference pdbs for offsets). 

    `pdb_reference_point` and `sample_save_point` both refer to what point in the trajectory to save as a reference/sample. Two special values:
    if `pdb_reference_point` is None, it will simply save (copy) the original pdb (from the DFMDict condition) to the reference directory. Similarly,
    if `sample_save_point` is 'reference', the sample point will just be the reference (the pdb_reference if saved, otherwise the original pdb from the condition).

    If save_pdb_references is True, reference pdbs will be saved to out_dir/'pdb'/'reference', and all offsets in samples/trajectories will point to
    these reference pdbs as the 0 point. Otherwise, the original pdb from the DFMDict condition will be used as the reference.

    Samples will be written to a top-level out_dir/'sample.csv' file, with sample id in the 'index' column and reference pdb indicated by the 'Filename'
    column. If `sample_save_type` == 'pdb', this pdb file will contain the sample and be written* to out_dir/'pdb'/'samples'. If
    `sample_save_type` == 'offset', the pdb file will instead be a reference pdb (existing or newly saved), and the file will additionally have columns
    ["Offset_Tr_X", "Offset_Tr_Y", "Offset_Tr_Z"] and/or ["Offset_Rot_X", "Offset_Rot_Y", "Offset_Rot_Z"] which store the translational/rotational
    components of the offset vector between the reference and the sampled point.

    Individual trajectory csvs will be written to out_dir/'trajectories'/trajectory_{id}.csv, with an index/indices maintained in top-level trajectory_index.csv
    and trajectory_index_{num}.csv files. Each trajectory file will have columns 'Index', with an integer index from 0 to N-1 of each step of the trajectory,
    'Timestep', with the time t of each point, and then columns depending on `trajectory_save_type`. If `trajectory_save_type` == 'offset', there will be
    additional offset columns as described above for samples, relative to the 'PDB_File' column in the trajectory index. If `trajectory_save_type` == 'pdb',
    there will instead be a 'PDB_File' column in the trajectory csv itself, pointing to the pdb file corresponding to each step in the trajectory.

    *SPECIAL CASE: if the sample would be saved in pdb format, but it would match an existing pdb file - e.g. if `sample_save_point` == 'reference'
    OR `save_pdb_reference` == True and `sample_save_point` = `pdb_reference_point` - rather than writing a new pdb file, the samples.csv file will simply
    point to the existing pdb. This behavior can be disabled by setting force_copy_duplicate_sample to True.
    """

    if not (save_samples or save_trajectories):
        if save_pdb_references:
            warnings.warn("Skipping sample/trajectory saving, no reference pdb will be written! To silence this, set 'save_reference_pdb' to False")
        return (None,None)

    trajectory_dir = Path(trajectory_dir)
    pdb_dir = Path(pdb_dir)
    reference_dir = pdb_dir/'reference'
    sample_dir = pdb_dir/'samples'
    pdb_trajectory_dir = pdb_dir/'trajectories'

    def relative_to(path,reference):
        return Path(path).absolute().relative_to(Path(reference).absolute(),walk_up=True)
    
    xtraj = torch.stack([to_array_nobatch(x) for x in trajectory]).detach().cpu() #Nx3 or Nx6
    ttraj = torch.as_tensor(times).detach().cpu()

    pdb_reference_file = condition["orig_pdb"]
    
    if save_pdb_references:
        if (((not save_samples) or (sample_save_type == 'pdb' and (sample_save_point != pdb_reference_point and sample_save_point != 'reference'))) and
                ((not save_trajectories) or (trajectory_save_type == 'pdb'))): 
            #complex conditional, but boils down to whether we will use the reference pdb in either samples/trajectories
            #the extra sample logic is because the samples can optionally be set to point to the reference pdb
            warnings.warn("Saving reference pdbs despite neither samples nor trajectories using them as reference!")

        if pdb_reference_point is None:
            reference_offset = None
        elif pdb_reference_point == 'end':
            reference_offset = xtraj[-1].clone() #clone so -= doesn't raise an error when it xtraj subtracts from itself
        elif pdb_reference_point == 'start':
            reference_offset = xtraj[0].clone()
        else:
            raise ValueError(f"{pdb_reference_point:}")

        reference_dir.mkdir(parents=True,exist_ok=True)
        ref_file = reference_dir/f"{id}.pdb"

        if reference_offset is not None:
            ref_struct = get_offset_pdb(pdb_reference_file,*split_offset(reference_offset,offset_type))
            save_structure(ref_file,ref_struct)
        else:
            shutil.copy(pdb_reference_file,ref_file)

        pdb_reference_file = ref_file

        if reference_offset is not None:
            xtraj -= reference_offset
    
    sample_res:Optional[dict[str,Any]] = None
    if save_samples:
        sample_res = {"index":id}
        if not force_copy_duplicate_sample and sample_save_type == 'pdb' and (sample_save_point == 'reference' or (save_pdb_references and sample_save_point == pdb_reference_point)):
                #SPECIAL CASE - in other words, the samples would precisely match the reference. This is silly - just have the samples point to the reference!
                sample_res["Filename"] = relative_to(pdb_reference_file,pdb_dir)
        else:
            if sample_save_point == 'reference':
                sample_offset = None
            elif sample_save_point == 'end':
                sample_offset = xtraj[-1]
            elif sample_save_point == 'start':
                sample_offset = xtraj[0]
            else:
                raise ValueError(f"{sample_save_point=}")
            
            if sample_save_type == 'pdb':
                sample_dir.mkdir(parents=True,exist_ok=True)
                sample_file = sample_dir/f"{id}.pdb"
                sample_res["Filename"] = relative_to(sample_file,pdb_dir)

                if sample_offset is not None:
                    assert force_copy_duplicate_sample
                    sample_struct = get_offset_pdb(pdb_reference_file,*split_offset(sample_offset,offset_type))
                    save_structure(sample_file,sample_struct)
                else:
                    shutil.copy(pdb_reference_file,sample_file)
            elif sample_save_type == 'offset':
                sample_res["Filename"] = relative_to(pdb_reference_file,pdb_dir)

                sample_res.update(zip(offset_trajectory_columns(offset_type),(sample_offset.tolist() if sample_offset is not None else itertools.repeat(0))))
            else:
                raise ValueError(f"{sample_save_type=}")
            
    trajectory_res:Optional[dict[str,Any]] = None
    if save_trajectories:
        trajectory_dir.mkdir(parents=True,exist_ok=True)
        trajectory_csv = trajectory_dir/f'trajectory_{id}.csv'

        trajectory_res = {"index":id, "Trajectory_File":relative_to(trajectory_csv,trajectory_dir)}#,"PDB_File":pdb_path}

        if trajectory_save_type == 'offset':
            assert xtraj.ndim == 2
            assert xtraj.shape[1] == (6 if offset_type == "Translation+Rotation" else 3)
            
            columns = ["Timestep"] + offset_trajectory_columns(offset_type)
            data = torch.cat([ttraj[...,None],xtraj],dim=1)                    
            trajectory_df = pd.DataFrame(columns=columns,data=data.numpy(force=True))
            trajectory_res["PDB_File"] = relative_to(pdb_reference_file,pdb_dir)

        elif trajectory_save_type == 'pdb':
            pdb_trajectory_dir.mkdir(parents=True,exist_ok=True)

            filenames = [pdb_trajectory_dir/f'{id}_{i}.pdb' for i in range(len(xtraj))]
            for name,offset in zip(filenames,xtraj):
                offset_pdb = get_offset_pdb(pdb_reference_file,*split_offset(offset,offset_type))
                save_structure(trajectory_dir/name,offset_pdb)
            
            trajectory_df = pd.DataFrame({"Timestep":ttraj.numpy(force=True),"PDB_File":map(lambda f: str(relative_to(f,pdb_dir)),filenames)})
        else:
            raise ValueError(f"{trajectory_save_type=}")

        if integrand_results is not None:
            for name,result in integrand_results.items():
                trajectory_df[f"accumulated_integrand: {name}"] = result

        trajectory_df.to_csv(trajectory_csv,index_label="Index")

        


    return (sample_res, trajectory_res)

def write_likelihood_outputs(config:DictConfig,
                            likelihoods:Iterable[tuple[str,Sequence[LigDict],Sequence[float],DFMDict,dict[str,Sequence[float|ArrayLike]]]],
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


    ## GENERAL PDB WRITING PREP
    pdb_folder = out_dir/"pdb" #the logic of when to make this directory is complicated, it's handled in write_dfmdock_samples

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
        if config.get("sample_save_type",'offset') == 'offset':
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
    save_trajectory_index = config.get("write_trajectory_index",True) and save_trajectories
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
        i = 0
        for ids,trajectories,times,conditions,integrand_resultss in likelihoods:
            with torch.profiler.record_function("Writing Likelihoods"):
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
                        row = {"id":id,
                            "prior_position":to_array_nobatch(prior_endpoint[0]).tolist(),
                                "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                                **{f"prior:{name}":val for name,val in prior_results.items()},
                                **{f"integrand:{name}":val[-1] for name,val in integrand_results.items()}} #write last accumulated likelihood
                        likelihoods_writer.writerow(row)

                    sample_out, traj_out = write_dfmdock_samples(
                        trajectory_folder,
                        pdb_folder,
                        id,
                        trajectory,
                        time,
                        conditions,
                        offset_type,
                        write_samples,
                        save_trajectories,
                        integrand_results=integrand_results if config.get("save_trajectory_likelihoods") else None,
                        save_pdb_references=config.get("save_pdb_references",False),
                        pdb_reference_point=config.get("pdb_reference_point",None),
                        sample_save_point=config.get("sample_save_point","end"),
                        sample_save_type=config.get("sample_save_type","offset"),
                        force_copy_duplicate_sample=config.get("force_copy_duplicate_sample",False),
                        trajectory_save_type=config.get("trajectory_save_type","offset"),
                    )

                    if sample_out:
                        samples_writer.writerow(sample_out)

                    if traj_out:
                        for cutoff,file,writer in trajectory_indices:
                            if cutoff is None or acc_trajnum <= cutoff:
                                writer.writerow(traj_out)
                        acc_trajnum += 1
                    
                    i += 1
                    if i == 2:
                        return


    finally:
        #make sure the files are closed!!!
        if likelihoods_handle is not None:
            likelihoods_handle.close()
        if samples_handle is not None:
            samples_handle.close()
        for cutoff,file,writer in trajectory_indices:
            file.close()


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
    


    

    # with torch.profiler.profile(activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA, # Only include if CUDA is available
    #         ],
    #         # schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=2), #don't start recording until we've done at least one 'warmup' cycle
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,) as prof:
        write_likelihood_outputs(config,
                                likelihoods,
                                integrands,
                                priors,
                                offset_type,
                                batch_size)


    # try:
    #     torch.cuda.memory._dump_snapshot(f"dfmdock_snapshot.pickle")
    # except Exception as e:
    #     logging.error(f"Failed to capture memory snapshot {e}")

    # with open("memory_summary.txt","w") as f:
    #     f.write(torch.cuda.memory_summary())


    # from IPython import embed; embed()

    # # Stop recording memory snapshot history.
    # torch.cuda.memory._record_memory_history(enabled=None)

    # prof.export_memory_timeline(f"dfmdock_timeline.html", device="cuda:0")
    # prof.export_chrome_trace("dfmdock_trace.json")
        




# def write_dfmdock_trajectory(trajectory_dir:Path,
#                              id:str,
#                              trajectory:Sequence[LigDict],
#                              times:Sequence[float],
#                              condition:DFMDict,
#                              offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
#                              relative_to:Optional[str|Path]=None,
#                              save_type:Literal['offset','pdb','offset+reference_pdb']='offset',
#                              offset_reference:Literal['start','end','condition']='condition',
#                              warn_modify_endpoint_unsave:bool=True,
#                              )->tuple[Path,Path]:
#     """Write a DFMDock trajectory to disk. Recall that for likelihood computation, the fundamental data representation
#     is that of a 3d or 6d offset vector representing [translation or rotation] or [translation + rotation] respectively, 
#     and is thus described relative to some conditioned pdb file's receptor and ligand coordinates (the data passed in 'condition').

#     Depending on the value of 'offset_reference', the offsets will be recalculated in terms of a particular receptor/ligand 
#     configuration 'reference'. For offset_reference='condition' (default), the reference is the default configuration specfied in
#     `condition`, so no recalculation is necessary. For offset_reference = 'end', the last point in the path is used; for 
#     offset_reference = 'start', the first point is used. UNUSED IF save_type = 'pdb' (since no reference is necessary for raw pdbs).
#     Additionally, if save_type = 'offset' and offset_reference != 'condition', offsets will be recalculated, but the reference
#     WILL NOT BE SAVED! A warning will be displayed unless warn_modify_endpoint_unsave is set to False.
    
#     The `save_type` parameter describes how you want the data to be saved:
    
#     - With save_type = 'offset' (the default), timepoints ('Timestep') and offset vectors (see offset_trajectory_columns()) 
#     will be saved to the csv file 'trajectory_{id}.csv', and no pdb data will be saved. Returns the path to the csv file
    
#     - With save_type = 'pdb', the trajectory will instead be saved as a sequence of pdb files in a subfolder of the given 
#     trajectory directory. These pdbs are generated by offsetting the original pdb file ligand by the offset vector.
#     NOTE THAT THIS WILL SIGNIFICANTLY SLOW DOWN THE PROCESS OF WRITING TRAJECTORIES! Instead of offsets, the csv file 
#     will be populated with timepoints ('Timestep') and filenames ('PDB_File') which point to the pdb files. Pdbs will 
#     be saved with the naming convention {id}_{index}.pdb. Returns the path to the csv file.

#     - With save_type = 'offset+reference_pdb', the trajectory will be saved as with save_type = 'offset' along with a pdb file
#     containing the "reference" as specified by the value of `offset_reference`. This allows the offsets and reference file to be used
#     as the trajectory and condition respectively for likelihood computation. Returns a tuple of the paths to the trajectory 
#     and reference files.

#     Returns: tuple of (trajectory_csv, pdb_file). The pdb file will be relative to the pdb dir, trajectory_dir/'pdb', 
#     even when the pdb is external (using relative syntax, '..'). If relative_to is provided, trajectory_csv will be returned
#     relative to 'relative_to'.
    
#     """


    

#     trajectory_csv = trajectory_dir/f'trajectory_{id}.csv'
#     pdb_folder = trajectory_dir/'pdb'
#     if 'pdb' in save_type:
#         pdb_folder.mkdir(exist_ok=True)
    

#     xtraj = torch.stack([to_array_nobatch(x) for x in trajectory]).cpu() #Nx3 or Nx6
#     ttraj = torch.as_tensor(times).cpu()
    
#     if 'offset' in save_type:
#         if offset_reference != 'condition' and save_type != 'offset+reference_pdb' and warn_modify_endpoint_unsave:
#                 warnings.warn("Recalculating offsets without saving new relative pdb when writing dfmdock trajectory!")
                    
#         if offset_reference == 'end':
#             reference_offset = xtraj[-1] #yes I'm great at names
#         elif offset_reference == 'start':
#             reference_offset = xtraj[0]
#         elif offset_reference == 'condition':
#             reference_offset = torch.zeros((xtraj.shape[-1]))
#         else:
#             raise ValueError(reference_offset)
        
#         xtraj -= reference_offset #ummm this probably just, like, *works* for rotation haha :nervous:
            
#         assert xtraj.ndim == 2
#         assert xtraj.shape[1] == (6 if offset_type == "Translation+Rotation" else 3)
        
#         columns = ["Timestep"] + offset_trajectory_columns(offset_type)
#         data = torch.cat([ttraj[...,None],xtraj],dim=1)                    
#         trajectory_df = pd.DataFrame(columns=columns,data=data.numpy(force=True))
#         trajectory_df.to_csv(trajectory_csv,index_label="Index")
        
        
#         if save_type == 'offset+reference_pdb':
#             offset_pdb = get_offset_pdb(condition["orig_pdb"],*split_offset(reference_offset,offset_type))
#             pdb_file = pdb_folder/f'{id}.pdb'
#             save_structure(pdb_file,offset_pdb)
#         else:
#             pdb_file = Path(condition["orig_pdb"])
#     else:
#         filenames = [Path('pdb')/f'{id}_{i}.pdb' for i in range(len(xtraj))]
#         for name,offset in zip(filenames,xtraj):
#             offset_pdb = get_offset_pdb(condition["orig_pdb"],*split_offset(offset,offset_type))
#             save_structure(trajectory_dir/name,offset_pdb)
        
#         trajectory_df = pd.DataFrame({"Timestep":ttraj.numpy(force=True),"PDB_File":map(str,filenames)})
#         trajectory_df.to_csv(trajectory_csv,index_label="Index")
#         pdb_file = Path(condition["orig_pdb"])

#     if relative_to is not None:
#         try:
#             trajectory_csv = trajectory_csv.absolute().relative_to(Path(relative_to).absolute(),walk_up=True)
#         except ValueError:
#             pass
#     if pdb_file is not None:
#         try:
#             pdb_file = pdb_file.absolute().relative_to(pdb_folder.absolute(),walk_up=True)
#         except ValueError:
#             pass

#     return trajectory_csv,pdb_file
        

# def make_dfmdock_sample(id:str,
#                         trajectory:Sequence[LigDict],
#                         times:Sequence[float],
#                         condition:DFMDict,
#                         offset_type:Literal["Translation", "Rotation", "Translation+Rotation"],
#                         #TODO: UNIFY THIS POINT SELECTION AND ANCHOR CALCULATION WITH WRITE_DFMDOCK_TRAJECTORY!
#                         save_point:Literal["start","end"]="start", 
#                         save_new_pdb:bool=False,
#                         new_pdb_file:Optional[str|Path]=None,#samples_pdb_folder/f"{id}.pdb",
#                         relative_to:Optional[str|Path]=None,
#                         include_offset:bool=False):#=config.get("write_sample_with_offset",False)):
#     """Returned pdb filename will always be relative to 'relative_to' (no matter how distantly) if provided."""

#     if save_point == 'end':
#         point = trajectory[-1] 
#     elif save_point == 'start':
#         point = trajectory[0]
#     else:
#         raise ValueError(save_point)
    
#     offset = to_array_nobatch(point)

#     if not (save_new_pdb or include_offset or torch.allclose(offset,torch.zeros_like(offset))):
#         raise ValueError("Can't faithfully save sample with nonzero offset without either writing the offset or saving the pdb!")

#     if save_new_pdb:
#         ##if we're going to save the offset, don't *also* offset the pdb!
#         updates = split_offset(offset,offset_type) if not include_offset else (None,None)
#         new_pdb = get_offset_pdb(condition["orig_pdb"],*updates)
#         save_structure(new_pdb_file,new_pdb)
#         pdb_file = new_pdb_file
#     else:
#         pdb_file = condition["orig_pdb"]

#     if pdb_file is not None and relative_to is not None:
#         try:
#             pdb_file = Path(pdb_file).absolute().relative_to(Path(relative_to).absolute(),walk_up=True)
#         except ValueError:
#             pass

#     result = {"index":id,"Filename":pdb_file}
#     if include_offset:
#         result.update(zip(offset_trajectory_columns(offset_type),offset.tolist()))

#     return result

    

if __name__ == '__main__':
    main()
