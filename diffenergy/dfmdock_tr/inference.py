from contextlib import contextmanager
from csv import DictWriter
import functools
from functools import cached_property
import itertools
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence
import warnings

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import pandas as pd
import torch
from tqdm import tqdm
from diffenergy.dfmdock_tr.docked_dataset import DockedDatum, PDBImporter
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel
from diffenergy.dfmdock_tr.likelihood_helpers import DFMDict, LigDict, DFMDockModelEval, split_offset
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.dfmdock_tr.utils.biotite_utils import get_chain_coords, get_offset_pdb
from diffenergy.dfmdock_tr.utils.esm_utils import load_coords
from diffenergy.dfmdock_tr.utils.metrics import METRICS_KEYS, compute_metrics
from diffenergy.helper import diffusion_coeff, prior_gaussian_nd
from diffenergy.inference import DiffEnergyLikelihood, ForcesMixin, MapDataset, SizeWrappedIter, get_integrands, get_paths, unzip, SizedIter
from diffenergy.likelihood import run_diff_likelihood, run_ode_likelihood

from biotite.structure.io import save_structure
from biotite.structure import AtomArray

def get_sample_metrics(gt_pdb:str|Path|AtomArray,sample_pdb:str|Path|AtomArray, rec_chain:str="A", lig_chain:str="B"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt_rec,_ = get_chain_coords(gt_pdb,rec_chain)
        gt_lig,_ = get_chain_coords(gt_pdb,lig_chain)
        sample_rec,_ = get_chain_coords(sample_pdb,rec_chain)
        sample_lig,_ = get_chain_coords(sample_pdb,lig_chain)
    return compute_metrics((torch.as_tensor(sample_rec),torch.as_tensor(sample_lig)),(torch.as_tensor(gt_rec),torch.as_tensor(gt_lig)))

class DFMDockLikelihood(DiffEnergyLikelihood[LigDict,DFMDict]):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    @classmethod
    def to_array(cls,x:LigDict)->torch.Tensor:
        return x['offset']

    @classmethod
    def from_array(cls,a,device:str|torch.device='cuda')->LigDict:
        return {'offset':torch.as_tensor(a,dtype=torch.float,device=torch.device(device))}

    
    @cached_property
    def out_pdb_folder(self):
        pdb_folder = self.out_dir/"pdb"
        pdb_folder.mkdir(parents=True,exist_ok=True)
        return pdb_folder
    
    @property
    def offset_type(self)->Literal["Translation","Rotation","Translation+Rotation"]:
        offset_type:str = self.config.offset_type
        if offset_type not in ("Translation","Rotation","Translation+Rotation"):
            raise ValueError("DFMDock offset_type must be one of 'Translation', 'Rotation', or 'Translation+Rotation'")
        return offset_type
    
    @property
    def offset_trajectory_columns(self):
        tr_columns = ["Offset_Tr_X", "Offset_Tr_Y", "Offset_Tr_Z"]
        rot_columns = ["Offset_Rot_X", "Offset_Rot_Y", "Offset_Rot_Z"]

        match self.offset_type:
            case "Translation": columns = tr_columns  # noqa: E701
            case "Rotation": columns = rot_columns  # noqa: E701
            case "Translation+Rotation": columns = tr_columns + rot_columns  # noqa: E701
            case _: raise ValueError  # noqa: E701
        return columns
    
    @classmethod
    def dockeddatum_to_condition(cls,datum:DockedDatum,device:str|torch.device)->DFMDict:
        return {
            "orig_pdb": datum["pdb_file"],
            "lig_pos_orig": datum["lig_pos"].to(device),
            "rec_pos": datum["rec_pos"].to(device),
            "lig_x": datum["lig_x"].to(device),
            "rec_x": datum["rec_x"].to(device),
            "position_matrix": datum["position_matrix"].to(device),
        }

    def trajectory_index_writers(self,write_indices:bool,extra_fieldnames:Iterable[str]=[]):
        return super().trajectory_index_writers(write_indices,extra_fieldnames=['PDB_File','Trajectory_File',*extra_fieldnames])

    def sample_index_writer(self,write_samples:bool,extra_fieldnames:Iterable[str]=[],offset_columns=False):
        return super().sample_index_writer(write_samples,extra_fieldnames=['Filename',*(self.offset_trajectory_columns if offset_columns else []),*extra_fieldnames])
    
    def get_sample_metrics(self,sample:LigDict|AtomArray,condition:DFMDict):
        gt_pdb = condition["orig_pdb"] #so glad I put that in there from the beginning
        sample_struct = sample if isinstance(sample,AtomArray) else get_offset_pdb(gt_pdb,*self.split_offset(sample['offset']))
        return get_sample_metrics(gt_pdb,sample_struct)

    @property
    def metrics_file(self):
        return self.out_dir/"metrics.csv"

    @contextmanager
    def metrics_writer(self, write_metrics:bool): #TODO: DFMDock Energy support? "energy" is one of the outputs of model(batch), and it just uses the last one
        if write_metrics:
            with open(self.metrics_file,'w',newline='') as f:
                writer = DictWriter(f,fieldnames=['index',*METRICS_KEYS])
                writer.writeheader()
                yield writer
        else:
            yield None
    
    def load_samples(self, data_file:str|Path, pdb_dir:str|Path, importer:PDBImporter, device:str|torch.device='cuda')->Sequence[tuple[str,LigDict,DFMDict]]|Sequence[tuple[Sequence[str],LigDict,DFMDict]]:
        """Loads pdbs from a CSV file containing filenames. Returns tuples of (id, LigDict, DFMDict),
        where LigDict contains the offset vector for the ligand (zero by default!) and DFMDict contains the
        (NOT 0-CENTERED!) ligand and receptor coordinates as well as the other conditioning information (ESM embeddings, etc)
        required for model evaluation.
        """
        
        df = pd.read_csv(data_file, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
        ids = df.loc[:, "index"].values  # Extract the first column as ids
        paths = df.loc[:, "Filename"].values  # Extract the second column as filenames
        offset_columns = self.offset_trajectory_columns
        if any([col.startswith("Offset") for col in df.columns]):
            offsets = df.loc[:,offset_columns]
        else:
            offsets = None

        pdb_dir = Path(pdb_dir)
        def getpdb(idx:int)->tuple[str,LigDict,DFMDict]:
            id = str(ids[idx])
            pdb_path = pdb_dir/paths[idx]
            dfmdict = self.dockeddatum_to_condition(importer.get_pdb(str(pdb_path),id),device)
            if offsets is None:
                offset = torch.zeros(len(offset_columns),device=device,dtype=dfmdict["lig_pos_orig"].dtype)
            else:
                offset = torch.as_tensor(offsets[idx],device=device,dtype=dfmdict["lig_pos_orig"].dtype)
            return (id,
                    self.from_array(offset,device=device),
                    dfmdict)
        
        return MapDataset([(i,) for i in range(len(ids))],getpdb)
    
    @classmethod
    def load_trajectories(cls, trajectory_index_file:str|Path,pdb_dir:str|Path,trajectory_dir:str|Path,pdb_importer:PDBImporter,device:str|torch.device='cuda')->SizedIter[tuple[str,Path,DFMDict]]:
        trajectory_index_file = Path(trajectory_index_file)
        pdb_dir = Path(pdb_dir)
        trajectory_dir = Path(trajectory_dir)

        assert trajectory_index_file.suffix == '.csv'
        df = pd.read_csv(trajectory_index_file)
        
        #since we're reading the condition pdbs as we load the trajectories, needs to be a generator so we don't load them all at once!
        res:Iterable[tuple[str,Path,DFMDict]] = (
            (id, 
             trajectory_dir/trajectory_filename, 
             cls.dockeddatum_to_condition(pdb_importer.get_pdb(str(pdb_dir/pdb_filename),id), device=device))
                for id,pdb_filename,trajectory_filename in zip(df["index"],df["PDB_File"],df["Trajectory_File"]))
        #add length hint for progress bar
        return SizeWrappedIter(res,len(df['index']))
    
    @classmethod
    def load_trajectories_batched(cls, trajectory_index_file:str|Path,pdb_dir:str|Path,trajectory_dir:str|Path,pdb_importer:PDBImporter,batch_size:int,device:str|torch.device='cuda')->SizedIter[tuple[tuple[str,...],tuple[Path,...],tuple[DFMDict,...]]]:
        unbatched = cls.load_trajectories(trajectory_index_file,pdb_dir,trajectory_dir,pdb_importer,device=device)
        num_batches = math.ceil(len(unbatched)//batch_size)
        #add length hint for progress bar
        return SizeWrappedIter(((tuple(b[0] for b in batch),tuple(b[1] for b in batch), tuple(b[2] for b in batch)) for batch in itertools.batched(unbatched,batch_size)),num_batches)


    def load_trajectory(self,data_path:str|Path|tuple[str|Path,...], pdb_dir:str|Path, reference:DFMDict, device:str|torch.device='cuda')->Sequence[tuple[LigDict|tuple[LigDict,...],torch.Tensor]]:
        paths = data_path if isinstance(data_path,tuple) else [data_path]
        alltimes:Optional[torch.Tensor] = None
        sampleslist:list[list[LigDict]] = []
        
        columns = self.offset_trajectory_columns
        
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

            if all(coli in df.columns for coli in columns):
                data = torch.as_tensor(df[columns].values,dtype=torch.float32,device=device) #this just works yesssss
            elif not any(coli in df.columns for coli in columns) and "PDB_File" in df.columns: #we're reading a PDB trajectory, turn into offset
                data = torch.empty((len(df),3),dtype=torch.float32,device=device)
                for i,file in enumerate(df['PDB_File']):
                    pdb_file = Path(pdb_dir)/file
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        step_pos, _ = load_coords(str(pdb_file),"B")
                    dx = (torch.as_tensor(step_pos) - reference['lig_pos_orig'].cpu())[...,1,:].mean(dim=0) # we do /not/ need to move every pdb to the gpu lmao
                    data[i] = dx
                assert data.ndim == 2 and data.shape[1] == 3
            else:
                raise ValueError(f"incomplete columns in csv: {df.columns}")

            #make list of LigDict
            sampleslist.append([self.from_array(data[i],device=device) for i in range(data.shape[0])])
        assert alltimes is not None
        samples:list[LigDict]|list[tuple[LigDict,...]] = [batch for batch in zip(*sampleslist)] if isinstance(data_path,tuple) else sampleslist[0] #needs to be NxB if batched so make sure to stack at dimension 1
        return list(zip(samples,alltimes))

    def load_priors(self,
                    sigma_min:float,
                    sigma_max:float,
                    batch_size:int|None):
        batched = batch_size is not None

        ### LOAD PRIORS
        priors:list[tuple[str,Callable[[LigDict|Sequence[LigDict],float,DFMDict],float]]] = []
        functions:DictConfig = self.config.get("prior_fns","smax_gaussian")
        if isinstance(functions,str):
            functions = functions.split(" ") # pyright: ignore[reportAssignmentType]
        if not isinstance(functions,Mapping):
            functions = DictConfig({f:{} for f in functions}) #ensure types is a DictConfig of Dicts
        
        for prior_fn,params in functions.items():
            match prior_fn:
                case "smax_gaussian":
                    def prior_likelihood_fn(x:LigDict|Sequence[LigDict],t:float, condition:DFMDict):
                        offset = self.to_array(x) # pyright: ignore[reportArgumentType] #assume x matches batchness and has been dealt with
                        tt = torch.as_tensor(t)
                        assert torch.allclose(tt, torch.ones_like(tt)), t #diffeq errors might mean it's not *quite* 1 but that's fine
                        if self.offset_type == "Rotation": #Assume rotational prior is just 0!
                            res = torch.zeros(offset.shape[:-1],dtype=offset.dtype)
                        else:
                            res = (prior_gaussian_nd(offset[...,:3],sigma_max)[0]) #first three components are always x,y,z. Assume rotational prior is just 0!
                        return res.numpy(force=True) if batched else res.item()
                    priors.append((prior_fn,prior_likelihood_fn)) # pyright: ignore[reportArgumentType]
        
        return priors

    def split_offset(self,offset:torch.Tensor|None,device:str|torch.device|None='cpu',detach:bool=True):
        return split_offset(offset,self.offset_type,device=device,detach=detach)

    def write_samples(self,
                      trajectory_dir:str|Path,
                      pdb_dir:str|Path,
                      id:str,
                      trajectory:Sequence[LigDict],
                      times:Sequence[float],
                      condition:DFMDict,
                      save_samples:bool,
                      save_trajectories:bool,
                      save_sample_metrics:bool,
                      integrand_results:Optional[Mapping[str,Sequence[float|np.ndarray|torch.Tensor]]]=None,
                      save_pdb_references:bool=False,
                      pdb_reference_point:Literal[None,'start','end']=None,
                      sample_save_point:Literal['reference','start','end']='end', #None here means match the original pdb!! set to the same value as pdb_reference_point or explicitly 'reference' to save the reference as the sample
                      sample_save_type:Literal['pdb','offset']='offset',
                      force_copy_duplicate_sample:bool=False,
                      trajectory_save_type:Literal['pdb','offset']='offset',
                      )->tuple[Optional[dict[str,Any]],Optional[dict[str,Any]],Optional[dict[str,Any]]]:
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
            return (None,None,None)

        trajectory_dir = Path(trajectory_dir)
        pdb_dir = Path(pdb_dir)
        reference_dir = pdb_dir/'reference'
        sample_dir = pdb_dir/'samples'
        pdb_trajectory_dir = pdb_dir/'trajectories'

        def relative_to(path,reference):
            return Path(path).absolute().relative_to(Path(reference).absolute(),walk_up=True)
        
        xtraj = torch.stack([self.to_array(x) for x in trajectory]).detach().cpu() #Nx3 or Nx6
        ttraj = torch.as_tensor(times).detach().cpu()

        pdb_reference_file = condition["orig_pdb"]
        
        ref_struct = None
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
                ref_struct = get_offset_pdb(pdb_reference_file,*self.split_offset(reference_offset))
                save_structure(ref_file,ref_struct)
            else:
                shutil.copy(pdb_reference_file,ref_file)

            pdb_reference_file = ref_file

            if reference_offset is not None:
                xtraj -= reference_offset
        
        sample_res:Optional[dict[str,Any]] = None
        metrics_res:Optional[dict[str,Any]] = None
        if save_samples:
            sample_struct = None
            sample_offset = None
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
                        sample_struct = get_offset_pdb(
                            ref_struct if ref_struct is not None else pdb_reference_file,
                            *self.split_offset(sample_offset))
                        save_structure(sample_file,sample_struct)
                    else:
                        shutil.copy(pdb_reference_file,sample_file)
                elif sample_save_type == 'offset':
                    sample_res["Filename"] = relative_to(pdb_reference_file,pdb_dir)

                    sample_res.update(zip(self.offset_trajectory_columns,(sample_offset.tolist() if sample_offset is not None else itertools.repeat(0))))
                else:
                    raise ValueError(f"{sample_save_type=}")
                
            if save_sample_metrics:
                #we need the pdb structure of the sample - compute if it hasn't been already
                if sample_struct is None:
                    sample_struct = get_offset_pdb(
                        ref_struct if ref_struct is not None else pdb_reference_file,
                        *self.split_offset(sample_offset)) #note that if the sample offset is None, it will return the original structure
                metrics_res = self.get_sample_metrics(sample_struct,condition) #TODO: Add configuration for ground-truth selection?
                metrics_res['index'] = id
                #this only really works during sampling tbh where the reference is guaranteed to be the gt pdb
                
        trajectory_res:Optional[dict[str,Any]] = None
        if save_trajectories:
            trajectory_dir.mkdir(parents=True,exist_ok=True)
            trajectory_csv = trajectory_dir/f'trajectory_{id}.csv'

            trajectory_res = {"index":id, "Trajectory_File":relative_to(trajectory_csv,trajectory_dir)}#,"PDB_File":pdb_path}

            if trajectory_save_type == 'offset':
                assert xtraj.ndim == 2
                assert xtraj.shape[1] == (6 if self.offset_type == "Translation+Rotation" else 3)
                
                columns = ["Timestep"] + self.offset_trajectory_columns
                data = torch.cat([ttraj[...,None],xtraj],dim=1)                    
                trajectory_df = pd.DataFrame(columns=columns,data=data.numpy(force=True))
                trajectory_res["PDB_File"] = relative_to(pdb_reference_file,pdb_dir)

            elif trajectory_save_type == 'pdb':
                pdb_trajectory_dir.mkdir(parents=True,exist_ok=True)

                filenames = [pdb_trajectory_dir/f'{id}_{i}.pdb' for i in range(len(xtraj))]
                for name,offset in zip(filenames,xtraj):
                    offset_pdb = get_offset_pdb(
                        ref_struct if ref_struct is not None else pdb_reference_file,
                        *self.split_offset(offset))
                    save_structure(name,offset_pdb)
                
                trajectory_df = pd.DataFrame({"Timestep":ttraj.numpy(force=True),"PDB_File":map(lambda f: str(relative_to(f,pdb_dir)),filenames)})
            else:
                raise ValueError(f"{trajectory_save_type=}")

            if integrand_results is not None:
                for name,result in integrand_results.items():
                    trajectory_df[f"accumulated_integrand:{name}"] = result

            trajectory_df.to_csv(trajectory_csv,index_label="Index")


        return (sample_res, trajectory_res, metrics_res)



    def compute_likelihoods(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        if batched:
            raise ValueError("Batched DFMDock evaluation not supported!")

        to_array = self.to_array
        from_array = functools.partial(self.from_array,device=device)# if not batched else from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        score_model = Score_Model.load_from_checkpoint(self.config.checkpoint,deterministic=self.config.get("deterministic_score",False))
        score_model.freeze()
        score_model.to(device)

        model_eval = DFMDockModelEval(score_model,offset_type=self.offset_type,reset_seed_each_eval=self.config.get("reset_seed_each_eval",False),manual_seed=self.config.get("seed",0))
        
        scorefn = model_eval.score# if not batched else model_eval.batch_score
        divergencefn = model_eval.divergence# if not batched else model_eval.batch_divergence

        esm_model = ESMLanguageModel()
        pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

        assert batch_size is None
        load_samples_fn = lambda: self.load_samples(self.config.data_samples, self.config.pdb_dir, pdb_importer, device=device)  # noqa: E731
        load_trajectories_fn = lambda: self.load_trajectories(self.config.trajectory_index_file, self.config.pdb_dir, self.config.trajectory_dir, pdb_importer)  # noqa: E731
        get_trajectory_fn = lambda trajectory_file, condition: self.load_trajectory(trajectory_file, self.config.pdb_dir, condition, device=device)  # noqa: E731

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))

        priors = self.load_priors(
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
                        get_trajectory_fn, # pyright: ignore[reportArgumentType]
                        device)

        ### RUN LIKELIHOOD COMPUTATION

        self.initialize_out_dir()
        self.write_config(self.out_config_file)
                
        int_type = self.config.integral_type

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        write_samples = self.config.get("write_samples",False)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories
        sample_save_type = self.config.get("sample_save_type","offset")
        write_metrics = self.config.get("write_sample_metrics",False) and write_samples
        
        ## WRITE OUTPUT
        acc_trajnum = 0
        with (  #open the various global output csv.DictWriters
                self.likelihoods_writer(True, 
                    prior_names=[name for (name,_) in priors],
                    integrand_names=[integrand.name() for integrand in integrands])     as likelihoods_writer,
                self.sample_index_writer(write_samples,
                                         offset_columns=sample_save_type == 'offset')   as samples_writer, 
                self.trajectory_index_writers(write_trajectory_index)                   as trajectory_indices,
                self.metrics_writer(write_metrics)                                      as metrics_writer,
            ):
            for (id,path) in tqdm(paths):
                if reset_seed_each_path:
                    torch.manual_seed(seed)
                if int_type == "ode":
                    #just assume paths are ode integrable. error will be thrown otherwise
                    trajectory, time, likelihood_result = run_ode_likelihood(path,integrands,accumulate=save_trajectories) # pyright: ignore[reportArgumentType]
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
                        "prior_position":self.to_array(prior_endpoint[0]).tolist(),
                            "prior_time":torch.as_tensor(prior_endpoint[1]).item(), 
                            **{f"prior:{name}":val for name,val in prior_result.items()},
                            **{f"integrand:{name}":val[-1] for name,val in likelihood_result.items()}} #write last accumulated likelihood
                    likelihoods_writer.writerow(row)
                    
                    
                sample_out, traj_out, metrics_out, = self.write_samples(
                    self.out_trajectory_folder,
                    self.out_pdb_folder,
                    id, # pyright: ignore[reportArgumentType]
                    trajectory,
                    time,
                    condition,
                    write_samples,
                    save_trajectories,
                    write_metrics,
                    integrand_results=likelihood_result if self.config.get("save_trajectory_likelihoods") else None,
                    save_pdb_references=self.config.get("save_pdb_references",False),
                    pdb_reference_point=self.config.get("pdb_reference_point",None),
                    sample_save_point=self.config.get("sample_save_point","end"),
                    sample_save_type=sample_save_type,
                    force_copy_duplicate_sample=self.config.get("force_copy_duplicate_sample",False),
                    trajectory_save_type=self.config.get("trajectory_save_type","offset"),
                )

                if sample_out:
                    samples_writer.writerow(sample_out) # pyright: ignore[reportOptionalMemberAccess]

                if traj_out:
                    for cutoff,writer in trajectory_indices.items():
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow(traj_out)
                    acc_trajnum += 1
                    
                if metrics_out:
                    metrics_writer.writerow(metrics_out) # pyright: ignore[reportOptionalMemberAccess]


class DFMDockForces(ForcesMixin, DFMDockLikelihood):
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
            for (id,P) in tqdm(paths):
                if reset_seed_each_path:
                    torch.manual_seed(seed)

                c = P.condition
                forces_csv_file = f'{id}.csv'
                index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
                with open(self.forces_folder/forces_csv_file,'w',newline='') as f2:
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



class DFMDockSampler(DFMDockLikelihood):

    def sample_random_offset(self,rec_pos, lig_pos, sigma:float)->torch.Tensor:
        device=rec_pos.device

        # get center of mass
        rec_cen = torch.mean(rec_pos, dim=(0, 1))
        lig_cen = torch.mean(lig_pos, dim=(0, 1))

        # get rotat update: random rotation vector
        restensors = []

        if "Translation" in self.offset_type:
            # get trans update: random noise + translate x2 to x1
            restensors.append(torch.normal(0.0, sigma, size=(3,), device=device) + (rec_cen - lig_cen))
        if "Rotation" in self.offset_type:
            raise NotImplementedError()

        return torch.cat(restensors,dim=0)

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

            ## Other relevant parameters: (see write_samples for details)

            # save_pdb_references: bool, default False. whether to save reference pdbs to the output folder. Default False, meaning offsets will be relative to the pdbs provided during training
            # pdb_reference_point: Literal[null,'start','end'], default null. what point on the trajectory path to save the reference, or null to just copy the original pdb
            # sample_save_point: Literal['reference','start','end'], default end. Save as above, but for the sampled point. 'reference' saves the reference point as the sample 
            # ^IF YOU WANT TRAJECTORIES RELATIVE TO A SAMPLE PDB, THIS IS HOW TO DO IT! SET SAVE PDB REFERENCE TO TRUE AND SAMPLE SAVE POINT TO REFERENCE
            # sample_save_type: Literal['offset','pdb'], default offset. Whether to save the samples as pdbs, or as offsets relative to the reference pdb.
            # force_copy_duplicate_sample: bool, default False. Whether to force duplicate the pdb if the sample matches the pdb (See write_samples)
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

        to_array = self.to_array
        from_array = functools.partial(self.from_array,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        score_model = Score_Model.load_from_checkpoint(self.config.checkpoint,deterministic=self.config.get("deterministic_score",False))
        score_model.freeze()
        score_model.to(device)

        model_eval = DFMDockModelEval(score_model,offset_type=self.offset_type)

        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        esm_model = ESMLanguageModel()
        pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        assert batch_size is None
        def load_noised_samples()->SizedIter[tuple[str,LigDict,DFMDict]]:
            dataset = self.load_samples(self.config.data_samples, self.config.pdb_dir, pdb_importer, device=device)

            def get_noised_sample(id:str,lig:LigDict,cond:DFMDict):
                assert torch.all(lig["offset"] == 0)
                rand_offset = self.sample_random_offset(cond["rec_pos"],cond["lig_pos_orig"],sigma_max)
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


        ### RUN SAMPLING

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        ## WRITE OUTPUT
        write_samples = self.config.get("write_samples",False)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories
        sample_save_type = self.config.get("sample_save_type","offset")
        write_metrics = self.config.get("write_sample_metrics",False) and write_samples

        acc_trajnum = 0
        with (  #open the various global output csv.DictWriters
                self.sample_index_writer(write_samples,
                                         offset_columns=sample_save_type == 'offset')   as samples_writer, 
                self.trajectory_index_writers(write_trajectory_index)                   as trajectory_indices,
                self.metrics_writer(write_metrics)                                      as metrics_writer,
            ):
            for (id,path) in tqdm(paths):
                trajectory,time = unzip(path)
                condition = path.condition

                sample_out, traj_out, metrics_out = self.write_samples(
                    self.out_trajectory_folder,
                    self.out_pdb_folder,
                    id,
                    trajectory,
                    time,
                    condition,
                    write_samples,
                    save_trajectories,
                    write_metrics,
                    integrand_results=None,
                    save_pdb_references=self.config.get("save_pdb_references",False),
                    pdb_reference_point=self.config.get("pdb_reference_point",None),
                    sample_save_point=self.config.get("sample_save_point","end"),
                    sample_save_type=sample_save_type,
                    force_copy_duplicate_sample=self.config.get("force_copy_duplicate_sample",False),
                    trajectory_save_type=self.config.get("trajectory_save_type","offset"),
                )

                if sample_out:
                    samples_writer.writerow(sample_out) # pyright: ignore[reportOptionalMemberAccess]

                if traj_out:
                    for cutoff,writer in trajectory_indices.items():
                        if cutoff is None or acc_trajnum < cutoff:
                            writer.writerow(traj_out)
                    acc_trajnum += 1
                    
                if metrics_out:
                    metrics_writer.writerow(metrics_out) # pyright: ignore[reportOptionalMemberAccess]
                    