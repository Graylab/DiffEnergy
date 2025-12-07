

import abc
from contextlib import contextmanager
import csv
from io import TextIOWrapper
import os
from pathlib import Path
import shutil
from typing import Iterable, Optional
from functools import cached_property
import warnings

from omegaconf import DictConfig


class DiffEnergyLikelihood(abc.ABC):
    def __init__(self,config:DictConfig) -> None:
        self.config = config
        self._out_dir = None
        
        # Path where config, likelihoods, trajectories, samples, etc will be saved. 
        # By default, instantiating this class will require the out_dir to be empty; 
        
        
        
    def initialize_out_dir(self,allow_existing=False):
        """Initialize output directory, where config, likelihoods, trajectories, samples, etc will be saved.
        By default, initialization requires the output directory to be empty, and (if overwrite_output=True) will clear it otherwise.
        Use allow_existing=True to overwrite this behavior; existing files in the output directory will be overwritten upon write,
        though extraneous files might remain from the old folder.
        """
        
        out_dir = Path(self.config.out_dir)
        if out_dir.exists() and not allow_existing:
            if not self.config.get("overwrite_output",False):
                raise FileExistsError(out_dir,"""Pass '++overwrite_output=True' in the command line (recommended over config) or use config.overwrite_output to overwrite existing output.""")
            else:
                backup_out = out_dir.with_stem(out_dir.stem + "_backup")
                warnings.warn(f"Moving dir {out_dir} to backup directory {backup_out}. Subsequent calls will DELETE THIS BACKUP, so be careful!!")
                if backup_out.exists():
                    shutil.rmtree(backup_out)
                os.rename(out_dir,backup_out)
        out_dir.mkdir(parents=True,exist_ok=True)
        self._out_dir = out_dir
    
    @property
    def out_dir(self):
        if not self._out_dir:
            raise ValueError(f"Output directory not initialized! Please call initialize_out_dir() before using {type(self).__name__}.out_dir")
        return self._out_dir
        
    @contextmanager
    def likelihoods_writer(self,prior_names:Iterable[str]=[],integrand_names:Iterable[str]=[],extra_fieldnames:Iterable[str]=[]):
        ## WRITE LIKELIHOODS PREP
        likelihoods_file = self.out_dir/"likelihood.csv"
        write_likelihoods = self.config.get("write_likelihoods",True)
        likelihoods_handle: Optional[TextIOWrapper] = None
        likelihoods_writer: Optional[csv.DictWriter] = None
        if write_likelihoods:
            with open(likelihoods_file,"w") as likelihoods_handle:
                fieldnames = ['id',"prior_position","prior_time"] + [f"prior:{name}" for name in prior_names] + [f"integrand:{name}" for name in integrand_names]
                fieldnames.extend(extra_fieldnames)
                likelihoods_writer = csv.DictWriter(likelihoods_handle,fieldnames=fieldnames)
                likelihoods_writer.writeheader()
                yield likelihoods_writer
        else:
            yield None
    
    @contextmanager
    def sample_index_writer(self,write_samples:bool,extra_fieldnames:Iterable[str]=[]):
        ## WRITE SAMPLES PREP
        samples_file = self.out_dir/"samples.csv"
        samples_handle: Optional[TextIOWrapper] = None
        samples_writer: Optional[csv.DictWriter] = None

        if write_samples:
            with open(samples_file,"w") as samples_handle:
                fieldnames = ["index","Samples"]
                fieldnames.extend(extra_fieldnames)
                samples_writer = csv.DictWriter(samples_handle,fieldnames=fieldnames) #TODO: regularize capitalization aaaa
                samples_writer.writeheader()
                yield samples_writer
        else:
            yield None
    
    
    @cached_property
    def trajectory_folder(self):
        res = self.out_dir/"trajectories"
        res.mkdir(parents=True,exist_ok=True)
        return res
    
    @contextmanager
    def trajectory_index_writers(self,write_indices:bool,extra_fieldnames:Iterable[str]=[]):
        if write_indices:
            index_handles: list[tuple[int|None,TextIOWrapper]] = [(None,open(self.trajectory_folder/"trajectory_index.csv","w"))]
            if (extra := self.config.get("trajectory_extra_indices",None)):
                index_handles += [(ind,open(self.trajectory_folder/f"trajectory_index_{ind}.csv","w")) for ind in extra]
            extras = list(extra_fieldnames)
            trajectory_indices = [(ind,csv.DictWriter(f,fieldnames=["index","PDB_File","Trajectory_File"] + extras)) for (ind,f) in index_handles]
            [index[1].writeheader() for index in trajectory_indices]
            
            try:
                yield trajectory_indices
            finally:
                for _,handle in index_handles:
                    handle.close()
        else:
            yield []
            
            
    
    