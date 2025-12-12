# It's bad practice to inherit from the base class as a mixin but oh well. 
# I don't really feel like these belong in DiffEnergyLikelihood though


import abc
import csv
import functools
from contextlib import contextmanager
from csv import DictWriter
import os
import shutil
import warnings

from omegaconf import DictConfig, OmegaConf
from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from typing import Generic, Iterable, Optional, TypeVar


X = TypeVar("X") #data type of point
C = TypeVar("C") #data type of diffusion conditioning

class DiffEnergyLikelihood(abc.ABC, Generic[X,C]):
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

    @property
    def out_config_file(self):
        return self.out_dir/"config.yaml"

    def write_config(self,file:str|Path):
        with open(file,"w") as f:
            f.write(OmegaConf.to_yaml(self.config))

    @property
    def out_likelihoods_file(self):
        return self.out_dir/"likelihood.csv"

    @contextmanager
    def likelihoods_writer(self,write_likelihoods:bool,prior_names:Iterable[str]=[],integrand_names:Iterable[str]=[],extra_fieldnames:Iterable[str]=[]):
        ## WRITE LIKELIHOODS PREP
        likelihoods_handle: Optional[TextIOWrapper] = None
        likelihoods_writer = None
        if write_likelihoods:
            with open(self.out_likelihoods_file,"w") as likelihoods_handle:
                fieldnames = ['id',"prior_position","prior_time"] + [f"prior:{name}" for name in prior_names] + [f"integrand:{name}" for name in integrand_names]
                fieldnames.extend(extra_fieldnames)
                likelihoods_writer = csv.DictWriter(likelihoods_handle,fieldnames=fieldnames)
                likelihoods_writer.writeheader()
                yield likelihoods_writer
        else:
            yield None

    @property
    def out_samples_file(self):
        return self.out_dir/"samples.csv"

    @contextmanager
    def sample_index_writer(self,write_samples:bool,extra_fieldnames:Iterable[str]=[]):
        ## WRITE SAMPLES PREP
        samples_handle: Optional[TextIOWrapper] = None
        samples_writer: Optional[csv.DictWriter] = None

        if write_samples:
            with open(self.out_samples_file,"w") as samples_handle:
                fieldnames = ["index","Samples"] #TODO: regularize capitalization aaaa
                fieldnames.extend(extra_fieldnames)
                samples_writer = csv.DictWriter(samples_handle,fieldnames=fieldnames)
                samples_writer.writeheader()
                yield samples_writer
        else:
            yield None


    @cached_property
    def out_trajectory_folder(self):
        res = self.out_dir/"trajectories"
        res.mkdir(parents=True,exist_ok=True)
        return res

    @property
    def out_trajectory_index(self):
        return self.out_trajectory_folder/"trajectory_index.csv"

    @property
    def out_trajectory_indices(self):
        indices: dict[int|None,Path] = {None:self.out_trajectory_index}
        for index_limit in self.config.get("trajectory_extra_indices",[]):
            indices[index_limit] = self.out_trajectory_folder/f"trajectory_index_{index_limit}.csv"
        return indices

    @contextmanager
    def trajectory_index_writers(self,write_indices:bool,extra_fieldnames:Iterable[str]=[]):
        if write_indices:
            index_handles: dict[int|None,TextIOWrapper] = {ind:open(file,"w",newline='') for ind,file in self.out_trajectory_indices.items()}

            extras = list(extra_fieldnames)
            trajectory_indices = {ind:csv.DictWriter(f,fieldnames=["index","PDB_File","Trajectory_File"] + extras) for (ind,f) in index_handles.items()}
            [writer.writeheader() for writer in trajectory_indices.values()]

            try:
                yield trajectory_indices
            finally:
                for handle in index_handles.values():
                    handle.close()
        else:
            yield {}


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

Y = TypeVar("Y")
def unzip(it:Iterable[tuple[X,Y]])->tuple[list[X],list[Y]]:
    x,y = zip(*it)
    return list(x),list(y)