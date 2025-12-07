from contextlib import contextmanager
import csv
from io import TextIOWrapper
from typing import Iterable, Literal, Optional, override

from omegaconf import DictConfig
from likelihood_class import DiffEnergyLikelihood
from scripts.likelihood_dfmdock import offset_trajectory_columns


class DFMDockLikelihood(DiffEnergyLikelihood):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
    
    @property
    def offset_type(self)->Literal["Translation","Rotation","Translation+Rotation"]:
        return self.config.offset_type

    def sample_index_writer(self,extra_fieldnames:Iterable[str]=[]):
        return super().sample_index_writer(extra_fieldnames=[*extra_fieldnames,*offset_trajectory_columns(self.offset_type)])
