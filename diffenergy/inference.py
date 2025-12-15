from __future__ import annotations
import abc
import csv
import functools
from contextlib import contextmanager
from csv import DictWriter
import itertools
import os
import shutil
import warnings

from omegaconf import DictConfig, OmegaConf, open_dict
from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Literal, Mapping, Optional, Protocol, Sequence, TypeVar, TypeVarTuple, Unpack, overload

import omegaconf
import torch
from torch.utils.data import Dataset

from torch import Tensor
from diffenergy.likelihood import ArrayLike, EnsembledIntegrablePath, FlowEquivalentODEPath, ForwardSDEPath, IntegrablePath, IntegrableSequence, InterpolatedIntegrableSequence, LikelihoodIntegrand, LinearPath, LinearizedFlowPath, PerturbedPath, PiecewiseDifferentiableSequence, ReverseSDEPath, ScoreDivDiffIntegrand, SpaceIntegrand, TimeIntegrand, TotalIntegrand


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
        Use allow_existing=True to overwrite this behavior; when writing, existing files in the output directory will be overwritten,
        though extraneous files might remain from the old folder. 
        
        allow_existing=True can be useful when instantiating DiffEnergyLikelihood with an existing config file
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

    def write_config(self,file:str|Path,strip_overwrite:bool=True):
        config = self.config.copy()
        if strip_overwrite:
            with open_dict(config):
                if "overwrite_output" in config: #make sure overwrite_output doesn't get propagated
                    del config.overwrite_output
        with open(file,"w") as f:
            f.write(OmegaConf.to_yaml(config))

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
                fieldnames = ["index"] #TODO: regularize capitalization aaaa
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
            trajectory_indices = {ind:csv.DictWriter(f,fieldnames=["index"] + extras) for (ind,f) in index_handles.items()}
            [writer.writeheader() for writer in trajectory_indices.values()]

            try:
                yield trajectory_indices
            finally:
                for handle in index_handles.values():
                    handle.close()
        else:
            yield {}


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


Y = TypeVar("Y")
def unzip(it:Iterable[tuple[X,Y]])->tuple[list[X],list[Y]]:
    x,y = zip(*it)
    return list(x),list(y)


P = TypeVarTuple("P")
B = TypeVarTuple("B")
T = TypeVar("T")
I = TypeVar("I")  # noqa: E741

def get_integrands(
        config:DictConfig,
        from_array:Callable[[ArrayLike],X],
        to_array:Callable[[X],Tensor],
        scorefn:Callable[[X,float,C],Tensor],
        divergencefn:Callable[[X,float,C],float|Tensor],
        diffusion_coeff_fn:Callable[[float],float|Tensor],
        ):

    ### LOAD INTEGRANDS
    integrands:list[LikelihoodIntegrand[X,C]] = []
    types:DictConfig = config.integrand_types
    if types is None:
        types = []
    if not isinstance(types,Mapping):
        types = DictConfig({t:{} for t in types}) #ensure types is a DictConfig of Dicts

    for integrand_type,params in types.items():
        def param(p:str,*args):
            if params is not None:
                if params.get(p,None) is not None:
                    return p
            return config.get(p,*args)

        intclasses:dict[str,type[ScoreDivDiffIntegrand[X,C]]] = {cls.__name__:cls for cls in [TotalIntegrand[X,C],TimeIntegrand[X,C],SpaceIntegrand[X,C]]}

        if integrand_type in intclasses:
            intcls = intclasses[integrand_type]
            integrand = intcls(scorefn,divergencefn,diffusion_coeff_fn,to_array,from_array)
        else:
            raise ValueError("Unknown integrand type:",integrand_type)

        integrands.append(integrand)


    if len(integrands) == 0 and not config.get("no_integrands",False):
        raise ValueError("""Must specify at least one integrand type!\n
                         `integrand_types` can either be a list of strings (classnames) or a list of mappings from classnames to
                         integrand-specific parameters. If an integrand parameter is not provided in the integrand-specific section,
                         it will look for that parameter in the global scope instead (hence defaults/shared parameters can be provided globally).
                         To specify an empty mapping, simply include the `ClassName:` line without any subitems below it.""")

    return integrands


class SizedIter(Protocol,Generic[X]):
    def __len__(self)->int:
        ...
    def __iter__(self)->Iterator[X]:
        ...


class MapDataset(Generic[X,Unpack[P]], Dataset[X], SizedIter[X], Sequence[X]):
    source:Sequence[tuple[Unpack[P]]]
    map:Callable[[Unpack[P]],X]

    @classmethod
    def chain(cls:type[MapDataset[X,Unpack[P]]],source:MapDataset[tuple[Unpack[B]],Unpack[P]],map:Callable[[Unpack[B]],X])->MapDataset[X,Unpack[P]]:
        map2 = source.map
        def map_composed(*args:Unpack[P]):
            return map(*map2(*args))
        return cls(source.source,map_composed)

    def __init__(self,source:Sequence[tuple[Unpack[P]]], map:Callable[[Unpack[P]],X]):
        self.source = source
        self.map = map

    def __len__(self):
        return len(self.source)

    @overload
    def __getitem__(self,index:int) -> X: ...
    @overload
    def __getitem__(self,index:slice) -> MapDataset[X,Unpack[P]]: ...
    def __getitem__(self, index:int|slice) -> X|MapDataset[X,Unpack[P]]:
        if isinstance(index,slice):
            return MapDataset(self.source[index],self.map)
        else:
            return self.map(*self.source[index])

    def __iter__(self) -> Iterator[X]:
        yield from (self[i] for i in range(len(self)))


class SizeWrappedIter(SizedIter[X]):
    def __init__(self,iter:Iterable[X],length:int):
        self.iter = iter
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(self.iter)


def get_paths(
        config:DictConfig,
        from_array:Callable[[ArrayLike],X],
        to_array:Callable[[X],Tensor],
        scorefn:Callable[[X,float,C],Tensor],
        divergencefn:Callable[[X,float,C],float|Tensor],
        diffusion_coeff_fn:Callable[[float],float|Tensor],
        load_samples:Callable[[],SizedIter[tuple[I,X,C]]],
        load_trajectories:Callable[[],SizedIter[tuple[I,T,C]]],
        get_trajectory:Callable[[T,C],Sequence[tuple[X,float]]],
        device:str|torch.device)->SizedIter[tuple[I,IntegrablePath[X,C]]]:

    device = torch.device(device)

    ### LOAD PATHS
    #ode integration: needs a timeschedule
    try:
        match config.get("ode_timeschedule","uniform"):
            case "uniform":
                ode_times = torch.linspace(0,1,config.ode_steps,device=device)
            case "reverse_uniform":
                ode_times = torch.linspace(1,0,config.ode_steps,device=device)
            case default:
                raise ValueError("Unknown ODE timeschedule method:",default)
    except omegaconf.errors.ConfigAttributeError:
        ode_times = None

    #sde integration: also needs a timeschedule sometimes.
    try:
        match config.get("sde_timeschedule","uniform"):
            case "uniform":
                sde_times = torch.linspace(0,1,config.sde_steps,device=device)
            case "reverse_uniform":
                sde_times = torch.linspace(1,0,config.sde_steps,device=device)
            case default:
                raise ValueError("Unknown SDE timeschedule method:",default)
    except omegaconf.errors.ConfigAttributeError:
        sde_times = None

    def load_endpoints(path,condition):
        trajectory = get_trajectory(path,condition)
        (x0,t0), (x1,t1) = trajectory[0], trajectory[-1]
        assert t0 == 0 and t1 == 1
        return x0, x1

    if (num_subsample := config.get('num_subsample')) is not None: #subsample loaded trajectory with num_subsample points
        #try to evenly distribute the subsampled points (as much as possible). from https://stackoverflow.com/a/9873935/13682828
        from math import ceil
        _get_trajectory = get_trajectory
        def subsampled_trajectory(arg:T,condition:C)->Sequence[tuple[X,float]]:
            #make sure we keep the endpoints
            assert num_subsample >= 2

            traj = _get_trajectory(arg,condition)
            indices = [(int(ceil(i * (len(traj)-1) / (num_subsample-1))),) for i in range(num_subsample)]

            #make sure we keep the endpoints!!
            assert (0,) in indices
            assert ((len(traj) - 1),) in indices

            return MapDataset(indices,lambda i: traj[i])

        get_trajectory = subsampled_trajectory


    int_args = config.get("integration",DictConfig({}))

    #integration parameter aliases for backwards compatibility
    for argname,topname in [("method","integration_method"),("atol","odeint_atol"),("rtol","odeint_rtol"),("method","odeint_method")]:
        if (newarg := config.get(topname)):
            if (arg := int_args.get(argname,None)) and newarg != arg:
                raise ValueError(f"Mismatch between top-level {topname} ({newarg}) and integration parameter {argname} ({arg})")
            int_args[argname] = newarg

    if "method" not in int_args and config.get("integral_type",None) == "diff":
        #for backwards compatibility, default method for "diff" integral type is "euler"
        int_args["method"] = "euler"

    int_args = dict(int_args)
    int_method = int_args.pop("method",None)

    # load path and associated dataset
    paths:SizedIter[tuple[I,IntegrablePath[X,C]]]
    match config.path_type:
        case "flow_ode":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            samples = load_samples()

            pathgen = ( #maybe this should be a dataloader or something idk
                (id,FlowEquivalentODEPath[X,C](
                    scorefn,
                    diffusion_coeff_fn,
                    ode_times,
                    (initial,0),
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                    for (id,initial,condition) in (samples)
                )
            paths = SizeWrappedIter(pathgen,len(samples))

        case "sde_trajectories":
            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],config.num_interpolants)

            pathgen = (
                (id,pathclass(get_trajectory(path,condition),
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in  (trajectories)
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "sde_trajectories_unreversed":
            #sde: get paths from diffusion tajectories, going from *1 to 0* 
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],config.num_interpolants)

            pathgen = (
                (id,pathclass(get_trajectory(path,condition)[::-1], #REVERSE
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in  (trajectories)
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "piecewise_trajectories":
            #piecewise sde: linear paths between points on diffusion trajectory
            trajectories = load_trajectories()

            pathgen = (
                (id,PiecewiseDifferentiableSequence(
                    get_trajectory(path,condition),
                    config.num_interpolants,
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                for id,path,condition in  (trajectories)
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "linear_trajectories":
            #linear: take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories()

            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,condition),condition) for id,trajectory,condition in  (trajectories))

            pathgen = (
                (id,LinearPath[X,C]((start,0),(end,1),ode_times,
                            to_array,
                            from_array,
                            condition,
                            int_method,
                            int_args))
                for (id,(start,end),condition) in endpoints
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "linearized_flow":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            samples = load_samples()

            pathgen = ( #maybe this should be a dataloader or something idk
                (id,LinearizedFlowPath[X,C](
                    scorefn,
                    diffusion_coeff_fn,
                    ode_times,
                    (sample,0),
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                    for id,sample,condition in  (samples)
                )
            paths = SizeWrappedIter(pathgen,len(samples))

        case "diff_data_translation":
            # Diffusion trajectory solely in data space: like sde_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling [e.g. ground truth]

            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],n_interp=config.num_interpolants)

            pathgen = (
                (id,pathclass([(x,0) for x,t in (get_trajectory(path,condition))],
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in  (trajectories)
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "data_translation":
            # Linear translation in data space: like linear_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling [e.g. ground truth]

            # take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories()

            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,condition),condition) for id,trajectory,condition in  (trajectories))

            pathgen = (
                (id,LinearPath((start,0),(end,0),ode_times, #start and end both 0!
                            to_array,
                            from_array,
                            condition,
                            int_method,
                            int_args))
                for (id,(start,end),condition) in endpoints
            )
            paths = SizeWrappedIter(pathgen,len(trajectories))

        case "reverse_sde":
            samples = load_samples()

            pathgen = (
                (id,ReverseSDEPath(
                    scorefn,
                    diffusion_coeff_fn,
                    config.get("noise_scale",1),
                    sde_times,
                    initial,
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                for (id,initial,condition) in  (samples))
            paths = SizeWrappedIter(pathgen,len(samples))

        case "forward_sde":
            samples = load_samples()

            pathgen = (
                (id,ForwardSDEPath(
                    diffusion_coeff_fn,
                    config.get("noise_scale",1),
                    sde_times,
                    initial,
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                for (id,initial,condition) in  (samples))
            paths = SizeWrappedIter(pathgen,len(samples))

        case "ensembled_forward_sde":
            samples = load_samples()
            n_paths = config.ensemble_num_paths
            noise_scale = config.get("noise_scale",1)
            pathgen = (
                (id,EnsembledIntegrablePath(
                    (
                        ForwardSDEPath(diffusion_coeff_fn,
                                    noise_scale,
                                    sde_times,
                                    initial,
                                    to_array,
                                    from_array,
                                    condition,
                                    int_method,
                                    int_args)
                        for _ in range(n_paths)
                    ),
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args,
                ))
                for (id,initial,condition) in  (samples)
            )
            paths = SizeWrappedIter(pathgen,len(samples))

        case "flow_along_trajectory": #that is, calculate the flowtime integral from t=0 to t=1 for each point in each diffusion trajectory
            trajectories = load_trajectories() #each trajectory of N timesteps will produce N flow results

            #note we replace each id with a tuple (id,t)
            numtraj = len(trajectories)
            trajectories,copy = itertools.tee(trajectories)
            trajsize = len(get_trajectory(*next(copy)[1:])) # get length of first trajectory for length hinting reasons
            initials =  (((f"{id}_{float(t)}",x,c) for (id,traj,c) in trajectories for (x,t) in get_trajectory(traj,c)))
            pathgen = (
                (id,
                 FlowEquivalentODEPath(scorefn,
                                       diffusion_coeff_fn,
                                       ode_times,
                                       (x,0),
                                       to_array,
                                       from_array,
                                       condition,
                                       int_method,
                                       int_args))
                for (id,x,condition) in initials
            ) #technically we've broken typing here by changing the id but w/e. I guess really this should be a custom load_samples fn but I can't be bothered
            paths = SizeWrappedIter(pathgen, trajsize*numtraj)

        case _:
            raise ValueError("Unknown path type:",config.path_type)

    if config.get("perturb_path",False):
        if config.integral_type == "ode":
            raise ValueError("Can't stochastically perturb an ODE! However, ODEIntegrablePaths can be used in discrete integral mode. Please set integral_type to 'diff' or disable perturbation")
        sigma:float = config.perturbation_sigma
        schedule:Literal['uniform','data'] = config.get("perturbation_schedule","data")
        paths = SizeWrappedIter(((id,PerturbedPath[X,C](path,schedule,sigma,path.condition,int_method,int_args)) for id,path in paths),len(paths)) #god I love generators

    return paths