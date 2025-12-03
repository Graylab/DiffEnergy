import itertools
import omegaconf
from torch.utils.data import Dataset
from diffenergy.likelihoodv3 import EnsembledIntegrablePath, FlowEquivalentODEPath, ForwardSDEPath, IntegrablePath, IntegrableSequence, InterpolatedIntegrableSequence, LikelihoodIntegrand, LinearPath, LinearizedFlowPath, PerturbedPath, PiecewiseDifferentiableSequence, ReverseSDEPath, ScoreDivDiffIntegrand, SpaceIntegrand, TimeIntegrand, TotalIntegrand, run_diff_likelihood, run_diff_likelihoods, run_ode_likelihood, run_ode_likelihoods

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from diffenergy.likelihoodv3 import (
    ArrayLike,
    Array)

import functools
from typing import Callable, Generic, Iterable, Iterator, Literal, Mapping, Protocol, Sequence, Optional, TypeVar, TypeVarTuple, overload

#some convenience wrappers for sized iterables
X = TypeVar("X",covariant=True)
class SizedIter(Protocol,Generic[X]):
    def __len__(self)->int:
        ...
    def __iter__(self)->Iterator[X]:
        ...

class SizeWrappedIter(SizedIter):
    def __init__(self,iter:Iterable[X],length:int):
        self.iter = iter
        self.length = length

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self):
        return iter(self.iter)
    
P = TypeVarTuple("P")
X = TypeVar("X")
class MapDataset(Generic[X,*P], Dataset[X], SizedIter[X], Sequence[X]):
    source:Sequence[tuple[*P]]
    map:Callable[[*P],X]

    @classmethod
    def chain[*B](cls:type["MapDataset[X,*P]"],source:"MapDataset[tuple[*B],*P]",map:Callable[[*B],X])->"MapDataset[X,*P]":
        map2 = source.map
        def map_composed(*args:*P):
            return map(*map2(*args))
        return cls(source.source,map_composed)

    def __init__(self,source:Sequence[tuple[*P]], map:Callable[[*P],X]):
        self.source = source
        self.map = map

    def __len__(self):
        return len(self.source)

    @overload
    def __getitem__(self,index:int) -> X: ...
    @overload
    def __getitem__(self,index:slice) -> "MapDataset[X,*P]": ...
    def __getitem__(self, index:int|slice) -> X|"MapDataset[X,*P]":
        if isinstance(index,slice):
            return MapDataset(self.source[index],self.map)
        else:
            return self.map(*self.source[index])

    def __iter__(self) -> Iterator[X]:
        yield from (self[i] for i in range(len(self)))
        


def get_paths[X,C,T,I](
        config:DictConfig,
        from_array:Callable[[ArrayLike],X],
        to_array:Callable[[X],Array],
        scorefn:Callable[[X,float,C],Array],
        divergencefn:Callable[[X,float,C],float|Array],
        diffusion_coeff_fn:Callable[[float],float|Array],
        load_samples:Callable[[],SizedIter[tuple[I,X,C]]],
        load_trajectories:Callable[[],SizedIter[tuple[I,T,C]]],
        get_trajectory:Callable[[T,C],Sequence[tuple[X,float]]],
        device:str|torch.device)->Iterable[tuple[I,IntegrablePath[X,C]]]:
    
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
    
    #ode integration: also needs a timeschedule sometimes.
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

    if "method" not in int_args and config.integral_type == "diff":
        #for backwards compatibility, default method for "diff" integral type is "euler"
        int_args["method"] = "euler"
        
    int_args = dict(int_args)
    int_method = int_args.pop("method")

    # load path and associated dataset
    paths:Iterable[tuple[I,IntegrablePath[X,C]]]
    match config.path_type:
        case "flow_ode":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_samples()

            paths = ( #maybe this should be a dataloader or something idk
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
                    for (id,initial,condition) in tqdm(dataloader)
                )
        case "sde_trajectories":
            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],config.num_interpolants)

            paths = (
                (id,pathclass(get_trajectory(path,condition),
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in tqdm(trajectories)
            )
        case "sde_trajectories_unreversed":
            #sde: get paths from diffusion tajectories, going from *1 to 0* 
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],config.num_interpolants)

            paths = (
                (id,pathclass(get_trajectory(path,condition)[::-1], #REVERSE
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in tqdm(trajectories)
            )
        case "piecewise_trajectories":
            #piecewise sde: linear paths between points on diffusion trajectory
            trajectories = load_trajectories()

            paths = (
                (id,PiecewiseDifferentiableSequence(
                    get_trajectory(path,condition),
                    config.num_interpolants,
                    to_array,
                    from_array,
                    condition,
                    int_method,
                    int_args))
                for id,path,condition in tqdm(trajectories)
            )

        case "linear_trajectories":
            #linear: take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories()

            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,condition),condition) for id,trajectory,condition in tqdm(trajectories))

            paths = (
                (id,LinearPath[X,C]((start,0),(end,1),ode_times,
                            to_array,
                            from_array,
                            condition,
                            int_method,
                            int_args))
                for (id,(start,end),condition) in endpoints
            )
        case "linearized_flow":
            #flow ode: get data samples from diffusion endpoints, run the flow forwards
            dataloader = load_samples()

            paths = ( #maybe this should be a dataloader or something idk
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
                    for id,sample,condition in tqdm(dataloader)
                )
            
        case "diff_data_translation":
            # Diffusion trajectory solely in data space: like sde_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling

            #sde: get paths from diffusion tajectories
            trajectories = load_trajectories()

            pathclass = IntegrableSequence[X,C]
            if config.get("interpolate_trajectories",False):
                pathclass = functools.partial(InterpolatedIntegrableSequence[X,C],n_interp=config.num_interpolants)

            paths = (
                (id,pathclass([(x,0) for x,t in (get_trajectory(path,condition))],
                              to_array,
                              from_array,
                              condition,
                              int_method,
                              int_args))
                for id,path,condition in tqdm(trajectories)
            )

        case "data_translation":
            # Linear translation in data space: like linear_trajectories, but always at time=0. 
            # Requires a prior function compatible with t0 sampling

            # take sampled paths, and just make a straight line from start to end
            trajectories = load_trajectories()

            #we love inline generators
            endpoints = ((id,load_endpoints(trajectory,condition),condition) for id,trajectory,condition in tqdm(trajectories))

            paths = (
                (id,LinearPath((start,0),(end,0),ode_times, #start and end both 0!
                            to_array,
                            from_array,
                            condition,
                            int_method,
                            int_args))
                for (id,(start,end),condition) in endpoints
            )

        case "reverse_sde":
            samples = load_samples()

            paths = (
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
                for (id,initial,condition) in tqdm(samples))
            
        case "forward_sde":
            samples = load_samples()

            paths = (
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
                for (id,initial,condition) in tqdm(samples))
        case "ensembled_forward_sde":
            samples = load_samples()
            n_paths = config.ensemble_num_paths
            noise_scale = config.get("noise_scale",1)
            paths = (
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
                for (id,initial,condition) in tqdm(samples)
            )

        case "flow_along_trajectory": #that is, calculate the flowtime integral from t=0 to t=1 for each point in each diffusion trajectory
            trajectories = load_trajectories() #each trajectory of N timesteps will produce N flow results

            #note we replace each id with a tuple (id,t)
            numtraj = len(trajectories)
            trajectories,copy = itertools.tee(trajectories)
            trajsize = len(get_trajectory(*next(copy)[1:]))
            initials = tqdm(((f"{id}_{float(t)}",x,c) for (id,traj,c) in trajectories for (x,t) in get_trajectory(traj,c)),total=numtraj*trajsize)
            paths = (
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

        case _:
            raise ValueError("Unknown path type:",config.path_type)

    if config.get("perturb_path",False):
        if config.integral_type == "ode":
            raise ValueError("Can't stochastically perturb an ODE! However, ODEIntegrablePaths can be used in discrete integral mode. Please set integral_type to 'diff' or disable perturbation")
        sigma:float = config.perturbation_sigma
        schedule:Literal['uniform','data'] = config.get("perturbation_schedule","data")
        paths = ((id,PerturbedPath[X,C](path,schedule,sigma,path.condition,int_method,int_args)) for id,path in paths) #god I love generators

    return paths

def get_integrands[X,C](
        config:DictConfig,
        from_array:Callable[[ArrayLike],X],
        to_array:Callable[[X],Array],
        scorefn:Callable[[X,float,C],Array],
        divergencefn:Callable[[X,float,C],float|Array],
        diffusion_coeff_fn:Callable[[float],float|Array],
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
    

def get_likelihoods[X,C,I](
        config:DictConfig,
        paths:Iterable[tuple[I,IntegrablePath[X,C]]],
        integrands:Sequence[LikelihoodIntegrand],
        device:str|torch.device,
                        ):
    device = torch.device(device)

    ### RUN LIKELIHOOD COMPUTATION

    #parallel setup (currently nonfunctional)
    parallel = config.get("parallel",False)
    cluster_kwargs = config.get("cluster_kwargs",{})
    if parallel:
        import ray
        ray.init(**cluster_kwargs)
    actor_kwargs = config.get("actor_kwargs",{})
    if device.type == 'cuda' and 'num_gpus' not in actor_kwargs:
        actor_kwargs['num_gpus'] = 1 #assume each actor will consume an entire gpu



    reset_seed_each_path = config.get("reset_seed_each_path",False)
    seed = config.get("seed",0)

    #attempted optimization, hope it works
    accumulate_path = config.get("save_trajectories",False)

    int_type = config.get("integral_type")
    if int_type == "ode":
        #just assume paths are ode integrable. error will be thrown otherwise
        for id,path in paths:
            if reset_seed_each_path:
                torch.manual_seed(seed)
            yield run_ode_likelihood(id,path,integrands,accumulate=accumulate_path)
        
    elif int_type == "diff":
        #use standard integration
        for id,path in paths:
            if reset_seed_each_path:
                torch.manual_seed(seed)
            yield run_diff_likelihood(id,path,integrands,accumulate=accumulate_path)
    else:
        raise ValueError(f"Unknown integral type: {int_type}. For standard (non-ode solver) numerical integration, use integral_type: \"diff\" (the default).")

