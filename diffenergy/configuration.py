from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from omegaconf import DictConfig
from diffenergy.unionenum import UnionEnumMeta, enum_union, extend_enum

class SamplesPath(str,Enum):
    FLOW_ODE = "flow_ode"
    STILL = "still"
    LINEARIZED_FLOW = "linearized_flow"
    REVERSE_SDE = "reverse_sde"
    FORWARD_SDE = "forward_sde"
    ANCHORED_LINEAR = "anchored_linear"


class TrajectoriesPath(str,Enum):
    SDE_TRAJECTORIES = "sde_trajectories"
    SDE_TRAJECTORIES_UNREVERSED = "sde_trajectories_unreversed"
    PIECEWISE_TRAJECTORIES = "piecewise_trajectories"
    LINEAR_TRAJECTORIES = "linear_trajectories" #endpoints only

    DATA_TRANSLATION = "data_translation" #endpoints only
    DIFF_DATA_TRANSLATION = "diff_data_translation"

    FLOW_ALONG_TRAJECTORY = "flow_along_trajectory"


# @extend_enum(SamplesPaths)
# @extend_enum(TrajectoriesPaths)
# class PathType(str,Enum): # Pa
#     pass


class IntegralType(str,Enum):
    diff = 'diff'
    ode = 'ode'
    piecewise_diff = 'piecewise_diff'
    piecewise_ode = 'piecewise_ode'

@dataclass(kw_only=True)
class DiffEnergyConfig:
    out_dir: str
    "output directory"

    overwrite_output: bool = False
    "whether to overwrite the output directory"

    resume_existing: bool = False
    "whether to resume existing inference found in the output directory"

    ## paths
    integral_type: IntegralType
    integration: dict[str,Any] = field(default_factory=dict)

    path_type:SamplesPath|TrajectoriesPath

    ode_timeschedule: str = 'uniform'
    ode_steps: Optional[int] = None
    
    sde_timeschedule: str = 'uniform'
    sde_steps: Optional[int] = None

    interpolate_trajectories: bool = False
    """Used to automatically interpolate IntegrableSequence paths 
    (like `sde_trajectories` into InterpolatedIntegrableSequence paths)"""
    num_interpolants: Optional[int] = None
    """Interpolation factor; e.g., the number of steps (#points - 1) is multiplied
    by a factor of num_interpolants"""

    piecewise_interpolants: int = 1
    """Interpolation factor for the PiecewiseDifferentiablePath wrapper. Equivalent rule to num_interpolants,
    multiplies number of steps by piecewise_interpolants."""

    anchor_point: float = 0
    "Anchor point used for anchored_linear path"

    noise_scale: float = 1
    "Noise scale value used for Reverse/ForwardSDEPath"

    perturb_path: bool = False
    "Whether to add a small amount of random noise to the generated path"
    perturbation_sigma: Optional[float] = None
    "standard deviation of added noise"
    perturbation_schecule: Literal['uniform','data'] = 'data'
    """Whether to use the path-provided time t as the time deltas for added noise [data] or
    to use a linear spacing of times [uniform]"""

    num_subsample: Optional[int] = None
    "If provided, subsample loaded trajectory [e.g. skip points] to have num_subsample points"
    
    ## sample writing    
    trajectory_extra_indices: list[int] = field(default_factory=list)
    "extra 'trajectory index' files to write containing this many trajectories [for subsampling]"

@dataclass(kw_only=True)
class _BaseLikelihoodMixin:
    integrand_types: Optional[list[str]|dict[str,dict[str,Any]]] = None
    """[Deprecated] which integrands to use during integration. 
    Can either be a list of class names (e.g. ["TotalIntegrand"]) or a dictionary
    of classnames and instantiation arguments, e.g. {"ParameterizedIntegrand":[args]}.
    DEPRECATED: All other integrands are equivalent to TotalIntegrand. Soon only values 
    of ["TotalIntegrand"] will be accepted."""

    no_integrands: bool = False
    "whether to allow 'integration' with no integrands [advanced use only]"


@dataclass(kw_only=True)
class DiffEnergyLikelihoodConfig(DiffEnergyConfig,_BaseLikelihoodMixin):
    pass
    
# d = DiffEnergyLikelihoodConfig(out_dir='wow',integral_type=IntegralType.diff,path_type=TrajectoriesPath.SDE_TRAJECTORIES)
# d.integrand_types
# d.out_dir