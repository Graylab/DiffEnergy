from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, Protocol, Sequence, TypeVar, TypeVarTuple, Unpack, overload
from typing_extensions import ParamSpec

import torch
from torch.utils.data import Dataset
from diffenergy.torch_decorator import torch_fn
from diffenergy.gaussian_helper import gaussian_logpdf

#--------------------------------------------------------------------------
# Helper functions: marginal probability, diffusion coefficients, prior likelihood

def marginal_prob_std(t, sigma_min, sigma_max):
    r"""
    Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
        
    Returns:
        The standard deviation.
    """    
    return sigma_min * (sigma_max / sigma_min) ** t

def diffusion_coeff(t, sigma_min, sigma_max, clamp = False):
    r"""
    Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        
    Returns:
        The vector of diffusion coefficients.
    """
    t = torch.as_tensor(t)
    sigma_min = torch.as_tensor(sigma_min)
    sigma_max = torch.as_tensor(sigma_max)
    std = sigma_min * (sigma_max / sigma_min) ** t
    diff_coeff = std * torch.sqrt(torch.as_tensor(2 * (torch.log(sigma_max/sigma_min)), device=t.device))
    if clamp:
        diff_coeff = torch.clamp(diff_coeff, min=1e-5)
    return diff_coeff


## this is sigma(t)^2 in the EDM formulation
def int_diffusion_coeff_sq(t, sigma_min, sigma_max):
    """
    Compute the integral of the squared diffusion coefficient of our SDE from t=0 to t.
    Required to determine the total level of gaussian noise added to our data distribution at time t for ground truth score calculation.
    RETURNS THE *VARIANCE*, use sqrt to get the standard deviation!
    
    Args:
        t: A vector of time steps
        sigma_min: the initial sigma of our SDE (at t=0)
        sigma_max: the noised sigma of our SDE (at t=1)
        
    Returns:
        the vector of integrated squared diffusion coefficients"""
    
    t = torch.as_tensor(t)
    var_min = torch.as_tensor(sigma_min)**2
    var_max = torch.as_tensor(sigma_max)**2
    int_diff_coeff = var_min * ((var_max / var_min) ** t - 1)
    return int_diff_coeff

@torch_fn
def prior_log_gaussian_1d(x, sigma):
    """The likelihood of a 1-dimensional Gaussian distribution with mean zero and 
            uniform standard deviation sigma. Assumes x is a B-length 1D array"""
    assert x.ndim == 1
    return gaussian_logpdf(x[:,None],0,std=sigma)

@torch_fn
def prior_log_gaussian_nd_batched(x, sigma):
    """The likelihood of a D-dimensional Gaussian distribution with mean zero and 
            uniform standard deviation sigma. Assumes x is a BxD-dimensional array"""
    return gaussian_logpdf(x,0,std=sigma)

@torch_fn
def prior_log_gaussian_nd(x, sigma):
    """The likelihood of a D-dimensional Gaussian distribution with mean zero and 
            uniform standard deviation sigma. Assumes x is a D-dimensional array"""
    return gaussian_logpdf(x[None,:],0,std=sigma)




X = TypeVar('X')
P = TypeVarTuple("P")
B = TypeVarTuple("B")


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
    