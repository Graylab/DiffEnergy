from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, Protocol, Sequence, TypeVar, overload
from typing_extensions import TypeVarTuple, Unpack

import torch
from torch.utils.data import Dataset
from diffenergy.torch_decorator import torch_fn
from diffenergy.gaussian_helper import gaussian_logpdf

#--------------------------------------------------------------------------
# Helper functions: marginal probability, diffusion coefficients, prior likelihood

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


def marginal_kernel_var(t, sigma_min, sigma_max):
    """
    Returns the variance of the perturbation kernel from time 0 to t, e.g.,
    the integral of the squared diffusion coefficient of the SDE from t=0 to t.
    
    Args:
        t: A vector of time steps
        sigma_min: the initial sigma of the SDE (at t=0)
        sigma_max: the noised sigma of the SDE (at t=1)
        
    Returns:
        a vector of variances"""
    
    t = torch.as_tensor(t)
    var_min = torch.as_tensor(sigma_min)**2
    var_max = torch.as_tensor(sigma_max)**2
    int_diff_coeff = var_min * (var_max / var_min) ** t - var_min
    return int_diff_coeff

def marginal_kernel_std(t, sigma_min, sigma_max):
    """
    Returns the std of the perturbation kernel from time 0 to t, e.g.,
    the square root of the integral of the squared diffusion coefficient of the SDE from t=0 to t.
    
    Args:
        t: A vector of time steps
        sigma_min: the initial sigma of the SDE (at t=0)
        sigma_max: the noised sigma of the SDE (at t=1)
        
    Returns:
        A vector of standard deviations"""
    return torch.sqrt(marginal_kernel_var(t,sigma_min,sigma_max))


def marginal_prob_std(t, sigma_min, sigma_max):
    r"""
    Returns the standard deviation of the marginal $p(x_t,t)$.
    This assumes the standard deviation of the data is sigma_min.

    Args:    
        t: A vector of time steps.
        sigma_min: the initial sigma of the SDE (at t=0, should match data distribution)
        sigma_max: the noised sigma of the SDE (at t=1, determines prior std)
        
    Returns:
        A vector standard deviations
    """    
    #if sigma_min = data std, then the total variance at time t is sigma_min^2 [the data variance] 
    # + (sigma_min^2)*(sigma_max^2/sigma_min^2)^t - sigma_min^2 [the perturbation kernel variance]
    # = (sigma_min^2)*(sigma_max^2/sigma_min^2)^t
    # and thus the std is as below
    return sigma_min * (sigma_max / sigma_min) ** t


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
    