

from dataclasses import dataclass
import functools
from typing import Callable, Optional, TypeVar

from torch import Tensor
from torch.profiler import record_function
import torch

from diffenergy.likelihoodv3 import BatchScoreModelEvaluator
from diffenergy.helper import int_diffusion_coeff_sq
from diffenergy.likelihoodv3 import ScoreModelEvaluator


def batched_normpdf_scalar(dx:Tensor,variances:Tensor): #dx: BxNxD, variances: N, result: BxN
    assert dx.ndim == 3, f"{dx.ndim=}"
    assert variances.ndim == 1, f"{variances.ndim=}"
    dim = dx.shape[-1]
    #scalar sigmas are much easier, no mucking about with inverses or decompositions
    nums = torch.linalg.vecdot(dx,dx)/variances[None,:]
    # print(f"{nums=}")
    logprob = -1/2*(dim*(torch.log(variances*2*torch.pi)) + nums)
    # print(f"{logprob=}")
    return torch.exp(logprob)


def batched_normpdf_matrix(dx:Tensor,covariances:Optional[Tensor]=None,scale_tril:Optional[Tensor]=None)->Tensor: #assume x has shape BxNxD, mean has shape NxD, and sigma has shape NxDxD (covarianve matrices)
    assert dx.ndim == 3, f"{dx.ndim=}"
    assert (covariances is None) + (scale_tril is None) == 1, "Exactly one of variances and scale_tril must be provided!"
    N = 0
    if covariances is not None:
        assert covariances.ndim == 3, f"{covariances.ndim=}"
        N = covariances.shape[0]
    elif scale_tril is not None:
        assert scale_tril.ndim == 3, f"{scale_tril.ndim=}"
        N = scale_tril.shape[0]

    means = torch.zeros((1,N,1),dtype=dx.dtype,device=dx.device)
        
    # add batch dimensions
    dist = torch.distributions.MultivariateNormal(means,covariance_matrix=covariances[None,...] if covariances is not None else None, scale_tril=scale_tril[None,...] if scale_tril is not None else None)
    # add dimension for #of gaussians
    return torch.exp(dist.log_prob(dx))
    

X = TypeVar("X")
def getcache(x:Tensor,t:float,cache:Optional[tuple[tuple[Tensor,float],X]])->Optional[X]:
    if cache and (x is cache[0][0]) and t == cache[0][1]: #don't bother checking tensor equality - assume it never comes up (shouldn't)
        return cache[1]
    return None

class MultimodalGaussianGroundTruthScoreModel(BatchScoreModelEvaluator[Tensor,Tensor,None,None]):
    def __init__(self,means,variances,weights,sigma_min:float,sigma_max:float) -> None:

        self.means = means
        self.variances = variances
        self.weights = weights
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.scorecache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.divcache: Optional[tuple[tuple[Tensor,float],float]] = None
        self.intcache: Optional[tuple[tuple[Tensor,float],dict[str,Tensor]]] = None

    def to(self,device=None,dtype=None):
        tensor = lambda d: torch.as_tensor(d,device=device,dtype=dtype)
        self.means = tensor(self.means)
        self.variances = tensor(self.variances)
        self.weights = tensor(self.weights)
        self.sigma_max = tensor(self.sigma_max)
        self.sigma_min = tensor(self.sigma_min)

    def _intermediates(self, x: Tensor, t: float) -> dict[str,Tensor]:
        with record_function("cache"):
            cache = getcache(x,t,self.intcache)
            if cache is not None: return cache

        tensor = lambda d: torch.as_tensor(d,device=x.device,dtype=x.dtype)

        
        with record_function("diffusion_coeff"):
            st = int_diffusion_coeff_sq(tensor(t),self.sigma_min,self.sigma_max)


        currvar = (tensor(st) + self.variances) #should be N or NxDxD for scalar or matrix covariance
      


        #add dimensions because x is batched
        dx = x[:,None,...] - self.means[None,...]
        with record_function("normpdf"):
            if self.variances.ndim == 1:
                prob = batched_normpdf_scalar(dx,currvar) #shape:BxN
            else:
                prob = batched_normpdf_matrix(dx,currvar) #shape:BxN
        wprobs = self.weights[None,...]*prob #shape:N

        with record_function("inverse"):
            if currvar.ndim == 1:
                currvar_inv = 1/currvar
                transf_dx = dx*currvar_inv[None,:,None]
            elif currvar.ndim == 3:
                #matrices will broadcast to the batch dim of x, adding 1 to the dim of x makes it a column vector
                currvar_inv = torch.linalg.inv_ex(currvar)[0]
                transf_dx = torch.matmul(currvar_inv,dx[...,None]).squeeze(-1)
            else:
                raise ValueError(currvar.shape) 

        intermediates = {"transf_dx":transf_dx,"wprobs":wprobs,"currvar_inv":currvar_inv,"wprobs_sum":torch.sum(wprobs,dim=-1),"prob":prob}
        self.intcache = ((x,t),intermediates)
        return intermediates

    def score(self, x: Tensor, t: float, conditioning:None) -> Tensor:
        assert x.shape[0] == 1
        return self.batch_score(x,t,conditioning).squeeze(0)
    
    def divergence(self, x: Tensor, t: float, conditioning:None) -> float:
        assert x.shape[0] == 1
        return self.batch_divergence(x,t,conditioning).squeeze(0)
    
    def batch_pdf(self, x: Tensor, t:float, conditioning:None) -> Tensor:
        return self._intermediates(x,t)["wprobs_sum"]

    def batch_score(self, batch: Tensor, t: float, conditioning:None) -> Tensor:
        cache = getcache(batch,t,self.scorecache)
        if cache is not None: return cache
        
        ints = self._intermediates(batch,t)
        wprobs = ints["wprobs"]; transf_dx = ints["transf_dx"]; wprobs_sum = ints["wprobs_sum"]
        
        with record_function("scoresum"):
            score = -torch.sum(wprobs[...,None]*transf_dx,dim=-2)/wprobs_sum[...,None]

        self.scorecache = ((batch,t),score)
        return score
    
    def batch_divergence(self, batch: Tensor, t: float, conditioning:None) -> Tensor:
        cache = getcache(batch,t,self.divcache)
        if cache is not None: return cache
        
        ints = self._intermediates(batch,t)
        score = self.batch_score(batch,t,conditioning) 

        wprobs = ints["wprobs"]; transf_dx = ints["transf_dx"]; 
        currvar_inv = ints["currvar_inv"]; wprobs_sum = ints["wprobs_sum"]

        with record_function("div_linalg"):
            dxnorm = torch.linalg.vecdot(transf_dx,transf_dx)
            if currvar_inv.ndim == 3:
                invtrace = torch.diagonal(currvar_inv,dim1=-1,dim2=-2).sum(-1)
            else:
                invtrace = currvar_inv
            dfcoeff = (dxnorm - invtrace)
            dfoverg = torch.linalg.vecdot(dfcoeff,wprobs)/wprobs_sum
            
            scoresq = torch.linalg.vecdot(score,score)
        
            div = dfoverg - scoresq

        self.divcache = ((batch,t),div)
        return div