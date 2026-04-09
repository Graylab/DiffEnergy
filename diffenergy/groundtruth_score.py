from typing import Callable, Optional

from torch import Tensor
from torch.profiler import record_function
import torch

from diffenergy.gaussian_helper import batched_pdf
from diffenergy.scoremodels import CachedScoreModelEvaluator

class MultimodalGaussianGroundTruthScoreModel(CachedScoreModelEvaluator[Tensor,None]):
    def __init__(self,weights,means,variances,int_diff_coeff_sq:Callable[[Tensor],float],batched:bool=False) -> None:
        super().__init__()
        self.weights = weights
        self.means = means
        self.variances = variances
        self.int_diff_coeff_sq = int_diff_coeff_sq
        self.batched = batched

        self.intcache: Optional[tuple[tuple[Tensor,float,None],dict[str,Tensor]]] = None

    def to(self,device=None,dtype=None):
        tensor = lambda d: torch.as_tensor(d,device=device,dtype=dtype)
        self.means = tensor(self.means)
        self.variances = tensor(self.variances)
        self.weights = tensor(self.weights)

    def _intermediates(self, x: Tensor, t: float) -> dict[str,Tensor]:
        if (cache := self._check_cache(self.intcache,(x,t,None))) is not None: return cache

        tensor = lambda d: torch.as_tensor(d,device=x.device,dtype=x.dtype)

        with record_function("diffusion_coeff"):
            gsq = tensor(self.int_diff_coeff_sq(tensor(t))) #Scalar

        if self.variances.ndim == 3: #matrix, need to embed the diffusion coefficient into a diagonal matrix
            gsq = gsq[...,None].expand(*gsq.shape,self.variances.shape[-1]).diag_embed()
        currvar = gsq + self.variances #should be N or NxDxD for scalar or matrix covariance
      
        #precompute dx and use mean=0 so we can reuse it later [tiny optimization]
        #add dimensions because x is batched
        dx = x[:,None,...] - self.means[None,...]
        with record_function("normpdf"):
            prob = batched_pdf(dx,torch.zeros_like(self.means),currvar) #shape:BxN
        wprobs = self.weights[None,...]*prob #shape:BxN

        with record_function("inverse"):
            if currvar.ndim == 1:
                currvar_inv = 1/currvar #shape: N
                transf_dx = dx*currvar_inv[None,:,None] #shape: BxNxD
            elif currvar.ndim == 3:
                #matrices will broadcast to the batch dim of x, adding 1 to the dim of x makes it a column vector
                currvar_inv = torch.linalg.inv_ex(currvar)[0] #shape: NxDxD
                transf_dx = torch.matmul(currvar_inv,dx[...,None]).squeeze(-1) #shape: BxNxD
            else:
                raise ValueError(currvar.shape) 

        intermediates = {
            "currvar_inv":currvar_inv, #(Sigma_i+int_diff_coeff(t)^2)^-1, for each Sigma_i of the N gaussians. shape: N or Nx3x3
            "transf_dx":transf_dx, # currvar_inv[i] @ (x-means[i]). E.g. the z-score of each x in each gaussian. Shape: BxNxD
            "prob":prob, #probability at all B points for all N gaussians at time t. shape: BxN
            "wprobs":wprobs, #w_i*prob_i, batched; shape: BxN
            "wprobs_sum":torch.sum(wprobs,dim=-1), #sum(w_i*prob_i) [pdf of multimodal at time t], batched; shape: B
        }
        self.intcache = ((x,t,None),intermediates)
        return intermediates

    def score(self, x: Tensor, t: float, conditioning:None) -> Tensor:
        if not self.batched:
            assert x.shape[0] == 1
            return self.batch_score(x,t,conditioning).squeeze(0)
        else:
            return self.batch_score(x,t,conditioning)
    
    def divergence(self, x: Tensor, t: float, conditioning:None) -> Tensor:
        if not self.batched:
            assert x.shape[0] == 1
            return self.batch_divergence(x,t,conditioning).squeeze(0)
        else:
            return self.batch_divergence(x,t,conditioning).squeeze(0)
    
    def batch_pdf(self, batch: Tensor, t:float, conditioning:None) -> Tensor: 
        return self._intermediates(batch,t)["wprobs_sum"]
    
    def batch_grad(self, batch: Tensor, t:float, conditioning:None) -> Tensor: #grad_x \cdot p(x,t)
        ints = self._intermediates(batch,t)
        wprobs = ints["wprobs"]; transf_dx = ints["transf_dx"];    
        return torch.sum(wprobs[...,None]*transf_dx,dim=-2)
    
    def batch_pdf_divergence(self, batch: Tensor, t:float, conditioning:None) -> Tensor: #laplacian(p(x,t)) = grad_x \cdot (grad_x \cdot p(x,t))
        #non-logarithmic divergence
        ints = self._intermediates(batch,t)
        wprobs = ints["wprobs"] # shape: BxN
        transf_dx = ints["transf_dx"]  #shape: BxNxD
        currvar_inv = ints["currvar_inv"] #shape: N or NxDxD

        with record_function("div_linalg"):
            dxnorm = torch.linalg.vecdot(transf_dx,transf_dx) #shape: BxN
            if currvar_inv.ndim == 3:
                invtrace = torch.diagonal(currvar_inv,dim1=-1,dim2=-2).sum(-1) #shape: N
            else:
                ## NEED TO MULTIPLY SCALAR VARIANCE BY NUMBER OF DIMENSIONS TO GET PROPER TRACE
                invtrace = currvar_inv*transf_dx.shape[-1] #shape: N
            dfcoeff = (dxnorm - invtrace) # shape: BxN
            df = torch.linalg.vecdot(dfcoeff,wprobs) #shape: B
        return df



    def batch_score(self, batch: Tensor, t: float, conditioning:None) -> Tensor: # grad_x \cdot log p(x,t)
        if (cache := self._cached_score(batch,t,conditioning)) is not None: return cache[1]
        
        ints = self._intermediates(batch,t)
        wprobs = ints["wprobs"]; transf_dx = ints["transf_dx"]; wprobs_sum = ints["wprobs_sum"]

        with record_function("scoresum"):
            s = torch.sum(wprobs[...,None]*transf_dx,dim=-2)
            score = -s/wprobs_sum[...,None]

        self._put_score(batch,t,None,batch,score)
        return score
    
    def batch_divergence(self, batch: Tensor, t: float, conditioning:None) -> Tensor: # laplacian(log p(x,t)) = grax_x \cdot (grad_x \cdot log p(x,t))
        #technically this is the divergence of the log pdf but w/e
        if (cache := self._cached_divergence(batch,t,conditioning)) is not None: return cache
        
        ints = self._intermediates(batch,t)
        score = self.batch_score(batch,t,conditioning) 

        wprobs = ints["wprobs"]; transf_dx = ints["transf_dx"] 
        currvar_inv = ints["currvar_inv"]; wprobs_sum = ints["wprobs_sum"]

        with record_function("div_linalg"):
            dxnorm = torch.linalg.vecdot(transf_dx,transf_dx)
            if currvar_inv.ndim == 3:
                invtrace = torch.diagonal(currvar_inv,dim1=-1,dim2=-2).sum(-1)
            else:
                ## NEED TO MULTIPLY SCALAR VARIANCE BY NUMBER OF DIMENSIONS TO GET PROPER TRACE
                invtrace = currvar_inv*transf_dx.shape[-1] #shape: N
            dfcoeff = (dxnorm - invtrace)
            dfoverg = torch.linalg.vecdot(dfcoeff,wprobs)/wprobs_sum
            
            scoresq = torch.linalg.vecdot(score,score)
        
            div = dfoverg - scoresq

        self._put_divergence(batch,t,conditioning,div)
        return div