

from typing import Optional, TypedDict

from torch import Tensor
import torch
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP


def to_array(x:Tensor)->Tensor:
    return x.squeeze(0)
def from_array(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device))[None,...]

class ModelEval:
    def __init__(self,score_model:ScoreNetMLP|NegativeGradientMLP, always_grad:bool=True) -> None:
        self.score_model = score_model
        self.scorecache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.divcache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.always_grad = always_grad
        self.dtype = self.score_model.parameters()

    def score(self,x:Tensor,t:float,grad:bool=False):
        if self.scorecache and (x,t) == self.scorecache[0]:
            return self.scorecache[1]
        
        
        # enable grad to cache the gradients
        if self.always_grad or grad:
            grad_ctx = torch.enable_grad
            x.requires_grad_(True)
        else:
            grad_ctx = torch.no_grad
            x.requires_grad_(False)
        
        with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
            score = self.score_model(x, torch.as_tensor([t],device=x.device,dtype=x.dtype)).squeeze(0) #x is batched, but we assume with a size of 1

        self.scorecache = ((x,t),score)
        return score
    
    def divergence(self,x:Tensor,t:float)->float:
        if self.divcache and (x,t) == self.divcache[0]:
            return self.divcache[1]
        
        if self.scorecache and (x,t) == self.scorecache[0] and self.scorecache[1].grad_fn is not None: #make sure score was calculated with gradient
            score = self.scorecache[1]
        else:
            self.scorecache = None #make sure to invalidate a bad cache
            score = self.score(x,t,grad=True)

        grad_score = torch.autograd.grad(score, x)[0]
        trace = torch.sum(grad_score, dim=1).squeeze(0) #since x is batched

        self.divcache = ((x,t),trace)

        return trace
        

        


    



        

