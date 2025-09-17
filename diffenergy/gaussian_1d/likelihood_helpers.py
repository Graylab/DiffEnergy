

import abc
from typing import Generic, Optional, Protocol, TypeVar, TypedDict

import IPython
from torch import Tensor
import torch
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP


def to_array(x:Tensor)->Tensor:
    return x.squeeze(0)
def from_array(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device))[None,...]

X = TypeVar("X",contravariant=True)


def getcache(x:Tensor,t:float,cache:Optional[tuple[tuple[Tensor,float],X]])->Optional[X]:
    if cache and (x is cache[0][0]) and t == cache[0][1]: #don't bother checking tensor equality - assume it never comes up (shouldn't)
        return cache[1]
    return None

class ScoreModelEvaluator(Protocol,Generic[X]):
    def score(self,x:X,t:float)->Tensor:  ...
    def divergence(self,x:X,t:float)->float:  ...


class ModelEval(ScoreModelEvaluator[Tensor]):
    def __init__(self,score_model:ScoreNetMLP|NegativeGradientMLP, always_grad:bool=True) -> None:
        self.score_model = score_model
        self.scorecache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.divcache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.always_grad = always_grad
        self.dtype = self.score_model.parameters()

    def score(self,x:Tensor,t:float,grad:bool=False):
        cache = getcache(x,t,self.scorecache)
        if cache is not None: return cache
        
        
        # enable grad to cache the gradients
        if self.always_grad or grad:
            grad_ctx = torch.enable_grad
            x.requires_grad_(True)
        else:
            grad_ctx = torch.no_grad
            x.requires_grad_(False)

        with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
            score = self.score_model(x, torch.as_tensor([t],device=x.device,dtype=x.dtype).expand((x.shape[0],))).squeeze(0) #x is batched, but we assume with a size of 1

        self.scorecache = ((x,t),score)
        return score
    
    def divergence(self,x:Tensor,t:float)->float:
        cache = getcache(x,t,self.divcache)
        if cache is not None: return cache
        
        score = getcache(x,t,self.scorecache)
        if not score:
            self.scorecache = None #make sure to invalidate a bad cache
            score = self.score(x,t,grad=True)

        grad_scores = torch.empty(x.shape,dtype=x.dtype,device=x.device)
        assert x.shape[0] == 1
        for i in range(x.shape[1]):
            grad_scores[0,i] = torch.autograd.grad(score[i], x, retain_graph=True, create_graph=False)[0][i]
        trace = torch.sum(grad_scores, dim=1).squeeze(0) #since x is batched

        self.divcache = ((x,t),trace)

        return trace #essentially a float teehee
        

        


    



        

