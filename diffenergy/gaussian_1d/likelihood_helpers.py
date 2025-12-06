
from typing import Optional, TypeVar

from torch import Tensor
import torch
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.scoremodels import CachedScoreModelEvaluator


def to_array(x:Tensor)->Tensor:
    return x.squeeze(0)
def from_array(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device))[None,...]

def to_array_batch(x:Tensor)->Tensor:
    return x #don't un-batch
def from_array_batch(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device)) #don't re-batch

X = TypeVar("X") #X 


class ModelEval(CachedScoreModelEvaluator[Tensor,None]):
    def __init__(self,score_model:ScoreNetMLP|NegativeGradientMLP,batched:bool=False) -> None:
        super().__init__()
        self.score_model = score_model
        self.batched = batched

    def batchscore(self, x: Tensor, t: float, conditioning: None, grad: bool = True, return_grad: bool = False) -> Tensor:
        if (cache := self._cached_score(x,t,conditioning, needs_grad=grad)) is not None: #make sure to ensure the is populated with tensor w/ gradient if grad=True
            return cache if return_grad else cache.detach()
        
        # enable grad to cache the gradients
        if grad:
            grad_ctx = torch.enable_grad
            x.requires_grad_(True)
        else:
            grad_ctx = torch.no_grad
            x.requires_grad_(False)

        with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
            score:Tensor = self.score_model(x, torch.as_tensor([t],device=x.device,dtype=x.dtype).expand((x.shape[0],)))
        
        self._put_score(x,t,conditioning,score)
        return score if return_grad else score.detach()

    def score(self, x: Tensor, t: float, conditioning: None, grad: bool = True, return_grad: bool = False) -> Tensor:
        score = self.batchscore(x, t, conditioning, grad=grad, return_grad=return_grad)
        if self.batched:
            return score
        
        if score.shape[0] == 1:
            return score[0]
        else:
            raise ValueError("x must have size 1 in dimension 1 to use unbatched score! To use batched data, please initialize the model with batched=True.")

    def batchdivergence(self, x: Tensor, t: float, conditioning: None, return_grad: bool = False) -> Tensor:
        if (cache := self._cached_divergence(x,t,conditioning)) is not None: return cache
        score = self.score(x,t,conditioning,grad=True,return_grad=True)

        grad_scores = torch.empty(x.shape,dtype=x.dtype,device=x.device)
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                grad_scores[b,i] = torch.autograd.grad(score[b,i], x, retain_graph=True, create_graph=False)[0][b,i]
        trace = torch.sum(grad_scores, dim=1)

        self._put_divergence(x,t,conditioning,trace)
        return trace if return_grad else trace.detach()
    
    def divergence(self,x:Tensor,t:float,conditioning:None,return_grad:bool=False) -> Tensor:
        div = self.batchdivergence(x, t, conditioning, return_grad=return_grad)
        if self.batched:
            return div

        if div.shape[0] == 1:
            return div[0]
        else:
            raise ValueError("x must have size 1 in dimension 1 to use unbatched divergence! To use batched data, please initialize the model with batched=True.")


        
#TODO: TEST MODEL EVAL

# class ModelEval(BatchScoreModelEvaluator[Tensor,Tensor,None,None]): #unbatched has a size of 1 in first dim, batched has size of N
#     def __init__(self,score_model:ScoreNetMLP|NegativeGradientMLP, always_grad:bool=True) -> None:
#         self.score_model = score_model
#         self.scorecache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
#         self.divcache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
#         self.always_grad = always_grad
#         self.dtype = self.score_model.parameters()


#     def batch_score(self,batch:Tensor,t:float,conditioning:None,grad:bool=False,return_grad:bool=False):
#         cache = getcache(batch,t,self.scorecache)
#         if cache is not None: return cache
        
#         x = batch

#         # enable grad to cache the gradients
#         if self.always_grad or grad:
#             grad_ctx = torch.enable_grad
#             x.requires_grad_(True)
#         else:
#             grad_ctx = torch.no_grad
#             x.requires_grad_(False)

#         with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
#             batchscore = self.score_model(x, torch.as_tensor([t],device=x.device,dtype=x.dtype).expand((x.shape[0],)))

#         self.scorecache = ((x,t),batchscore)
#         return batchscore if return_grad else batchscore.detach()
    

#     def score(self,x:Tensor,t:float,conditioning:None,grad:bool=False,return_grad:bool=False):
#         score = self.batch_score(x,t,conditioning,grad=grad,return_grad=return_grad)
#         if score.shape[0] == 1:
#             return score[0]
#         else:
#             raise ValueError("x must have size 1 in dimension 1 to use unbatched score! Noneall batch_score otherwise!")
    
#     def batch_divergence(self,batch:Tensor,t:float,conditioning:None,return_grad:bool=False)->Tensor:
#         cache = getcache(batch,t,self.divcache)
#         if cache is not None: return cache

#         x = batch
        
#         batchscore = getcache(x,t,self.scorecache)
#         if batchscore is None:
#             self.scorecache = None #make sure to invalidate a bad cache
#             batchscore = self.batch_score(x,t,conditioning,grad=True,return_grad=True)

#         grad_scores = torch.empty(x.shape,dtype=x.dtype,device=x.device)
#         for b in range(x.shape[0]):
#             for i in range(x.shape[1]):
#                 grad_scores[b,i] = torch.autograd.grad(batchscore[b,i], x, retain_graph=True, create_graph=False)[0][b,i]
#         batchtrace = torch.sum(grad_scores, dim=1)

#         self.divcache = ((x,t),batchtrace)

#         return batchtrace if return_grad else batchtrace.detach()#essentially a float teehee

#     def divergence(self,x:Tensor,t:float,conditioning:None,return_grad:bool=False)->float:
#         div = self.batch_divergence(x,t,conditioning,return_grad=return_grad)
#         if div.shape[0] == 1:
#             return div[0]
#         else:
#             raise ValueError("x must have size 1 in dimension 1 to use unbatched divergence! Call batch_divergence otherwise!")
        

        


    



        

