

import abc
from typing import Generic, Iterable, Optional, Protocol, Sequence, TypeVar, TypedDict

import IPython
from torch import FloatTensor, Tensor
import torch
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP


def to_array(x:Tensor)->Tensor:
    return x.squeeze(0)
def from_array(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device))[None,...]

def to_array_batch(x:Tensor)->Tensor:
    return x #don't un-batch
def from_array_batch(a,device:str|torch.device='cuda')->Tensor:
    return torch.as_tensor(a,dtype=torch.float,device=torch.device(device)) #don't re-batch

X = TypeVar("X",contravariant=True) #X 
XB = TypeVar("XB") #X Batched (e.g. Iterable[X])
C = TypeVar("C",contravariant=True) #Conditioning
CB = TypeVar("CB",contravariant=True) #Batched Conditioning (e.g. Iterable[C])


def getcache(x:Tensor,t:float,cache:Optional[tuple[tuple[Tensor,float],X]])->Optional[X]:
    if cache and (x is cache[0][0]) and t == cache[0][1]: #don't bother checking tensor equality - assume it never comes up (shouldn't)
        return cache[1]
    return None

class ScoreModelEvaluator(Protocol,Generic[X,C]):
    def score(self,x:X,t:float,conditioning:C)->Tensor:  ...
    def divergence(self,x:X,t:float,conditioning:C)->float:  ...

class BatchScoreModelEvaluator(ScoreModelEvaluator[X,C],Generic[X,XB,C,CB]):
    def batch_score(self,batch:XB,t:float,conditioning:CB)->Iterable[Tensor]:  ...
    def batch_divergence(self,batch:XB,t:float,conditioning:CB)->Iterable[float]:  ...

class ModelEval(BatchScoreModelEvaluator[Tensor,Tensor,None,None]): #unbatched has a size of 1 in first dim, batched has size of N
    def __init__(self,score_model:ScoreNetMLP|NegativeGradientMLP, always_grad:bool=True) -> None:
        self.score_model = score_model
        self.scorecache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.divcache: Optional[tuple[tuple[Tensor,float],Tensor]] = None
        self.always_grad = always_grad
        self.dtype = self.score_model.parameters()


    def batch_score(self,batch:Tensor,t:float,conditioning:None,grad:bool=False):
        cache = getcache(batch,t,self.scorecache)
        if cache is not None: return cache
        
        x = batch

        # enable grad to cache the gradients
        if self.always_grad or grad:
            grad_ctx = torch.enable_grad
            x.requires_grad_(True)
        else:
            grad_ctx = torch.no_grad
            x.requires_grad_(False)

        with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
            batchscore = self.score_model(x, torch.as_tensor([t],device=x.device,dtype=x.dtype).expand((x.shape[0],)))

        self.scorecache = ((x,t),batchscore)
        return batchscore
    

    def score(self,x:Tensor,t:float,conditioning:None,grad:bool=False):
        score = self.batch_score(x,t,conditioning,grad=grad)
        if score.shape[0] == 1:
            return score[0]
        else:
            raise ValueError("x must have size 1 in dimension 1 to use unbatched score! Noneall batch_score otherwise!")
    
    def batch_divergence(self,batch:Tensor,t:float,conditioning:None)->Tensor:
        cache = getcache(batch,t,self.divcache)
        if cache is not None: return cache

        x = batch
        
        batchscore = getcache(x,t,self.scorecache)
        if batchscore is None:
            self.scorecache = None #make sure to invalidate a bad cache
            batchscore = self.batch_score(x,t,conditioning,grad=True)

        grad_scores = torch.empty(x.shape,dtype=x.dtype,device=x.device)
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                grad_scores[b,i] = torch.autograd.grad(batchscore[b,i], x, retain_graph=True, create_graph=False)[0][b,i]
        batchtrace = torch.sum(grad_scores, dim=1)

        self.divcache = ((x,t),batchtrace)

        return batchtrace #essentially a float teehee

    def divergence(self,x:Tensor,t:float,conditioning:None)->float:
        div = self.batch_divergence(x,t,conditioning)
        if div.shape[0] == 1:
            return div[0]
        else:
            raise ValueError("x must have size 1 in dimension 1 to use unbatched divergence! Call batch_divergence otherwise!")
        

        


    



        

