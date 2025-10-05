

import abc
from typing import Generic, Iterable, Optional, Protocol, Sequence, TypeVar, TypedDict

import IPython
from torch import FloatTensor, Tensor
import torch
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.likelihoodv3 import ScoreModelEvaluator

class DFMDict(TypedDict):
    rec_x: Tensor
    lig_x: Tensor
    rec_pos: Tensor
    lig_pos: Tensor
    position_matrix: Tensor
    t: Tensor

class LigDict(TypedDict):
    lig_pos: Tensor
    
# not gonna deal with shape nonsense, just gonna keep everything in its default (batched?) shape.
# all the likelihood stuff can use arrays of any shape so it's fine
def to_array(x:LigDict)->Tensor:
    return x['lig_pos']
def from_array(a,device:str|torch.device='cuda')->LigDict:
    return {'lig_pos':torch.as_tensor(a,dtype=torch.float,device=torch.device(device))}


X = TypeVar("X")
def getcache(x:Tensor,t:float,cache:Optional[tuple[tuple[Tensor,float],X]])->Optional[X]:
    if cache and (x is cache[0][0]) and t == cache[0][1]: #don't bother checking tensor equality - assume it never comes up (shouldn't)
        return cache[1]
    return None


class ModelEval(ScoreModelEvaluator[LigDict,DFMDict]): #unbatched has a size of 1 in first dim, batched has size of N
    def __init__(self,score_model:Score_Model, always_grad:bool=True) -> None:
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
        

        
def divergence_eval_tr(batch, score_model:Score_Model):
    # Compute the divergence of the score-based model

    # grab some input
    lig_pos = batch["lig_pos"]

    with torch.enable_grad():
        lig_pos.requires_grad_(True)
        tr_score = score_model(batch)["tr_score"].squeeze()

        # Compute the gradient of tr_score w.r.t lig_pos
        grad_score_lig = []
        for i in range(tr_score.shape[0]):
            grad = torch.autograd.grad(tr_score[i], lig_pos, retain_graph=True, create_graph=False)[0]
            grad_score_lig.append(grad)
        grad_score_lig = torch.stack(grad_score_lig, dim=0)  # shape[3,107,3,3]
        grad_score_lig_ca = grad_score_lig[:,:,1,:].squeeze() # shape[3,107,3]
        grad_score_lig_ca_mean = grad_score_lig_ca.mean(dim=1).squeeze() # shape[3,3]
        diagonal = torch.diagonal(grad_score_lig_ca_mean)
        trace = torch.sum(diagonal).unsqueeze(0)

    return trace

    



        

