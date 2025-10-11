

import abc
from typing import Generic, Iterable, Literal, Optional, Protocol, Sequence, TypeVar, TypeVarTuple, TypedDict

import IPython
from torch import FloatTensor, Tensor
import torch
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.likelihoodv3 import ScoreModelEvaluator

class DFMDict(TypedDict):
    orig_pdb: str
    rec_x: Tensor
    lig_x: Tensor
    rec_pos: Tensor
    lig_pos_orig: Tensor #the ORIGINAL POSITION of the ligand. Should be centered at zero!
    position_matrix: Tensor

class LigDict(TypedDict):
    offset: Tensor #either a 3d (Translation or Rotation) or 6d (Translation + Rotation) vector representing how to transform lig_pos in space
    
# not gonna deal with shape nonsense, just gonna keep everything in its default (batched?) shape.
# all the likelihood stuff can use arrays of any shape so it's fine
def to_array(x:LigDict)->Tensor:
    return x['offset']
def from_array(a,device:str|torch.device='cuda')->LigDict:
    return {'offset':torch.as_tensor(a,dtype=torch.float,device=torch.device(device))}


X = TypeVarTuple("X")
R = TypeVar("R")
def getcache(x:tuple[*X],t:float,cache:Optional[tuple[tuple[*X,float],R]])->Optional[R]:
    if cache and all(x[i] is cache[0][i] for i in range(len(x))) and t == cache[0][-1]: #don't bother checking tensor equality - assume it never comes up (shouldn't)
        return cache[1]
    return None


class ModelEval(ScoreModelEvaluator[LigDict,DFMDict]): #unbatched has a size of 1 in first dim, batched has size of N
    def __init__(self,
                 score_model:Score_Model, 
                 offset_type:Literal["Translation","Rotation","Translation+Rotation"], 
                 always_grad:bool=True,
                 reset_seed_each_eval:bool=False,
                 manual_seed:int=0) -> None:
        self.score_model = score_model
        self.scorecache: Optional[tuple[tuple[LigDict,DFMDict,float],Tensor]] = None
        self.divcache: Optional[tuple[tuple[LigDict,DFMDict,float],Tensor]] = None
        self.always_grad = always_grad
        self.dtype = self.score_model.parameters()
        self.offset_type = offset_type

        self.reset_seed_each_eval = reset_seed_each_eval
        self.manual_seed = manual_seed


    def score(self,x:LigDict,t:float,conditioning:DFMDict,grad:bool=False):
        with torch.profiler.record_function("ModelEval: Score"):
            batch = x

            cache = getcache((batch,conditioning),t,self.scorecache)
            if cache is not None: return cache
            
            if self.reset_seed_each_eval:
                torch.manual_seed(self.manual_seed)

            offset = batch["offset"]

            assert offset.ndim == 1 #no batch support yet

            # enable grad to cache the gradients
            if self.always_grad or grad:
                grad_ctx = torch.enable_grad
                offset.requires_grad_(True)
            else:
                grad_ctx = torch.no_grad
                offset.requires_grad_(False)


            offset_tr = None; offset_rot = None

            if self.offset_type == "Translation+Rotation":
                offset_tr,offset_rot = offset[...,:3],offset[...,3:]
            elif self.offset_type == "Translation":
                offset_tr = offset
            elif self.offset_type == "Rotation":
                offset_rot = offset
            else:
                raise ValueError(self.offset_type)


            new_batch = dict(**conditioning)
            new_batch["offset_tr"] = offset_tr
            new_batch["offset_rot"] = offset_rot

            new_batch["t"] = torch.as_tensor(t,device=offset.device,dtype=offset.dtype)
            
            with torch.profiler.record_function("ModelEval: Score: score_model"):
                with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
                    scores = self.score_model(new_batch)

            tr_score:Tensor = scores["tr_score"][0]; rot_score:Tensor = scores["rot_score"][0]

            score = None
            if self.offset_type == "Translation+Rotation":
                score = torch.cat((tr_score,rot_score))
            elif self.offset_type == "Translation":
                score = tr_score
            elif self.offset_type == "Rotation":
                score = rot_score
            else:
                raise ValueError(self.offset_type)

            self.scorecache = ((batch,conditioning,t),score)
            return score
    

    def divergence(self,x:LigDict,t:float,conditioning:DFMDict)->Tensor:
        with torch.profiler.record_function("ModelEval: Divergence"):
            batch = x

            cache = getcache((batch,conditioning),t,self.divcache)
            if cache is not None: return cache
            
            score = getcache((batch,conditioning),t,self.scorecache)
            if score is None:
                self.scorecache = None #make sure to invalidate a bad cache
                score = self.score(batch,t,conditioning,grad=True)

            offset = batch['offset']

            assert score.ndim == 1 #shape = (3,) or (6,)
            assert offset.ndim == 1

            with torch.profiler.record_function("ModelEval: Divergence: Autograd"):
                grad_scores = torch.empty(offset.shape,dtype=offset.dtype,device=offset.device)
                for i in range(offset.shape[0]):
                    grad_scores[i] = torch.autograd.grad(score[i], offset, retain_graph=True, create_graph=False)[0][i]
            trace = torch.sum(grad_scores, dim=0)

            self.divcache = ((batch,conditioning,t),trace)

            return trace #essentially a float teehee


    



        

