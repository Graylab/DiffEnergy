

from typing import Literal, Optional, TypeVar, TypeVarTuple, TypedDict

from torch import Tensor
import torch
from diffenergy.dfmdock_tr.score_model import Score_Model
from diffenergy.scoremodels import CachedScoreModelEvaluator

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

class ModelEval(CachedScoreModelEvaluator[LigDict,DFMDict]): #unbatched has a size of 1 in first dim, batched has size of N
    def __init__(self,
                 score_model:Score_Model, 
                 offset_type:Literal["Translation","Rotation","Translation+Rotation"], 
                 always_grad:bool=True,
                 reset_seed_each_eval:bool=False,
                 manual_seed:int=0,
                 divide_div_by_N=False) -> None:
        
        super().__init__()

        self.score_model = score_model

        # note that the cache will always store the tensors with grad if they were calculated with grad, 
        # but returned values from score and divergence will never have grad unless return_grad is True
        # score cache stores both the input (with grad) and output, since the input gets cloned before eval
        self.always_grad = always_grad
        self.dtype = self.score_model.parameters()
        self.offset_type = offset_type

        self.reset_seed_each_eval = reset_seed_each_eval
        self.manual_seed = manual_seed

        self.divide_div_by_N = divide_div_by_N


    def score(self, x:LigDict, t:float, conditioning:DFMDict, grad:bool=False, return_grad:bool=False):
        with torch.profiler.record_function("ModelEval: Score"):
            batch = x

            cache = self._cached_score(batch,t,conditioning)
            if cache is not None: return cache if return_grad else cache.detach()
            
            self.score_model.zero_grad(set_to_none=True) #new inputs, so clear the gradient cache as well
            
            if self.reset_seed_each_eval:
                torch.manual_seed(self.manual_seed)

            offset = batch["offset"] #make sure when we set requries_grad to True, it doesn't affect the original!

            assert offset.ndim == 1 #no batch support yet

            # enable grad to cache the gradients
            if self.always_grad or grad:
                grad_ctx = torch.enable_grad
                offset = offset.clone().detach()
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

            self.scorecache = ((batch,t,conditioning),score)
            return score if return_grad else score.detach()
    

    def divergence(self,x:LigDict,t:float,conditioning:DFMDict,return_grad:bool=False)->Tensor:
        with torch.profiler.record_function("ModelEval: Divergence"):
            batch = x

            cache = self._cached_divergence(batch,conditioning,t)#
            if cache is not None: return cache if return_grad else cache.detach()
            
            score = self._cached_score(batch,t,conditioning)
            if score is None:
                self.scorecache = None #make sure to invalidate a bad cache
                score = self.score(batch,t,conditioning,grad=True,return_grad=True)

            (offset,score) = score #use input from score, not from x!

            assert score.ndim == 1 #shape = (3,) or (6,)
            assert offset.ndim == 1

            with torch.profiler.record_function("ModelEval: Divergence: Autograd"):
                grad_scores = torch.empty(offset.shape,dtype=offset.dtype,device=offset.device)
                for i in range(offset.shape[0]):
                    grad_scores[i] = torch.autograd.grad(score[i], offset, retain_graph=True, create_graph=False)[0][i]
            trace = torch.sum(grad_scores, dim=0)
            if self.divide_div_by_N:
                trace = trace/conditioning["lig_pos_orig"].shape[0] #divide by N

            self.divcache = ((batch,t,conditioning),trace)

            return trace if return_grad else trace.detach() #essentially a float teehee


    



        

