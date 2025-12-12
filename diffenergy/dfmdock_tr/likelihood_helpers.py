

from typing import Literal, TypedDict

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

def split_offset(offset:torch.Tensor, offset_type:Literal["Translation","Rotation","Translation+Rotation"],device:str|torch.device|None=None,detach:bool=False):
    tr_update = offset[:3].to(device=device) if 'Translation' in offset_type else None
    rot_update = offset[-3:].to(device=device) if 'Rotation' in offset_type else None
    if detach and tr_update is not None: tr_update = tr_update.detach()
    if detach and rot_update is not None: rot_update = rot_update.detach()
    
    return (tr_update,rot_update)

def join_offset(tr:Tensor,rot:Tensor,offset_type:Literal["Translation","Rotation","Translation+Rotation"]):
    if offset_type == "Translation+Rotation":
        return torch.cat((tr,rot))
    elif offset_type == "Translation":
        return tr
    elif offset_type == "Rotation":
        return rot
    else:
        raise ValueError(offset_type)

class DFMDockModelEval(CachedScoreModelEvaluator[LigDict,DFMDict]): #unbatched has a size of 1 in first dim, batched has size of N
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
        self.offset_type:Literal["Translation","Rotation","Translation+Rotation"] = offset_type

        self.reset_seed_each_eval = reset_seed_each_eval
        self.manual_seed = manual_seed

        self.divide_div_by_N = divide_div_by_N


    def score(self, x:LigDict, t:float, conditioning:DFMDict, grad:bool=False, return_grad:bool=False):
        with torch.profiler.record_function("ModelEval: Score"):
            batch = x

            if (cache := self._cached_score(batch,t,conditioning, needs_grad=grad)) is not None: #make sure to ensure the is populated with tensor w/ gradient if grad=True
                return cache[1] if return_grad else cache[1].detach()
            
            self.score_model.zero_grad(set_to_none=True) #new inputs, so clear the gradient cache as well
            
            if self.reset_seed_each_eval:
                torch.manual_seed(self.manual_seed)

            offset = batch["offset"] 

            assert offset.ndim == 1 #no batch support yet

            # enable grad to cache the gradients
            if self.always_grad or grad:
                grad_ctx = torch.enable_grad
                offset = offset.clone().detach() #prevent memory leak from leftover gradients in the original x
                offset.requires_grad_(True)
            else:
                grad_ctx = torch.no_grad
                offset.requires_grad_(False)


            offset_tr,offset_rot = split_offset(offset,self.offset_type)

            # assemble batch for model eval
            eval_batch = dict(**conditioning)
            eval_batch["offset_tr"] = offset_tr
            eval_batch["offset_rot"] = offset_rot
            eval_batch["t"] = torch.as_tensor(t,device=offset.device,dtype=offset.dtype)
            
            with torch.profiler.record_function("ModelEval: Score: score_model"):
                with grad_ctx(): #not sure if this actually is that important, might be enough to just set requires grad to true/false. don't think it can hurt, though
                    scores = self.score_model(eval_batch)

            tr_score:Tensor = scores["tr_score"][0]; rot_score:Tensor = scores["rot_score"][0]
            score = join_offset(tr_score,rot_score,self.offset_type)

            self._put_score(batch,t,conditioning,offset,score)
            return score if return_grad else score.detach()
    

    def divergence(self,x:LigDict,t:float,conditioning:DFMDict,return_grad:bool=False)->Tensor:
        with torch.profiler.record_function("ModelEval: Divergence"):
            if (cache := self._cached_divergence(x,t,conditioning)) is not None: return cache if return_grad else cache.detach()
            self.score(x,t,conditioning,grad=True)
            assert self.scorecache is not None
            offset,score = self.scorecache[1] #make sure we use the corresponding input tensor with the grad enabled

            assert score.ndim == 1 #shape = (3,) or (6,)
            assert offset.ndim == 1

            with torch.profiler.record_function("ModelEval: Divergence: Autograd"):
                grad_scores = torch.empty(offset.shape,dtype=offset.dtype,device=offset.device)
                for i in range(offset.shape[0]):
                    grad_scores[i] = torch.autograd.grad(score[i], offset, retain_graph=True, create_graph=False)[0][i]
            trace = torch.sum(grad_scores, dim=0)
            if self.divide_div_by_N:
                trace = trace/conditioning["lig_pos_orig"].shape[0] #divide by N

            self._put_divergence(x,t,conditioning,trace)

            return trace if return_grad else trace.detach() #essentially a float


    



        

