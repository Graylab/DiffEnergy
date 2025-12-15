import operator
import warnings
from torch import Tensor


from typing import Callable, Generic, Optional, Protocol, TypeVar


# Wrappers upon wrappers upon wrappers... Unified class providing score (grad log p) and divergence (grad dot grad log p) functionality, 
# wrapping an internal neural network. Cache is super valuable since you need the score (with gradient on) to compute the divergence.
X = TypeVar("X")
C = TypeVar("C",contravariant=True)
class ScoreModelEvaluator(Protocol,Generic[X,C]):
    def score(self,x:X,t:float,conditioning:C) -> Tensor:  ...
    def divergence(self,x:X,t:float,conditioning:C) -> Tensor:  ...

# Super barebones one-entry cache functionality. Score and divergence should only be called at the "current" point and never again so
# larger cache shouldn't be necessary (which is good, since the score we're saving has a big gradient attached to it!)
# Implementing classes call _cached_score/_cached_divergence at the beginning of their divergence function and update them with _put_cache 
# at the end. _check_cache can be used if you want to cache other stuff.
# The main reason to have this centralized is cause of x_eq, t_eq, c_eq - crucially, **the default equality check for the data (x) and the condition (c)
# is *is*, not == !** This is mostly cause I didn't want to have to deal with torch.all() when checking tensor equality, but it also helped a surprising amount with performance
V = TypeVar("V") #cache value for general _check_cache
class CachedScoreModelEvaluator(ScoreModelEvaluator[X,C]):
    def __init__(self,
                 x_eq:Callable[[X,X],bool]=operator.is_,
                 t_eq:Callable[[float,float],bool]=operator.__eq__,
                 c_eq:Callable[[C,C],bool]=operator.is_):
        self.scorecache: Optional[tuple[tuple[X,float,C],tuple[Tensor,Tensor]]] = None #Save the input to the score as well to retain the computation graph
        self.divcache: Optional[tuple[tuple[X,float,C],Tensor]] = None

        self.x_eq = x_eq
        self.t_eq = t_eq
        self.c_eq = c_eq

    def _check_cache(self,cache:tuple[tuple[X,float,C],V]|None,key:tuple[X,float,C]):
        if cache is not None and (
            self.x_eq(key[0],cache[0][0]) and
            self.t_eq(key[1],cache[0][1]) and
            self.c_eq(key[2],cache[0][2])
        ):
            return cache[1]

    def _cached_score(self,x:X,t:float,c:C,needs_grad:bool=False) -> tuple[Tensor,Tensor] | None:
        res = self._check_cache(self.scorecache,(x,t,c))
        if res is not None and needs_grad and (res[0].requires_grad is False or res[1].grad_fn is None):  #cache miss if we need to return a tensor with a gradient history
            warnings.warn("Cache miss because the cached tensor is missing a requested gradient! This greatly reduces performance - please make the original call require grad to prevent recalculation!")
            return None
        return res
    
    def _put_score(self,x:X,t:float,c:C,input:Tensor,score:Tensor):
        self.scorecache = ((x,t,c),(input,score))

    def _cached_divergence(self,x:X,t:float,c:C) -> Tensor | None:
        return self._check_cache(self.divcache,(x,t,c))
    
    def _put_divergence(self,x:X,t:float,c:C,divergence:Tensor):
        self.divcache = ((x,t,c),divergence)
