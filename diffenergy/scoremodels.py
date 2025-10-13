import operator
from torch import Tensor


from typing import Callable, Generic, Iterable, Optional, Protocol, TypeVar

X = TypeVar("X")
C = TypeVar("C")
XB = TypeVar("XB") #X Batched (e.g. Iterable[X])
CB = TypeVar("CB",contravariant=True) #Batched Conditioning (e.g. Iterable[C])

class ScoreModelEvaluator(Protocol,Generic[X,C]):
    def score(self,x:X,t:float,conditioning:C)->Tensor:  ...
    def divergence(self,x:X,t:float,conditioning:C)->float:  ...

class BatchScoreModelEvaluator(ScoreModelEvaluator[X,C],Generic[X,XB,C,CB]):
    def batch_score(self,batch:XB,t:float,conditioning:CB)->Iterable[Tensor]:  ...
    def batch_divergence(self,batch:XB,t:float,conditioning:CB)->Iterable[float]:  ...


class CachedScoreModelEvaluator(ScoreModelEvaluator[X,C]):
    def __init__(self,x_eq:Optional[Callable[[X,X],bool]]=operator.is_,c_eq:Optional[Callable[[C,C],bool]]=operator.is_):
        self.scorecache: Optional[tuple[tuple[X,C,float],tuple[Tensor,Tensor]]] = None
        self.divcache: Optional[tuple[tuple[X,C,float],Tensor]] = None

        self.x_eq = x_eq if x_eq is not None else operator.__eq__
        self.c_eq = c_eq if c_eq is not None else operator.__eq__

    def _cached_score(self,x:X,c:C,t:float):
        if self.scorecache is not None and (
            self.x_eq(x,self.scorecache[0][0]) and
            self.c_eq(c,self.scorecache[0][1]) and
            t == self.scorecache[0][2]
        ):
            return self.scorecache[1]
        return None

    def _cached_divergence(self,x:X,c:C,t:float):
        if self.divcache is not None and (
            self.x_eq(x,self.divcache[0][0]) and
            self.c_eq(c,self.divcache[0][1]) and
            t == self.divcache[0][2]
        ):
            return self.divcache[1]
        return None
