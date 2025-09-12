

from dataclasses import dataclass

from torch import Tensor

from diffenergy.gaussian_1d.likelihood_helpers import ScoreModelEvaluator


@dataclass
class TrimodalGaussianConfig:
    mu1:float
    sigma1:float
    w1:float
    mu2:float
    sigma2:float
    w2:float
    mu3:float
    sigma3:float
    w3:float

class GroundTruthScoreModel(ScoreModelEvaluator[Tensor]):
    def __init__(self,config:TrimodalGaussianConfig,sigma_min:float,sigma_max:float) -> None:
        self.config = config
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def score(self, x: Tensor, t: float, grad: bool = False) -> Tensor:

        
        return super().score(x, t, grad)
    
    def divergence(self, x: Tensor, t: float) -> float:
        return super().divergence(x, t)