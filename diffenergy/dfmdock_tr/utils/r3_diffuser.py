"""R^3 diffusion methods."""
import numpy as np
import torch

move_to_np = lambda x: x.cpu().detach().numpy()

class R3Diffuser:
    """VE-SDE diffuser class for translations."""

    def __init__(self, conf):
        self.min_sigma = conf.min_sigma
        self.max_sigma = conf.max_sigma

    def sigma(self, t):
        #return self.min_sigma * (self.max_sigma / self.min_sigma) ** t
        return self.min_sigma ** (1 - t) * self.max_sigma ** t

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return self.sigma(t) * np.sqrt(2 * (np.log(self.max_sigma) - np.log(self.min_sigma))) 

    def torch_score(self, tr_t, t):
        return -tr_t / self.sigma(t)**2

    def score_scaling(self, t: float):
        return 1 / self.sigma(t)

    def forward_marginal(self, t: float):
        """Samples marginal p(x(t) | x(0)).
        Args:
            t: continuous time in [0, 1].
            x0: ligand center [..., 3]
        Returns:
            tr_t: [..., 3] positions at time t in Angstroms.
            tr_score: [..., 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')

        z = np.random.randn(1, 3)
        tr_t = self.sigma(t) * z 
        tr_score = self.torch_score(tr_t, t)
        return tr_t, tr_score

    def torch_reverse(
            self,
            score_t: torch.tensor,
            dt: torch.tensor,
            t: float,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step
        Args:
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.
        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')

        g_t = self.diffusion_coef(t)
        z = noise_scale * torch.randn(1, 3, device=score_t.device)
        perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(dt) * z

        return perturb.float()

