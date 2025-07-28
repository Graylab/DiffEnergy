import torch
import numpy as np

#--------------------------------------------------------------------------
# Helper functions: marginal probability, diffusion coefficients, prior likelihood

def marginal_prob_std(t, sigma_min, sigma_max):
    r"""
    Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
        
    Returns:
        The standard deviation.
    """    
    return sigma_min * (sigma_max / sigma_min) ** t

def diffusion_coeff(t, sigma_min, sigma_max, clamp = False):
    r"""
    Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        
    Returns:
        The vector of diffusion coefficients.
    """
    t = t.clone().detach()
    std = sigma_min * (sigma_max / sigma_min) ** t
    diff_coeff = std * torch.sqrt(torch.tensor(2 * (np.log(sigma_max/sigma_min)), device=t.device))
    if clamp:
        diff_coeff = torch.clamp(diff_coeff, min=1e-5)
    return diff_coeff

def prior_dfmdock_tr(batch, sigma):
    """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
    # shape = z.shape
    # N = np.prod(shape)
    # return -N / 2. * torch.log(torch.tensor(2 * np.pi * sigma ** 2, device = z.device)) - torch.sum(z**2, dim=(0,1,2)) / (2 * sigma**2)
    lig_pos = batch['sample']
    rec_pos = batch['rec_pos']
    com_diff = lig_pos[...,1,:].mean(dim=0) - rec_pos[...,1,:].mean(dim=0)
    N = com_diff.numel()
    prior_logp = -N / 2. * torch.log(torch.tensor(2 * np.pi * sigma ** 2, device = lig_pos.device)) - torch.sum(com_diff**2) / (2 * sigma**2)
    return prior_logp, N
    

def prior_gaussian_1d(batch, sigma):
    """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
    # shape = z.shape
    # N = np.prod(shape)
    z = batch['sample']
    N = z.numel()
    prior_logp = -N / 2. * torch.log(torch.tensor(2 * np.pi * sigma ** 2, device = z.device)) - (z**2) / (2 * sigma**2)
    return prior_logp, N
