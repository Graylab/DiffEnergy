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
    t = torch.as_tensor(t)
    sigma_min = torch.as_tensor(sigma_min)
    sigma_max = torch.as_tensor(sigma_max)
    std = sigma_min * (sigma_max / sigma_min) ** t
    diff_coeff = std * torch.sqrt(torch.as_tensor(2 * (torch.log(sigma_max/sigma_min)), device=t.device))
    if clamp:
        diff_coeff = torch.clamp(diff_coeff, min=1e-5)
    return diff_coeff


def int_diffusion_coeff_sq(t, sigma_min, sigma_max):
    """
    Compute the integral of the squared diffusion coefficient of our SDE from t=0 to t.
    Required to determine the total level of gaussian noise added to our data distribution at time t for ground truth score calculation.
    RETURNS THE *VARIANCE*, use sqrt to get the standard deviation!
    
    Args:
        t: A vector of time steps
        sigma_min: the initial sigma of our SDE (at t=0)
        sigma_max: the noised sigma of our SDE (at t=1)
        
    Returns:
        the vector of integrated squared diffusion coefficients"""
    
    t = torch.as_tensor(t)
    var_min = torch.as_tensor(sigma_min)**2
    var_max = torch.as_tensor(sigma_max)**2
    int_diff_coeff = var_min * ((var_max / var_min) ** t - 1)
    return int_diff_coeff

def prior_gaussian_nd(x, sigma):
    """The likelihood of a D-dimensional Gaussian distribution with mean zero and 
            uniform standard deviation sigma. Assumes x is a D-dimensional or (*B)xD-dimensional array"""
    D = x.shape[-1]
    prior_logp = -D / 2. * torch.log(torch.as_tensor(2 * np.pi * sigma ** 2, device = x.device)) - (torch.linalg.vecdot(x,x)) / (2 * sigma**2)
    return prior_logp, D