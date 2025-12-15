from typing import Iterable
from matplotlib import pyplot as plt

import numpy as np
from scipy.stats import norm
from diffenergy.helper import int_diffusion_coeff_sq

def get_gaussian(x,means:Iterable[float],sigmas:Iterable[float],weights:Iterable[float]):
    return sum([weight*norm.pdf(x,loc=mean,scale=sigma) for (mean,sigma,weight) in zip(means,sigmas,weights)])

def get_gt_gaussian(x,means=[-30.0,0.0,40.0],sigmas=[8.0,5.0,10.0],weights=[0.4,0.3,0.3]):
    return get_gaussian(x,means,sigmas,weights)

def get_prior_gaussian(x,sigma_max=70):
    return norm.pdf(x, loc=0, scale=sigma_max)

def get_timed_gt_gaussian(x,t,means=[-30.0,0.0,40.0],t0_sigmas=[8.0,5.0,10.0],weights=[0.4,0.3,0.3],sigma_min=0.1,sigma_max=70):
    t_sigmas = np.sqrt((np.array(t0_sigmas,dtype=float)**2)[:,None] + int_diffusion_coeff_sq(t,sigma_min=sigma_min,sigma_max=sigma_max).numpy()[None,:])
    return get_gaussian(x,means,t_sigmas,weights)

def setfont():
    from matplotlib import font_manager
    font_dirs = ['./figures/arial']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # set font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["mathtext.fontset"] = 'cm'
    
def marginal_prob_std(t, sigma_min, sigma_max):
    """Compute the standard deviation at time t."""
    return sigma_min * (sigma_max / sigma_min) ** t