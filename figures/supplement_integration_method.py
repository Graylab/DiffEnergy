from pathlib import Path
import string
from typing import Iterable, Optional
import warnings
import pandas as pd

from matplotlib import pyplot as plt, rcParams
from matplotlib.figure import Figure, SubFigure
import numpy as np

from figures.figure3 import load_sample_likelihoods, plot_comparison

try:
    from shared import setfont
    from figure3 import plot_sample_result, plot_correlation
except ImportError:
    from figures.shared import setfont
    from figures.figure3 import plot_sample_result, plot_correlation


def compare_likelihoods(fig:Figure,label:str,title:str,likelihood_1:str|Path|pd.Series,label_1:str,likelihood_2:str|Path|pd.Series,label_2:str,integrand:str|Iterable[str]="integrand:TotalIntegrand",prior:str|Iterable[str]="prior:smax_gaussian",
            lim:Optional[tuple[float,float]]=(0,0.03),ticks:Optional[list[float]]=[0,0.01,0.02,0.03],exp=True,
            other_corner=False):
    ax = fig.add_subplot()

    integrands = [integrand,integrand] if isinstance(integrand,str) else list(integrand)
    priors = [prior,prior] if isinstance(prior,str) else list(prior)

    likelihoods1 = load_sample_likelihoods(likelihood_1,integrand_column=integrands[0],prior_column=priors[0])[0] if not isinstance(likelihood_1,pd.Series) else likelihood_1
    likelihoods2 = load_sample_likelihoods(likelihood_2,integrand_column=integrands[1],prior_column=priors[1])[0] if not isinstance(likelihood_2,pd.Series) else likelihood_2
    index = likelihoods1.index.intersection(likelihoods2.index)
    if len(likelihoods1) != len(likelihoods2):
        warnings.warn(f"Missing some samples! L1={len(likelihoods1)}, L2={len(likelihoods2)}")
    # from IPython import embed; embed() 
    prob_1 = likelihoods1[index]
    prob_2 = likelihoods2[index]

    if exp:
        prob_1 = np.exp(prob_1)
        prob_2 = np.exp(prob_2)

    # filter probabilities >> 1
    # filter_idx = (prob_1 < 10) & (prob_2 < 10)
    # prob_1 = prob_1[filter_idx]
    # prob_2 = prob_2[filter_idx]

    try:
        r,p = plot_comparison(prob_1,prob_2,ax=ax, lim=lim, ticks=ticks, title=title, xlabel=label_1, ylabel=label_2)

        if other_corner:
            ax.text(.05,.9, f"$r_p={r:0.2f}$",ha='left',va='center',transform=ax.transAxes)
        else:    
            ax.text(.95,.10, f"$r_p={r:0.2f}$",ha='right',va='center',transform=ax.transAxes)
    except ValueError:
        ax.text(0.5,0.5,"NO LIKELIHOODS",ha="center",va="center",transform=ax.transAxes)

    fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})

    # print(rcParams['axes.titlesize'])
    #override font sizes for labels
    ax.set_title(title,fontsize='medium')
    ax.set_xlabel(label_1,fontsize='small')
    ax.set_ylabel(label_2,fontsize='small')

