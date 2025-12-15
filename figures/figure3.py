from matplotlib.figure import SubFigure
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Callable, Literal
from scipy.stats import norm, laplace
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic, gaussian_kde, kstest
from scipy.optimize import curve_fit

from yaml import load,CLoader as Loader
from pathlib import Path
from typing import Optional
from matplotlib.axes import Axes

from shared import get_gaussian, get_gt_gaussian, marginal_prob_std, setfont

def get_binline(nbins,x,y):
    prob_mean, bin_edges, binnumber = binned_statistic(x, y, statistic='mean', bins=nbins)
    prob_std = binned_statistic(x, y, statistic='std', bins=nbins).statistic
    prob_mean = gaussian_filter1d(prob_mean, sigma=0.5)
    prob_std = gaussian_filter1d(prob_std, sigma=0.5)
    # upper = prob_mean + prob_std
    # lower = prob_mean - prob_std
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, prob_mean
    

def plot_sample_result(parent_folder:str|Path, 
                       likelihood_subfolder:str|Path, 
                       integrand_column:str='integrand:TotalIntegrand', 
                       prior_column:str='prior:smax_gaussian', 
                       out_subfolder:str|Path|None=None, 
                       out_filename:str|None=None, 
                       binline=False, 
                       binline_color='darkred',
                       samples_file:Optional[str|Path]=None, 
                       ax_title:str|bool=True, 
                       fig_title:str|bool=True, 
                       ax:Optional[Axes]=None, 
                       save:bool=False, 
                       likelihood_offset=None,
                       plot_samples:bool=True,
                       plot_kde:bool=True,
                       plot_p0_gt:bool=True,
                       plot_p1_gt:bool=True,
                       legend:bool=False,
                       samples_color='salmon',
                       plot_value:Literal["probability","energy"]="probability"):
    parent_folder = Path(parent_folder)
    out_subfolder = out_subfolder or likelihood_subfolder
    likelihood_folder = parent_folder/likelihood_subfolder
    likelihood_file = likelihood_folder/"likelihood.csv"
    likelihood_config = likelihood_folder/"config.yaml"
    with open(likelihood_config,"r") as f:
        config = load(f.read(), Loader)
    if not samples_file:
        samples_file = config["data_samples"]

    
    x = np.linspace(-60, 60, 500)
    if not ax:
        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot()
    else:
        fig = None

    #THESE TITLES ARE NOT IN THE PAPER FIGURES
    if fig and fig_title:
        fig_title = fig_title if isinstance(fig_title,str) else str(likelihood_subfolder)
        fig.suptitle(fig_title,fontsize='small')
    if ax_title:
        ax_title = ax_title if isinstance(ax_title,str) else f"{integrand_column} w/ {prior_column}"
        ax.set_title(ax_title,fontsize='small')

    # Plot the samples data
    # path = '/scratch4/jgray21/ssarma4/repos/DiffEnergy/src/exps/laplacian/data/'
    # filename_1 = path + 'trinormal.csv'
    samples_df = pd.read_csv(samples_file, index_col=0)
    samples_1 = samples_df.values.flatten()
    
    # Load the likelihoods
    # path = '/scratch4/jgray21/ssarma4/repos/DiffEnergy/src/exps/laplacian/results/'
    # filename_2 = path + 'difftime_trinormal.csv'
    filename_2 = likelihood_file
    likelihood_df = pd.read_csv(filename_2)
    samples_1 = samples_df.loc[likelihood_df['id']].values.flatten() #I think this should work
    
    likelihoods = likelihood_df[integrand_column] + likelihood_df[prior_column]
    if likelihood_offset is not None:
        if isinstance(likelihood_offset,str):
            likelihood_offset = likelihood_df.loc[:,likelihood_offset]
        if isinstance(likelihood_offset,Callable):
            likelihood_offset = likelihood_offset(likelihood_df,integrand_column,prior_column)
        likelihoods += likelihood_offset.loc[likelihood_df['id']]
    likelihoods = likelihoods.values.flatten()

    sigma_min, sigma_max = 0.1, 30.0  # Define sigma limits
    gaussian_pdf = get_gt_gaussian(x)

    if plot_value == "probability":
        if plot_p0_gt:
            ax.plot(x, gaussian_pdf, 'k', linestyle='--', alpha=0.6, label="GT $p_0(x_0)$", zorder = 0)
            #plt.fill_between(x, gaussian_pdf, color='pink', alpha=0.6)

        if plot_p1_gt:
            sigma = marginal_prob_std(1, sigma_min, sigma_max)
            gaussian = norm.pdf(x, 0, sigma)  # Zero mean, std = sigma_t1
            ax.plot(x, gaussian, color='gray', linestyle='dashed', alpha=0.8, label="Prior $p_1(x_1)$", zorder = 2)
            #plt.fill_between(x, gaussian, color='green', alpha=0.2)

        if plot_kde:
            sns.kdeplot(samples_1, color='steelblue', alpha=0.8, label="Sample KDE", zorder = 1, ax=ax, bw_adjust=0.3)

        if plot_samples:
            probability = np.exp(likelihoods) #this was negative likelihoods in the old data since they calculated nll instead of ll
            ax.scatter(samples_1, probability, color=samples_color, label="Learned $p_0(x_0)$", s=2, linewidth=0.1, edgecolors=(0.1, 0.1, 0.1), alpha=0.8, zorder=-1)
            
            if binline:
                clamp_index = np.where(probability < 0.03)[0]
                nbins = 150
                bin_centers,prob_mean = get_binline(nbins,samples_1[clamp_index],probability[clamp_index])
                ax.plot(bin_centers, prob_mean, color=binline_color, zorder = 3, alpha=0.5)

        mu1, mu2, mu3 = -30, 0, 40
        b1, b2, b3 = 8.0, 5.0, 10.0
        w1, w2, w3 = 0.4, 0.3, 0.3
        gaussian_cdf =lambda x: (w1 * norm.cdf(x, loc=mu1, scale=b1) + w2 * norm.cdf(x, loc=mu2, scale=b2) + w3 * norm.cdf(x, loc=mu3, scale=b3))
        res = kstest(samples_1, gaussian_cdf)
        print("kde match:", res)

        
        # # Plot sampled prior at t=1. If 'prior_position' not provided in the dataframe, only works for paths with diffusion start points
        # if 'prior_position' in likelihood_df.columns:
        #     prior_points = likelihood_df["prior_position"]
        # else:
        #     raise ValueError()
        #     if config["path_type"] in ["sde_trajectories", "data_translation", "linear_trajectories"]:
        #         print("warning: Prior position not provided in likelihood.csv. Using diffusion trajectory endpoint")
        #         #endpoints defined by trajectory file, we can replicate them
        #         index_file = Path('..')/config["trajectory_index_file"]
        #         trajectories = load_trajectories(index_file)
        #         prior_points = np.array([load_endpoints(trajectories[id])[1][0] for id in likelihood_df['id']])
        #     else:
        #         print("warning: Prior position not provided in likelihood.csv. Unable to plot prior values")
        #         prior_points = None
        # if prior_points is not None:
        #     sorted_idx = prior_points.argsort()
        #     ax.plot(prior_points[sorted_idx], np.exp(likelihood_df[prior_column].values.flatten())[sorted_idx],color='purple',linestyle='dotted',alpha=0.8)
        


    
        # Labels and title
        ax.set_xlabel("Sample $x$")
        ax.set_ylabel("Density $p_0(x)$")
        ax.set_xlim(-60,60)
        ax.set_ylim(0, 0.03)
        if fig:
            fig.tight_layout()
        ax.set_yticks([0.00, 0.01, 0.02, 0.03])
        formatter = FuncFormatter(lambda y, _: f'{y:.2f}')
        ax.yaxis.set_major_formatter(formatter)

    elif plot_value == "energy":
        
        # Plot ground-truth gaussian
        if plot_p0_gt:
            loggaussian_pdf = np.log(gaussian_pdf)
            ax.plot(x, -loggaussian_pdf, 'k', linestyle='--', alpha=0.6, label="GT $p_0(x_0)", zorder = 0)

        # Plot Gaussian noise (prior) at t=1
        if plot_p1_gt:
            sigma = marginal_prob_std(1, sigma_min, sigma_max)
            loggaussian = norm.logpdf(x, 0, sigma)  # Zero mean, std = sigma_t1
            ax.plot(x, -loggaussian, color='gray', linestyle='dashed', alpha=0.8, label="Prior $p_1(x_1)$", zorder = 2)

        #getting a logarithmic kde is annoying, have to do it manually
        if plot_kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples_1)
            kde.set_bandwidth(kde.factor * 0.3) #bw_adjust from sns kde for parity with above
            ax.plot(x,-kde.logpdf(x), color='steelblue', alpha=0.8, label="-log(Density) estimate from generated samples", zorder = 1)
            # sns.kdeplot(samples_1, color='steelblue', alpha=0.8, label="Density estimate from generated samples", zorder = 1, ax=ax, bw_adjust=0.3)

        if plot_samples:
            energy = -likelihoods
            ax.scatter(samples_1, energy, color=samples_color, label='nll', s=2, linewidth=0.1, edgecolors=(0.1,0.1,0.1), alpha=0.8, zorder=-1)

            if binline:
                nbins = 150
                bin_centers,prob_mean = get_binline(nbins,samples_1,energy)
                ax.plot(bin_centers, prob_mean, color=binline_color, zorder = 3, alpha=0.5)

        # Labels and title
        ax.set_xlabel("Sample $x$")
        ax.set_ylabel("Energy $E_0(x)$")
        ax.set_xlim(-60,60)
        ax.set_ylim(2.5, 12.5)
        # plt.grid(False)
        if fig:
            fig.tight_layout()
        ax.set_yticks([2.50, 5.0, 7.5, 10.00, 12.5])
        formatter = FuncFormatter(lambda y, _: f'{y:.2f}')
        ax.yaxis.set_major_formatter(formatter)

    if legend:
        ax.legend(fontsize=4,loc='upper right',numpoints=10)
    
    if save:
        fig.set_dpi(600)
        outfolder = Path(".")/out_subfolder
        outfolder.mkdir(exist_ok=True,parents=True)
        plt.savefig(outfolder/(out_filename or f"{integrand_column}_{plot_value}.png"), dpi=300)


def plot_correlation(parent_folder:str|Path, likelihood_subfolder:str|Path, integrand_column:str, prior_column:str, out_subfolder:str|Path|None=None, out_filename:str|None=None, samples_file:Optional[str|Path]=None, ax_title:str|bool=True, fig_title:str|bool=True, ax:Optional[Axes]=None, save:bool=False, likelihood_offset=None):
    parent_folder = Path(parent_folder)
    out_subfolder = out_subfolder or likelihood_subfolder
    likelihood_folder = parent_folder/likelihood_subfolder
    likelihood_file = likelihood_folder/"likelihood.csv"
    likelihood_config = likelihood_folder/"config.yaml"
    with open(likelihood_config,"r") as f:
        config = load(f.read(), Loader)
    if not samples_file:
        samples_file = config["data_samples"] #since this notebook is one nested

    
    # Plot the samples data
    filename_1 = samples_file
    samples_df = pd.read_csv(filename_1, index_col=0)
    samples_1 = samples_df.values.flatten()

    filename_2 = likelihood_file
    likelihood_df = pd.read_csv(filename_2)
    likelihoods = likelihood_df[integrand_column] + likelihood_df[prior_column]
    if likelihood_offset is not None:
        if isinstance(likelihood_offset,str):
            likelihood_offset = likelihood_df.loc[:,likelihood_offset]
        if isinstance(likelihood_offset,Callable):
            likelihood_offset = likelihood_offset(likelihood_df,integrand_column,prior_column)
        likelihoods += likelihood_offset.loc[likelihood_df['id']]
    likelihoods = likelihoods.values.flatten()
    probability = np.exp(likelihoods) #this was negative likelihoods in the old data since they calculated nll instead of ll
    samples_1 = samples_df.loc[likelihood_df['id']].values.flatten() #only get samples for whom the likelihood has been calculated

    gaussian_pdf = get_gt_gaussian(samples_1)


    if not ax:
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_subplot()
    else:
        fig = None

    #THESE TITLES ARE NOT IN THE PAPER FIGURES
    if fig and fig_title:
        fig_title = fig_title if isinstance(fig_title,str) else str(likelihood_subfolder)
        fig.suptitle(fig_title,fontsize='small')
    if ax_title:
        ax_title = ax_title if isinstance(ax_title,str) else f"{integrand_column} w/ {prior_column}"
        ax.set_title(ax_title,fontsize='small')


    ax.scatter(gaussian_pdf[:5000], probability[:5000], color='salmon', edgecolors=(0.1, 0.1, 0.1), linewidth=0.2, s=2, label="Scatter", alpha=0.8, zorder=1)

    # Draw y = x line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='y = x', zorder=0)

    def line_through_origin(x, a):
        return a * x

    # Fit the model
    params, pcov, infodict, mesg, ier  = curve_fit(line_through_origin, gaussian_pdf, probability, full_output=True)
    a = params[0]

    drange = (0,0.03)

    # Draw y = ax line
    min_val = drange[0]#min(gaussian_pdf.min(), probability.min())
    max_val = drange[1]#max(gaussian_pdf.max(), probability.max())
    
    #shorter dashes. default '--' dash pattern is (3.7,1.6) according to rcParams
    ax.plot([min_val, max_val], [a*min_val, a*max_val], color='k', linestyle='--', dashes=(2,1), label=f'y = {a:.2f}x')


    # Example
    corr, p_value = pearsonr(gaussian_pdf, probability)
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")

    ax.set_xlim(drange)
    ax.set_ylim(drange)
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel("Ground Truth $p_0(x)$")
    ax.set_ylabel("Recovered $p_0(x)$")

    ax.set_aspect(1.0)

    if save:
        outfolder = Path(".")/out_subfolder
        outfolder.mkdir(exist_ok=True,parents=True)
        plt.savefig(outfolder/(out_filename or f"{integrand_column}_correlation.png"), dpi=300, bbox_inches='tight')

    return (corr,p_value)

if __name__ == "__main__":
    setfont()
    
    # Figure 3
    likelihood_dir = Path('../likelihood_results/likelihoodv3')

    #going to try using subfigures. Hope this works!
    def add_comb_row(fig:SubFigure, parent_folder:Path, likelihood_subfolder:Path, integrand:str, prior:str, title:str, label:str, binline=False,):
        probax,corax = fig.subplots(nrows=1,ncols=2,width_ratios=[2,1.5])

        plot_sample_result(parent_folder,likelihood_subfolder,integrand,prior,binline=binline,binline_color='purple',
                        ax=probax,save=False,fig_title=False,ax_title=False)
        r,pval = plot_correlation(parent_folder,likelihood_subfolder,integrand,prior,ax=corax,save=False,fig_title=False,ax_title=False)

        corax.text(.95,.10, f"$r_p={r:0.2f}$",ha='right',va='center',transform=corax.transAxes)

        fig.suptitle(title)
        #add subfigure label
        fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})


    #probability size: (3,2); correlation size: (1.5,1.5)
    # Row size: (5.5,2.5)? extra .5 in width for spacing and .5 in height for titles
    f = plt.figure(figsize=(5,4),layout='constrained')
    subfs:list[SubFigure] = f.subfigures(nrows=2,wspace=0.5,squeeze=True)
    add_comb_row(subfs[0],likelihood_dir,Path('gaussian_1d_unnoised/3integrand_diff_trapezoidint'),'integrand:TotalIntegrand','prior:smax_gaussian','Diffusion Trajectory','(a)',binline=True)
    add_comb_row(subfs[1],likelihood_dir,Path('gaussian_1d_unnoised/3integrand_flow'),'integrand:TotalIntegrand','prior:smax_gaussian','Flow Trajectory','(b)',binline=True)

    f.savefig("figures/figure_3.png",dpi=600)