from pathlib import Path
import string
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

def add_comb_row(fig:SubFigure, label:str, title:str, parent_folder:Path, likelihood_subfolder:Path, integrand:str, prior:str, binline=False, other_corner=False):
    probax,corax = fig.subplots(nrows=1,ncols=2,width_ratios=[2,1.5])

    plot_sample_result(parent_folder,likelihood_subfolder,integrand,prior,binline=binline,binline_color='purple',
                    ax=probax,save=False,fig_title=False,ax_title=False)
    r,pval = plot_correlation(parent_folder,likelihood_subfolder,integrand,prior,ax=corax,save=False,fig_title=False,ax_title=False)

    if "backward" in title.lower():
        if "interpolated" in title.lower():
            probax.set_ylim([0,0.04])
            probax.set_yticks([0,0.01,0.02,0.03,0.04])
        else:
            probax.set_ylim([0,0.20])
            probax.set_yticks([0,0.05,0.10, 0.15, 0.20])

    if other_corner:
        corax.text(.05,.9, f"$r_p={r:0.2f}$",ha='left',va='center',transform=corax.transAxes)
    else:    
        corax.text(.95,.10, f"$r_p={r:0.2f}$",ha='right',va='center',transform=corax.transAxes)

    fig.suptitle(title)
    #add subfigure label
    fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})

def compare_likelihoods(fig:Figure,label:str,title:str,likelihood_1:str|Path|pd.Series,label_1:str,likelihood_2:str|Path|pd.Series,label_2:str,integrand="integrand:TotalIntegrand",prior="prior:smax_gaussian",
            lim:Optional[tuple[float,float]]=(0,0.03),ticks:Optional[list[float]]=[0,0.01,0.02,0.03],exp=True,
            other_corner=False):
    ax = fig.add_subplot()
    likelihoods1 = load_sample_likelihoods(likelihood_1,integrand_column=integrand,prior_column=prior)[0] if not isinstance(likelihood_1,pd.Series) else likelihood_1
    likelihoods2 = load_sample_likelihoods(likelihood_2,integrand_column=integrand,prior_column=prior)[0] if not isinstance(likelihood_2,pd.Series) else likelihood_2
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

if __name__ == "__main__":
    setfont()

    # Figure 3
    likelihood_dir = Path('results/likelihood/diff_integration_methods_smin30')

    #going to try using subfigures. Hope this works!







    f = plt.figure(figsize=(10,10),layout='constrained')
    subf_rows:list[SubFigure] = f.subfigures(nrows=5,ncols=1,height_ratios=[1,1,1,1,1.1])
    subfs_grid = [subf.subfigures(nrows=1,ncols=2) for subf in subf_rows[:-1]] #make all but the last row two column
    
    labelit = iter([f'({l})' for l in string.ascii_lowercase])

    add_comb_row(subfs_grid[0][0],next(labelit),'Forward Euler Integration',likelihood_dir,Path('gaussian_1d_diff_forward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True, other_corner=True)
    add_comb_row(subfs_grid[1][0],next(labelit),'Backward Euler Integration',likelihood_dir,Path('gaussian_1d_diff_backward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[2][0],next(labelit),'Trapezoidal Integration',likelihood_dir,Path('../noise_schedule_tests/gaussian_1d_diff_smin30_smax70'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[3][0],next(labelit),'Piewise ODE Integration',likelihood_dir,Path('gaussian_1d_diff_piecewise_ode'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[0][1],next(labelit),'Forward Euler Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_forward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[1][1],next(labelit),'Backward Euler Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_backward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[2][1],next(labelit),'Trapezoidal Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[3][1],next(labelit),'Flow Trajectory ODE Integration',likelihood_dir,Path('../noise_schedule_tests/gaussian_1d_flow_smin30_smax70'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)

    subf_bottom = subf_rows[-1]
    subf_bottom.get_constrained_layout()
    comparison_figs = subf_bottom.subfigures(nrows=1,ncols=5,);
    compit = iter(comparison_figs)
    compare_likelihoods(next(compit),next(labelit),"Forward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_forward_euler','Forward Euler',likelihood_dir/'gaussian_1d_diff_forward_euler_interpolated','Interpolated Forward Euler')
    compare_likelihoods(next(compit),next(labelit),"Backward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_backward_euler','Backward Euler',likelihood_dir/'gaussian_1d_diff_backward_euler_interpolated','Interpolated Backward Euler',other_corner=True)
    compare_likelihoods(next(compit),next(labelit),"Trapezoidal\nvs Interpolated",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Trapezoidal',likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated Trapezoidal')
    # compare(next(compit),"Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Original',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')
    compare_likelihoods(next(compit),next(labelit),"Trapezoidal\nvs Piecewise ODE",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Trapezoidal',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')
    compare_likelihoods(next(compit),next(labelit),"Interpolated Trapezoidal\nvs Piecewise ODE",likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated Trapezoidal',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')


    f.savefig("figures/supplement_integration_method.png",dpi=600)

    
    # exit()





    #TODO: Make 2 sets of subplots, 4 with overbar of "1D gaussian" and 2 with overbar of "DFMDock" <- comparing trapezoid vs trapezoid interpolated vs piecewise OD
    f = plt.figure(figsize=(12,2.5),layout='constrained')
    subfs = f.subfigures(ncols=5)

    labelit = iter([f'({l})' for l in string.ascii_lowercase])


    likelihood_dir = Path('results/likelihood')

    subit = iter(subfs)

    compare_likelihoods(next(subit),next(labelit),"Flow (40 steps) vs\nFlow (80 steps)",likelihood_dir/'dfmdock_flow','Trapezoidal',likelihood_dir/'dfmdock_flow_2interp','Flow (80 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False)
    compare_likelihoods(next(subit),next(labelit),"Trapezoidal vs\nPiecewsise ODE (120 steps)",likelihood_dir/'dfmdock_diff','Trapezoidal',likelihood_dir/'dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False)
    compare_likelihoods(next(subit),next(labelit),"Interpolated Trapezoidal\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_diff_interpolated','10x Interpolated Trapezoidal',likelihood_dir/'dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False)
    compare_likelihoods(next(subit),next(labelit),"Pieceiwse ODE (40 steps)\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_diff_piecewise_ode','Piecewise ODE (40 steps)',likelihood_dir/'dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False)
    compare_likelihoods(next(subit),next(labelit),"Piecewise ODE (80 steps)\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_diff_piecewise_ode_2interp','Piecewise ODE (80 steps)',likelihood_dir/'dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False)

    # f.tight_layout()

    # compare(axs[0],"Forward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_forward_euler','Original',likelihood_dir/'gaussian_1d_diff_forward_euler_interpolated','Interpolated')
    # compare(axs[1],"Backward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_backward_euler','Original',likelihood_dir/'gaussian_1d_diff_backward_euler_interpolated','Interpolated')
    # compare(axs[2],"Trapezoidal\nvs Interpolated",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Original',likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated')
    # compare(axs[3],"Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Original',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')
    # compare(axs[4],"DFMDock Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'..'/'dfmdock_diff','Original',likelihood_dir/'dfmdock_diff_piecewise_ode','Piecewise ODE',prior='prior:receptor_smax_gaussian')
    # compare(axs[4],"DFMDock Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'..'/'dfmdock_diff','Original',likelihood_dir/'dfmdock_diff_interpolated','Interpolated',prior='prior:receptor_smax_gaussian')
    # compare_likelihoods(subfs[4],next(labelit),"DFMDock Interpolated Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'dfmdock_diff_interpolated','Interpolated',likelihood_dir/'dfmdock_diff_piecewise_ode','Piecewise ODE',prior='prior:receptor_smax_gaussian')


    f.savefig("figures/supplement_integration_comparison.png",dpi=600)

