from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.figure import SubFigure
import numpy as np

from figures.figure3 import load_sample_likelihoods, plot_comparison

try:
    from shared import setfont
    from figure3 import plot_sample_result, plot_correlation
except ImportError:
    from figures.shared import setfont
    from figures.figure3 import plot_sample_result, plot_correlation

if __name__ == "__main__":
    setfont()

    # Figure 3
    likelihood_dir = Path('results/likelihood/diff_integration_methods')

    #going to try using subfigures. Hope this works!
    def add_comb_row(fig:SubFigure, parent_folder:Path, likelihood_subfolder:Path, integrand:str, prior:str, title:str, label:str, binline=False, other_corner=False):
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

    f = plt.figure(figsize=(10,8),layout='constrained')
    subfs:list[list[SubFigure]] = f.subfigures(nrows=4,ncols=2)
    
    add_comb_row(subfs[0][0],likelihood_dir,Path('gaussian_1d_diff_forward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration','(a)',binline=True, other_corner=True)
    add_comb_row(subfs[1][0],likelihood_dir,Path('gaussian_1d_diff_backward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration','(c)',binline=True)
    add_comb_row(subfs[2][0],likelihood_dir,Path('../gaussian_1d_diff'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration','(e)',binline=True)
    add_comb_row(subfs[3][0],likelihood_dir,Path('gaussian_1d_diff_piecewise_ode'),'integrand:TotalIntegrand','prior:smax_gaussian','Piewise ODE Integration','(g)',binline=True)
    add_comb_row(subfs[0][1],likelihood_dir,Path('gaussian_1d_diff_forward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration, 10x Interpolated','(b)',binline=True)
    add_comb_row(subfs[1][1],likelihood_dir,Path('gaussian_1d_diff_backward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration, 10x Interpolated','(d)',binline=True)
    add_comb_row(subfs[2][1],likelihood_dir,Path('gaussian_1d_diff_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration, 10x Interpolated','(f)',binline=True)
    add_comb_row(subfs[3][1],likelihood_dir,Path('../gaussian_1d_flow'),'integrand:TotalIntegrand','prior:smax_gaussian','Flow Trajectory ODE Integration','(h)',binline=True)


    f.savefig("figures/supplement_integration_method.png",dpi=600)


    #TODO: Make 2 sets of subplots, 4 with overbar of "1D gaussian" and 2 with overbar of "DFMDock" <- comparing trapezoid vs trapezoid interpolated vs piecewise ODE

    f,axs = plt.subplots(figsize=(10,2),layout='constrained',ncols=4,squeeze=True)

    def compare(ax,title,likelihood_1,label_1,likelihood_2,label_2,integrand="integrand:TotalIntegrand",prior="prior:smax_gaussian"):
        samples1, likelihoods1 = load_sample_likelihoods(likelihood_1,integrand_column=integrand,prior_column=prior)
        samples2, likelihoods2 = load_sample_likelihoods(likelihood_2,integrand_column=integrand,prior_column=prior)
        index = likelihoods1.index.intersection(likelihoods2.index)
        if len(likelihoods1) != len(likelihoods2):
            warnings.warn(f"Missing some samples! L1={len(likelihoods1)}, L2={len(likelihoods2)}")
        # from IPython import embed; embed() 
        plot_comparison(np.exp(likelihoods1[index]),np.exp(likelihoods2[index]),ax=ax)
        ax.set_title(title)
        ax.set_xlim([0,0.03])
        ax.set_xticks([0,0.01,0.02,0.03])
        ax.set_xlabel(label_1)
        ax.set_ylim([0,0.03])
        ax.set_yticks([0,0.01,0.02,0.03])
        ax.set_ylabel(label_2)

    compare(axs[0],"Forward Euler vs Interpolated",likelihood_dir/'gaussian_1d_diff_forward_euler','Original',likelihood_dir/'gaussian_1d_diff_forward_euler_interpolated','Interpolated')
    compare(axs[1],"Backward Euler vs Interpolated",likelihood_dir/'gaussian_1d_diff_backward_euler','Original',likelihood_dir/'gaussian_1d_diff_backward_euler_interpolated','Interpolated')
    compare(axs[2],"Trapezoidal vs Interpolated",likelihood_dir/'..'/'gaussian_1d_diff','Original',likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated')
    compare(axs[3],"Trapezoidal vs Piecewsise ODE",likelihood_dir/'..'/'gaussian_1d_diff','Original',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')


    f.savefig("figures/supplement_trapezoid_integration.png",dpi=600)

