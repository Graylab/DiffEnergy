from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import SubFigure

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
    def add_comb_row(fig:SubFigure, parent_folder:Path, likelihood_subfolder:Path, integrand:str, prior:str, title:str, label:str, binline=False,):
        probax,corax = fig.subplots(nrows=1,ncols=2,width_ratios=[2,1.5])

        plot_sample_result(parent_folder,likelihood_subfolder,integrand,prior,binline=binline,binline_color='purple',
                        ax=probax,save=False,fig_title=False,ax_title=False)
        r,pval = plot_correlation(parent_folder,likelihood_subfolder,integrand,prior,ax=corax,save=False,fig_title=False,ax_title=False)

        corax.text(.95,.10, f"$r_p={r:0.2f}$",ha='right',va='center',transform=corax.transAxes)

        fig.suptitle(title)
        #add subfigure label
        fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})

    f = plt.figure(figsize=(5,14),layout='constrained')
    subfs:list[SubFigure] = f.subfigures(nrows=7,wspace=0.5,squeeze=True)
    
    add_comb_row(subfs[0],likelihood_dir,Path('gaussian_1d_diff_forward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration','(a)',binline=True)
    add_comb_row(subfs[1],likelihood_dir,Path('gaussian_1d_diff_forward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration,\n10x Interpolated Trajectory','(b)',binline=True)
    add_comb_row(subfs[2],likelihood_dir,Path('gaussian_1d_diff_backward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration','(c)',binline=True)
    add_comb_row(subfs[3],likelihood_dir,Path('gaussian_1d_diff_backward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration,\n10x Interpolated Trajectory','(d)',binline=True)
    add_comb_row(subfs[4],likelihood_dir,Path('gaussian_1d_diff_trapezoid'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration','(e)',binline=True)
    add_comb_row(subfs[5],likelihood_dir,Path('gaussian_1d_diff_trapezoid_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration,\n10x Interpolated Trajectory','(f)',binline=True)
    add_comb_row(subfs[6],likelihood_dir,Path('gaussian_1d_diff_piecewise_ode'),'integrand:TotalIntegrand','prior:smax_gaussian','Piewise ODE Integration','(g)',binline=True)


    f.savefig("figures/supplement_integration_method.png",dpi=600)