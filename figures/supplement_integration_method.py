try:
    from shared import setfont
    from figure3 import plot_sample_result, 
except ImportError:
    from figures.shared import setfont


if __name__ == "__main__":
    setfont()

    # Figure 3
    likelihood_dir = Path('results/likelihood')

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

    f = plt.figure(figsize=(5,4),layout='constrained')
    subfs:list[SubFigure] = f.subfigures(nrows=2,wspace=0.5,squeeze=True)
    add_comb_row(subfs[0],likelihood_dir,Path('gaussian_1d_diff'),'integrand:TotalIntegrand','prior:smax_gaussian','Diffusion Trajectory','(a)',binline=True)
    add_comb_row(subfs[1],likelihood_dir,Path('gaussian_1d_flow'),'integrand:TotalIntegrand','prior:smax_gaussian','Flow Trajectory','(b)',binline=True)

    f.savefig("figures/figure_3.png",dpi=600)