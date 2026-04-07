from pathlib import Path
#run from either parent directory or figures directory
try:
    from shared import setfont
    from gridplots import load_dfmdock_stats, plot_all_grids
except ImportError:
    from figures.shared import setfont
    from figures.gridplots import load_dfmdock_stats, plot_all_grids



if __name__ == "__main__":
    setfont()
    
    # Supplemental Figures
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats()
    plot_all_grids(sample_stats,gt_stats,labels,limits,save=True,
                   outdir=Path('figures/supplement_gridplots/'),close_plots=True)