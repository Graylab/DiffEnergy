from pathlib import Path
from gridplots import load_dfmdock_stats, plot_all_grids
from shared import setfont


if __name__ == "__main__":
    setfont()
    
    # Supplemental Figures
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats()
    plot_all_grids(sample_stats,gt_stats,labels,limits,save=True,
                   outdir=Path('figures/supplement_gridplots/'),close_plots=True,
                   extra_pairs=[('diff_piecewise_ode_nll','diff_trapezoid_nll'),('diff_piecewise_ode_nll','flow_nll'),('diff_trapezoid_nll','flow_nll'),])