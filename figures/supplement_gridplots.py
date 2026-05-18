from pathlib import Path
import re

import pandas as pd
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
    # sample_stats, gt_stats, labels, limits = load_dfmdock_stats()
    # plot_all_grids(sample_stats,gt_stats,labels,limits,save=True,
    #                outdir=Path('figures/supplement_gridplots/'),close_plots=True)
    
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(
                           prior='receptor_smax_gaussian')
    # from IPython import embed; embed()
    extra_pairs = [('flow_nll','diff_trapezoid_nll'),('flow_nll','diff_10interp_nll'),('flow_nll','diff_piecewise_ode_nll'),('flow_nll','diff_15ensembled_nll')]
    plot_all_grids(sample_stats,gt_stats,labels,limits,save=True,title_graphs=True,
                   outdir=Path('figures/supplement_gridplots/'),close_plots=True,extra_pairs=extra_pairs)
    
    # ensemble_stats, gt_stats, labels, limits = load_dfmdock_stats(sample_likelihoods_sources={
    #                        'diff_ensembled_nll':'dfmdock_forwardsde_ensembled'},
    #                        prior='receptor_smax_gaussian',)
    
    # average_stats = {}
    # ensemble_regex = re.compile(r"(.*)_r\d+")
    # def same_or_mean(s:pd.Series):
    #     try:
    #         return s.mean()
    #     except:
    #         assert len(s.unique()) == 1
    #         return s.loc[s.index[0]]
        
    
    # for key, stats in ensemble_stats.items():
    #     average_stats[key] = stats.groupby(stats['index'].str.extract(ensemble_regex)[0]).aggregate(same_or_mean).reset_index(drop=True)
    
    # plot_all_grids(average_stats,gt_stats,labels,limits,save=True,title_graphs=True,)