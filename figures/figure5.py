from pathlib import Path
from matplotlib import pyplot as plt
import itertools

#run from either parent directory or figures directory
try:
    from shared import setfont
    from gridplots import load_dfmdock_stats, plot_samples
except ImportError:
    from figures.shared import setfont
    from figures.gridplots import load_dfmdock_stats, plot_samples

if __name__ == "__main__":
    setfont()
    
    # outdir = Path("figures/figure5_subs")
    outdir = Path("figures/figure5_subs")
    outdir.mkdir(exist_ok=True)
    
    # Figure 5 Subfigures
    # sample_stats, gt_stats, labels, limits = load_dfmdock_stats()  
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(prior='receptor_smax_gaussian')

    y = ['flow_nll','diff_nll','diff_15ensembled_nll','rosetta_Isc']
    x = ['DockQ','i_rmsd']

    limits['diff_nll'] = None

    pairs = list(itertools.product(x,y))

    for id in ('2SIC','2A1A'):
        print(id)
        # if id == '2A1A':
        #     limits['diff_nll'] = (-5,7)
        for x,y in pairs:
            f,ax = plt.subplots(figsize=(2.5,2))
            try:
                plot_samples(sample_stats,gt_stats,id,x,y,markersize=30,ax=ax,
                            custom_xlabel=labels[x],custom_ylabel=labels[y],
                            custom_xlim=limits[x],custom_ylim=limits[y],save=False)
            except KeyError:
                pass
            if 'flow' in y:
                ax.yaxis.labelpad = 13 #shift flow label away from xaxis so spacing is same as diff and rosetta w/ negative number ticklabels
            if 'diff' in y and id == '2SIC':
                ax.yaxis.labelpad = 5 #shift flow label away from xaxis so spacing is same as diff and rosetta w/ negative number ticklabels
            name = f"{id}_{y}_v_{x}"
            f.savefig(outdir/f'{name}.png',dpi=600,bbox_inches='tight')
            f.suptitle(name)
    