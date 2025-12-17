from pathlib import Path
from matplotlib import pyplot as plt
from gridplots import load_dfmdock_stats, plot_samples
from shared import setfont
import itertools

if __name__ == "__main__":
    setfont()
    
    outdir = Path("figures/figure5_subs")
    outdir.mkdir(exist_ok=True)
    
    # Figure 5 Subfigures
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats()
    
    from IPython import embed; embed()
    

    y = ['flow_nll','rosetta_Isc']
    x = ['DockQ','i_rmsd']

    pairs = list(itertools.product(x,y))

    for id in ('2SIC','2AYO'):
        print(id)
        for x,y in pairs:
            f,ax = plt.subplots(figsize=(2.5,2))
            try:
                plot_samples(sample_stats,gt_stats,id,x,y,markersize=30,ax=ax,
                            custom_xlabel=labels[x],custom_ylabel=labels[y],
                            custom_xlim=limits[x],custom_ylim=limits[y],save=False)
            except KeyError:
                pass
            name = f"{id}_{y}_v_{x}"
            f.savefig(outdir/f'{name}.png',dpi=300,bbox_inches='tight')
            f.suptitle(name)
    