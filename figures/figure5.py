from matplotlib import pyplot as plt
from gridplots import load_dfmdock_stats, plot_samples
from shared import setfont
import itertools

if __name__ == "__main__":
    setfont()
    
    # Figure 5 Subfigures
    samples_source = 'dfmdock_inference_trtrained_deterministic'
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(samples_source)
    

    assert samples_source == 'dfmdock_inference_trtrained_deterministic'

    y = ['flow_nll','rosetta_Isc']
    x = ['DockQ','i_rmsd']

    pairs = list(itertools.product(x,y))

    for id in ('2SIC','2AYO'):
        print(id)
        for x,y in pairs:
            f,ax = plt.subplots(figsize=(2.5,2))
            plot_samples(sample_stats,gt_stats,id,x,y,markersize=30,ax=ax,
                        custom_xlabel=labels[x],custom_ylabel=labels[y],
                        custom_xlim=limits[x],custom_ylim=limits[y],save=False)
            name = f"{id}_{y}_v_{x}"
            f.savefig(f'figures/figure5_subs/{name}.png',dpi=300,bbox_inches='tight')
            f.suptitle(name)
    