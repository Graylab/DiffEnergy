from gridplots import load_dfmdock_stats
from shared import setfont


if __name__ == "__main__":
    setfont()
    
    samples_source = 'dfmdock_inference_trtrained_deterministic'
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(samples_source)
    
    def get_ranking_scores(rank_metric:str,invert=False):
        dockq:dict[str,float] = {} #pdb_id: (sample_id,DockQ,metric_score)
        metric:dict[str,float] = {}
        idx:dict[str,str|int] = {}
        for id in sample_stats:
            if not invert:
                best = sample_stats[id][rank_metric].idxmin()
            else: 
                best = sample_stats[id][rank_metric].idxmax()
            best_dockq = sample_stats[id]['DockQ'].loc[best]
            best_metric = sample_stats[id][rank_metric].loc[best]

            dockq[id] = best_dockq
            metric[id] = best_metric
            idx[id] = best
        
        return dockq,metric,idx

    min_dockq_flow,_,min_idx_flow = get_ranking_scores('flow_nll')
    min_dockq_diff,_,min_idx_diff = get_ranking_scores('diff_trapezoid_nll')
    min_dockq_rosetta,_,min_idx_rosetta = get_ranking_scores('rosetta_Isc')
    max_dockq_oracle,_,max_idx_oracle = get_ranking_scores('DockQ',invert=True)
    
    key_sorted_oracle = list(dict(sorted(max_dockq_oracle.items(), key=lambda x: x[1], reverse=True)).keys())
    
    import matplotlib.pyplot as plt
    # plt.scatter(key_sorted_oracle, [min_dockq_diff[key] for key in key_sorted_oracle], label='Learned Energy (Diff)')
    plt.scatter(key_sorted_oracle, [min_dockq_flow[key] for key in key_sorted_oracle], label='Learned Energy (Flow)')
    plt.scatter(key_sorted_oracle, [min_dockq_rosetta[key] for key in key_sorted_oracle], marker='x', label='Rosetta Energy')
    plt.scatter(key_sorted_oracle, [max_dockq_oracle[key] for key in key_sorted_oracle], marker="1", c='k', label='Oracle', alpha=0.8)
    
    plt.hlines(0.23, xmin=-1, xmax=25, color='gray', linestyle='--', alpha=0.5)
    plt.hlines(0.49, xmin=-1, xmax=25, color='gray', linestyle='--', alpha=0.5)
    plt.hlines(0.80, xmin=-1, xmax=25, color='gray', linestyle='--', alpha=0.5)

    plt.text(1.15, 0.26, 'Acceptable', fontsize=12, horizontalalignment='left')
    plt.text(1.15, 0.52, 'Medium', fontsize=12, horizontalalignment='left')
    plt.text(1.15, 0.83, 'High', fontsize=12, horizontalalignment='left')

    plt.legend()
    plt.ylabel('DockQ')
    plt.xlim(-1, 25)
    plt.xticks(rotation=45)
    fig = plt.gcf()
    fig.set_size_inches(7.5, 3)
    fig.set_dpi(300)
    plt.savefig('figures/figure6_ranking.png', dpi=600, bbox_inches='tight')