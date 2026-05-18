#run from either parent directory or figures directory
import os
from pathlib import Path
import shutil


try:
    from shared import setfont
    from gridplots import load_dfmdock_stats
except ImportError:
    from figures.shared import setfont
    from figures.gridplots import load_dfmdock_stats


if __name__ == "__main__":
    setfont()
    
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(prior='receptor_smax_gaussian')
    
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
    min_dockq_diff,_,min_idx_diff = get_ranking_scores('diff_nll')
    min_dockq_ensembled,_,min_idx_ensembled = get_ranking_scores('diff_15ensembled_nll')
    min_dockq_rosetta,_,min_idx_rosetta = get_ranking_scores('rosetta_Isc')
    max_dockq_oracle,_,max_idx_oracle = get_ranking_scores('DockQ',invert=True)
    
    key_sorted_oracle = list(dict(sorted(max_dockq_oracle.items(), key=lambda x: x[1], reverse=True)).keys())

    # print(f"====== 2SIC ranking ======")
    # print(f"  Flow: {min_idx_flow['2SIC']}")
    # print(f"  Diffusion: {min_idx_diff['2SIC']}")
    # print(f"  Rosetta: {min_idx_rosetta['2SIC']}")
    # print(f"  Oracle: {max_idx_oracle['2SIC']}")

    # print(f"====== 2A1A ranking ======")
    # print(f"  Flow: {min_idx_flow['2A1A']}")
    # print(f"  Diffusion: {min_idx_diff['2A1A']}")
    # print(f"  Rosetta: {min_idx_rosetta['2A1A']}")
    # print(f"  Oracle: {max_idx_oracle['2A1A']}")
    
    sample_dir = Path('results/sample_results/dfmdock/pdb/sample_pdbs')
    copy_files = True
    keys = [('Flow',min_idx_flow), ('Diffusion',min_idx_diff), ('Ensembled',min_idx_ensembled), ('Rosetta',min_idx_rosetta), ('Oracle',max_idx_oracle)]
    for id in '2SIC', '2A1A':
        outfolder = Path('figures/figure6_ranked_poses')/id
        if copy_files: outfolder.mkdir(parents=True,exist_ok=True)
        
        print(f"====== {id} ranking ======")
        for k,v in keys:
            print(f"  {k}: {v[id]}")
            if copy_files: shutil.copy(sample_dir/f"{v[id]}.pdb",outfolder/f"{k}.pdb")
    
    import matplotlib.pyplot as plt
    plt.scatter(key_sorted_oracle, [min_dockq_flow[key] for key in key_sorted_oracle], label='Learned Energy (Flow)')
    plt.scatter(key_sorted_oracle, [min_dockq_diff[key] for key in key_sorted_oracle], color='green', label='Learned Energy (Diff)')
    plt.scatter(key_sorted_oracle, [min_dockq_ensembled[key] for key in key_sorted_oracle], color='red', label='Learned Energy (Ensembled)')
    plt.scatter(key_sorted_oracle, [min_dockq_rosetta[key] for key in key_sorted_oracle], marker='x', label='Rosetta Energy')
    plt.scatter(key_sorted_oracle, [max_dockq_oracle[key] for key in key_sorted_oracle], marker="1", c='k', label='Oracle', alpha=0.8)
    
    plt.hlines(0.23, xmin=-2, xmax=26, color='gray', linestyle='--', alpha=0.5)
    plt.hlines(0.49, xmin=-2, xmax=26, color='gray', linestyle='--', alpha=0.5)
    plt.hlines(0.80, xmin=-2, xmax=26, color='gray', linestyle='--', alpha=0.5)

    plt.text(-1.9, 0.26, 'Acceptable', fontsize=12, horizontalalignment='left')
    plt.text(-1.9, 0.52, 'Medium', fontsize=12, horizontalalignment='left')
    plt.text(-1.9, 0.83, 'High', fontsize=12, horizontalalignment='left')

    # plt.legend(fontsize=10,loc='upper right')
    plt.legend(fontsize=8,loc='upper right')
    plt.ylabel('DockQ')
    plt.xlim(-2, 26)
    plt.xticks(rotation=45)
    fig = plt.gcf()
    fig.set_size_inches(7.5, 3)
    fig.set_dpi(300)
    plt.savefig('figures/figure6_ranking.png', dpi=600, bbox_inches='tight')