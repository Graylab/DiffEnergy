import itertools
import pickle
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from typing import Any, Literal, DefaultDict, Mapping, MutableMapping, Optional, TypeVar


METRIC_LABELS = {
    'DockQ': 'DockQ',
    'i_rmsd': "Interface RMSD (Å)",
    'rosetta_Isc': "Rosetta Energy (REU)",
    'dfmdock_energy': "DFMDock Predicted Energy"
}

METRIC_LIMITS = {
    'DockQ': (-0.05, 1.05),
    'i_rmsd': (-0.2,None),
    'rosetta_Isc': (-70,5),
    'dfmdock_energy':None #TODO: set reasonable limit
}

NEWCODE_LABELS = {"flow_nll": "Learned Energy (Flow)",
                 "diff_nll": "Learned Energy (Diffusion) [Euler]",
                 "diff_piecewise_ode_nll": "Learned Energy (Diffusion) [Piecewise ODE]",
                 "diff_trapezoid_nll": "Learned Energy (Diffusion)",
                 "forwardsde_nll": "Learned Energy (Forward SDE)",
                 "diff_10interp_nll": "Learned Energy (Diffusion, Interpolated 10x)",
                 "diff_50interp_nll": "Learned Energy (Diffusion, Interpolated 50x)"}


def get_likelihoods(likelihood_folder,integrand="TotalIntegrand",prior="smax_gaussian")->tuple[dict[str,pd.DataFrame],dict[str,pd.DataFrame]]:
    lfile = likelihood_folder/"likelihood.csv"
    ldf = pd.read_csv(lfile)
    ldf.index = ldf['id'].str.rsplit("/",n=1).str[-1]
    pdb_ids = ldf.index.str.split("_").str[0];

    ldf['pdb_id'] = pdb_ids
    ldf.sort_values(['pdb_id'])

    nlls = {}
    priors = {}
    for id in np.unique(pdb_ids):
        arr = ldf[ldf['pdb_id'] == id]

        #use the index inherent to the series for ordering!
        nlls[id] = -((arr[f'integrand:{integrand}'] + arr[f'prior:{prior}']))/3 #negate because the code is expecting negative log-likelihood; divide by 3 because original code divided by N
        priors[id] = -arr[f'prior:{prior}']/3

    return nlls, priors

def get_metrics_csv_sample_stats(metrics_csv):
    metrics_df = pd.read_csv(metrics_csv)
    metrics_df.index = metrics_df['index']
    pdb_ids = metrics_df.index.str.split("_").str[0];
    sample_index = metrics_df.index.str.split("_").str[1];
    
    metrics_df['sample_index'] = sample_index.astype(int)
    metrics_df['pdb_id'] = pdb_ids
    metrics_df.sort_values(['pdb_id'])
    unique_ids = np.unique(pdb_ids)
    
    results = {}
    for id in unique_ids:
        iddf = metrics_df[metrics_df['pdb_id'] == id].sort_values('sample_index')
        assert np.all(np.diff(iddf['sample_index']) == 1) #make sure no gaps (paranoia incarnate)
        results[id] = iddf
    return results
    
    

def get_dfmdock_sample_stats(dfmdock_stats_csv):
    dfmdock_stats_df = pd.read_csv(dfmdock_stats_csv)
    dfmdock_stats_df.rename(columns={'id':'pdb_id','index':'sample_index','energy':'dfmdock_energy'},inplace=True)
    dfmdock_stats_df.index = dfmdock_stats_df['pdb_id'].astype(str) + '_p' + dfmdock_stats_df['sample_index'].astype(str)
    unique_ids = np.unique(dfmdock_stats_df['pdb_id'])
    results = {}
    for id in unique_ids:
        iddf = dfmdock_stats_df[dfmdock_stats_df['pdb_id'] == id].sort_values('sample_index')
        assert np.all(np.diff(iddf['sample_index']) == 1) #make sure no gaps (paranoia incarnate)
        results[id] = iddf
    return results

def get_neighborhood_sample_stats(sample_folder):
    sample_folder = Path(sample_folder)
    metrics_df = pd.read_csv(sample_folder/"metrics.csv",index_col='index')
    index_df = pd.read_csv(sample_folder/"index.csv",index_col='index')

    df = metrics_df.join(index_df)
    df.rename(columns={'PDB_id':'pdb_id','index':'id'},inplace=True)
    unique_ids = df['pdb_id'].unique()
    results = {}
    for id in unique_ids:
        iddf = df[df['pdb_id'] == id]
        results[id] = iddf
    return results

def load_rosetta_isc(rosetta_csv):
    rosetta_df = pd.read_csv(rosetta_csv)
    rosetta_df.index = rosetta_df['id']
    unique_ids = np.unique(rosetta_df['pdb_id'])
    
    results = {}
    for pdb_id in unique_ids:
        iddf = rosetta_df[rosetta_df['pdb_id'] == pdb_id]
        
        #Missing Fnat means scoring failed
        iddf = iddf[iddf['Fnat'].notna()]
    
        #assign values by index. Missing rows will be assigned 'nan', since rosetta doesn't return a value for every structure
        results[pdb_id] = iddf['I_sc']
    return results


def concat_dfdicts(*dicts:dict[str,pd.DataFrame]):
    ids = set().union(*(d.keys() for d in dicts))
    dest:dict[str,pd.DataFrame] = {}
    for id in ids:
        dest[id] = pd.concat([d[id] for d in dicts if id in d])
    return dest

def copy_keys(dest,source,*keys):
    #convenience function to copy a bunch of key-value pairs from source to dest
    for key in keys:
        dest[key] = source[key]

def insert_dfdict(dest:Mapping[str,pd.DataFrame|MutableMapping[str,Any]],destkey:str,source:Mapping[str,pd.DataFrame]):
    for id in source:
        if id in dest:
            d = dest[id]
            if isinstance(d,pd.DataFrame):
                d.loc[source[id].index,destkey] = source[id]
            else:
                d[destkey] = source[id]
        else:
            raise ValueError(id)

T = TypeVar("T")
def insert_dict(dest:Mapping[str,MutableMapping[str,T]],destkey:str,source:Mapping[str,T]):
    for id in source:
        if id in dest:
            dest[id][destkey] = source[id]

DFMDOCK_STATS_LOCATIONS = {
    'dfmdock_inference_trtrained_deterministic': "../sample_results/dfmdock_inference_trtrained_deterministic/csv_files/db5_test_diffenergy_tr_0_0.5_120_samples_40_steps_dips_.csv",
    'darren_inference': "../sample_results/darren_inference/csv_files/db5_test_DFMDock_model_0_0.5_120_samples_40_steps_dips_.csv",
    'dfmdock_tr_inference': "../sample_results/dfmdock_inference/csv_files/db5_test_DFMDock_model_0_0.5_120_samples_40_steps_dips_.csv",
}
def load_dfmdock_stats(likelihoods_folder='results/likelihood/', #parent folder for likelihoods
                       #sample paths
                       sample_likelihoods_sources:dict[str,str|Path|tuple[str|Path,...]]={
                           'flow_nll':'dfmdock_flow',
                           'diff_nll':'dfmdock_diff'},
                       sample_metrics_csv='results/sample_results/dfmdock/metrics.csv',
                       sample_rosetta_csv='results/rosetta/dfmdock/refined_score.csv',
                       #gt paths
                       gt_likelihoods_sources:dict[str,str|Path|tuple[str|Path,...]]={'flow_nll':'dfmdock_gt_flow'},
                       #no gt metrics because they're all either 0 or 1
                       gt_rosetta_csv='results/rosetta/dfmdock_gt/refined_score.csv',
                    ):
    
    ## One big dict which will hold all the stats for the *generated samples*. Start by populating the test metrics generated during sampling:
    sample_stats:dict[str,pd.DataFrame] = {}
    sample_stats = get_metrics_csv_sample_stats(sample_metrics_csv)
    
    # Add rosetta energies
    insert_dfdict(sample_stats,'rosetta_Isc',load_rosetta_isc(sample_rosetta_csv))
    
    
    ## Same idea as sample_stats, but for the ground truth structures. just contains dicts of floats, since only one structure per id and thus no dataframe required
    gt_stats:dict[str,dict[str,float]] = {}
    for id in sample_stats:
        gt_stats[id] = {'DockQ': 1, 'c_rmsd': 0, 'i_rmsd': 0, 'l_rmsd': 0, 'fnat': 1, 'num_clashes': 0} #some dummy data
        
    # Add rosetta energies
    for id,isc in load_rosetta_isc(gt_rosetta_csv).items():
        gt_stats[id]['rosetta_Isc'] = isc.item() if len(isc) > 0 else np.nan #get the one value for the one-element series
    
    # Dict to hold: 1) the display labels for each inference type, and 2) the axis limits (optional) for each display label
    labels:dict[str,str|None] = DefaultDict(lambda: None, **METRIC_LABELS)
    limits:dict[str,tuple[float|None,float|None]|None] = DefaultDict(lambda: None, **METRIC_LIMITS)    
    
    ### Add Calculated Likelihoods
    labels.update(NEWCODE_LABELS)
    limits['flow_nll'] = (-1,8)
    limits['diff_10interp_nll'] = (-4,8)
    limits['diff_trapezoid_nll'] = (-4,8)
    limits['diff_piecewise_ode_nll'] = (-4,8)
    
    default_integrand = 'TotalIntegrand' #TotalIntegrand is always right, no need to mess with total vs flow on different trajectories
    default_prior = 'smax_gaussian'

    likelihoods_folder = Path(likelihoods_folder)

    ## Sample Likelihoods
    load_newcode_likelihoods(sample_likelihoods_sources,default_integrand,default_prior,sample_stats,srcfolder=likelihoods_folder)
    
    ## Ground Truth Likelihoods
    load_newcode_likelihoods(gt_likelihoods_sources,default_integrand,default_prior,gt_stats,srcfolder=likelihoods_folder)
    
    return sample_stats, gt_stats, labels, limits

def load_neighborhood_stats(samples_source:Literal['centerline_shifts', 'transverse_plane']):
    
    ## One big dict which will hold all the stats for the *generated samples*. Start by populating the test metrics generated during sampling:
    sample_stats:dict[str,pd.DataFrame] = {}
    
    if samples_source == 'centerline_shifts':
        sample_stats = concat_dfdicts(
            get_neighborhood_sample_stats("../neighborhood_sampling/centerline_shifts/"),
            get_neighborhood_sample_stats("../neighborhood_sampling/dense/centerline_shifts/"),
            get_neighborhood_sample_stats("../neighborhood_sampling/centered/centerline_shifts/")
        )
    elif samples_source == 'transverse_plane':
        sample_stats = concat_dfdicts(
            get_neighborhood_sample_stats("../neighborhood_sampling/transverse_plane/"),
            get_neighborhood_sample_stats("../neighborhood_sampling/dense/transverse_plane/"),
            get_neighborhood_sample_stats("../neighborhood_sampling/centered/transverse_plane/")
        )
    else:
        raise ValueError()
    
    # Add rosetta energies
    try:
        if samples_source == 'centerline_shifts':
            rosetta_csv = "../dfmdock_perturb_tr_likelihood/refined_scores_centerline.csv"
        elif samples_source == 'transverse_plane':
            rosetta_csv = "../dfmdock_perturb_tr_likelihood/refined_scores_transplane.csv"
        else:
            raise ValueError()
        
        rosetta_df = pd.read_csv(rosetta_csv)
        for pdb_id in sample_stats:
            iddf = rosetta_df[rosetta_df['pdb_id'] == pdb_id]
            
            #Missing Fnat means scoring failed
            iddf = iddf[iddf['Fnat'].notna()]
        
            #assign values by index. Missing rows will be assigned 'nan', since rosetta doesn't return a value for every structure
            iddf.index = iddf['id']
            sample_stats[pdb_id]['rosetta_Isc'] = iddf['I_sc']
    except FileNotFoundError:
        print("Unable to load rosetta energies from nonexistent file:",rosetta_csv)
        pass
    
    
    ## Same idea as sample_stats, but for the ground truth structures
    gt_stats:dict[str,dict[str,float]] = {}
    for id in sample_stats:
        gt_stats[id] = {'DockQ': 1, 'c_rmsd': 0, 'i_rmsd': 0, 'l_rmsd': 0, 'fnat': 1, 'num_clashes': 0} #some dummy data
    
    # Use original gt_results' rosetta energy since it's independent of samples and method
    with open('../dfmdock_perturb_tr_likelihood/dfmdock_perturb_tr_likelihood_gt.pkl', 'rb') as f:
        old_gt_results = pickle.load(f)
    for id in sample_stats:
        gt_stats[id]['rosetta_Isc'] = old_gt_results['rosetta'][id]
    
    
    # Dict to hold: 1) the display labels for each inference type, and 2) the axis limits (optional) for each display label
    labels:dict[str,str|None] = DefaultDict(lambda: None, **METRIC_LABELS)
    limits:dict[str,tuple[float|None,float|None]|None] = DefaultDict(lambda: None, **METRIC_LIMITS)
    
    
    ### Add Calculated Likelihoods

    ## New Code
    labels.update(NEWCODE_LABELS)
    limits['flow_nll'] = (1,6)
    
    default_integrand = 'TotalIntegrand'
    default_prior = 'smax_gaussian'
    
    #add likelihoods from standard sources where available
    srcfolder = Path("../likelihood_results/likelihoodv3/dfmdock_trtrained_deterministic/neighborhood/")
    if samples_source == 'centerline_shifts':
        sources = {'flow_nll': ('centerline_flow_40',), 'forwardsde_nll': ('centerline_forward_sde',)}
    elif samples_source == 'transverse_plane':
        sources = {'flow_nll': ('transverse_plane_flow_40','transverse_plane_centered_flow_40','transverse_plane_dense_flow_40'), 'forwardsde_nll': ('transverse_plane_forward_sde',)}
    else:
        raise ValueError()
    
    load_newcode_likelihoods(sources,default_integrand,default_prior,sample_stats,srcfolder=srcfolder)

    #add any misc likelihoods here / override additions from above

    return sample_stats, gt_stats, labels, limits


def load_newcode_likelihoods(sources:Mapping[str,str|Path|tuple[str|Path,...]],integrand:str,prior:str,
                             sample_stats:MutableMapping[str,pd.DataFrame]|MutableMapping[str,dict[str,float]],srcfolder:Path=Path(".")):
    for key,srcs in sources.items():
        if isinstance(srcs,str|Path):
            srcs = [srcs]
        for src in srcs:
            try:
                nlls,priors = get_likelihoods(srcfolder/src,integrand=integrand,prior=prior)
                insert_dfdict(sample_stats,key,nlls)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"missed source: {src}")
                pass

def plot_samples(sample_results:Mapping[str,pd.DataFrame], gt_results:Optional[Mapping[str,dict[str,float]]],
                  id:str, xtype:str, ytype:str,
                  ctype:Optional[str]=None, ztype:Optional[str]=None,
                  ax:Optional[Axes]=None, #TODO: also add subfigure support? not sure if necessary
                  axtitle:Optional[str]=None,
                  custom_xlabel:Optional[str]=None, custom_ylabel:Optional[str]=None,
                  custom_xlim:Optional[tuple[float|None,float|None]]=None, custom_ylim:Optional[tuple[float|None,float|None]]=None,
                  #plot_gt = 'if_present' will try to plot ytype (or gt_ytype if specified) from the gt_results dict if the key is present, but will not error if it's not.
                  # plot_gt = True will raise an error if the key is not present in gt_results; plot_gt = False prevents plotting altogether.
                  plot_gt:bool|Literal['if_present']='if_present', gt_ytype:Optional[str]=None,
                  markersize=30,
                  save=False, out_file:str|Path='dfmdock_sample_plots/gridplot.png'):
    if ax is None: ax = plt.gca()
    idx_to_keep = (sample_results[id][xtype].notna() & sample_results[id][ytype].notna())
    if 'rosetta_Isc' in sample_results[id].columns:
        idx_to_keep &= (sample_results[id]['rosetta_Isc'] > -10000)# & (_results[_id]['flow_nll' if not old_results else 'flowtime_nll'] < 6.4)

    if ctype is None:
        colors = ['gray' if dockq < 0.23 
                else 'tab:blue' if dockq < 0.49
                else 'goldenrod' if dockq < 0.80
                else 'tab:red' for dockq in sample_results[id]['DockQ'][idx_to_keep]]
    else:
        colors = sample_results[id][ctype][idx_to_keep]

    if not ztype:
        ax.scatter(sample_results[id][xtype][idx_to_keep], sample_results[id][ytype][idx_to_keep], s=markersize, edgecolors='k', linewidths=1, c=colors)
    else:
        ax.scatter(sample_results[id][xtype][idx_to_keep], sample_results[id][ytype][idx_to_keep], sample_results[id][ztype][idx_to_keep], s=markersize, edgecolors='k', linewidths=1, c=colors)

    if gt_results is not None and plot_gt != False:
        yt = gt_ytype or ytype
        if (plot_gt == 'if_present' and xtype in gt_results[id] and yt in gt_results[id] and (ztype is None or ztype in gt_results[id])) or plot_gt == True:
            if not ztype:
                ax.scatter(gt_results[id][xtype], gt_results[id][yt], s=100, c='yellow', marker='*', edgecolors='k', linewidths=1) 
            else:
                ax.scatter(gt_results[id][xtype], gt_results[id][yt], gt_results[id][ztype], s=100, c='yellow', marker='*', edgecolors='k', linewidths=1) 

    if axtitle: ax.set_title(axtitle)

    ax.set_xlabel(custom_xlabel or xtype)
    ax.set_ylabel(custom_ylabel or ytype)
    if custom_xlim: ax.set_xlim(*custom_xlim)
    if custom_ylim: ax.set_ylim(*custom_ylim)
    if save:
        plt.savefig(out_file,dpi=300,bbox_inches='tight')
    return ax


def make_gridplot(sample_results:Mapping[str,pd.DataFrame],gt_results:Optional[Mapping[str,dict[str,float]]],
                  xtype:str,
                  ytype:str,
                  ctype:Optional[str]=None, ztype:Optional[str]=None,
                  custom_xlabel:Optional[str]=None, custom_ylabel:Optional[str]=None,
                  custom_xlim:Optional[tuple[float|None,float|None]]=None,custom_ylim:Optional[tuple[float|None,float|None]]=None,
                  #plot_gt = 'if_present' will try to plot ytype (or gt_ytype if specified) from the gt_results dict if the key is present, but will not error if it's not.
                  # plot_gt = True will raise an error if the key is not present in gt_results; plot_gt = False prevents plotting altogether.
                  plot_gt:bool|Literal['if_present']='if_present',gt_ytype:Optional[str]=None, 
                  skip_plot_if_missing:bool=True, #if true, will simply skip individual plots whose x or y value is not present in the sample_results dict. Otherwise, will error.
                  save=False,out_file:str|Path='gridplots/gridplot.png'):
    
    fig, axes = plt.subplots(5, 5, figsize=(14, 14),subplot_kw={'projection':'3d' if ztype else None})
    axdict = {}
    for i, _id in enumerate(sorted(sample_results.keys())):
        if skip_plot_if_missing and (xtype not in sample_results[_id].columns or ytype not in sample_results[_id].columns): print(f"skipping {_id}"); continue
        ax = axes[i // 5, i % 5]
        plot_samples(sample_results,gt_results,
                     _id,xtype,ytype,ctype,ztype,
                     ax,_id,
                     custom_xlabel,custom_ylabel,
                     custom_xlim,custom_ylim,
                     plot_gt,gt_ytype,
                     save=False
                     )
        axdict[_id] = ax
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.yaxis.set_tick_params(which='both', labelbottom=True)
        
            
    plt.tight_layout()
    if save:
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True,parents=True)
        plt.savefig(out_file, dpi=300, bbox_inches='tight')

    return fig,axes,axdict


def plot_all_grids(sample_stats:Mapping[str,pd.DataFrame],gt_stats:Mapping[str,dict[str,float]],labels:Mapping[str,str|None],limits:Mapping[str,tuple[float|None,float|None]|None],save:bool=True,outdir:Path|None=None,close_plots:bool=False,
                   y_stats='auto',x_stats='auto',extra_pairs:list[tuple[str,str]]=[]):
    if save and outdir is None:
        raise ValueError("Must specify an output directory if saving gridplots! Standard destination is f\'gridplots/{samples_source}'")

    xs = []
    ys = []

    if y_stats == 'auto' or x_stats == 'auto':
        allstats = list(np.unique([k for ks in sample_stats.values() for k in ks.columns]))
        exclude = ('pdb_id','Filename','index','sample_index','Basis File','Plane Offset','X','Y')
        allstats = [a for a in allstats if a not in exclude]

        ys = [f for f in allstats if f.endswith('nll')]
        xs = list(set(allstats).difference(ys)) #these are interesting both to compare with the nll directly and to act as a proxy for nll when comparing to the other metrics

        excludex = ('num_clashes','c_rmsd','fnat') #uninteresting metrics, take up lots of space in directories
        xs = [x for x in xs if x not in excludex]

        if 'dfmdock_energy' in allstats: ys.append('dfmdock_energy')
        if 'rosetta_Isc' in allstats: ys.append('rosetta_Isc')

    if y_stats != 'auto': ys = y_stats
    if x_stats != 'auto': xs = x_stats

    pairs = list(itertools.product(xs,ys))

    pairs.extend(extra_pairs)

    
    for x,y in tqdm(pairs):
        if x == y: continue
        print(x,y)
        f,axes,axdict = make_gridplot(sample_stats,gt_stats,
                  x,y,
                  custom_xlabel=labels[x],custom_ylabel=labels[y],
                  custom_xlim=limits[x],custom_ylim=limits[y],
                  save=True,out_file=Path(outdir)/f'{y}_vs_{x}.png')
        if close_plots:
            plt.close(f)
            del axes
            del axdict
            import gc
            gc.collect()
        

if __name__ == "__main__":
    source_type:Literal['dfmdock','neighborhood'] = 'neighborhood'
    if source_type == 'dfmdock':
        #Literal['darren_inference', 'dfmdock_tr_inference', 'dfmdock_inference_trtrained_deterministic']
        samples_source = 'dfmdock_inference_trtrained_deterministic'
        sample_stats, gt_stats, labels, limits = load_dfmdock_stats(samples_source)
    elif source_type == 'neighborhood':
        #Literal['centerline_shifts', 'transverse_plane']
        samples_source = 'centerline_shifts'
        sample_stats, gt_stats, labels, limits = load_neighborhood_stats(samples_source)

    plot_all_grids(sample_stats,gt_stats,labels,limits,save=False,outdir=Path(f'gridplots/{samples_source}/'))