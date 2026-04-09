import itertools
import re
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from typing import Any, Iterable, Literal, DefaultDict, Mapping, MutableMapping, Optional, TypeVar


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

LIKELIHOOD_LABELS = {"flow_nll": "Learned Energy (Flow)",
                 "diff_nll": "Learned Energy (Diffusion) [Euler]",
                 "diff_piecewise_ode_nll": "Learned Energy (Diffusion) [Piecewise ODE]",
                 "diff_trapezoid_nll": "Learned Energy (Diffusion)",
                 "forwardsde_nll": "Learned Energy (Forward SDE)",
                 "diff_10interp_nll": "Learned Energy (Diffusion, Interpolated 10x)",
                 "diff_50interp_nll": "Learned Energy (Diffusion, Interpolated 50x)"}


def get_likelihoods(likelihood_file:Path|str,
                    integrand:str="TotalIntegrand",prior:str="smax_gaussian",
                    index_col:str='id',id_regex:Optional[str|re.Pattern]=None,sequential_sample_index_check=True)->tuple[dict[str,pd.Series],dict[str,pd.Series]]:
    """
    Read csv file with data about each sample. Returns a tuple of dicts ({pdb_id:nll_array}, {pdb_id:prior_array}); nll_array
    and prior_array are both pd.Series containing learned negative log-likelihood and prior negative log-likelihood values respectively,
    both indexed by the specified by 'index_col'. 
    
    To extract the pdb_id of each sample, either the csv must contain
    a 'pdb_id' column, or the 'id_regex' must be specified: the capturing groups of id_regex will be used to populate
    the pdb_id column (required) and the sample_index column (optional). If named capturing groups are used, 'pdb_id' and
    'sample_index' will be assigned correspondingly; otherwise, the pdb_id and sample_index will be assigned 
    by the first and second groups respectively. The 'sample_index' column represents the sub-index of each sample
    within a particular pdb_id; it will be sorted alphanumerically before returning, and if sequential_sample_index_check == True,
    sample_index will be asserted to be integer-valued and checked to ensure there are no gaps in the index.
    
    :param likelihood_file: Location of likelihood csv file to read
    :type likelihood_file: Path | str
    :param integrand: Name of the integrand to use for delta nll. Will use the f"integrand:{integrand}" column in the csv file.
    default: "TotalIntegrand"
    :type integrand: str
    :param prior: Name of the prior to use. Will use the f"prior:{prior}" column in the csv file.
    default: "smax_gaussian"
    :type prior: str
    :param index_col: Column to use for indexing the csv, default: 'index'
    :type index_col: str
    :param id_regex: Regex to parse 'pdb_id' and (optionally) 'sample_index' from the index column. 
    If not provided, 'pdb_id' MUST be a column of the csv file. default: None
    :type id_regex: Optional[str|re.Pattern]
    :param sequential_sample_index_check: Whether to check that the 'sample_index' column A) exists, B) is integer-valued, and C)
    is sequential (e.g. has no gaps from the smallest to the largest value). default: True
    :type sequential_sample_index_check: bool
    """
    likelihoods_df = load_csv(likelihood_file,index_col=index_col,id_regex=id_regex,sequential_sample_index_check=sequential_sample_index_check)
    
    nlls:dict[str,pd.Series] = {}
    priors:dict[str,pd.Series] = {}

    for pdb_id, df in likelihoods_df.items():
        ## NOTE: / 3 is purely a remnant!!
        nlls[pdb_id] = -(df[f'integrand:{integrand}'] + df[f'prior:{prior}'])/3 
        priors[pdb_id] = -df[f'prior:{prior}']/3
    
    return nlls, priors

def load_csv(samples_csv:str|Path|Iterable[str|Path|pd.DataFrame],index_col:str|list[str]='index',id_regex:Optional[str|re.Pattern]=r'([A-Z0-9]+)_p(\d+)',exclude_cols=[],sequential_sample_index_check=True)->dict[str,pd.DataFrame]:
    """
    Read csv file(s) with data about each sample. Returns a dict of {pdb_id:DataFrame}, where each dataframe
    is indexed by the specified by 'index_col'. To extract the pdb_id of each sample, either the csv must contain
    a 'pdb_id' column, or the 'id_regex' must be specified: the capturing groups of id_regex will be used to populate
    the pdb_id column (required) and the sample_index column (optional). If named capturing groups are used, 'pdb_id' and
    'sample_index' will be assigned correspondingly; otherwise, the pdb_id and sample_index will be assigned 
    by the first and second groups respectively. The 'sample_index' column represents the sub-index of each sample
    within a particular pdb_id; it will be sorted alphanumerically before returning, and if sequential_sample_index_check == True,
    sample_index will be asserted to be integer-valued and checked to ensure there are no gaps in the index.

    If multiple csvs provided, all will be read and merged by 'index_col' before any other operations take place
    
    :param samples_csv: Location of csv file to read
    :type samples_csv: Path|str
    :param index_col: Column(s) to use for (multi-)indexing the csv, default: 'index'
    :type index_col: str
    :param id_regex: Regex to parse 'pdb_id' and (optionally) 'sample_index' from the index column. 
    If not provided, 'pdb_id' MUST be a column of the csv file. default: None
    :type id_regex: Optional[str|re.Pattern]
    :param sequential_sample_index_check: Whether to check that the 'sample_index' column A) exists, B) is integer-valued, and C)
    is sequential (e.g. has no gaps from the smallest to the largest value). default: True
    :type sequential_sample_index_check: bool
    """
    
    samples_df = pd.concat([csv if isinstance(csv,pd.DataFrame) else pd.read_csv(csv,index_col=index_col) for csv in listify(samples_csv)],axis='columns')
    samples_df['index'] = samples_df.index

    if id_regex is not None:
        match_df = samples_df['index'].str.extract(id_regex,expand=True)
        match_df = match_df.rename(columns={0:'pdb_id',1:'sample_index'})

        for key in ['pdb_id','sample_index']:
            if key not in samples_df and key in match_df:
                samples_df[key] = match_df[key]

    assert 'pdb_id' in samples_df, f"column 'pdb_id' not in {samples_csv} and provided regex {id_regex} could not parse index column to produce it!"
    unique_ids = np.unique(samples_df['pdb_id'])

    if sequential_sample_index_check:
        assert 'sample_index' in samples_df
        samples_df['sample_index'] = samples_df['sample_index'].astype(int)

    samples_df = samples_df.sort_values(['pdb_id','sample_index'] if 'sample_index' in samples_df else ['pdb_id'])

    results = {}
    for id in unique_ids:
        iddf = samples_df[samples_df['pdb_id'] == id]
        if sequential_sample_index_check:
            assert np.all(np.diff(iddf['sample_index']) == 1) #make sure no gaps
        results[id] = iddf.drop(columns=exclude_cols)
    return results
    
    

def get_dfmdock_sample_stats(dfmdock_stats_csv:str|Path)->dict[str,pd.DataFrame]:
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

def get_neighborhood_sample_stats(sample_folder:str|Path)->dict[str,pd.DataFrame]:
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

def load_rosetta_isc(rosetta_csv:str|Path)->dict[str,pd.Series]:
    rosetta_df = pd.read_csv(rosetta_csv)
    
    #get only top-scoring structures for each index
    rosetta_df = rosetta_df.sort_values('I_sc',na_position='last').drop_duplicates('id')
    
    rosetta_df.index = rosetta_df['id']
    unique_ids = np.unique(rosetta_df['pdb_id'])
    
    results:dict[str,pd.Series] = {}
    for pdb_id in unique_ids:
        iddf = rosetta_df[rosetta_df['pdb_id'] == pdb_id]
        
        #Missing Fnat means scoring failed
        iddf = iddf[iddf['Fnat'].notna()]
    
        #assign values by index. Missing rows will be assigned 'nan', since rosetta doesn't return a value for every structure
        results[pdb_id] = iddf['I_sc']
    return results


def concat_dfdicts(dicts:Iterable[Mapping[str,pd.DataFrame]],duplicates_keep:Literal['first','last','all',False]='last'):
    """
    Concatenate Dataframes nested in dictionary {str,pd.DataFrame} structure.
    duplicates_keep specifies priority for duplicate indices; False removes all duplicates,
    'all' keeps all duplicates, while 'first' and 'last' give earlier and later entries in 
    'dicts' priority respectively
    
    :param dicts: Dictionary-nested Dataframes to concatenate
    :type dicts: Iterable[dict[str, pd.DataFrame]]
    :param duplicates_keep: Priority for duplicate indices
    :type duplicates_keep: Literal['first', 'last', 'all', False]
    """
    dicts = list(dicts)
    ids = set().union(*(d.keys() for d in dicts))
    dest:dict[str,pd.DataFrame] = {}
    for id in ids:
        to_concat = [d[id] for d in dicts if id in d]
        if len(to_concat) == 0: continue
        c = pd.concat(to_concat)
        if duplicates_keep != 'all': c = c.loc[~c.index.duplicated(keep=duplicates_keep)] #see https://stackoverflow.com/questions/13035764/remove-pandas-rows-with-duplicate-indices#34297689 for performance over drop_duplicates
        dest[id] = c
    return dest

def concat_seridicts(dicts:Iterable[Mapping[str,pd.Series]],duplicates_keep:Literal['first','last','all',False]='last'):
    """
    Concatenate Series nested in dictionary {str,pd.Series} structure.
    duplicates_keep specifies priority for duplicate indices; False removes all duplicates,
    'all' keeps all duplicates, while 'first' and 'last' give earlier and later entries in 
    'dicts' priority respectively
    
    :param dicts: Dictionary-nested Series to concatenate
    :type dicts: Iterable[dict[str, pd.Series]]
    :param duplicates_keep: Priority for duplicate indices
    :type duplicates_keep: Literal['first', 'last', 'all', False]
    """
    dicts = list(dicts)
    ids = set().union(*(d.keys() for d in dicts))
    dest:dict[str,pd.Series] = {}
    for id in ids:
        to_concat = [d[id] for d in dicts if id in d]
        if len(to_concat) == 0: continue
        c = pd.concat(to_concat)
        if duplicates_keep != 'all': c = c.loc[~c.index.duplicated(keep=duplicates_keep)] #see https://stackoverflow.com/questions/13035764/remove-pandas-rows-with-duplicate-indices#34297689 for performance over drop_duplicates
        dest[id] = c
    return dest


def insert_seridict(dest:Mapping[str,pd.DataFrame|MutableMapping[str,Any]],destkey:str,source:Mapping[str,pd.Series]):
    """
    Assign values of a dictionary to a particular key of a doubly-nested mapping. 
    Alternatively, assign members of a dictionary-nested Series to a particular column of a dictionary-nested DataFrame,
    using the indices as a guide.
    
    
    :param dest: Dictionary of mappings into which to assign each corresponding value
    :type dest: Mapping[str, pd.DataFrame | MutableMapping[str, Any]]
    :param destkey: Key to which each value will be assigned
    :type destkey: str
    :param source: Source dictionary from which to take the values
    :type source: Mapping[str, pd.Series]
    """
    for id in source:
        if id in dest:
            d = dest[id]
            if isinstance(d,pd.DataFrame):
                d.loc[source[id].index,destkey] = source[id]
            else:
                d[destkey] = source[id]
        else:
            raise ValueError(f"key {id} not in dest keys: {dest.keys()}")

def merge_dfdicts(dest:Mapping[str,pd.DataFrame|MutableMapping[str,Any]],keys:Optional[Iterable[str]],source:Mapping[str,pd.DataFrame]):
    """
    Like insert_dfdict, but copies specific keys from source to dest
    
    :param dest: Dictionary of mappings into which to assign corresponding values
    :type dest: Mapping[str, pd.DataFrame | MutableMapping[str, Any]]
    :param keys: Key of which values will be copied
    :type keys: Iterable[str]
    :param source: Source dictionary from which to copy the values
    :type source: Mapping[str, pd.DataFrame]
    """
    keys = list(keys) if keys is not None else None
    for id in source:
        if id in dest:
            d = dest[id]
            ks = keys if keys is not None else source[id].columns
            if isinstance(d,pd.DataFrame):
                d.loc[source[id].index,ks] = source[id][ks]
            else:
                for k in ks:
                    d[k] = source[id][k]
        
                

T = TypeVar("T")
def insert_dict(dest:Mapping[str,MutableMapping[str,T]],destkey:str,source:Mapping[str,T]):
    for id in source:
        if id in dest:
            dest[id][destkey] = source[id]

def listify(s:str|Path|Iterable[str|Path|T]):
    return [s] if isinstance(s,str|Path) else s


def load_dfmdock_stats(likelihoods_folder:str|Path|Iterable[str|Path]='results/likelihood/', #parent folder(s) for likelihoods
                       #sample paths
                       sample_likelihoods_sources:Mapping[str,str|Path|Iterable[str|Path]]={
                           'flow_nll':'dfmdock_flow',
                           'diff_nll':'dfmdock_diff'},
                       sample_metrics_csv:str|Path|Iterable[str|Path]='results/sample_results/dfmdock/metrics.csv',
                       sample_rosetta_csv:Optional[str|Path|Iterable[str|Path]]='results/rosetta/dfmdock/refined_score.csv',
                       #gt paths
                       gt_likelihoods_sources:Optional[Mapping[str,str|Path|Iterable[str|Path]]]={'flow_nll':'dfmdock_gt_flow'},
                       gt_metrics_csv:Optional[str|Path|Iterable[str|Path]]=None,
                       gt_rosetta_csv:Optional[str|Path|Iterable[str|Path]]='results/rosetta/dfmdock_gt/refined_score.csv',
                    ):
    """
    Load likelihoods, rosetta energies, and any other provided metrics csv files (containing e.g. DockQ, interface RMSD, etc) into a combined DataFrame
    and group by original pdb id, with different csv files associated with each other via the 'index' column corresponding to the id of the relevant sample.
    Also optionally return a similarly grouped dictionary likelihoods, metrics, and rosetta energies for "ground truth" structures; interprets 'index' column 
    of ground truth csv files as corresponding to the pdb id of samples. Finally, return dictionary of axis labels and limits for use in plotting. 
    Returns a tuple of ``(sample_stats, ground_truth_stats, axis_labels, axis_limits)``.

    :param likelihoods_folder: Parent folder for likelihoods in ``sample_likelihood_sources`` and ``gt_likelihoods_sources``; allows likelihood paths to be specified
        as a relative path instead of absolute.
    :type likelihoods_folder: str|Path|Iterable[str|Path]
    :param sample_likelihoods_sources: A dictionary of {likelihood_key: likelihood_folder(s)} pointing to where likelihoods should be read.
        Computes negative log likelihood [energy] as the negative sum of the ``integrand:TotalIntegrand`` and ``prior:smax_gaussian`` columns of the likelihood csv, 
        and stores the nll in the resulting sample_stats DataFrame with column label ``likelihood_key``.
    :type sample_likelihoods_sources: Mapping[str,str|Path|Iterable[str|Path]]
    :param sample_metrics_csv: Path(s) to additional csv files with stats about the samples to be added to the sample_stats dataframes. Rows will be aligned
        with likelihoods using the ``index`` column, then copied into the sample_stats dataframes. Will preserve original column names from the metrics csv files,
        meaning if there are duplicate column names among csv files, only the *last* (following the order given in ``sample_metrics_csv``) will be kept!
    :type sample_metrics_csv: str|Path|Iterable[str|Path]
    :param sample_rosetta_csv: [Optional] Path(s) to csv files containing rosetta scoring outputs, as generated by ``scripts/rosetta_refine/*``.
        Copies the ``I_sc`` [interface score] column from the rosetta csv into the ``rosetta_Isc`` column of sample_stats dataframes.
    :type sample_rosetta_csv: Optional[str|Path|Iterable[str|Path]], default: None
    :param gt_likelihoods_sources: [Optional] A dictionary of ``{likelihood_key: gt_likelihood_folder(s)}`` pointing to likelihoods of ground truth structures. 
    Use the same ``likelihood_key`` as in sample_likelihoods_sources to label that they were computed with the same method.
    :type gt_likelihoods_sources: Optional[Mapping[str,str|Path|Iterable[str|Path]]], default: None
    :param gt_metrics_csv: [Optional] Path(s) to additional csv files with stats about ground truth structures. Note that unless overwritten by other files 
        (or if none specified), the following stats will be assumed for ground truth structures: ``DockQ = 1``, ``c_rmsd = 0``, ``i_rmsd = 0``, ``l_rmsd = 0``, 
        ``fnat = 1``, and ``num_clashess = 0``.
    :type gt_metrics_csv: Optional[str|Path|Iterable[str|Path]], default: None
    :param gt_rosetta_csv: [Optional] Path(s) to csv files containing rosetta scoring outputs for ground truth structures.
    :type gt_rosetta_csv: Optional[str|Path|Iterable[str|Path]], default: None
    """
    ### Assemble non-likelihood stats

    ## One big dict which will hold all the stats for the *generated samples*. Start by populating the test metrics generated during sampling:
    sample_stats:dict[str,pd.DataFrame] = {}
    sample_stats = concat_dfdicts(load_csv(csv) for csv in listify(sample_metrics_csv))

    ## Add sample rosetta energies
    if sample_rosetta_csv:
        sample_rosetta_isc = concat_seridicts(load_rosetta_isc(csv) for csv in listify(sample_rosetta_csv) if Path(csv).exists())
        insert_seridict(sample_stats,'rosetta_Isc',sample_rosetta_isc)
    

    ## Same idea as sample_stats, but for the ground truth structures. just contains dicts of floats, since only one structure per id and thus no dataframe required
    gt_stats:dict[str,dict[str,float]] = {}
    for id in sample_stats:
        gt_stats[id] = {'DockQ': 1, 'c_rmsd': 0, 'i_rmsd': 0, 'l_rmsd': 0, 'fnat': 1, 'num_clashes': 0} #some dummy data
    
    ## Add actual GT metrics, if specified
    if gt_metrics_csv:
        gt_stats_dicts = concat_dfdicts(load_csv(csv) for csv in listify(gt_metrics_csv))
        for id, df in gt_stats_dicts:
            #store one-row dataframe into dict of scalars
            for column in df.columns:
                gt_stats[id][column] = df[column].item()

    ## Add GT rosetta energies
    if gt_rosetta_csv:
        gt_rosetta_isc = concat_seridicts(load_rosetta_isc(csv) for csv in listify(gt_rosetta_csv) if Path(csv).exists())
        for id,isc in gt_rosetta_isc.items():
            gt_stats[id]['rosetta_Isc'] = isc.item() if len(isc) > 0 else np.nan #get the one value for the one-element series
    
    ## Dict to hold: 1) the display labels for each inference type, and 2) the axis limits (optional) for each display label
    labels:dict[str,str|None] = DefaultDict(lambda: None, **METRIC_LABELS)
    limits:dict[str,tuple[float|None,float|None]|None] = DefaultDict(lambda: None, **METRIC_LIMITS)
    
    ### Add Calculated Likelihoods
    labels.update(LIKELIHOOD_LABELS)
    limits['flow_nll'] = (-1,8)
    limits['diff_10interp_nll'] = (-4,8)
    limits['diff_trapezoid_nll'] = (-4,8)
    limits['diff_piecewise_ode_nll'] = (-4,8)
    
    default_integrand = 'TotalIntegrand' #TotalIntegrand is always right, no need to mess with total vs flow on different trajectories
    default_prior = 'smax_gaussian'

    likelihood_id_regex = r'.*?(\w+)_p?(\d+)'

    ## Sample Likelihoods
    sample_nlls = get_newcode_likelihoods(sample_likelihoods_sources,default_integrand,default_prior,srcfolder=likelihoods_folder,id_regex=likelihood_id_regex)[0]
    for key,nlls in sample_nlls.items():
        insert_seridict(sample_stats,key,nlls)

    ## Ground Truth Likelihoods
    if gt_likelihoods_sources is not None:
        gt_nlls = get_newcode_likelihoods(gt_likelihoods_sources,default_integrand,default_prior,srcfolder=likelihoods_folder,id_regex=r'(.*)',sequential_sample_index_check=False)[0]
        for key,nlls in gt_nlls.items():
            insert_seridict(gt_stats,key,nlls)
    
    return sample_stats, gt_stats, labels, limits


def load_neighborhood_stats(
                        likelihoods_folder:str|Path|Iterable[str|Path]='../likelihood_results/dfmdock_neighborhood/', #parent folder(s) for likelihoods
                        #sample paths
                        sample_likelihoods_sources:Mapping[str,str|Path|Iterable[str|Path]]={},
                        samples_folder:str|Path|Iterable[str|Path]=[], #index and metrics assumed to be 'samples_folder/index.csv' and 'samples_folder/metrics.csv' respectively
                        sample_rosetta_csv:Optional[str|Path|Iterable[str|Path]]=None,

                        #gt paths
                        gt_likelihoods_sources:Optional[Mapping[str,str|Path|Iterable[str|Path]]]=None,
                        #no gt metrics because they're all either 0 or 1
                        gt_rosetta_csv:Optional[str|Path|Iterable[str|Path]]=None,

                        sample_index_regex:Optional[str|re.Pattern]=r'.*?(\w+?)_plane_.*',
                        gt_index_regex:Optional[str|re.Pattern]=r'.*?([a-zA-Z0-9]+)',
                        ):

    ## One big dict which will hold all the stats for the *generated samples*. Start by populating the test metrics generated during sampling:
    sample_stats = concat_dfdicts(load_csv((Path(folder)/'index.csv',Path(folder)/'metrics.csv'),id_regex=sample_index_regex,sequential_sample_index_check=False) for folder in listify(samples_folder)) #load index and metrics together so metrics has 'pdb_id' column

    print(sample_stats.keys())

    ## Add sample rosetta energies
    if sample_rosetta_csv is not None:
        sample_rosetta_isc = concat_seridicts(load_rosetta_isc(csv) for csv in listify(sample_rosetta_csv) if Path(csv).exists()) 
        insert_seridict(sample_stats,'rosetta_Isc',sample_rosetta_isc)
    
    ## Same idea as sample_stats, but for the ground truth structures
    gt_stats:dict[str,dict[str,float]] = {}
    for id in sample_stats:
        gt_stats[id] = {'DockQ': 1, 'c_rmsd': 0, 'i_rmsd': 0, 'l_rmsd': 0, 'fnat': 1, 'num_clashes': 0} #some dummy data
    
    ## Add GT rosetta energies
    if gt_rosetta_csv is not None:
        gt_rosetta_isc = concat_seridicts(load_rosetta_isc(csv) for csv in listify(gt_rosetta_csv) if Path(csv).exists())
        for id,isc in gt_rosetta_isc.items():
            gt_stats[id]['rosetta_Isc'] = isc.item() if len(isc) > 0 else np.nan #get the one value for the one-element series
    
    ## Dict to hold: 1) the display labels for each inference type, and 2) the axis limits (optional) for each display label
    labels:dict[str,str|None] = DefaultDict(lambda: None, **METRIC_LABELS)
    limits:dict[str,tuple[float|None,float|None]|None] = DefaultDict(lambda: None, **METRIC_LIMITS)
    
    ### Add Calculated Likelihoods
    labels.update(LIKELIHOOD_LABELS)
    limits['flow_nll'] = (-1,8)
    limits['diff_10interp_nll'] = (-4,8)
    limits['diff_trapezoid_nll'] = (-4,8)
    limits['diff_piecewise_ode_nll'] = (-4,8)
    
    default_integrand = 'TotalIntegrand' #TotalIntegrand is always right, no need to mess with total vs flow on different trajectories
    default_prior = 'smax_gaussian'

    ## Sample Likelihoods
    sample_nlls = get_newcode_likelihoods(sample_likelihoods_sources,default_integrand,default_prior,srcfolder=likelihoods_folder,id_regex=sample_index_regex,sequential_sample_index_check=False)[0]
    for key,nlls in sample_nlls.items():
        insert_seridict(sample_stats,key,nlls)

    ## Ground Truth Likelihoods
    if gt_likelihoods_sources is not None:
        gt_nlls = get_newcode_likelihoods(gt_likelihoods_sources,default_integrand,default_prior,srcfolder=likelihoods_folder,id_regex=gt_index_regex,sequential_sample_index_check=False)[0]
        for key,nlls in gt_nlls.items():
            insert_seridict(gt_stats,key,nlls)
    
    return sample_stats, gt_stats, labels, limits



def get_newcode_likelihoods(sources:Mapping[str,str|Path|Iterable[str|Path]],integrand:str,prior:str,id_regex:Optional[str|re.Pattern]=None,srcfolder:str|Path|Iterable[str|Path]=Path("."),**load_csv_kwargs):
    newcode_nlls:dict[str,dict[str,pd.Series]] = {}
    newcode_priors:dict[str,dict[str,pd.Series]] = {}
    for key,srcs in sources.items():
        nlls = []
        priors = []
        for srcf in listify(srcfolder):
            for src in listify(srcs):
                try:
                    n,p = get_likelihoods(Path(srcf)/src/'likelihood.csv',integrand=integrand,prior=prior,id_regex=id_regex,**load_csv_kwargs)
                    nlls.append(n)
                    priors.append(p)
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    print(f"missed source: {src}")
                    pass
        newcode_nlls[key] = concat_seridicts(nlls)
        newcode_priors[key] = concat_seridicts(priors)
    return newcode_nlls,newcode_priors

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
        exclude = ('pdb_id','filename','index','sample_index','Basis File','Plane Offset','X','Y')
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