from pathlib import Path
from typing import Any, Iterable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from shared import setfont

oranges = (0.99,0.68,0.42,1.0), (0.99,0.55,0.24,1.0), (0.85,0.41,0.07,1.0)
pinks = (0.73,0.54,1.0,1.0), (0.63,0.65,1.0,1.0), (0.80,0.7,0.95,1.0)
greens = (0.13,0.54,0.13,1.0), (0.23,0.64,0.23,1.0), (0.03,0.44,0.03,1.0)
blues = (0.28,0.51,0.71,1.0), (0.38,0.61,0.81,1.0), (0.18,0.41,0.61,1.0)

def get_traj_likelihoods(folders):
    if isinstance(folders,Path|str):
        folders = (folders,)

    resdict = {}
    for folder in folders:
        single_traj_likelihoods = pd.read_csv(Path(folder)/'likelihood.csv')

        expanded_id = single_traj_likelihoods['id'].str.rsplit("_",n=1,expand=True)
        
        times = expanded_id[1].astype(float)
        ids = expanded_id[0]
        single_traj_likelihoods['id'] = ids
        single_traj_likelihoods['Timestep'] = times
        single_traj_likelihoods['nll'] = -(single_traj_likelihoods['prior:smax_gaussian'] + single_traj_likelihoods['integrand:TotalIntegrand'])/3
        
        single_traj_likelihoods = single_traj_likelihoods.iloc[::-1,:]

        all_ids = np.unique(ids)

        for id in all_ids:
            resdict[id] = single_traj_likelihoods[single_traj_likelihoods['id']==id].reset_index()
        
    return resdict

def read_forces(forces_folders):
    if isinstance(forces_folders,Path|str):
        forces_folders = (forces_folders,)
    traj_force_dict = {}
    for forces_folder in forces_folders:
        forces_folder = Path(forces_folder)
        index = pd.read_csv(forces_folder/'force_index.csv')
        
        cols = ["Offset_Tr_X", "Offset_Tr_Y", "Offset_Tr_Z"]
        score_cols = [f"score:{col}" for col in cols]
        pos_cols = [f"pos:{col}" for col in cols]
        
        for row in index.itertuples():
            id = row.id
            file = row.Forces_CSV
            forcedict = traj_force_dict[id] = {}
            forces = pd.read_csv(forces_folder/'forces'/file)
            forcedict['id'] = id
            forcedict['time'] = np.array(forces['Timestep'].values[::-1])
            forcedict['score'] = np.array(forces[score_cols].values[::-1])
            forcedict['divergence'] = np.array(forces['Divergence'][::-1])
            forcedict['diff_coeff'] = np.array(forces['Diffusion_Coeff'][::-1])
            forcedict['pos'] = np.array(forces[pos_cols].values[::-1])
    return traj_force_dict

def plot_likelihood(likelihood_df:pd.DataFrame,color,name=None):
    plt.plot(likelihood_df.Timestep,likelihood_df.nll, marker='o', linestyle='-', markersize=1.5, alpha=0.8, linewidth=0.7, c=color, label=name)

def plot_forces_xyz(id_dict:dict[str,Any],colors,scaled=False,name=None):
    scale = -1/2*np.squeeze(id_dict['diff_coeff'])**2 if scaled else 1
    plt.plot(id_dict['time'], scale*id_dict['score'][:,0], marker='o', linestyle='-', c=colors[0], markersize=1.5, alpha=0.8, linewidth=0.7,label=f"{name} x" if name else name)
    plt.plot(id_dict['time'], scale*id_dict['score'][:,1], marker='s', linestyle='-', c=colors[1], markersize=1.5, alpha=0.8, linewidth=0.7,label=f"{name} y" if name else name)
    plt.plot(id_dict['time'], scale*id_dict['score'][:,2], marker='v', linestyle='-', c=colors[2], markersize=1.5, alpha=0.8, linewidth=0.7,label=f"{name} z" if name else name)

def plot_forces_mag(id_dict:dict[str,Any],color,scaled=False,name=None):
    scale = -1/2*np.squeeze(id_dict['diff_coeff'])[:,None]**2 if scaled else 1
    plt.plot(id_dict['time'],np.linalg.norm(scale*id_dict['score'],axis=1),marker='o', linestyle='-', c=color, markersize=1.5, alpha=0.8, linewidth=0.7,label=name)

def plot_divergence(id_dict:dict[str,Any],color,scaled=False,name=None):
    scale = -1/2*np.squeeze(id_dict['diff_coeff'])**2 if scaled else 1
    plt.plot(id_dict['time'],scale*np.squeeze(id_dict['divergence']),marker='o', linestyle='-', c=color, markersize=1.5, alpha=0.8, linewidth=0.7,label=name)


#plotting functions
def plot_traj_likelihoods(id:str,trajectories:Iterable[str],outdir:Path,trajdict:dict[str,pd.DataFrame],colors:Iterable[Any]=(greens[0],blues[0])):
    fig = plt.figure()

    for traj,color in zip(trajectories,colors):
        plot_likelihood(trajdict[traj], color=color)

    plt.xlabel('Diffusion time $t$')
    plt.ylabel(r'Learned Energy $-\log\,p_{0}(x_{t})$')
    plt.ylim(1, 7)
    fig.set_size_inches(2.85, 2)
    fig.set_dpi(300)
    plt.savefig(outdir/f'{id}_trajectories.png', dpi=300, bbox_inches='tight')

def plot_traj_forces_mag(id:str,trajectories:Iterable[str],outdir:Path,trajdict:dict[str,dict[str,Any]],colors:Iterable[Any]=(greens[0],blues[0]),scaled=False):
    fig = plt.figure()
    for traj,color in zip(trajectories,colors):
        plot_forces_mag(trajdict[traj], color=color, scaled=scaled)
        
    if scaled:
        plt.ylabel(r'$||-\frac{1}{2}g(t)^2 \nabla_x \log p(x_{t},t)||$')
    else:
        plt.ylabel(r'$||s_{\theta} (x_t,t)||=||\nabla_x \log p(x_{t},t)||$')

        
    plt.xlabel('Diffusion time $t$')

    fig.set_size_inches(2.85, 1.9)
    fig.set_dpi(300)
    plt.savefig(outdir/f"{id}_single_traj{'_scaled' if scaled else ''}_force_mag.png", dpi=300, bbox_inches='tight')
    
def plot_traj_forces_xyz(id:str,trajectories:Iterable[str],outdir:Path,trajdict:dict[str,dict[str,Any]],colors:Iterable[Any]=(greens,blues),scaled=False):
    fig = plt.figure()
    for traj,color in zip(trajectories,colors):
        plot_forces_xyz(trajdict[traj], colors=color, scaled=scaled)
    
    plt.plot(np.linspace(0,1,100),np.zeros(100),linestyle='--',color='gray',linewidth=0.8)
    if scaled:
        plt.ylabel(r'Scaled score $\tilde{s}_{\theta} (x_t,t)$')
        # plt.ylabel(r'$\tilde{s}_{\theta} (x_t,t) = -\frac{1}{2}g(t)^2 \nabla_x \log p(x_t,t)$',fontdict={'size':'small'})
    else:
        plt.ylabel(r'Score $s_{\theta} (x_t,t)$')
        # plt.ylabel(r'$s_{\theta} (x_t,t)=\nabla_x \log p (x_t, t)$')
    plt.xlabel('Diffusion time $t$')

    fig.set_size_inches(2.85, 1.9)
    fig.set_dpi(300)
    plt.savefig(outdir/f"{id}_single_traj{'_scaled' if scaled else ''}_force_xyz.png", dpi=300, bbox_inches='tight')
    
def plot_traj_divergence(id:str,trajectories:Iterable[str],outdir:Path,trajdict:dict[str,dict[str,Any]],colors:Iterable[Any]=(greens[0],blues[0]),scaled=False):
    fig = plt.figure()
    for traj,color in zip(trajectories,colors):
        plot_divergence(trajdict[traj], color=color, scaled=scaled)
    
    plt.plot(np.linspace(0,1,100),np.zeros(100),linestyle='--',color='gray',linewidth=0.8)
    if scaled:
        plt.ylabel(r'$-\frac{1}{2}g(t)^2 \nabla_{x} \cdot \nabla_{x} \log p(x_{t},t)$')
    else:
        plt.ylabel(r'$\nabla_{x} \cdot \nabla_{x} \log p(x_{t},t)$')
    plt.xlabel('Diffusion time $t$')

    fig.set_size_inches(2.85, 1.9)
    fig.set_dpi(300)
    plt.savefig(outdir/f"{id}_single_traj{'_scaled' if scaled else ''}_divergence.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    setfont()
    
    # Figure 4 Subfigures
    outdir = Path('figures/figure4_subs/')
    outdir.mkdir(exist_ok=True)
    
    traj_likelihoods_folder = (
        Path('results/likelihood/dfmdock_traj_flow')
    )    
    forces_folder = (
        Path('results/forces/dfmdock'),
    )
    
    traj_likelihoods = get_traj_likelihoods(traj_likelihoods_folder)
    traj_force_dict = read_forces(forces_folder)
    
    ## New Trajectories (generated with deterministic score model)
    trajectories:list[tuple[str,tuple[str,str]]] = [('2SIC',('2SIC_61','2SIC_119')),
                                                    ('2A1A',('2A1A_91','2A1A_103'))]
    
    # from IPython import embed; embed()

    #make subfigure pieces
    plot_traj_likelihoods(*trajectories[0],outdir,traj_likelihoods)
    plot_traj_likelihoods(*trajectories[1],outdir,traj_likelihoods)
    
    plot_traj_forces_xyz(*trajectories[0],outdir,traj_force_dict,scaled=False)
    plot_traj_forces_xyz(*trajectories[0],outdir,traj_force_dict,scaled=True)
    plot_traj_forces_xyz(*trajectories[1],outdir,traj_force_dict,scaled=False)
    plot_traj_forces_xyz(*trajectories[1],outdir,traj_force_dict,scaled=True)
    
    plot_traj_forces_mag(*trajectories[0],outdir,traj_force_dict,scaled=False)
    plot_traj_forces_mag(*trajectories[0],outdir,traj_force_dict,scaled=True)
    plot_traj_forces_mag(*trajectories[1],outdir,traj_force_dict,scaled=False)
    plot_traj_forces_mag(*trajectories[1],outdir,traj_force_dict,scaled=True)
    
    plot_traj_divergence(*trajectories[0],outdir,traj_force_dict,scaled=False)
    plot_traj_divergence(*trajectories[0],outdir,traj_force_dict,scaled=True)
    plot_traj_divergence(*trajectories[1],outdir,traj_force_dict,scaled=False)
    plot_traj_divergence(*trajectories[1],outdir,traj_force_dict,scaled=True)
    
    
