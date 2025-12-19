import pandas as pd
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from shared import get_gt_gaussian, get_prior_gaussian, setfont


def get_traj_likelihoods(folder:Path):
    single_traj_likelihoods = pd.read_csv(folder/'likelihood.csv')

    expanded_id = single_traj_likelihoods['id'].str.rsplit("_",n=1,expand=True)
    
    times = expanded_id[1].astype(float)
    ids = expanded_id[0]
    single_traj_likelihoods['id'] = ids
    single_traj_likelihoods['Timestep'] = times
    single_traj_likelihoods['nll'] = -(single_traj_likelihoods['prior:smax_gaussian'] + single_traj_likelihoods['integrand:TotalIntegrand'])/3
    
    single_traj_likelihoods = single_traj_likelihoods.iloc[::-1,:]

    all_ids = np.unique(ids)

    resdict = {}
    for id in all_ids:
        resdict[id] = single_traj_likelihoods[single_traj_likelihoods['id']==id].reset_index()
        
    return resdict

def get_forces(forces_folder:Path):
    index = pd.read_csv(forces_folder/'force_index.csv')
    
    traj_force_dict = {}
    for row in index.itertuples():
        id = row.id
        file:str = row.Forces_CSV
        forcedict = traj_force_dict[id] = {}
        forces = pd.read_csv(file)
        forcedict['id'] = id
        forcedict['time'] = forces['Timestep'].values
        forcedict['score'] = forces[['score']].values
        forcedict['divergence'] = forces['Divergence']
        forcedict['diff_coeff'] = forces['Diffusion_Coeff']
        forcedict['pos'] = forces[['pos']].values

    return traj_force_dict


class TrajectoryIndex:
    def __init__(self,traj_folder:Path,index_file:str|Path,index_offset=0):
        self.traj_folder = traj_folder
        self.index_file = traj_folder/index_file
        index_df = pd.read_csv(self.index_file,index_col="index")
        self.files = index_df.to_dict(orient='index')
        self.offset = 0

    def get(self,id:int):
        fname = self.files[id + self.offset]['filename'] #because of annoying df.to_dict ugh
        traj_df = pd.read_csv(self.traj_folder/fname)
        t = np.array(traj_df['Timestep']); x = np.array(traj_df['Sample'])
        return t,x

    def __getitem__(self,id:int):
        return self.get(id)
        
def get_trajectories(traj_folder:str|Path,index_file:str|Path,index_offset=0): return TrajectoryIndex(Path(traj_folder),index_file,index_offset)

if __name__ == "__main__":
    setfont()
    

    flow_trajectories = get_trajectories('results/likelihood/gaussian_1d_flow/trajectories','trajectory_index.csv')
    diff_trajectories = get_trajectories('results/sample_results/gaussian_1d/trajectories','trajectory_index.csv')
    difftraj_forces = get_forces(Path('results/forces/gaussian_1000'))
    
    from typing import Iterable
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.figure import SubFigure

    plot_trajs = [0,75,50,100]
    paircolors = [ #diff,flow
        ('black',     'dimgray',       ),
        ('steelblue', 'cornflowerblue',),
        ('orange',    'goldenrod',     ),
        ('orangered', 'coral',         ),
    ]
    colors = [p[0] for p in paircolors]

    f = plt.figure(figsize=(4.8,8))
    f.tight_layout()
    subs = f.subfigures(ncols=2,nrows=4,width_ratios=[0.1,1])
    axes:list[Axes] = [s[1].subplots() for s in subs]

    def make_trajrow(ax:Axes,ylabel):
        ax.set_xlim(0,1)
        for traj,(diffcolor,flowcolor) in zip(plot_trajs,paircolors):
            diff = diff_trajectories[traj]
            flow = flow_trajectories[traj]
            ax.plot(diff[0],diff[1],color=diffcolor, alpha=0.8, label=f"Diff Trajectory {traj}")
            ax.plot(flow[0],flow[1],color=flowcolor, linestyle='--', alpha=0.8, label = f"Flow Trajectory {traj}")
        ax.set_xlabel('Diffusion time $t$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # ax.yaxis.tick_right()

    def make_row(ax,plots:Iterable[tuple[Any,Any]],colors,ylabel):
        ax.set_xlim(0,1)
        for (t,val),color in zip(plots,colors):
            ax.plot(t,val,color=color, alpha=0.8)
        ax.set_xlabel('Diffusion time $t$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # ax.yaxis.tick_right()

    make_trajrow(axes[0],'$x_{t}$')
    vrange = (-150,150)
    axes[0].set_ylim(*vrange)

    #now for the vertical distributions
    x = np.linspace(*vrange,10000)
    prior_pdf = get_prior_gaussian(x)
    prior_max = np.max(prior_pdf)
    gt_pdf = get_gt_gaussian(x)
    gt_max = np.max(gt_pdf)

    gt_scale = -0.075
    prior_scale = 0.03
    # axes[0].plot(-gt_scale*gt_pdf/gt_max,x,color=(0.8,0,0.8),zorder=0,clip_on=False)
    # axes[0].fill_betweenx(x,-gt_scale*gt_pdf/gt_max,0,color=(0.8,0,0.8),alpha=0.6,zorder=0,clip_on=False)
    # axes[0].plot(1.0+prior_scale*prior_pdf/prior_max,x,color='green',alpha=0.5,zorder=0,clip_on=False)
    # axes[0].fill_betweenx(x,1.0+prior_scale*prior_pdf/prior_max,1.0,color='green',alpha=0.2,zorder=0,clip_on=False)

    axes[0].plot(-gt_scale*gt_pdf/gt_max,x,color='black',alpha=0.9,zorder=0,clip_on=False)
    axes[0].fill_betweenx(x,-gt_scale*gt_pdf/gt_max,0,color='gray',alpha=0.6,zorder=0,clip_on=False)
    axes[0].plot(1.0+prior_scale*prior_pdf/prior_max,x,color='black',alpha=0.9,linestyle='-',zorder=0,clip_on=False)
    axes[0].fill_betweenx(x,1.0+prior_scale*prior_pdf/prior_max,1.0,color='gray',alpha=0.6,zorder=0,clip_on=False)
    # axes[0].tick_params(axis='y', which='major', pad=15)

    #push labels out and draw fake axes boundaries
    # left,right = (-gt_scale*1.15,1+prior_scale*1.15)
    # axes[0].plot([left,left],[*vrange],color='black',zorder=0,clip_on=False,linewidth=0.9)
    # axes[0].plot([right,right],[*vrange],color='black',zorder=0,clip_on=False,linewidth=0.9)
    # axes[0].plot([left,right],[vrange[0],vrange[0]],color='black',zorder=0,clip_on=False,linewidth=0.9)
    # axes[0].plot([left,right],[vrange[1],vrange[1]],color='black',zorder=0,clip_on=False,linewidth=0.9)

    # axes[0].tick_params(axis='y', which='major', pad=25)
    # #move the ticks lol
    # for tickline in axes[0].get_yticklines():
    #     print(tickline.get_linewidth())
    #     tickline.set_xdata(np.array(tickline.get_xdata())+left) #perhaps?


    # plot x centered on its t0 point
    # make_row(axes[1],[(diff_trajectories[i][0],diff_trajectories[i][1] - diff_trajectories[i][1][0]) for i in plot_trajs],colors,'$x_{t}-x_{0}$')


    make_row(axes[1],[(difftraj_forces[id]['time'],difftraj_forces[id]['score']) for id in plot_trajs],colors,
            #  r'$s_{\theta} (x_t,t)=\nabla_{x} \log p(x_{t},t)$')
            r'Score $s_{\theta}(x_t,t)$')
    axes[1].set_ylim(-0.4,0.4)



    make_row(axes[2],[(difftraj_forces[id]['time'],-1/2*(difftraj_forces[id]['diff_coeff']**2)*difftraj_forces[id]['score'][:,0]) for id in plot_trajs],colors,
            r'Scaled score $\tilde{s}_{\theta}(x_t,t)$')
            #  r'$-\frac{1}{2}g(t)^2 \nabla_{x} \log p_{t}(x_{t})$')
    # axes[2].set_ylim(-500,500)

    # make_row(axes[3],[(diff_trajectories[traj][0],-np.log(get_timed_gt_gaussian(diff_trajectories[traj][1],diff_trajectories[traj][0]))) for traj in plot_trajs],
    #          colors,r'$-\log p^{GT}(x_t,t)$')

    make_row(axes[3],[(diff_trajectories[traj][0],-np.log(get_gt_gaussian(diff_trajectories[traj][1]))) for traj in plot_trajs],colors,
            r'$-\log p_0^{GT}(x_t)$')


    #add labels
    for (s,_),label in zip(subs,("(a)","(b)","(c)","(d)")):
        assert isinstance(s,SubFigure)
        s.text(.025, .95, label, ha='left', va='top', transform=s.transSubfigure,fontdict={"fontsize":"large"})

    # make_row(axes[2],[diff_trajectories[i] for i in plot_trajs],colors,'dum')

    f.savefig('figures/figure_2.png',dpi=600,bbox_inches='tight')