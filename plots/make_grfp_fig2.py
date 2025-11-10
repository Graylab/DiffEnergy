from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
from gridplots import load_dfmdock_stats, plot_all_grids

source_type:Literal['dfmdock','neighborhood'] = 'dfmdock'
if source_type == 'dfmdock':
    #Literal['darren_inference', 'dfmdock_tr_inference', 'dfmdock_inference_trtrained_deterministic']
    samples_source = 'dfmdock_inference_trtrained_deterministic'
    sample_stats, gt_stats, labels, limits = load_dfmdock_stats(samples_source)
# elif source_type == 'neighborhood':
#     #Literal['centerline_shifts', 'transverse_plane']
#     samples_source = 'centerline_shifts'
#     sample_stats, gt_stats, labels, limits = load_neighborhood_stats(samples_source)


## GRFP Figure 2

from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import BboxTransformTo, IdentityTransform
from gridplots import plot_samples


xkey = "i_rmsd"
ykey = "flow_nll"
f,(ax1,ax2) = plt.subplots(ncols=2,figsize=(4,2),layout="constrained",dpi=600)

assert isinstance(ax1,Axes) and isinstance(ax2,Axes)

plot_samples(sample_stats,None,'2SIC',xkey,ykey,markersize=30,ax=ax1,
             custom_xlabel="Interface RMSD (Å)",custom_ylabel="DFMDock Learned Energy",
             custom_xlim=(0,3),custom_ylim=(1,6))

ykey = "rosetta_Isc"
plot_samples(sample_stats,None,'2SIC',xkey,ykey,markersize=30,ax=ax2,
             custom_xlabel="Interface RMSD (Å)",custom_ylabel="Rosetta Energy (REU)",
             custom_xlim=(0,3),custom_ylim=limits[ykey])

# f.suptitle("Energy vs Docking Quality")

f.draw(f.canvas.get_renderer())

bbox1 = ax1.get_tightbbox()
trans1 = BboxTransformTo(bbox1)
f.text(0.05,0.05,r'$\bf{A)}$',transform=trans1,fontdict={'fontsize':'x-large'})

bbox2 = ax2.get_tightbbox()
trans2 = BboxTransformTo(bbox2)
f.text(0.05,0.05,r'$\bf{B)}$',transform=trans2,fontdict={'fontsize':'x-large'})

f.savefig("GRFP_Fig2.png",dpi=600,bbox_inches='tight')
