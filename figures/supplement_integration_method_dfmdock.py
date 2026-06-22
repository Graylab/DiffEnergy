
from pathlib import Path
import string

from matplotlib import pyplot as plt

from figures.shared import setfont
from figures.supplement_integration_method import compare_likelihoods


if __name__ == "__main__":
    setfont()

    f = plt.figure(figsize=(12,2.5),layout='constrained')
    subfs = f.subfigures(ncols=5)

    labelit = iter([f'({l})' for l in string.ascii_lowercase])


    likelihood_dir = Path('results/likelihood')

    subit = iter(subfs)

    compare_likelihoods(next(subit),next(labelit),"Flow (40 steps) vs\nFlow (80 steps)",likelihood_dir/'dfmdock_flow','Trapezoidal',likelihood_dir/'dfmdock_integration_methods/dfmdock_flow_2interp','Flow (80 steps)',prior='prior:receptor_smax_gaussian',lim=(-70,30),ticks=None,exp=False,markersize=10)
    compare_likelihoods(next(subit),next(labelit),"Trapezoidal vs\nPiecewsise ODE (120 steps)",likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_trapezoid','Trapezoidal',likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False,markersize=10)
    compare_likelihoods(next(subit),next(labelit),"Interpolated Trapezoidal\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_trapezoid_interpolated','10x Interpolated Trapezoidal',likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False,markersize=10)
    compare_likelihoods(next(subit),next(labelit),"Pieceiwse ODE (40 steps)\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_diff_piecewise_ode','Piecewise ODE (40 steps)',likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False,markersize=10)
    compare_likelihoods(next(subit),next(labelit),"Piecewise ODE (80 steps)\nvs Piecewsise ODE (120 steps)",likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_piecewise_ode_2interp','Piecewise ODE (80 steps)',likelihood_dir/'dfmdock_integration_methods/dfmdock_diff_piecewise_ode_3interp','Piecewise ODE (120 steps)',prior='prior:receptor_smax_gaussian',lim=None,ticks=None,exp=False,markersize=10)

    f.savefig("figures/supplement_integration_method_dfmdock.png",dpi=600)