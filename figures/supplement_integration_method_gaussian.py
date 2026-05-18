from pathlib import Path
import string

from matplotlib import pyplot as plt
from matplotlib.figure import SubFigure

from figures.shared import setfont
from figures.supplement_integration_method import add_comb_row, compare_likelihoods


if __name__ == "__main__":
    setfont()

    # Figure 3
    likelihood_dir = Path('results/likelihood/diff_integration_methods_smin30')

    #going to try using subfigures. Hope this works!







    f = plt.figure(figsize=(10,10),layout='constrained')
    subf_rows:list[SubFigure] = f.subfigures(nrows=5,ncols=1,height_ratios=[1,1,1,1,1.1])
    subfs_grid = [subf.subfigures(nrows=1,ncols=2) for subf in subf_rows[:-1]] #make all but the last row two column
    
    labelit = iter([f'({l})' for l in string.ascii_lowercase])

    add_comb_row(subfs_grid[0][0],next(labelit),'Forward Euler Integration',likelihood_dir,Path('gaussian_1d_diff_forward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True, other_corner=True)
    add_comb_row(subfs_grid[1][0],next(labelit),'Backward Euler Integration',likelihood_dir,Path('gaussian_1d_diff_backward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[2][0],next(labelit),'Trapezoidal Integration',likelihood_dir,Path('../noise_schedule_tests/gaussian_1d_diff_smin30_smax70'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[3][0],next(labelit),'Piewise ODE Integration',likelihood_dir,Path('gaussian_1d_diff_piecewise_ode'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[0][1],next(labelit),'Forward Euler Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_forward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[1][1],next(labelit),'Backward Euler Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_backward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[2][1],next(labelit),'Trapezoidal Integration, 10x Interpolated',likelihood_dir,Path('gaussian_1d_diff_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)
    add_comb_row(subfs_grid[3][1],next(labelit),'Flow Trajectory ODE Integration',likelihood_dir,Path('../noise_schedule_tests/gaussian_1d_flow_smin30_smax70'),'integrand:TotalIntegrand','prior:smax_gaussian',binline=True)

    subf_bottom = subf_rows[-1]
    subf_bottom.get_constrained_layout()
    comparison_figs = subf_bottom.subfigures(nrows=1,ncols=5,);
    compit = iter(comparison_figs)
    compare_likelihoods(next(compit),next(labelit),"Forward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_forward_euler','Forward Euler',likelihood_dir/'gaussian_1d_diff_forward_euler_interpolated','Interpolated Forward Euler')
    compare_likelihoods(next(compit),next(labelit),"Backward Euler\nvs Interpolated",likelihood_dir/'gaussian_1d_diff_backward_euler','Backward Euler',likelihood_dir/'gaussian_1d_diff_backward_euler_interpolated','Interpolated Backward Euler',other_corner=True)
    compare_likelihoods(next(compit),next(labelit),"Trapezoidal\nvs Interpolated",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Trapezoidal',likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated Trapezoidal')
    # compare(next(compit),"Trapezoidal\nvs Piecewsise ODE",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Original',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')
    compare_likelihoods(next(compit),next(labelit),"Trapezoidal\nvs Piecewise ODE",likelihood_dir/'..'/'noise_schedule_tests/gaussian_1d_diff_smin30_smax70','Trapezoidal',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')
    compare_likelihoods(next(compit),next(labelit),"Interpolated Trapezoidal\nvs Piecewise ODE",likelihood_dir/'gaussian_1d_diff_interpolated','Interpolated Trapezoidal',likelihood_dir/'gaussian_1d_diff_piecewise_ode','Piecewise ODE')


    f.savefig("figures/supplement_integration_method_gaussian.png",dpi=600)