from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
import numpy as np

from scipy.stats import norm

from diffenergy.helper import marginal_kernel_var
from figures.shared import get_timed_gt_gaussian, marginal_prob_std




try:
    from shared import setfont
    from figure3 import plot_sample_result, load_sample_likelihoods
except ImportError:
    from figures.shared import setfont
    from figures.figure3 import plot_sample_result, load_sample_likelihoods


if __name__ == "__main__":
    setfont()

    # Figure 3
    likelihood_dir = Path('results/likelihood/noise_schedule_tests')

    # #going to try using subfigures. Hope this works!
    # def add_comb_row(fig:SubFigure, parent_folder:Path, sigma_min: float, sigma_max:float, diff_subfolder:Path, flow_subfolder:Path, integrand:str, prior:str, title:str, label:str):
    #     parent_folder = Path(parent_folder)
    #     priorax,sampax = fig.subplots(nrows=1,ncols=2)

    #     #plot prior
    #     x = np.linspace(-60,60,500)

    #     sigma = np.sqrt(int_diffusion_coeff_sq(1, sigma_min, sigma_max))
    #     assumed_smax_prior = norm.pdf(x, 0, sigma)  # Zero mean, std = sigma_t1

    #     #plot samples
    #     #plot diffusion samples & GT
    #     plot_sample_result(parent_folder,diff_subfolder,integrand,prior,binline=False,plot_kde=False,
    #                     ax=sampax,save=False,fig_title=False,ax_title=False,plot_p0_gt=True,plot_p1_gt=False)
    #     #plot flow samples
    #     samples, flow_likelihoods = load_sample_likelihoods(parent_folder/flow_subfolder,integrand_column=integrand,prior_column=prior)
    #     flow_probability = np.exp(flow_likelihoods) #this was negative likelihoods in the old data since they calculated nll instead of ll
    #     sampax.scatter(samples, flow_probability, color='cyan', label="Flow", s=2, linewidth=0.1, edgecolors=(0.1, 0.1, 0.1), alpha=0.8, zorder=10)
        
    #     fig.suptitle(title)
    #     #add subfigure label
    #     fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})

    #going to try using subfigures. Hope this works!
    def add_comb_row(fig:SubFigure, parent_folder:Path, sigma_min: float, sigma_max:float, integrand:str, title:str, label:str, *subfolders:tuple[str,Path], assumed_prior:str="prior:smax_gaussian", gt_prior:str="prior:convolved_data", data_variance:float|None=None):
        parent_folder = Path(parent_folder)
        axs:list[Axes] = fig.subplots(nrows=1,ncols=1+len(subfolders),squeeze=True)
        priorax,*sampaxs = axs

        #plot prior
        x = np.linspace(-120,120,500)

        for priortype, prior in zip(("assumed","ground truth"),(assumed_prior,gt_prior)):
            match prior:
                case "prior:smax_gaussian":
                    sigma = marginal_prob_std(1, sigma_min, sigma_max)
                    prior_pdf = norm.pdf(x, 0, sigma)  # Zero mean, std = sigma_t1
                case "prior:estimated_data":
                    assert data_variance is not None
                    sigma = np.sqrt(marginal_kernel_var(1, sigma_min, sigma_max) + data_variance)
                    prior_pdf = norm.pdf(x, 0, sigma)  # Zero mean, std = sigma_t1
                case "prior:convolved_data":
                    prior_pdf = get_timed_gt_gaussian(x,[1],sigma_min=sigma_min,sigma_max=sigma_max)
                case _:
                    raise ValueError()
            priorax.plot(x,prior_pdf,label=f"{priortype} prior")
        
        priorax.legend(fontsize=8)
        priorax.set_title("Prior")

        #plot samples
        for sampax, (name,subfolder) in zip(sampaxs,subfolders):
            #format plot, plot GT
            plot_sample_result(parent_folder,subfolder,integrand,"prior:smax_gaussian",binline=False,plot_kde=True,plot_samples=False,
                            ax=sampax,save=False,fig_title=False,ax_title=False,plot_p0_gt=True,plot_p1_gt=False,compute_kde=False)
            
            priornames = {"prior:smax_gaussian":"Assumed Prior",
                          "prior:estimated_data":"Estimated Prior",
                          "prior:convolved_data":"True Prior"}

            #plot assumed prior samples
            samples, assumed_prior_likelihoods = load_sample_likelihoods(parent_folder/subfolder,integrand_column=integrand,prior_column=assumed_prior)
            assumed_prior_probability = np.exp(assumed_prior_likelihoods)
            sampax.scatter(samples, assumed_prior_probability, color='salmon', label=priornames[assumed_prior], s=2, linewidth=0.1, edgecolors=(0.1, 0.1, 0.1), alpha=0.8,zorder=-3)

            #plot GT prior samples
            samples, convolved_prior_likelihoods = load_sample_likelihoods(parent_folder/subfolder,integrand_column=integrand,prior_column=gt_prior)
            convolved_prior_probability = np.exp(convolved_prior_likelihoods)
            sampax.scatter(samples, convolved_prior_probability, color='cyan', label=priornames[gt_prior], s=2, linewidth=0.1, edgecolors=(0.1, 0.1, 0.1), alpha=0.8, zorder=-2)

            # sampax.legend(fontsize=8,numpoints=10)
            sampax.set_title(name)
        
        fig.suptitle(title)
        #add subfigure label
        fig.text(.01, .99, label, ha='left', va='top', transform=fig.transSubfigure,fontdict={"fontsize":"large"})

    f = plt.figure(figsize=(8,7),layout='constrained')
    subfs:list[SubFigure] = f.subfigures(nrows=3,wspace=0.5,squeeze=True)


    # add_comb_row(subfs[0],likelihood_dir,0.1,30,Path('gaussian_1d_diff_smax30'),Path('gaussian_1d_flow_smax30'),'integrand:TotalIntegrand','prior:smax_gaussian',r'$\sigma_{max}=30$','(a)')
    # add_comb_row(subfs[1],likelihood_dir,0.1,70,Path('../gaussian_1d_diff'),Path('../gaussian_1d_flow'),'integrand:TotalIntegrand','prior:smax_gaussian',r'$\sigma_{max}=70$','(b)')
    # add_comb_row(subfs[2],likelihood_dir,0.1,70,Path('gaussian_1d_diff_smax30'),Path('gaussian_1d_flow_smax30'),'integrand:TotalIntegrand','prior:convolved_data',r'$\sigma_{max}=30$, "perfect" prior','(c)')
    # add_comb_row(subfs[3],likelihood_dir,0.1,70,Path('../gaussian_1d_diff'),Path('../gaussian_1d_flow'),'integrand:TotalIntegrand','prior:convolved_data',r'$\sigma_{max}=70$, "perfect" prior','(d)')
    
    data_variance = 898.532 #approximate variance of the training dataset

    add_comb_row(subfs[0],likelihood_dir,0.1,30,'integrand:TotalIntegrand',r'$\sigma_{max}=30$','(a)',("Diffusion", Path('gaussian_1d_diff_smax30')),("Flow",Path('gaussian_1d_flow_smax30')))
    add_comb_row(subfs[1],likelihood_dir,30,70,'integrand:TotalIntegrand',r'$\sigma_{min}=30$, $\sigma_{max}=70$','(b)',("Diffusion", Path('gaussian_1d_diff_smin30_smax70')),("Flow", Path('gaussian_1d_flow_smin30_smax70')))
    add_comb_row(subfs[2],likelihood_dir,0.1,30,'integrand:TotalIntegrand',r'$\sigma_{max}=30$ w/ Data Variance','(c)',("Diffusion", Path('gaussian_1d_diff_smax30')),("Flow",Path('gaussian_1d_flow_smax30')),assumed_prior='prior:estimated_data',data_variance=data_variance)
    # add_comb_row(subfs[3],likelihood_dir,0.1,70,'integrand:TotalIntegrand',r'$\sigma_{max}=70$ w/ Data Variance','(d)',("Diffusion",Path('../gaussian_1d_diff')),("Flow",Path('../gaussian_1d_flow')),assumed_prior='prior:estimated_data',data_variance=data_variance)


    # add_comb_row(subfs[2],likelihood_dir,0.1,70,Path('gaussian_1d_diff_smax30'),Path('gaussian_1d_flow_smax30'),'integrand:TotalIntegrand','prior:convolved_data',r'$\sigma_{max}=30$, "perfect" prior','(c)')
    # add_comb_row(subfs[3],likelihood_dir,0.1,70,Path('../gaussian_1d_diff'),Path('../gaussian_1d_flow'),'integrand:TotalIntegrand','prior:convolved_data',r'$\sigma_{max}=70$, "perfect" prior','(d)')

    # add_comb_row(subfs[0],likelihood_dir,Path('gaussian_1d_diff_forward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration','(a)',binline=True)
    # add_comb_row(subfs[1],likelihood_dir,Path('gaussian_1d_diff_forward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Forward Euler Integration,\n10x Interpolated Trajectory','(b)',binline=True)
    # add_comb_row(subfs[2],likelihood_dir,Path('gaussian_1d_diff_backward_euler'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration','(c)',binline=True)
    # add_comb_row(subfs[3],likelihood_dir,Path('gaussian_1d_diff_backward_euler_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Backward Euler Integration,\n10x Interpolated Trajectory','(d)',binline=True)
    # add_comb_row(subfs[4],likelihood_dir,Path('gaussian_1d_diff_trapezoid'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration','(e)',binline=True)
    # add_comb_row(subfs[5],likelihood_dir,Path('gaussian_1d_diff_trapezoid_interpolated'),'integrand:TotalIntegrand','prior:smax_gaussian','Trapezoidal Integration,\n10x Interpolated Trajectory','(f)',binline=True)
    # add_comb_row(subfs[6],likelihood_dir,Path('gaussian_1d_diff_piecewise_ode'),'integrand:TotalIntegrand','prior:smax_gaussian','Piewise ODE Integration','(g)',binline=True)


    f.savefig("figures/supplement_noisy_prior.png",dpi=600)