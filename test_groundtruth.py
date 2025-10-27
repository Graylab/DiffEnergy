import functools
from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval
from diffenergy.gaussian_1d.network import ScoreNetMLP
from diffenergy.helper import prior_gaussian_1d
from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel
from scipy.stats import norm, laplace

from diffenergy.helper import int_diffusion_coeff_sq, marginal_prob_std


def get_gaussian(x,means:Iterable,sigmas:Iterable,weights:Iterable):
    return sum([weight*norm.pdf(x,loc=mean,scale=sigma) for (mean,sigma,weight) in zip(means,sigmas,weights)])
    
    # return w1 * norm.pdf(x, loc=mu1, scale=sigma1) + w2 * norm.pdf(x, loc=mu2, scale=sigma2) + w3 * norm.pdf(x, loc=mu3, scale=sigma3)

def get_convolved_gaussian(x,t,means,sigmas,weights,sigma_min=0.1,sigma_max=30):
    gauss = get_gaussian(x,means,sigmas,weights,)
    if t == 0:
        return gauss
    kernel = norm.pdf(x,loc=0,scale=np.sqrt(int_diffusion_coeff_sq(t,sigma_min,sigma_max).numpy()))
    return np.convolve(gauss,kernel,mode='same') / np.sum(kernel)

if __name__ == "__main__":
    plt.close('all')
    x = np.linspace(-100, 100, 800)
    time = 1.0

    sigma_min = 0.1
    sigma_max = 22.5

    means = torch.tensor([-30.0,0.0,40.0])
    sigmas = torch.tensor([8.0,5.0,10.0])
    weights = torch.tensor([0.4,0.3,0.3])
    # means = [0,15]
    # sigmas = [2,2]
    # weights = [1,1]

    t_sigmas = np.sqrt(np.array(sigmas,dtype=float)**2 + int_diffusion_coeff_sq(time,sigma_min=sigma_min,sigma_max=sigma_max).numpy())
    print(f"{t_sigmas=}, {t_sigmas**2=}")

    pdf = get_convolved_gaussian(x,time,means,sigmas,weights,sigma_min=sigma_min,sigma_max=sigma_max)
    pdf2 = get_gaussian(x,means, t_sigmas, weights)

    means = torch.as_tensor(means)[:,None]
    variances = torch.as_tensor(sigmas)**2
    # variances = torch.diag_embed(variances[:,None])
    gt_model_eval = MultimodalGaussianGroundTruthScoreModel(means,variances,weights,sigma_min,sigma_max)



    dpdf = np.gradient(pdf,x)
    logpdf = np.log(pdf)
    dlogpdf = np.gradient(logpdf,x)





    weights_path = 'checkpoints/1d_trinormal.ckpt'
    score_model_ckpt = torch.load(weights_path, map_location = 'cuda')
    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in score_model_ckpt.keys()):
        score_model_ckpt = {key.replace("module.", ""): value for key, value in score_model_ckpt.items()}

    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)
    score_model = ScoreNetMLP(
                input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to('cuda')
    score_model.load_state_dict(score_model_ckpt)
    model_eval = ModelEval(score_model)


    xt_cuda = torch.as_tensor(x,device='cuda',dtype=torch.float)[:,None] #make 1d not 0d
    xt_cpu = torch.as_tensor(x,device='cpu',dtype=torch.float)[:,None] #make 1d not 0d
    gt_model_eval.to(dtype=xt_cpu.dtype,device=xt_cpu.device)

    if True:
        ## compare convolved_gaussian prior with assumed smax prior
        prior_pdf = get_gaussian(x,means=[0],sigmas=[sigma_max],weights=[1.0])

        # plt.plot(x,pdf,label="pdf")
        plt.plot(x,pdf2,label="Data + Noise",linestyle='-')
        # plt.plot(x,gt_pdf,label="gt_pdf",linestyle=':')
        plt.plot(x,prior_pdf,label='Assumed Gaussian Prior',linestyle='-')
    
    if False:
        ## Match analytical score and model score with the numerical score
        
        gt_score = gt_model_eval.batch_score(xt_cpu,time).detach().cpu().numpy()
        score = model_eval.score(xt_cuda,time).detach().cpu().numpy()

        gt_pdf = gt_model_eval.batch_pdf(xt_cpu,time).detach().cpu().numpy()
        lgt_pdf = np.log(gt_pdf)

        dgt_pdf = np.gradient(np.log(gt_pdf),x)

        prior_pdf = get_gaussian(x,means=[0],sigmas=[sigma_max],weights=[1.0])

        # plt.plot(x,pdf,label="pdf")
        plt.plot(x,pdf2,label="convolved_trimodal",linestyle='-')
        # plt.plot(x,gt_pdf,label="gt_pdf",linestyle=':')
        plt.plot(x,prior_pdf,label='prior_fn',linestyle='-')
        # plt.plot(x,lgt_pdf,label="lgt_pdf")
        # plt.plot(x,logpdf,label="log")
        # plt.plot(x,dlogpdf,label="dgradientlog")
        # plt.plot(x,dgt_pdf,label="gt_dgradientlog")
        # plt.plot(x,gt_score,label="gt_score",linestyle='--')
        # plt.plot(x,score,label="score")
        # plt.plot(x,score - gt_score,label="error")
        plt.legend()
        plt.show()

    if False:
        ## plot scores and error over time

        times = np.linspace(0,1,40)
        # scores = np.ndarray((x.shape[0],times.shape[0]))
        # gt_scores = np.ndarray((x.shape[0],times.shape[0]))
        score_errors = np.ndarray((x.shape[0],times.shape[0]))
        div_errors = np.ndarray((x.shape[0],times.shape[0]))
        for i,t in enumerate(tqdm(times)):
            gt_score = gt_model_eval.batch_score(xt_cpu,t).detach().cpu().numpy()
            score = model_eval.score(xt_cuda,t).detach().cpu().numpy()

            gt_div = gt_model_eval.batch_divergence(xt_cpu,t).numpy()
            
            div = np.ndarray(gt_div.shape)#,device=xt.device,dtype=xt.dtype)
            for j in range(div.shape[0]):
                div[j:j+1] = model_eval.divergence(xt_cuda[j:j+1],time).detach().cpu().numpy()

            

            # scores[:,i] = score[:,0]
            # gt_scores[:,i] = gt_score[:,0]
            score_errors[:,i] = (score - gt_score)[:,0]
            div_errors[:,i] = (div - gt_div)
        fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
        tm,xm = np.meshgrid(times,x)
        ax.plot_surface(xm,tm,score_errors,label="score_error")
        ax.set_xlabel("X")
        ax.set_ylabel("t")
        ax.set_title("score")
        plt.legend()

        fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
        tm,xm = np.meshgrid(times,x)
        ax.plot_surface(xm,tm,div_errors,label="div_error")
        ax.set_xlabel("X")
        ax.set_ylabel("t")
        ax.set_title("divergence")
        plt.legend()
        
        plt.show()

    if False:
        ## Match analytical divergence and model divergence with the numerical second derivative
        ddlogpdf = np.gradient(dlogpdf,x)

        # with torch.autograd.profiler.profile(with_stack=True, record_shapes=True, profile_memory=True, use_device='cuda') as p:
        #     for i in tqdm(range(500)):
                
        # pav = p.key_averages(group_by_input_shape=True,group_by_stack_n=5)
        # from IPython import embed; embed()
        gt_div = gt_model_eval.batch_divergence(xt_cpu,time).detach().cpu().numpy()
        # div = div.detach().cpu().numpy()
        div = np.ndarray(gt_div.shape)#,device=xt.device,dtype=xt.dtype)
        for i in tqdm(range(div.shape[0])):
            div[i:i+1] = model_eval.divergence(xt_cuda[i:i+1],time).detach().cpu().numpy()
        # print(div.shape)
        # print(gt_div.shape)
        trim = np.arange(200,x.shape[0]-200)
        plt.plot(x[trim],pdf[trim],label="pdf")
        # plt.plot(x,logpdf,label="log")
        plt.plot(x[trim],dlogpdf[trim],label="gradientlog")
        plt.plot(x[trim],ddlogpdf[trim],label="d2log")
        plt.plot(x[trim],gt_div[trim],label="gt_div",linestyle='--')
        plt.plot(x[trim],div[trim],label="div")
        plt.plot(x[trim],(div - gt_div)[trim],label="error")
        plt.legend()
        plt.show()


    if False:
        #precisely assure alignment of the numerical derivatives and the analytical ones
        xt_cpu = torch.as_tensor(x,device='cpu',dtype=torch.float)[:,None] #make 1d not 0d
        gt_model_eval.to(device=xt_cpu.device,dtype=xt_cpu.dtype)
        times = np.linspace(0,1,2000)
        score_surface = np.ndarray((xt_cpu.shape[0],times.shape[0]),dtype=xt_cpu.numpy().dtype)
        div_surface = np.ndarray((xt_cpu.shape[0],times.shape[0]),dtype=xt_cpu.numpy().dtype)
        for i,t in enumerate(tqdm(times)):
            pdf = get_convolved_gaussian(x,t,means,sigmas,weights,sigma_min=sigma_min,sigma_max=sigma_max)
            logpdf = np.log(pdf)
            dlogpdf = np.gradient(logpdf,x) #score
            ddlogpdf = np.gradient(dlogpdf,x) #divergence
            gt_score = gt_model_eval.batch_score(xt_cpu,t).numpy()
            gt_div = gt_model_eval.batch_divergence(xt_cpu,t).numpy()

            score_surface[:,i] = dlogpdf - gt_score[:,0] #since score is vector!
            div_surface[:,i] = ddlogpdf - gt_div

        tm, xm = np.meshgrid(times,x)
        
        
        score_surface[np.isnan(score_surface)] = 0
        div_surface[np.isnan(div_surface)] = 0
        
        score_error = score_surface[20:-20,:] #remove edge artifacts
        div_error = div_surface[20:-20,:] #remove edge artifacts
        

        print(f"{np.abs(score_error).max()=}")
        # print(f"{np.abs(score_error).argmax()=}")
        print(f"{np.abs(div_error).max()=}")
        # print(f"{np.abs(div_error).argmax()=}")
        
        
        f1,ax1 = plt.subplots(subplot_kw={"projection":"3d"})
        ax1.plot_surface(xm,tm,score_surface)
        ax1.set_title("score")
        f1.show()
        
        f2,ax2 = plt.subplots(subplot_kw={"projection":"3d"})
        ax2.plot_surface(xm,tm,div_surface)
        ax1.set_title("divergence")
        f2.show()

        plt.show()

    if False:
        # range = (-15.5,-14.5)
        range = (-0.5,0.5)
        trim = (x > range[0]) & (x < range[1]) #we can't just make x small because we have to convolve

        x = x[trim]
        pdf = pdf[trim]
        logpdf = logpdf[trim]
        dlogpdf = dlogpdf[trim]

        time = 0.8
        xt_cpu = torch.as_tensor(x,device='cpu',dtype=torch.float)[:,None] #make 1d not 0d
        gt_model_eval.to(device=xt_cpu.device,dtype=xt_cpu.dtype)


        score = gt_model_eval.batch_score(xt_cpu,time)

        plt.plot(x,dlogpdf)
        plt.plot(x,score.numpy())
        plt.show()

        # from IPython import embed; embed()
        
            

            

            

        