# --------------------------------------------------------------------------
# author Sudeep Sarma
# sample from a 1D diffusion model
# --------------------------------------------------------------------------
import torch
import functools
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP
from diffenergy.helper import marginal_prob_std, diffusion_coeff

# --------------------------------------------------------------------------

def Euler_Maruyama_sampler(score_model,
                            marginal_prob_std,
                            diffusion_coeff,
                            batch_size,
                            num_steps,
                            device='cuda',
                            eps=1e-3,
                            save_trajectory=False,
                            outpath=None):

    """Generate samples from score-based models with Euler-Maruyama solvers.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE
        batch_size: The number of samplers to generate by calling this function once
        num_steps: The number of sampling steps
            Equivalent to the number of discretized time steps
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability
        save_trajectory: If True, saves the entire trajectory over time steps

    Returns:
        Final samples, optionally saves full trajectory
    """

    if save_trajectory:
        if outpath is None:
            raise ValueError("outpath must be provided if save_trajectory is True")

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, device=device) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    trajectory = [[] for _ in range(batch_size)] if save_trajectory else None  # Store trajectory per sample

    for time_step in tqdm(time_steps):      
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step)
        score = score_model(x, batch_time_step)

        with torch.no_grad():
            mean_x = x + (g**2) * score * step_size
            mean_x = mean_x[:, :1]
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)
            # Ensure x remains (batch_size, 1)
            x = x[:, :1]
            
            if save_trajectory:
                for i in range(batch_size):
                    trajectory[i].append(x[i].cpu().item())  # Store per-sample trajectory

    if save_trajectory:
        traj_dir = outpath
        if not traj_dir.exists():
            traj_dir.mkdir()

        filepaths = [f"{traj_dir}/lp{i+1}.csv" for i in range(len(trajectory))]
        for traj,path in zip(trajectory,filepaths):
            df_traj = pd.DataFrame({"Index": np.arange(num_steps), "Timestep": time_steps.numpy(), "Sample": traj})
            df_traj.to_csv(path, index=False)
        
        index_file = f"{outpath}/trajectory_index.txt"
        with open(index_file,"w") as f:
            f.write("\n".join(filepaths))
        

        

    return x

# ----------------------------------------------------------------------------------
# Main
@hydra.main(version_base=None, config_path="../configs", config_name="sample_gaussian_1d")
def main(config: DictConfig):

    # Print the entire configuration
    print(OmegaConf.to_yaml(config))

    # The number of sampling steps
    num_steps = config.num_steps
    sigma_min = config.sigma_min
    sigma_max = config.sigma_max
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max)

    device = 'cuda'# if torch.cuda.is_available() else 'cpu'
    wt_file = config.wt_file
    ckpt = torch.load(wt_file, map_location = device)

    outpath = Path(config.out_dir)
    if not outpath.exists():
        outpath.mkdir()

    # Remove "module." prefix if necessary
    if any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
    
    tr_type = config.tr_type
    # Initialize score model
    if tr_type == 'non_conservative':
        score = ScoreNetMLP(input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
    elif tr_type == 'conservative':
        score = NegativeGradientMLP(input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)

    # Load the checkpoint weights into the model    
    score.load_state_dict(ckpt)
    score_model =  score

    sample_batch_size = config.sample_num
    save_trajectory = config.save_trajectory

    traj_outpath = outpath / (Path(config.sample_file).stem + "_traj")

    # Generate samples using Euler-Maruyama sampler
    samples = Euler_Maruyama_sampler(score_model,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,
                        sample_batch_size,
                        num_steps,
                        device = device,
                        save_trajectory=save_trajectory,
                        outpath=traj_outpath)

    samples_np = samples.cpu().detach().numpy()
    if samples_np.ndim == 1:
        samples_np =  samples_np[:, None]   # Convert to 2D by adding a second axis

    # Create a DataFrame with 'index' and 'Samples' columns
    df = pd.DataFrame(samples_np, columns=["Samples"])
    df.reset_index(inplace=True)
    df.index += 1 #match trajectory indexing
    df.rename(columns={"index":"index"}, inplace=True) #name the index as an actual column
    sample_file = config.sample_file
    df.to_csv(outpath / sample_file, index=False)

if __name__ == '__main__':
    main()
