

# code from https://gist.github.com/delta2323/6bb572d9473f3b523e6e
# generate a random ["Brownian Bridge"](https://en.wikipedia.org/wiki/Brownian_bridge); e.g., a random walk
# with both t=0 and t=N pinned at 0
import numpy as np
import six
import torch
from torchdiffeq import odeint
from tqdm import tqdm


def brownian_bridge(N:int, in_shape:tuple[int,...]|int,sigma=1):
    if not isinstance(in_shape,tuple):
        in_shape = (in_shape,)
    dt = 1.0 / (N - 1)
    dt_sqrt = np.sqrt(dt)
    B = np.empty((N,) + in_shape, dtype=np.float32)
    B[0] = np.zeros(in_shape)
    for n in six.moves.range(N - 2): #six: py2-py3 compatibility layer. 
         t = n * dt
         xi = np.random.randn(*in_shape) * sigma * dt_sqrt
         B[n + 1] = B[n] * (1 - dt / (1 - t)) + xi
    B[-1] = 0                                                 
    return B


def plot_brownian_bridge(ntimesteps:int=50,nsamples:int=200):
    import matplotlib.pyplot as plt
    times = np.linspace(0,1,ntimesteps)
    #[...,0] to turn it into an array of scalars instead of length-1 vectors, .T to make axes [time,sampleno] instead of [sampleno,time]
    trajectories = np.array([brownian_bridge(ntimesteps,1) for _ in range(nsamples)])[...,0].T 
    plt.plot(times,trajectories)
    plt.show()


# -----------------------------------------------------------------------------------
# FlowPerturbation likelihood calculation
class FlowPerturbationIntegral:

    def __init__(self,
                dataloader,
                batch_process_fn,
                score_model,
                diffusion_coeff_fn,
                prior_likelihood_fn,
                score_eval_wrapper,
                divergence_eval_wrapper,
                ode_steps=100,
                odeint_rtol=1e-5,
                odeint_atol=1e-5,
                odeint_method='rk4',
                num_perturbations=10, #will always include flow trajectory, so 10 perturbations really means 9 new perturbations
                perturbation_sigma=1, 
                reset_seed_each_sample=False,
                seed=0,
                device='cuda'):

        self.dataloader = dataloader
        self.batch_process_fn = batch_process_fn
        self.score_model = score_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.prior_likelihood_fn = prior_likelihood_fn
        self.score_eval_wrapper = score_eval_wrapper
        self.divergence_eval_wrapper = divergence_eval_wrapper
        self.ode_steps = ode_steps
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.odeint_method = odeint_method
        self.num_perturbations = num_perturbations
        self.perturbation_sigma = perturbation_sigma
        self.reset_seed_each_sample = reset_seed_each_sample
        self.seed = seed
        self.device = device

    def ode_trajectory(self, batch)->tuple[torch.Tensor,torch.Tensor]:
        # Returns:
        #   t: Tensor, timepoints used for the ode
        #   flow_trajectory: Tensor, the solution to the flow ODE


        sample = batch['sample']
        sample_shape = sample.shape

        # print("sample shape:",sample)

        def ode_func(t, x):
            # The ODE function for the black-box solver
            time_steps = torch.ones((1,), device = self.device) * t
            sample = x.reshape(sample_shape).to(self.device)
            g = self.diffusion_coeff_fn(t)
            batch['sample'] = sample
            batch['time_steps'] = time_steps
            sample_grad = -0.5 * g**2 * self.score_eval_wrapper(batch, self.score_model, device = self.device)
            # logp_grad = -0.5 * g**2 * self.divergence_eval_wrapper(batch, self.score_model, device = self.device)
            # sample_logp_concat = torch.cat([sample_grad, logp_grad], dim=0)

            return sample_grad

        # init = torch.cat([sample.reshape((-1,)), torch.zeros((1,), device = self.device)]) 
        init = sample.reshape((-1,)).to(device=self.device)
        eps = 1e-2
        t_eval = torch.linspace(eps, 1.0, steps=self.ode_steps, device=self.device)

        # Black-box ODE solver
        res = odeint(ode_func, init, t_eval, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.odeint_method)
        return t_eval,res
    
    def ode_diff_likelihood(self, batch, num_steps = 0):

        t_step = torch.tensor([1.0 / self.ode_steps], device = self.device)
        t_final = torch.tensor([1.0], device = self.device)
        time_steps = t_final - t_step * num_steps
        batch['time_steps'] = time_steps

        g = self.diffusion_coeff_fn(time_steps)
        logp_grad = -0.5 * g**2 * self.divergence_eval_wrapper(batch, self.score_model, device = self.device)
        logp_grad_t = logp_grad * t_step

        # Return the output as a dictionary
        return logp_grad_t
    
    def run_likelihood(self):

        data_list = []

        for batch in tqdm(self.dataloader,desc="Iterating samples..."):
            batch = self.batch_process_fn(batch, self.device)
            if self.reset_seed_each_sample:
                torch.manual_seed(self.seed)

            time,flow_trajectory = self.ode_trajectory(batch)
            _id = batch['id'].item() if isinstance(batch['id'], torch.Tensor) else batch['id']

            data_shape = batch['sample'].shape
            trajectories = [flow_trajectory] + [
                flow_trajectory + 
                torch.tensor(brownian_bridge(self.ode_steps,data_shape,sigma=self.perturbation_sigma),device=self.device)
                for _ in range(self.num_perturbations-1)
            ]

            trajectory_integrals:list[dict] = []
            for trajectory in tqdm(trajectories,desc="Integrating Trajectories...",leave=False):
                integral_list = []
                if self.reset_seed_each_sample:
                    torch.manual_seed(self.seed)

                for num_steps, x in enumerate(trajectory):
                    batch = {"sample":x}
                    batch = self.batch_process_fn(batch, self.device)
                    # Call ode_diff_likelihood
                    logp_grad_t = self.ode_diff_likelihood(batch, num_steps)

                    if num_steps == 0:
                        prior_logp, N = self.prior_likelihood_fn(batch)

                    # Append the full output to the list
                    integral_list.append(logp_grad_t)

                # Sum tensors for each key in 'out'
                integral = torch.stack(integral_list).sum(dim=0).item()
                bpd = -(prior_logp + integral)
                bpd = bpd.to(self.device)
                bpd = bpd / N

                # Define and update out_2
                out = {
                    "nll": bpd.item(),
                    "prior_logp": prior_logp.item(),
                    "integral": integral,
                }

                trajectory_integrals.append(out)

            #these are all calculated from the same point, so should be the same
            assert np.allclose(trajectory_integrals[0]["prior_logp"],[traj["prior_logp"] for traj in trajectory_integrals])

            result = {
                "id": _id,
                "prior_logp":trajectory_integrals[0]["prior_logp"],
            }

            flow_trajectory=trajectory_integrals[0]
            all_trajectories={
                "nll":[traj["nll"] for traj in trajectory_integrals],
                "integral":[traj["integral"] for traj in trajectory_integrals],
            }

            result.update({
                "nll":np.average(all_trajectories["nll"]),
                "integral":np.average(all_trajectories["integral"]),
                "nll_variance":np.var(all_trajectories["nll"]),
                "integral_variance":np.var(all_trajectories["integral"])
            })

            result.update({
                "flow_trajectory": flow_trajectory,
                "all_trajectories": all_trajectories
            })

            data_list.append(result) 

        return data_list
    


    # def run_likelihood(self):

    #     data_list = []

    #     for batch in tqdm(self.dataloader):
    #         batch = self.batch_process_fn(batch, self.device)
    #         if self.reset_seed_each_sample:
    #             torch.manual_seed(self.seed)
    #         out = self.ode_likelihood(batch)
    #         _id = batch['id'].item() if isinstance(batch['id'], torch.Tensor) else batch['id']
    #         output = {'id': _id}
    #         output.update(out)
    #         data_list.append(output)

    #     return data_list