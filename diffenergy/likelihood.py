# --------------------------------------------------------------------------
# author Sudeep Sarma
# likelihood calculation for laplacian distribution
# --------------------------------------------------------------------------
from tqdm import tqdm
import torch
from torchdiffeq import odeint

# -----------------------------------------------------------------------------------
# FlowTime likelihood calculation
class FlowTimeIntegral:

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
        self.device = device

    def ode_likelihood(self, batch):

        sample = batch['sample']
        sample_shape = sample.shape

        def ode_func(t, x):
            # The ODE function for the black-box solver
            time_steps = torch.ones((1,), device = self.device) * t
            sample = x[:-1].reshape(sample_shape).to(self.device)
            g = self.diffusion_coeff_fn(t)
            batch['sample'] = sample
            batch['time_steps'] = time_steps
            sample_grad = -0.5 * g**2 * self.score_eval_wrapper(batch, self.score_model, device = self.device)
            logp_grad = -0.5 * g**2 * self.divergence_eval_wrapper(batch, self.score_model, device = self.device)
            sample_logp_concat = torch.cat([sample_grad, logp_grad], dim=0)

            return sample_logp_concat

        init = torch.cat([sample.reshape((-1,)), torch.zeros((1,), device = self.device)]) 
        N = sample.numel()
        eps = 1e-2
        t_eval = torch.linspace(eps, 1.0, steps=self.ode_steps, device=self.device)

        # Black-box ODE solver
        res = odeint(ode_func, init, t_eval, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.odeint_method)
        zp = res[-1]
        z = zp[:-1].reshape(sample_shape)
        delta_logp = zp[-1:].reshape(1)
        prior_logp = self.prior_likelihood_fn(z)
        bpd = - (prior_logp + delta_logp)
        bpd = bpd / N

        return {'bpd': bpd.item(), 'delta_logp': delta_logp.item()}

    def run_likelihood(self):

        data_list = []

        for batch in tqdm(self.dataloader):
            batch = self.batch_process_fn(batch, self.device)
            out = self.ode_likelihood(batch)
            _id = batch['id'].item() if isinstance(batch['id'], torch.Tensor) else batch['id']
            out.update({'id': _id}) 
            data_list.append(out)

        return data_list

# -----------------------------------------------------------------------------------
# DiffusionSpace likelihood calculation
class DiffSpaceIntegral:

    def __init__(self,
                dataloaders,
                batch_process_fn,
                score_model,
                diffusion_coeff_fn,
                prior_likelihood_fn,
                score_eval_wrapper,
                del_sample_fn,
                diffusion_steps=100,
                device='cuda'):

        self.dataloaders = dataloaders
        self.batch_process_fn = batch_process_fn
        self.score_model = score_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.prior_likelihood_fn = prior_likelihood_fn
        self.score_eval_wrapper = score_eval_wrapper
        self.del_sample_fn = del_sample_fn
        self.tot_steps = diffusion_steps
        self.device = device
    
    def diff_likelihood(self, batch, prev_sample = None, num_steps = 0):

        # grab some input
        sample = batch['sample'].clone().detach()
        t_step = torch.tensor([1.0 / self.tot_steps], device = self.device)
        t_final = torch.tensor([1.0], device = self.device)
        time_steps = t_final - t_step * num_steps
        batch['time_steps'] = time_steps

        score = self.score_eval_wrapper(batch, self.score_model, device = self.device)

        # Compute del_position as the difference between current and previous sample
        del_sample = self.del_sample_fn(sample, prev_sample)

        force_del_sample = torch.dot(score, del_sample)

        return force_del_sample

    def run_likelihood(self):
        
        data_list = []
    
        for i, (_id, single_traj) in enumerate(tqdm(self.dataloaders.items())):

            integral_list = []
            prev_sample = None

            for num_steps, batch in enumerate(single_traj):
                batch = self.batch_process_fn(batch, self.device)
                # Call diff_likelihood with the previous ligand position
                sample = batch['sample'].clone().detach()
                force_del_sample = self.diff_likelihood(batch, prev_sample, num_steps)

                if prev_sample is None:
                    prior_logp = self.prior_likelihood_fn(sample)
                    N = sample.numel()

                # Update previous ligand position for the next iteration
                prev_sample = sample.clone().detach()

                # Append the full output to the list
                integral_list.append(force_del_sample)

            # Sum tensors for each key in 'out'
            integral = torch.stack(integral_list).sum(dim=0).item()
            bpd = -prior_logp - integral
            bpd = bpd / N

            # Define and update out_2
            out = {
                "id": _id,
                "bpd": bpd.item(),
            }

            data_list.append(out)

        return data_list

# -----------------------------------------------------------------------------------
# DiffusionTime likelihood calculation
class DiffTimeIntegral:

    def __init__(self,
                dataloaders,
                batch_process_fn,
                score_model,
                diffusion_coeff_fn,
                prior_likelihood_fn,
                divergence_eval_wrapper,
                diffusion_steps=100,
                device='cuda'):

        self.dataloaders = dataloaders
        self.batch_process_fn = batch_process_fn
        self.score_model = score_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.prior_likelihood_fn = prior_likelihood_fn
        self.divergence_eval_wrapper = divergence_eval_wrapper
        self.tot_steps = diffusion_steps
        self.device = device

    def ode_diff_likelihood(self, batch, num_steps = 0):

        t_step = torch.tensor([1.0 / self.tot_steps], device = self.device)
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

        for i, (_id, single_traj) in enumerate(tqdm(self.dataloaders.items())):

            integral_list = []

            for num_steps, batch in enumerate(single_traj):
                batch = self.batch_process_fn(batch, self.device)
                # Call ode_diff_likelihood
                sample = batch['sample'].clone().detach()
                logp_grad_t = self.ode_diff_likelihood(batch, num_steps)

                if num_steps == 0:
                    prior_logp = self.prior_likelihood_fn(sample)
                    N = sample.numel()

                # Append the full output to the list
                integral_list.append(logp_grad_t)

            # Sum tensors for each key in 'out'
            integral = torch.stack(integral_list).sum(dim=0).item()
            bpd = -(prior_logp + integral)
            bpd = bpd.to(self.device)
            bpd = bpd / N

            # Define and update out_2
            out = {
                "id": _id,
                "bpd": bpd.item(),
            }

            data_list.append(out) 

        return data_list
