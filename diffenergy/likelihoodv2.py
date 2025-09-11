import abc
from typing import Any, Callable, Optional, Protocol

import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm

class ModelEval(Protocol):
    def __call__(self,batch:dict[str,Any],score_model:torch.nn.Module,device:torch.device=torch.device('cuda'))->torch.Tensor: ...

class FlowIntegral(abc.ABC):
    def __init__(self,
                dataloader:DataLoader,
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                ode_steps=100,
                odeint_rtol=1e-5,
                odeint_atol=1e-5,
                odeint_method='rk4',
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
        
        self.dataloader = dataloader
        self.batch_process_fn = batch_process_fn
        self.score_model = score_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.prior_likelihood_fn = prior_likelihood_fn
        self.score_eval_wrapper = score_eval_wrapper
        self.ode_steps = ode_steps
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.odeint_method = odeint_method
        self.reset_seed_each_sample = reset_seed_each_sample
        self.seed = seed
        self.device = torch.device(device)

    @abc.abstractmethod
    def ode_integrand(self, batch:dict[str,Any], g:float|torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        """Input the batch and diffusion coefficient corresponding to a particular point in the ode process and return [dx,dlogp] where dlogp is the computed change in log-probability at this step.
        Args:
            batch: dict[str,Any]
                batch["sample"]: torch.Tensor corresponding to current x in the diffusion trajectory
                batch["time_steps"]: torch.Tensor corresponding to current t in the diffusion trajectory
            g: float|torch.Tensor 
                diffusion coefficient corresponding to current time
        """
        ...


    def ode_likelihood(self, batch):
        sample = batch['sample']
        sample_shape = sample.shape

        def ode_func(t, x):
            # The ODE function for the black-box solver
            time_steps = torch.ones((1,), device = self.device) * t
            sample = x[:-1].reshape(sample_shape).to(self.device)
            batch['sample'] = sample
            batch['time_steps'] = time_steps

            sample_grad, dlogp = self.ode_integrand(batch, self.diffusion_coeff_fn(t))
            sample_logp_concat = torch.cat([sample_grad, dlogp], dim=0)

            return sample_logp_concat

        init = torch.cat([sample.reshape((-1,)), torch.zeros((1,), device = self.device)]) 
        eps = 1e-2
        t_eval = torch.linspace(eps, 1.0, steps=self.ode_steps, device=self.device)

        # Black-box ODE solver
        res = odeint(ode_func, init, t_eval, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.odeint_method)
        zp = res[-1]
        z = zp[:-1].reshape(sample_shape)
        batch['sample'] = z
        prior_logp, N = self.prior_likelihood_fn(batch)
        delta_logp = zp[-1:].reshape(1)
        bpd = - (prior_logp + delta_logp)
        bpd = bpd / N

        return {'nll': bpd.item(), 'prior_logp': prior_logp.item(), 'delta_logp': delta_logp.item()}

    def run_likelihood(self):

        data_list = []

        for batch in tqdm(self.dataloader):
            batch = self.batch_process_fn(batch, self.device)
            if self.reset_seed_each_sample:
                torch.manual_seed(self.seed)
            out = self.ode_likelihood(batch)
            _id = batch['id'].item() if isinstance(batch['id'], torch.Tensor) else batch['id']
            output = {'id': _id}
            output.update(out)
            data_list.append(output)

        return data_list
    

class FlowTimeIntegral(FlowIntegral):

    def __init__(self,
                dataloader:DataLoader,
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                divergence_eval_wrapper:ModelEval,
                ode_steps=100,
                odeint_rtol=1e-5,
                odeint_atol=1e-5,
                odeint_method='rk4',
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
        super().__init__(dataloader,
                         batch_process_fn,
                         score_model,
                         diffusion_coeff_fn,
                         prior_likelihood_fn,
                         score_eval_wrapper,
                         ode_steps=ode_steps,odeint_rtol=odeint_rtol,odeint_atol=odeint_atol,
                         odeint_method=odeint_method,
                         reset_seed_each_sample=reset_seed_each_sample,
                         seed=seed,
                         device=device);
        self.divergence_eval_wrapper = divergence_eval_wrapper

    def ode_integrand(self, batch: dict[str, Any], g: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device);
        divergence = self.divergence_eval_wrapper(batch, self.score_model, device = self.device);
        
        sample_grad = -0.5 * g**2 * score
        logp_grad = -0.5 * g**2 * divergence
        
        return sample_grad, logp_grad



# -----------------------------------------------------------------------------------
# FlowSpace likelihood calculation
class FlowSpaceIntegral(FlowIntegral):
    #default __init__

    def ode_integrand(self, batch: dict[str, Any], g: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device);

        sample_grad = -0.5 * g**2 * score

        dE = torch.dot(score,sample_grad)
        
        return sample_grad, dE


# -----------------------------------------------------------------------------------
# FlowSpace likelihood calculation
class FlowTotalIntegral(FlowIntegral):
    def __init__(self,
                dataloader:DataLoader,
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                divergence_eval_wrapper:ModelEval,
                ode_steps=100,
                odeint_rtol=1e-5,
                odeint_atol=1e-5,
                odeint_method='rk4',
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
        super().__init__(dataloader,
                         batch_process_fn,
                         score_model,
                         diffusion_coeff_fn,
                         prior_likelihood_fn,
                         score_eval_wrapper,
                         ode_steps=ode_steps,odeint_rtol=odeint_rtol,odeint_atol=odeint_atol,
                         odeint_method=odeint_method,
                         reset_seed_each_sample=reset_seed_each_sample,
                         seed=seed,
                         device=device);
        self.divergence_eval_wrapper = divergence_eval_wrapper
        
    def ode_integrand(self, batch: dict[str, Any], g: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device);
        divergence = self.divergence_eval_wrapper(batch, self.score_model, device = self.device);
        
        sample_grad = -0.5 * g**2 * score
        logp_grad = -0.5 * g**2 * divergence

        dE = torch.dot(score,sample_grad)
        
        return sample_grad, logp_grad + dE
    

class UnmovingIntegral(FlowIntegral):
    def __init__(self,
                dataloader:DataLoader,
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                divergence_eval_wrapper:ModelEval,
                ode_steps=100,
                odeint_rtol=1e-5,
                odeint_atol=1e-5,
                odeint_method='rk4',
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
        super().__init__(dataloader,
                         batch_process_fn,
                         score_model,
                         diffusion_coeff_fn,
                         prior_likelihood_fn,
                         score_eval_wrapper,
                         ode_steps=ode_steps,odeint_rtol=odeint_rtol,odeint_atol=odeint_atol,
                         odeint_method=odeint_method,
                         reset_seed_each_sample=reset_seed_each_sample,
                         seed=seed,
                         device=device);
        self.divergence_eval_wrapper = divergence_eval_wrapper

    def ode_integrand(self, batch: dict[str, Any], g: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device);
        divergence = self.divergence_eval_wrapper(batch, self.score_model, device = self.device);
        
        sample_grad = -0.5 * g**2 * score
        logp_grad = -0.5 * g**2 * divergence
        
        return sample_grad, logp_grad

    

class DiffIntegral(abc.ABC):
    def __init__(self,
                dataloaders:dict[str,DataLoader],
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                diffusion_steps=100,
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):

        self.dataloaders = dataloaders
        self.batch_process_fn = batch_process_fn
        self.score_model = score_model
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.prior_likelihood_fn = prior_likelihood_fn
        self.score_eval_wrapper = score_eval_wrapper
        self.tot_steps = diffusion_steps
        self.reset_seed_each_sample = reset_seed_each_sample
        self.seed = seed
        self.device = torch.device(device)

    @abc.abstractmethod
    def diffusion_integrand(self, batch:dict[str,Any], num_steps:int, prev_batch: Optional[dict[str,Any]]=None)->torch.Tensor: ...


    def diff_likelihood(self, batch:dict[str,Any], num_steps:int, prev_batch:Optional[dict[str,Any]] = None):
        # grab some input
        time_steps = self.t_final - self.t_step * num_steps
        batch['time_steps'] = time_steps

        dlogp = self.diffusion_integrand(batch,num_steps,prev_batch=prev_batch)

        return dlogp
    
    def run_likelihood(self):
        
        data_list = []

        self.t_step = torch.tensor([1.0 / self.tot_steps], device = self.device)
        self.t_final = torch.tensor([1.0], device = self.device)
    
        for i, (_id, single_traj) in enumerate(tqdm(self.dataloaders.items())):

            integral_list = []
            prev_batch = None
            if self.reset_seed_each_sample:
                torch.manual_seed(self.seed)

            for num_steps, batch in enumerate(single_traj):
                batch = self.batch_process_fn(batch, self.device)
                # Call diff_likelihood with the previous ligand position
                force_del_sample = self.diff_likelihood(batch, num_steps, prev_batch=prev_batch)

                if prev_batch is None:
                    prior_logp, N = self.prior_likelihood_fn(batch)

                # Update previous ligand position for the next iteration
                prev_batch = batch

                # Append the full output to the list
                integral_list.append(force_del_sample)

            # Sum tensors for each key in 'out'
            integral = torch.stack(integral_list).sum(dim=0).item()
            bpd = -prior_logp - integral
            bpd = bpd / N

            # Define and update out_2
            out = {
                "id": _id,
                "nll": bpd.item(),
                "prior_logp": prior_logp.item(),
                "integral": integral,
            }

            data_list.append(out)

        return data_list

class DiffTimeIntegral(DiffIntegral):
    def __init__(self,
                dataloaders:dict[str,DataLoader],
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                divergence_eval_wrapper:ModelEval,
                diffusion_steps=100,
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):

        super().__init__(dataloaders,
                        batch_process_fn,
                        score_model,
                        diffusion_coeff_fn,
                        prior_likelihood_fn,
                        score_eval_wrapper,
                        diffusion_steps=diffusion_steps,
                        reset_seed_each_sample=reset_seed_each_sample,
                        seed=seed,
                        device=device);
        self.divergence_eval_wrapper = divergence_eval_wrapper

    def diffusion_integrand(self, batch: dict[str, Any], num_steps: int, prev_batch: dict[str, Any] | None = None) -> torch.Tensor:
        time_steps = batch["time_steps"]
        score = self.divergence_eval_wrapper(batch, self.score_model, device = self.device)

        g = self.diffusion_coeff_fn(time_steps)
        logp_grad = -0.5 * g**2 * score
        logp_grad_t = logp_grad * self.t_step

        return logp_grad_t



class DiffSpaceIntegral(DiffIntegral):
        
    def __init__(self,
                dataloaders:dict[str,DataLoader],
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                del_sample_fn:Callable[[torch.Tensor,Optional[torch.Tensor]],torch.Tensor],
                diffusion_steps=100,
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
            
        super().__init__(dataloaders,
                        batch_process_fn,
                        score_model,
                        diffusion_coeff_fn,
                        prior_likelihood_fn,
                        score_eval_wrapper,
                        diffusion_steps=diffusion_steps,
                        reset_seed_each_sample=reset_seed_each_sample,
                        seed=seed,
                        device=device);
        self.del_sample_fn = del_sample_fn

    def diffusion_integrand(self, batch: dict[str, Any], num_steps: int, prev_batch: dict[str, Any] | None = None):
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device)

        sample = batch["sample"].clone().detach()
        prev_sample = prev_batch["sample"].clone().detach() if prev_batch else None

        # Compute del_position as the difference between current and previous sample
        del_sample = self.del_sample_fn(sample, prev_sample)

        force_del_sample = torch.dot(score, del_sample)

        return force_del_sample


class DiffTotalIntegral(DiffIntegral):
        
    def __init__(self,
                dataloaders:dict[str,DataLoader],
                batch_process_fn:Callable[[dict[str,Any],torch.device],dict[str,Any]],
                score_model:torch.nn.Module,
                diffusion_coeff_fn:Callable[[float],float|torch.Tensor],
                prior_likelihood_fn:Callable[[dict[str,Any]],tuple[float,int]],
                score_eval_wrapper:ModelEval,
                divergence_eval_wrapper:ModelEval,
                del_sample_fn:Callable[[torch.Tensor,Optional[torch.Tensor]],torch.Tensor],
                diffusion_steps=100,
                reset_seed_each_sample=False,
                seed=0,
                device:str|torch.device='cuda', **kwargs):
            
        super().__init__(dataloaders,
                        batch_process_fn,
                        score_model,
                        diffusion_coeff_fn,
                        prior_likelihood_fn,
                        score_eval_wrapper,
                        diffusion_steps=diffusion_steps,
                        reset_seed_each_sample=reset_seed_each_sample,
                        seed=seed,
                        device=device);
        self.del_sample_fn = del_sample_fn
        self.divergence_eval_wrapper = divergence_eval_wrapper

    def diffusion_integrand(self, batch: dict[str, Any], num_steps: int, prev_batch: dict[str, Any] | None = None):
        time_steps = batch["time_steps"]
        score = self.score_eval_wrapper(batch, self.score_model, device = self.device)

        sample = batch["sample"].clone().detach()
        prev_sample = prev_batch["sample"].clone().detach() if prev_batch else None

        # Compute del_position as the difference between current and previous sample
        del_sample = self.del_sample_fn(sample, prev_sample)
        force_del_sample = torch.dot(score, del_sample)

        g = self.diffusion_coeff_fn(time_steps)
        logp_grad = -0.5 * g**2 * score
        logp_grad_t = logp_grad * self.t_step

        return logp_grad_t + force_del_sample

