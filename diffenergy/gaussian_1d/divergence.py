import torch
from diffenergy.gaussian_1d.network import ScoreNetMLP, NegativeGradientMLP

# --------------------------------------------------------------------------
# Translation divergence calculation

def divergence_eval(sample, score_model, time_steps, epsilon):
    # Compute the divergence of the score-based model with Skilling-Hutchinson estimator

    with torch.enable_grad():
        sample.requires_grad_(True)
        score = score_model(sample, time_steps)
        grad_score = torch.autograd.grad(score, sample, retain_graph=True, create_graph=False)[0]
        trace = torch.sum(grad_score, dim=1)

    return trace

def score_eval_wrapper(batch, score_model, device="cuda"):
    # A wrapper for evaluating the score-based model for the black-box ODE solver
    
    sample = batch['sample'].unsqueeze(1).clone().detach()
    time_steps = batch['time_steps'].reshape((1,))
    
    if isinstance(score_model, ScoreNetMLP):
        with torch.no_grad():
            score = score_model(sample, time_steps)
    elif isinstance(score_model, NegativeGradientMLP):
        score = score_model(sample, time_steps)
    else:
        raise ValueError("Unknown score model type")
        
    return score.reshape((-1))

def divergence_eval_wrapper(batch, score_model, device="cuda"):
    # A wrapper for evaluating the divergence of score for the black-box ODE solver

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    sample = batch['sample'].unsqueeze(1).clone().detach()
    time_steps = batch['time_steps'].reshape((1,))
    epsilon = torch.randn_like(sample).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Obtain x(t) by solving the probability flow ODE
        # Compute likelihood
        div = divergence_eval(sample, score_model, time_steps, epsilon)

    return div.reshape((-1,))
