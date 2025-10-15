import torch

# --------------------------------------------------------------------------
# Translation divergence calculation

def divergence_eval_tr(batch, score_model):
    # Compute the divergence of the score-based model

    # grab some input
    lig_pos = batch["sample"]

    with torch.enable_grad():
        lig_pos.requires_grad_(True)
        tr_score = score_model(batch)["tr_score"].squeeze()

        # Compute the gradient of tr_score w.r.t lig_pos
        grad_score_lig = []
        for i in range(tr_score.shape[0]):
            grad = torch.autograd.grad(tr_score[i], lig_pos, retain_graph=True, create_graph=False)[0]
            grad_score_lig.append(grad)
        grad_score_lig = torch.stack(grad_score_lig, dim=0)  # shape[3,107,3,3]
        grad_score_lig_ca = grad_score_lig[:,:,1,:].squeeze() # shape[3,107,3]
        grad_score_lig_ca_sum = grad_score_lig_ca.sum(dim=1).squeeze() # shape[3,3]
        diagonal = torch.diagonal(grad_score_lig_ca_sum)
        trace = torch.sum(diagonal).unsqueeze(0)

    return trace

def score_eval_wrapper_tr_diffspace(batch, score_model, device="cuda"):
    #A wrapper for evaluating the score-based model for the black-box ODE solver
    
    # grab some input
    time_steps = batch["time_steps"].reshape((1,)) 

    # prepare for input
    batch["t"] = time_steps

    with torch.no_grad():    
        tr_score= score_model(batch)["tr_score"]

    return tr_score.reshape((-1,))

def score_eval_wrapper_tr_ode(batch, score_model, device="cuda"):
    #A wrapper for evaluating the score-based model for the black-box ODE solver
    
    # grab some input
    batch["sample"] = batch["sample"].clone().detach()
    lig_pos = batch["sample"]
    time_steps = batch["time_steps"].reshape((1,)) 

    # prepare for input
    batch["t"] = time_steps

    with torch.no_grad():    
        tr_score= score_model(batch)["tr_score"]
        tr_score = tr_score.expand(lig_pos.shape) 

    return tr_score.reshape((-1,))

def divergence_eval_wrapper_tr(batch, score_model, device="cuda"):
    #A wrapper for evaluating the divergence of score for the black-box ODE solver

    # grab some input
    batch["sample"] = batch["sample"].clone().detach()
    time_steps = batch["time_steps"].reshape((1,)) 

    # prepare for input
    batch["t"] = time_steps
    
    # compute divergence
    div = divergence_eval_tr(batch, score_model) 

    return div.reshape((-1,))
