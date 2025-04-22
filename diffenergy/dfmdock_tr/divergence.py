import torch

# --------------------------------------------------------------------------
# Translation divergence calculation

def divergence_eval_tr(batch, score_model):
	# Compute the divergence of the score-based model

	# grab some input
	lig_pos = batch["lig_pos"]

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
		grad_score_lig_ca_mean = grad_score_lig_ca.mean(dim=1).squeeze() # shape[3,3]
		diagonal = torch.diagonal(grad_score_lig_ca_mean)
		trace = torch.sum(diagonal).unsqueeze(0)

	return trace

def score_eval_wrapper_tr(batch, score_model, device="cuda"):
	#A wrapper for evaluating the score-based model for the black-box ODE solver
	
	# grab some input
	lig_pos = batch["lig_pos"]
	time_steps = batch["time_steps"]

	lig_pos = torch.tensor(lig_pos, device=device, dtype=torch.float32)
	time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((1,)) 

	# prepare for input
	batch["lig_pos"] = lig_pos
	batch["t"] = time_steps

	with torch.no_grad():    
		tr_score= score_model(batch)["tr_score"]
		tr_score = tr_score.expand(lig_pos.shape) 

	return tr_score.reshape((-1,))

def divergence_eval_wrapper_tr(batch, score_model, device="cuda"):
	#A wrapper for evaluating the divergence of score for the black-box ODE solver

	# grab some input
	lig_pos = batch["lig_pos"]
	time_steps = batch["time_steps"]

	lig_pos = torch.tensor(lig_pos, device=device, dtype=torch.float32)
	time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((1,)) 

	# prepare for input
	batch["lig_pos"] = lig_pos
	batch["t"] = time_steps
	
	# compute divergence
	div = divergence_eval_tr(batch, score_model) 

	return div.reshape((-1,))
