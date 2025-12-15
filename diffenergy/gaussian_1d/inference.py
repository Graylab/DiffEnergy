from csv import DictWriter
import warnings
from diffenergy.inference import MapDataset, get_integrands, get_paths, unzip
from diffenergy.gaussian_1d.likelihood_helpers import ModelEval
from diffenergy.gaussian_1d.network import NegativeGradientMLP, ScoreNetMLP
from diffenergy.groundtruth_score import MultimodalGaussianGroundTruthScoreModel, batched_normpdf_matrix, batched_normpdf_scalar
from diffenergy.helper import diffusion_coeff, int_diffusion_coeff_sq, marginal_prob_std, prior_gaussian_nd
from diffenergy.inference import DiffEnergyLikelihood, ForcesMixin
from diffenergy.likelihood import run_diff_likelihood, run_ode_likelihood
from diffenergy.inference import SizedIter


import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm


import functools
import itertools
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, override


class GaussianLikelihood(DiffEnergyLikelihood[torch.Tensor,None]):

    @classmethod
    def to_array(cls,x:torch.Tensor)->torch.Tensor:
        return x.squeeze(0)
    @classmethod
    def from_array(cls,a,device:str|torch.device='cuda')->torch.Tensor:
        return torch.as_tensor(a,dtype=torch.float,device=torch.device(device))[None,...]

    @classmethod
    def to_array_batch(cls,x:torch.Tensor)->torch.Tensor:
        return x #don't un-batch
    @classmethod
    def from_array_batch(cls,a,device:str|torch.device='cuda')->torch.Tensor:
        return torch.as_tensor(a,dtype=torch.float,device=torch.device(device)) #don't re-batch


    def load_model(self,
               sigma_min:float,
               sigma_max:float,
               batched:bool,
               device:torch.device):
        # set marginal probability distribution and diffusion coefficient distribution
        marginal_prob_std_fn = functools.partial(
            marginal_prob_std, sigma_min = sigma_min, sigma_max = sigma_max)

        # set models
        weights_path = self.config.checkpoint
        ckpt = torch.load(weights_path, map_location = device)

        # Remove "module." prefix if necessary
        if any(key.startswith("module.") for key in ckpt.keys()):
            ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}

        # Initialize score model, load the checkpoint weights into the model    
        tr_type = self.config.tr_type
        if tr_type == 'non_conservative':
            score_model = ScoreNetMLP(
                input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
            score_model.load_state_dict(ckpt)

            model_eval = ModelEval(score_model,batched=batched)
        elif tr_type == 'conservative':
            score_model = NegativeGradientMLP(
                input_dim = 1, marginal_prob_std = marginal_prob_std_fn, embed_dim = 512, layers = (512, 512, 512)).to(device)
            score_model.load_state_dict(ckpt)

            model_eval = ModelEval(score_model,batched=batched)
        elif tr_type == 'ground_truth':
            means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float)
            variances = torch.tensor([8.0,5.0,10.0])**2
            weights = torch.tensor([0.4,0.3,0.3])

            model_eval = MultimodalGaussianGroundTruthScoreModel(means,variances,weights,sigma_min,sigma_max,batched=batched)
            model_eval.to(device)
        else:
            raise ValueError(tr_type)

        return model_eval

    @classmethod
    def load_samples(cls,data_path:str|Path,device:torch.device)->list[tuple[str,torch.Tensor,None]]:
        """Loads dataset from a CSV file and returns an iterable of tuples ('id',x). Each x will be a tensor with shape (1,)."""
        df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
        ids = df.loc[:, "index"].values  # Extract the first column as ids
        samples = df.loc[:, "Samples"].values  # Extract the second column as samples

        # Convert to tensors
        ids = torch.tensor(ids, dtype=torch.int64)  # ids as integers
        samples = torch.tensor(samples, dtype=torch.float32, device=device)[...,None]  # samples as floats, add a dimension to make them vectors

        return [(str(int(id.item())),x,None) for id,x in zip(ids,samples)]

    @classmethod
    def load_samples_batched(cls,data_path:str|Path, batch_size:int,device:torch.device)->list[tuple[list[str],torch.Tensor,None]]:
        """Loads dataset from a CSV file and returns an iterable of tuples (['id1','id2',...],XB), where XB is a tensor with shape (batch_size,1)."""
        df = pd.read_csv(data_path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'
        ids = df.loc[:, "index"].values  # Extract the first column as ids
        samples = df.loc[:, "Samples"].values  # Extract the second column as samples

        # Convert to tensors
        ids = torch.tensor(ids, dtype=torch.int64)  # ids as integers
        samples = torch.tensor(samples, dtype=torch.float32, device=device)[...,None]  # samples as floats, add a dimension to make them 1D vectors

        assert batch_size > 0
        return [([str(int(id)) for id in ids[i*batch_size:(i+1)*batch_size].tolist()],samples[i*batch_size:(i+1)*batch_size],None)
                for i in range(math.ceil(len(samples)/batch_size))]

    @classmethod
    def load_trajectories(cls,index_file:str|Path):
        trajectory_index_file = Path(index_file)
        assert trajectory_index_file.suffix == '.csv'
        df = pd.read_csv(trajectory_index_file)
        dlist:dict[str,str] = df.set_index('index').T.to_dict(orient='records')[0] # pyright: ignore[reportAssignmentType] #make ids columnames, then convert to a dict of [{colname:value,colname:value}] and get the first result
        return [(str(id),trajectory_index_file.parent/p,None) for id,p in dlist.items()]

    @classmethod
    def load_trajectories_batched(cls,index_file:str|Path,batch_size:int):
        return [(tuple(b[0] for b in batch),tuple(b[1] for b in batch),None) for batch in itertools.batched(cls.load_trajectories(index_file),batch_size)]
    
    @override
    def sample_index_writer(self,write_samples:bool,extra_fieldnames:Iterable[str]=[]):
        return super().sample_index_writer(write_samples,extra_fieldnames=['Samples',*extra_fieldnames])
    
    @override
    def trajectory_index_writers(self,write_indices:bool,extra_fieldnames:Iterable[str]=[]):
        return super().trajectory_index_writers(write_indices,extra_fieldnames=['filename',*extra_fieldnames])

    @classmethod
    def load_trajectory(cls,data_path:str|Path|tuple[str|Path,...],device:str|torch.device='cuda')->tuple[torch.Tensor,torch.Tensor]:
        batched = isinstance(data_path,tuple)
        paths = data_path if batched else [data_path]
        alltimes:Optional[torch.Tensor] = None
        sampleslist:list[torch.Tensor] = []
        for path in paths:
            df = pd.read_csv(path, header=0)  # Load CSV keeping first column as 'id' and second column as 'samples'

            # we reverse the trajectory so it matches flow, with time going from 0 to 1. 
            # have to do this and numpy AND make a copy, cause tensors don't support negative stride -_-
            df = df.iloc[::-1].copy()

            if "Timestep" in df.columns:
                times = torch.as_tensor(df.loc[:, "Timestep"].values,dtype=torch.float32) #Extract Timestep column as times.
            else:
                steps = torch.as_tensor(df.loc[:,"Index"].values,dtype=torch.float32) # Extract the Index column, convert to timesteps
                times = 1 - steps/steps.max() #steps go from 0 to N, so divide by N and subtract from 1 to get time from 1 to 0

            if alltimes is not None:
                assert torch.allclose(alltimes,times)
            else:
                alltimes = times

            sampleslist.append(torch.as_tensor(df.loc[:, "Sample"].values,dtype=torch.float32,device=device))  # Extract the Sample column
        assert alltimes is not None
        samples = torch.stack(sampleslist,dim=1) if batched else sampleslist[0] #needs to be NxB 
        return samples[...,None],alltimes  #add dimension to samples so iteration of 1d vectors

    def load_priors(self,
                    sigma_min:float,
                    sigma_max:float,
                    device:torch.device,
                    batch_size:int|None):

        batched = batch_size is not None

        ### LOAD PRIORS
        priors:list[tuple[str,Callable[[torch.Tensor,float,None],float]]] = []
        functions:DictConfig = self.config.get("prior_fns","smax_gaussian")
        if isinstance(functions,str):
            functions = functions.split(" ") # pyright: ignore[reportAssignmentType]
        if not isinstance(functions,Mapping):
            functions = DictConfig({f:{} for f in functions}) #ensure types is a DictConfig of Dicts

        for prior_fn,params in functions.items():
            match prior_fn:
                case "convolved_data":
                    means = torch.tensor([[-30.0],[0.0],[40.0]],dtype=torch.float, device=device)
                    variances = torch.tensor([8.0,5.0,10.0], device=device)**2
                    weights = torch.tensor([0.4,0.3,0.3], device=device)

                    gt_time = params.get("time",1)
                    if gt_time == "Any":
                        gt_time = None
                    else:
                        gt_time = float(gt_time)
                        if not isinstance(gt_time,float) or not (0 <= gt_time and gt_time <= 1):
                            raise ValueError(f"'time' parameter to the convolved_data prior can only be a float in [0,1] or \"Any\", received {gt_time}")
                    gt_time = torch.as_tensor(gt_time, dtype=torch.float, device=device)

                    def gt_prior_fn(x:torch.Tensor,t:float,condition:None):
                        tt = torch.as_tensor(t)
                        if gt_time is not None: assert torch.allclose(tt, gt_time), t  # pyright: ignore[reportArgumentType] # noqa: E701
                        currvar = variances + int_diffusion_coeff_sq(tt,sigma_min=sigma_min,sigma_max=sigma_max)

                        assert x.ndim == 2,x.shape #BxD
                        assert means.ndim == 2,x.shape #NxD
                        dx = x[:,None,...] - means[None,...]
                        if variances.ndim == 1:
                            probs = batched_normpdf_scalar(dx,currvar) #shape:BxN
                        else:
                            probs = batched_normpdf_matrix(dx,currvar) #shape:BxN
                        logprob = torch.log(torch.sum(weights[None,...]*probs,dim=-1)) #shape:B

                        return logprob.squeeze().item() if not batched else logprob.numpy(force=True)

                    priors.append((prior_fn,gt_prior_fn)) # pyright: ignore[reportArgumentType]

                    print("using ground truth prior fn")


                case "smax_gaussian":
                    def prior_likelihood_fn(x:torch.Tensor,t:float,condition:None):
                        tt = torch.as_tensor(t)
                        assert torch.allclose(tt, torch.ones_like(tt)), t #diffeq errors might mean it's not *quite* 1 but that's fine
                        res = (prior_gaussian_nd(x,sigma_max)[0])
                        return res.numpy(force=True) if batched else res.item()
                    priors.append((prior_fn,prior_likelihood_fn))

        return priors

    def compute_likelihoods(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None

        to_array = self.to_array if not batched else self.to_array_batch
        from_array = functools.partial(self.from_array if not batched else self.from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        model_eval = self.load_model(sigma_min,sigma_max,batched,device)

        scorefn = model_eval.score
        divergencefn = model_eval.divergence


        load_samples_fn = (lambda: self.load_samples(self.config.data_samples, device=device)) if batch_size is None else (lambda: self.load_samples_batched(self.config.data_samples, batch_size=batch_size, device=device))
        load_trajectories_fn = (lambda: self.load_trajectories(self.config.trajectory_index_file)) if batch_size is None else (lambda: self.load_trajectories_batched(self.config.trajectory_index_file,batch_size))

        def get_trajectory(path,cond:None)->list[tuple[torch.Tensor,float]]:
            samples,times = self.load_trajectory(path,device=device)
            return list(zip(map(from_array,samples),times)) # pyright: ignore[reportReturnType]


        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        priors = self.load_priors(
                            sigma_min,
                            sigma_max,
                            device,
                            batch_size)

        integrands = get_integrands(self.config,
                                    from_array,
                                    to_array,
                                    scorefn,
                                    divergencefn,
                                    diffusion_coeff_fn)

        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_samples_fn,
                        load_trajectories_fn,
                        get_trajectory,
                        device)

        ### RUN LIKELIHOOD COMPUTATION

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        #TODO: remove from public release
        # #parallel setup (currently nonfunctional)
        # parallel = self.config.get("parallel",False)
        # cluster_kwargs = self.config.get("cluster_kwargs",{})
        # if parallel:
        #     import ray
        #     ray.init(**cluster_kwargs)
        # actor_kwargs = self.config.get("actor_kwargs",{})
        # if device.type == 'cuda' and 'num_gpus' not in actor_kwargs:
        #     actor_kwargs['num_gpus'] = 1 #assume each actor will consume an entire gpu


        int_type = self.config.integral_type

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        write_samples = self.config.get("write_samples",False)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories


        ## WRITE OUTPUT
        acc_trajnum = 0
        with (  #open the various global output csv.DictWriters
                self.likelihoods_writer(True,
                    prior_names=[name for (name,_) in priors],
                    integrand_names=[integrand.name() for integrand in integrands])  as likelihoods_writer,
                self.sample_index_writer(write_samples)                 as samples_writer,
                self.trajectory_index_writers(write_trajectory_index)   as trajectory_indices
            ):

            for (id_batch,path) in tqdm(paths):
                if reset_seed_each_path:
                    torch.manual_seed(seed)
                if int_type == "ode":
                    #just assume paths are ode integrable. error will be thrown otherwise
                    trajectory_batch, times, likelihoods_batch = run_ode_likelihood(path,integrands,accumulate=save_trajectories) # pyright: ignore[reportArgumentType]
                elif int_type == "diff":
                    #use standard integration
                    trajectory_batch, times, likelihoods_batch = run_diff_likelihood(path,integrands,accumulate=save_trajectories)
                else:
                    raise ValueError(f"Unknown integral type: {int_type}. For standard (non-ode solver) numerical integration, use integral_type: \"diff\" (the default).")

                condition_batch = path.condition

                # where to evaluate the prior. should be at time = 1, which is checked in the prior function; here's where the assumption
                # that likelihood paths always *end* at the prior matters, since we have to assume that that's where the prior point is
                # TODO: configurable property of the path, perhaps?
                prior_endpoint_batch:tuple[torch.Tensor,float,None] = (trajectory_batch[-1], times[-1], condition_batch)
                prior_batch:dict[str,float|list[float]] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_endpoint_batch))) for name,prior_fn in priors}

                #since trajectories are in 'array' form they always have a batch dimension
                trajectories = torch.stack(list(trajectory_batch),dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
                assert trajectories.ndim == 3 #BxNxD
                if batch_size is not None: #turn batched results into list of unbatched results
                    ids: Iterable[str|int] = id_batch
                    times = itertools.repeat(times) #only X is actually batched
                    likelihood_results = [
                        {name:np.array(result)[...,i] for name,result in likelihoods_batch.items()}
                        for i in range(batch_size)
                    ]
                    prior_results = [
                        {name:np.array(result)[...,i] for name,result in prior_batch.items()}
                        for i in range(batch_size)
                    ]
                else: #wrap unbatched results as lists
                    ids:Iterable[str|int] = [id_batch] # pyright: ignore[reportAssignmentType]
                    times = [times]
                    likelihood_results = [likelihoods_batch]
                    prior_results = [prior_batch]

                for id, trajectory, time, likelihood_result, prior_result in zip(ids,trajectories,times,likelihood_results,prior_results):

                    if likelihoods_writer:
                        prior_endpoint = trajectory[-1], time[-1]
                        row = {"id":int(id),
                            "prior_position":prior_endpoint[0].item(),  #since here we're in 1d, let's make it a scalar w/ item()
                                "prior_time":torch.as_tensor(prior_endpoint[1]).item(),
                                **{f"prior:{name}":val for name,val in prior_result.items()},
                                **{f"integrand:{name}":val[-1] for name,val in likelihood_result.items()}} #save the last accumulated likelihood
                        likelihoods_writer.writerow(row)

                    if samples_writer:
                        #TODO: CONFIGURATION FOR WHICH POINT TO SAVE
                        sample = {"index":id,"Samples":trajectory[-1].item()}
                        samples_writer.writerow(sample)

                    if save_trajectories: #save trajectory to folder
                        trajectory_name = f"trajectory_{id}.csv"
                        trajectory_file = self.out_trajectory_folder/f"trajectory_{id}.csv"
                        xtraj = torch.as_tensor(trajectory).numpy(force=True)
                        assert xtraj.ndim == 2
                        assert xtraj.shape[1] == 1
                        xtraj = xtraj[:,0] #1d position into scalar
                        ttraj = torch.as_tensor(time).numpy(force=True)
                        trajectory_df = pd.DataFrame({"Timestep":ttraj,"Sample":xtraj})
                        trajectory_df.to_csv(trajectory_file,index_label="index")


                        for cutoff,writer in trajectory_indices.items():
                            if cutoff is None or acc_trajnum < cutoff:
                                writer.writerow({"index":id,"filename":trajectory_name})
                        acc_trajnum += 1


class GaussianForces(ForcesMixin, GaussianLikelihood):
    def get_forces(self):
        # Print the entire configuration
        print(OmegaConf.to_yaml(self.config))

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None
        assert not batched

        to_array = self.to_array if not batched else self.to_array_batch
        from_array = functools.partial(self.from_array if not batched else self.from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        model_eval = self.load_model(sigma_min,sigma_max,batched,device)

        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        load_samples_fn = (lambda: self.load_samples(self.config.data_samples, device=device)) if batch_size is None else (lambda: self.load_samples_batched(self.config.data_samples, batch_size, device=device))
        load_trajectories_fn = (lambda: self.load_trajectories(self.config.trajectory_index_file)) if batch_size is None else (lambda: self.load_trajectories_batched(self.config.trajectory_index_file,batch_size))

        def get_trajectory(path,c:None)->list[tuple[torch.Tensor,float]]:
            samples,times = self.load_trajectory(path,device=device)
            return list(zip(map(from_array,samples),times))


        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_samples_fn,
                        load_trajectories_fn,
                        get_trajectory,
                        device)


        ### RUN FORCES

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        scorecols = ['score']
        poscols = ['pos']

        reset_seed_each_path = self.config.get("reset_seed_each_path",False)
        seed = self.config.get("seed",0)

        with self.forces_index_writer() as index_writer:
            for (id,P) in tqdm(paths):
                if reset_seed_each_path:
                    torch.manual_seed(seed)

                c = P.condition
                forces_csv_file = self.forces_folder/f'{id}.csv'
                index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
                with open(forces_csv_file,'w',newline='') as f2:
                    forces_writer = DictWriter(f2,fieldnames=['Index','Timestep','Diffusion_Coeff','Divergence'] + scorecols + poscols)
                    forces_writer.writeheader()
                    for i,(x,t) in enumerate(P):
                        force = scorefn(x,t,c)
                        div = divergencefn(x,t,c)

                        forcedict = {
                            "Index":i,
                            "Timestep":torch.as_tensor(t).item(),
                            'Diffusion_Coeff':diffusion_coeff_fn(t).item(),
                            'Divergence':torch.as_tensor(div).item(),
                            **dict(zip(scorecols,[force.item()])),
                            **dict(zip(poscols,[x.item()]))
                        }
                        forces_writer.writerow(forcedict)


class GaussianSampler(GaussianLikelihood):

    def sample(self):
        # Print the entire self.configuration
        print(OmegaConf.to_yaml(self.config))

        with open_dict(self.config):
            ## Set various sampling/trajectory output self.config parameters based on sampling parameters:
            # out_dir -> out_dir [unchanged]
            # wt_file | checkpoint -> checkpoint [take either for backwards compatibility]
            # sample_file: Raise warning if specified, all samples now have filename "samples.csv"
            # sample_num: Discard for dfmdock, dfmdock needs samples_per_pdb
            # num_steps -> sde_steps
            # save_trajectory -> save_trajectories: Also sets write_trajectory_index to True

            ## Other relevant parameters:
            # trajectory_extra_indices: list of index cutoffs (e.g. [25, 1000]), default=[]. Will always include a full index

            self.config.path_type = "reverse_sde"
            self.config.integral_type = "diff"
            self.config.sde_timeschedule = "reverse_uniform" #make sure to go from t=1 to t=0!

            if self.config.get("wt_file",None):
                self.config.checkpoint = self.config.wt_file

            if self.config.get("sample_file",None):
                warnings.warn("sample_file included, but will be ignored! Sample files are now always saved to {out_dir}/samples.csv!")

            if self.config.get("num_steps",None):
                self.config.sde_steps = self.config.num_steps

            if self.config.get("save_trajectory",False):
                self.config.save_trajectories = True

            if self.config.get("save_trajectories",False):
                self.config.write_trajectory_index = True

            self.config.write_samples=True
            self.config.write_likelihoods=False

        # set device
        device = torch.device(self.config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

        batched = self.config.get("batched",False)
        batch_size = int(self.config.batch_size) if batched else None

        if batch_size == -1:
            batch_size = self.config.sample_num #Do it in one **massive** batch because why not

        to_array = self.to_array if not batched else self.to_array_batch
        from_array = functools.partial(self.from_array if not batched else self.from_array_batch,device=device)

        # set sigma_values
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        # set models
        model_eval = self.load_model(sigma_min,sigma_max,batched,device)

        scorefn = model_eval.score
        divergencefn = model_eval.divergence

        diffusion_coeff_fn = functools.partial(
            diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = self.config.get("clamp_diffusion_coefficient",False))


        ## SAMPLE DIFFUSION TRAJECTORIES
        def load_noised_samples()->SizedIter[tuple[str|Sequence[str],torch.Tensor,None]]:
            bsize = batch_size or 1
            indices = [(i,) for i in range(math.ceil(self.config.sample_num//bsize))] #make into tuples grumble
            def getchunk(i:int):
                ids = [str(id) for id in range(self.config.sample_num)[i*bsize:(i+1)*bsize]]
                chunksize = len(ids)
                x = torch.randn(chunksize,1,device=device)*sigma_max
                if not batched:
                    ids = ids[0]
                return (ids,x,None)
            return MapDataset(indices,getchunk)

        def err(*args): raise ValueError()
        load_trajectories_fn = err
        get_trajectory_fn = err


        paths = get_paths(self.config,
                        from_array,
                        to_array,
                        scorefn,
                        divergencefn,
                        diffusion_coeff_fn,
                        load_noised_samples,
                        load_trajectories_fn,
                        get_trajectory_fn,
                        device)


        ### RUN SAMPLING

        self.initialize_out_dir()
        self.write_config(self.out_config_file)

        write_samples = self.config.get("write_samples",False)
        save_trajectories = self.config.get("save_trajectories",False)
        write_trajectory_index = self.config.get("write_trajectory_index",True) and save_trajectories

        ## WRITE OUTPUT
        acc_trajnum = 0
        with (  #open the various global output csv.DictWriters
                self.sample_index_writer(write_samples)                 as samples_writer,
                self.trajectory_index_writers(write_trajectory_index)   as trajectory_indices
            ):
            for (id_batch,path) in paths: #ids are potentially batched

                #trajectories are potentially batched
                trajectory_batch, times = unzip(path)

                ## Unbatch results for writing
                #since trajectories are in 'array' form they always have a batch dimension
                trajectories = torch.stack(trajectory_batch,dim=1) #put time-axis in dimension 1 so we can iterate over the batch dimension
                assert trajectories.ndim == 3 #BxNxD
                if not batched:
                    ids:Iterable[str] = [id_batch] # pyright: ignore[reportAssignmentType]
                    times = [times]
                else:
                    ids: Iterable[str] = id_batch
                    times = itertools.repeat(times)

                for id, trajectory, time in zip(ids, trajectories, times):
                    if samples_writer:
                        #TODO: CONFIGURATION FOR WHICH POINT TO SAVE
                        sample = {"index":id,"Samples":trajectory[-1].item()}
                        samples_writer.writerow(sample)

                    if save_trajectories: #save trajectory to folder
                        trajectory_name = f"trajectory_{id}.csv"
                        trajectory_file = self.out_trajectory_folder/f"trajectory_{id}.csv"
                        xtraj = torch.as_tensor(trajectory).numpy(force=True)
                        assert xtraj.ndim == 2
                        assert xtraj.shape[1] == 1
                        xtraj = xtraj[:,0] #1d position into scalar
                        ttraj = torch.as_tensor(time).numpy(force=True)
                        trajectory_df = pd.DataFrame({"Timestep":ttraj,"Sample":xtraj})
                        trajectory_df.to_csv(trajectory_file,index_label="index")

                        for cutoff,writer in trajectory_indices.items():
                            if cutoff is None or acc_trajnum < cutoff:
                                writer.writerow({"index":id,"filename":trajectory_name})
                        acc_trajnum += 1