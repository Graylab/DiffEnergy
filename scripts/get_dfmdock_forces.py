
import hydra
from omegaconf import DictConfig

from diffenergy.dfmdock_tr.inference import DFMDockForces

@hydra.main(version_base=None, config_path='../configs/likelihoodv3')
def main(config:DictConfig):
    forces = DFMDockForces(config)
    forces.get_forces()

    
    # # Print the entire configuration
    # print(OmegaConf.to_yaml(config))

    # # set device
    # device = torch.device(config.get("device","cuda" if torch.cuda.is_available() else "cpu"))

    # batched = config.get("batched",False)
    # batch_size = int(config.batch_size) if batched else None
    # if batched:
    #     raise ValueError("Batched DFMDock evaluation not supported!")

    # to_array = DFMDock_to_array_nobatch# if not batched else to_array_batch
    # from_array = functools.partial(DFMDock_from_array_nobatch,device=device)# if not batched else from_array_batch,device=device)

    # # set sigma_values
    # sigma_min = config.sigma_min
    # sigma_max = config.sigma_max

    # # set models
    # score_model = Score_Model.load_from_checkpoint(config.checkpoint,deterministic=config.get("deterministic_score",False))
    # score_model.freeze()
    # score_model.to(device)

    # offset_type:Literal["Translation","Rotation","Translation+Rotation"] = config.offset_type
    # valid_offsets = ["Translation", "Rotation", "Translation+Rotation"]
    # if offset_type not in valid_offsets:
    #     raise ValueError("offset_type must be one of",valid_offsets)

    # model_eval = DFMDockModelEval(score_model,offset_type=offset_type,reset_seed_each_eval=config.get("reset_seed_each_eval",False),manual_seed=config.get("seed",0))
    
    # scorefn = model_eval.score
    # divergencefn = model_eval.divergence

    # esm_model = ESMLanguageModel()
    # pdb_importer = PDBImporter(esm_model,esm_model.alphabet)

    # assert batch_size is None
    # load_samples_fn = lambda: load_samples(config.data_samples, config.pdb_dir, offset_type, pdb_importer, batch_size=batch_size, device=device)
    # load_trajectories_fn = lambda: load_trajectories(config.trajectory_index_file,config.pdb_dir,config.trajectory_dir,pdb_importer,batch_size=batch_size)
    # get_trajectory_fn = lambda trajectory_file,condition: load_trajectory(trajectory_file, config.pdb_dir, offset_type,condition, device=device) #TODO: add pdb trajectory support, add pdb_dir as parameter [filenames are relative to pdb_dir]

    # diffusion_coeff_fn = functools.partial(
    #     diffusion_coeff, sigma_min = sigma_min, sigma_max = sigma_max, clamp = config.get("clamp_diffusion_coefficient",False))


    # paths = get_paths(config,
    #                 from_array,
    #                 to_array,
    #                 scorefn,
    #                 divergencefn,
    #                 diffusion_coeff_fn,
    #                 load_samples_fn,
    #                 load_trajectories_fn,
    #                 get_trajectory_fn,
    #                 device)
    
    # out_folder = Path(config.out_dir)
    # forces_folder = out_folder/'forces'
    # forces_folder.mkdir(exist_ok=True,parents=True)
    # config_copy = out_folder/"config.yaml"
    # with open(config_copy,"w") as f:
    #     f.write(OmegaConf.to_yaml(config))
    # index_file = out_folder/'force_index.csv'
    # cols = offset_trajectory_columns(offset_type)
    # scorecols = [f'score:{col}' for col in cols]
    # poscols = [f'pos:{col}' for col in cols]

    # reset_seed_each_path = config.get("reset_seed_each_path",False)
    # seed = config.get("seed",0)
    # with open(index_file,'w',newline='') as f:
    #     index_writer = DictWriter(f,fieldnames=['id','Forces_CSV'])
    #     index_writer.writeheader()
    #     for (id,P) in paths:
    #         if reset_seed_each_path:
    #             torch.manual_seed(seed)

    #         c = P.condition
    #         forces_csv_file = forces_folder/f'{id}.csv'
    #         index_writer.writerow({"id":id,"Forces_CSV":forces_csv_file})
    #         with open(forces_csv_file,'w',newline='') as f2:
    #             forces_writer = DictWriter(f2,fieldnames=['Index','Timestep','Diffusion_Coeff','Divergence'] + scorecols + poscols)
    #             forces_writer.writeheader()
    #             for i,(x,t) in enumerate(P):
    #                 force = scorefn(x,t,c)
    #                 div = divergencefn(x,t,c)

    #                 forcedict = {
    #                     "Index":i,
    #                     "Timestep":torch.as_tensor(t).item(),
    #                     'Diffusion_Coeff':diffusion_coeff_fn(t).item(),
    #                     'Divergence':div.item(),
    #                     **dict(zip(scorecols,force.tolist())),
    #                     **dict(zip(poscols,x["offset"].tolist()))
    #                 }
    #                 forces_writer.writerow(forcedict)
            


if __name__ == "__main__":
    main()