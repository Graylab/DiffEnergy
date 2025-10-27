#bash sbatch_likelihood_v3_template_srun.sh likelihood_

#2SIC flow w/ 4 alternate seeds
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml \
    "seed=10" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/2SIC/3integrand_flow_40_pathreset/seed10" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml \
    "seed=20" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/2SIC/3integrand_flow_40_pathreset/seed20" &
    
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml \
    "seed=30" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/2SIC/3integrand_flow_40_pathreset/seed30" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml \
    "seed=40" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/2SIC/3integrand_flow_40_pathreset/seed40" &

#traj_flow (on diffusion trajectory) w/ 4 alternate seeds
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml \
    "seed=10" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_traj_flow_pathreset/seed10" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml \
    "seed=20" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_traj_flow_pathreset/seed20" &
    
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml \
    "seed=30" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_traj_flow_pathreset/seed30" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml \
    "seed=40" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_traj_flow_pathreset/seed40" &

#flowtraj_flow (on seed 0 flow trajectory) w/ 4 alternate seeds
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flowtraj_flow.yaml \
    "seed=10" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_flowtraj_flow_pathreset/seed10" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flowtraj_flow.yaml \
    "seed=20" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_flowtraj_flow_pathreset/seed20" &
    
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flowtraj_flow.yaml \
    "seed=30" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_flowtraj_flow_pathreset/seed30" &

bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flowtraj_flow.yaml \
    "seed=40" \
    "out_dir=../likelihood_results/likelihoodv3/dfmdock_tr/3integrand_flowtraj_flow_pathreset/seed40" &