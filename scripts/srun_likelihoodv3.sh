#bash sbatch_likelihood_v3_template_srun.sh likelihood_

#dfmdock_tr - GT structure prediction on flow trajectories
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/gtstruct_3integrand_flow_40.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/gtstruct_3integrand_flow.yaml        &

#dfmdock_tr - single traj
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_40_pathreset.yaml       &
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml       &


#dfmdock_tr - NN
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow_40_pathreset.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow_40.yaml              &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_linearized_flow.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_linearized_diff.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff_interp.yaml          &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff_50interp.yaml        &

#dfmdock_tr_2SIC - NN
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_linearized_flow.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_linearized_diff.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff_interp.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff_50interp.yaml   &

#gaussian - NN
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_flow.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_linearized_flow.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_linearized_diff.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp.yaml   '++overwrite_output=True'     &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_50interp.yaml '++overwrite_output=True'     &

#gaussian - GT
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/flow.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/linearized_flow.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/linearized_diff.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_50interp.yaml             &