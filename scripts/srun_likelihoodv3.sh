#bash sbatch_likelihood_v3_template_srun.sh likelihood_



##DFMDock_trtrained_deterministic - Deterministic Score on newly generated trajectories/samples

#Neighborhood Sampling - perturbing the gt structure in a neighborhood around the correct structure
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/centerline_flow_40.yaml              &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/centerline_dense_flow_40.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/centerline_forward_sde.yaml          &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/transplane_dense_flow_40.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/transplane_centered_flow_40.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/transplane_flow_40.yaml              &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/neighborhood/transplane_forward_sde.yaml          &


#DFMDock_trtrained_deterministic - GT structure prediction on flow trajectories
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/gtstruct/3integrand_flow_40.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/gtstruct/3integrand_flow.yaml        &

#DFMDock_trtrained_deterministic - single traj
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_traj_flow_40.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_traj_flow_40_2A1A.yaml &

#DFMDock_trtrained_deterministic - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_flow.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_flow_40.yaml              &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_diff.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_diff_trapezoid.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_linearized_flow.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_linearized_diff.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_diff_interp_trapezoid.yaml          &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/3integrand_diff_50interp.yaml        &

#DFMDock_trtrained_deterministic - 2SIC Only
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_flow.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_diff.yaml           ++overwrite_output=True &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_linearized_flow.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_diff_interp.yaml    ++overwrite_output=True &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_diff_50interp.yaml   &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/3integrand_linearized_diff.yaml &

#DFMDock_trtrained_deterministic - 1IRA Only
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_flow.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_diff.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_linearized_flow.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_diff_interp.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_diff_50interp.yaml   &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/1IRA/3integrand_linearized_diff.yaml &


# Deterministic score on OLD SAMPLES
#DFMDock_trtrained_deterministic on OLD SAMPLES - GT structure prediction on flow trajectories
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples/gtstruct/3integrand_flow_40.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples/gtstruct/3integrand_flow.yaml        &

##DFMDock_trtrained_deterministic on OLD SAMPLES - single traj
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples/3integrand_traj_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples/3integrand_flowtraj_flow_40.yaml     &

##DFMDock_trtrained_deterministic on OLD SAMPLES - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_flow.yaml                  &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_flow_40.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_diff.yaml                  &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_linearized_flow.yaml       &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_linearized_diff.yaml       &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_diff_interp.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/oldsamples3integrand_diff_50interp.yaml         &

##DFMDock_trtrained_deterministic on OLD SAMPLES - 2SIC Only
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_flow.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_diff.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_linearized_flow.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_diff_interp.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_diff_50interp.yaml   &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_trtrained_deterministic/2SIC/oldsamples/3integrand_linearized_diff.yaml &



##DFMDock_tr

#dfmdock_tr - GT structure prediction on flow trajectories
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/gtstruct_3integrand_flow_40.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/gtstruct_3integrand_flow.yaml        &

#dfmdock_tr - single traj
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_40_pathreset.yaml       &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_traj_flow_pathreset.yaml       &

#dfmdock_tr - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow_40_pathreset.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_flow_40.yaml              &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff.yaml                 &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_linearized_flow.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_linearized_diff.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff_interp.yaml          &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/3integrand_diff_50interp.yaml        &

#dfmdock_tr_2SIC - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40_pathreset.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_flow_40.yaml         &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff.yaml            &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_linearized_flow.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff_interp.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_diff_50interp.yaml   &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh dfmdock_tr/2SIC/3integrand_linearized_diff.yaml &

##Gaussian 1d

#gaussian - NN
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_flow.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_backwards.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_trapezoid.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_linearized_flow.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_linearized_diff.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp_trapezoid.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp_backwards.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_50interp.yaml      &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp_all.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp_all_trapezoid.yaml     &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_interp_all_backwards.yaml      &
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_piecewise_ode_1000.yaml   &
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh gaussian_1d/3integrand_diff_piecewise_ode_all.yaml   &

#gaussian - GT
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/flow.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/linearized_flow.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/linearized_diff.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp_all.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp_all_backwards.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp_all_trapezoid.yaml &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_interp.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_piecewise_ode_1000.yaml   &
# bash scripts/sbatch_likelihoodv3_template_srun.sh gaussian_1d/gtscore/diff_piecewise_ode_all.yaml   &