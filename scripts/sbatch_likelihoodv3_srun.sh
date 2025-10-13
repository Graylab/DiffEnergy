#bash sbatch_likelihood_v3_template_srun.sh likelihood_

#dfmdock_tr - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_flow.yaml               &
bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_diff.yaml               '++overwrite_output=True' &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_linearized_flow.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_linearized_diff.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_diff_interp.yaml        &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_3integrand_diff_50interp.yaml      &

#dfmdock_tr_2SIC - NN
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_flow.yaml              '++overwrite_output=True' &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_diff.yaml              '++overwrite_output=True' &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_linearized_flow.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_linearized_diff.yaml   '++overwrite_output=True' &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_diff_interp.yaml       '++overwrite_output=True' &
# bash scripts/sbatch_likelihoodv3_template_srun_dfmdock_gpu.sh likelihood_dfmdock_tr_2SIC_3integrand_diff_50interp.yaml      &

#gaussian - NN
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_flow.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_diff.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_linearized_flow.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_linearized_diff.yaml    &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_diff_interp.yaml   '++overwrite_output=True'     &
# bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_diff_50interp.yaml '++overwrite_output=True'     &

#gaussian - GT
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_flow.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_diff.yaml                      &
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_linearized_flow.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_linearized_diff.yaml           &
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_diff_interp.yaml               &
# bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_diff_50interp.yaml             &