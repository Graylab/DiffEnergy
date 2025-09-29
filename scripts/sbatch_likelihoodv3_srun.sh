#bash sbatch_likelihood_v3_template_srun.sh likelihood_gaussian_1d_

#NN
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_flow.yaml               '++num_gpus=4'&
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_diff.yaml               '++num_gpus=4'&
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_linearized_flow.yaml    '++num_gpus=4'&
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_linearized_diff.yaml    '++num_gpus=4'&
bash scripts/sbatch_likelihoodv3_template_srun_gpu.sh likelihood_gaussian_1d_3integrand_diff_interp.yaml        '++num_gpus=4'&

#GT
bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_flow.yaml                      &
bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_diff.yaml                      &
bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_linearized_flow.yaml           &
bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_linearized_diff.yaml           &
bash scripts/sbatch_likelihoodv3_template_srun.sh likelihood_gaussian_1d_gtscore_diff_interp.yaml               &