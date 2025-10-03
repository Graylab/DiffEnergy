#python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_flow.yaml && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_diff.yaml && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_linearized_flow.yaml && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_linearized_diff.yaml && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_diff_interp.yaml && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation.yaml '++overwrite_output=True' && \
# python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation_perturbed.yaml '++overwrite_output=True' && \
python scripts/likelihoodv3_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation_diffint.yaml '++overwrite_output=True' && \
echo "all likelihoods computed successfully!"