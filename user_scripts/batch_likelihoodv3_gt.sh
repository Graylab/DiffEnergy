#python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_flow.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_diff.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_linearized_flow.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_linearized_diff.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_diff_interp.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation.yaml '++overwrite_output=True' && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation_perturbed.yaml '++overwrite_output=True' && \
# python scripts/likelihood_gaussian_1d.py --config-name=likelihood_gaussian_1d_gtscore_translation_diffint.yaml '++overwrite_output=True' && \

python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/gtscore/diff_interp_trapezoid && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/gtscore/diff_interp_backwards && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/gtscore/diff_trapezoid && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/gtscore/diff_backwards && \
echo "all likelihoods computed successfully!"