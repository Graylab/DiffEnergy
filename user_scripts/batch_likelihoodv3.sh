#python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/
# python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_flow.yaml && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_diff_interp_all_dum.yaml && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_diff_interp_all_trapezoid.yaml && \
python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_diff_interp_all_backwards.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_linearized_flow.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_linearized_diff.yaml && \
# python scripts/likelihood_gaussian_1d.py --config-name=gaussian_1d/3integrand_diff_interp.yaml && \
echo "all likelihoods computed successfully!"