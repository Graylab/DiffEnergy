# python scripts/likelihood_dfmdock.py --config-name=testconf/dfmdock_diff_trapezoid.yaml   &
# python scripts/likelihood_dfmdock.py --config-name=testconf/dfmdock_flow_40.yaml   &
# python scripts/get_dfmdock_forces.py --config-name=testconf/dfmdock_traj_forces_2A1A.yaml   &
# python scripts/sample_dfmdock.py --config-name=testconf/sample_dfmdock_nonoise.yaml   &

python scripts/likelihood_gaussian_1d.py --config-name=testconf/gaussian_diff_trapezoid.yaml   &
# python scripts/likelihood_gaussian_1d.py --config-name=testconf/gaussian_flow.yaml   &
# python scripts/get_gaussian_forces.py --config-name=testconf/gaussian_traj_forces.yaml   &
# python scripts/sample_gaussian_1d.py --config-name=testconf/sample_gaussian_1d.yaml   &