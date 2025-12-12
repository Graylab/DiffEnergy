# bash scripts/gpu_configurable.sh likelihood_dfmdock.py testconf/dfmdock_diff_trapezoid.yaml   &
# bash scripts/gpu_configurable.sh likelihood_dfmdock.py testconf/dfmdock_flow_40.yaml   &
# bash scripts/gpu_configurable.sh get_dfmdock_forces.py testconf/dfmdock_traj_forces_2A1A.yaml   &
# bash scripts/gpu_configurable.sh sample_dfmdock.py testconf/sample_dfmdock_nonoise.yaml   &

bash scripts/gpu_configurable.sh likelihood_gaussian_1d.py testconf/gaussian_diff_trapezoid.yaml   &
# bash scripts/gpu_configurable.sh likelihood_gaussian_1d.py testconf/gaussian_flow.yaml   &
# bash scripts/gpu_configurable.sh get_gaussian_forces.py testconf/gaussian_traj_forces.yaml   &
# bash scripts/gpu_configurable.sh sample_gaussian_1d.py testconf/sample_gaussian_1d.yaml   &