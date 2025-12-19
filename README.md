# DiffEnergy

Source code for ["**Can We Extract Physics-like Energies from Generative Protein Diffusion Models?**"](http://www.biorxiv.org/content/10.1101/2025.11.28.690021) _S. Sarma, H. Truscott, D. Xu, K. Reid, L. Chu, J. Chen, J. Gray_

## Install
Set up conda enviroment and install as following:
```
conda create -n diffenergy python=3.10
conda activate diffenergy
cd /path/to/repo/root
pip install .
```
To install in editable mode, use:
```
pip install -e .
```

## Usage
Scripts for model inference, likelihood calculation, and other tools can be found in the `scripts` folder. `hydra` configs for training, inference, likelihood, and force calcluations are in the `configs` folder. Most scripts directly call classes in `diffenergy/gaussian_1d/inference.py` or `diffenergy/dfmdock/inference.py`. Scripts for generating figure plots are in the `figures` folder. Sampling and likelihood results as used in the paper can be found in the `results` directory, where inference scripts will output and the figure scripts will reference by default.

### Likelihood Calculation
Use `likelihood_gaussian_1d.py` or `likelihood_dfmdock.py` in the `scripts` folder for computing likelihood for the 1D Gaussian case or the translational DFMDock case respectively. These files require you to specify the integration method (ode vs diff, rk4 vs trapezoid) and integration path (flow, diffusion, and others) in a hydra config passed with the --config-name command line option. Configs for flow and diffusion (trapezoid integration) can be found in the `configs` folder. For example, to compute likelihoods of dfmdock samples using flow trajectories, use the following command:

```python scripts/likelihood_dfmdock.py --config-name=dfmdock_flow```

Settings like input/output directory, whether to save the computed trajectories, checkpoint file, and many others can be overwritten using the command line by way of [hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/), like so:

```python scripts/likelihood_dfmdock.py --config-name=dfmdock_diff_trapezoid out_dir=results_2 checkpoint=checkpoints/alternate_weights.ckpt```

Other likelihood configs include: computing the learned energy at each point along a diffusion trajectory using flow (`dfmdock_traj_flow`) and computing the learned energy of the ground truth structures (`dfmdock_gt_flow`).

Config options for alternate paths, integration methods, and other options can be found in the three `inference.py` files.

### Force Calculation
Use `scripts/get_gaussian_forces.py` and `scripts/get_dfmdock_forces.py` with configs `gaussian_traj_forces` and `dfmdock_traj_forces` to record the forces (among other metircs) along Gaussian and dfmdock trajectories respectively.

### Sampling
This code allows for sampling from our trimodel Gaussian model and the modified dfmdock with `python scripts/sample_gaussian_1d.py --config-name=sample_gaussian_1d` and `python scripts/sample_dfmdock.py --config-name=sample_dfmdock` respectively.

### Figure Generation
The code for generating the plots in figures 2-6 as well as the supplemental grid figures can be found in the `figures` folder.
