# DiffEnergy

Source code for "**Are Learned Energies from Diffusion Models Capturing Biophysics?**" _S. Sarma, D. Xu, L. Chu, J. Gray_

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

## Likelihood Calculation
Use `likelihood_gaussian_1d.py` or `likelihood_dfmdock_tr.py` in the `scripts` folder for computing likelihood for the 1D Gaussian case or the translational DFMDock case. Default `hydra` configs are in the `configs` folder as `likelihood_gaussian_1d.yaml` and `likelihood_dfmdock_tr.yaml`. Modify configs in command line as follows:
```
python likelihood_gaussian_1d.py \
    checkpoint=../checkpoints/1d_trinormal.ckpt \
    data_samples=../../data/trinormal_1d.csv \
    inference_type=FlowTimeIntegral \
    out_file=../../results/trinormal_1d_flowtime_nll.csv \
```
