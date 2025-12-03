#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=a100,ica100,v100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=72:00:00
#SBATCH --qos=qos_gpu
#SBATCH --output=slogs/%j.out

# Enable detailed Hydra error messages
export HYDRA_FULL_ERROR=1

# Define variables
DIR="../dfmdock_perturb_tr_likelihood/dfmdock_inference/no_noise"
TRAIN_SET="dips"
TEST_SET="db5_test"
MODEL="DFMDock_model_0"
NUM_SAMPLES=1 #deterministic - only one sample per pdb
NUM_STEPS=40
NOISE_SCALE=0 #no noise - deterministic output! Comparable with likelihoodv3

# Loop over three runs
srun python ../darren_DFMDock_sampler/src/inference_DFMDock.py \
    data.perturb_rot=False \
    data.ckpt=checkpoints/dfmdock_model_0.ckpt \
    data.dataset=${TEST_SET} \
    data.out_csv_dir=${DIR}/csv_files/ \
    data.out_csv=${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}_${RUN}.csv \
    data.num_samples=${NUM_SAMPLES} \
    data.num_steps=${NUM_STEPS} \
    data.out_pdb=True \
    data.out_trj=True \
    data.out_pdb_dir=${DIR}/pdbs/${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}/run${RUN} \
    data.out_trj_dir=${DIR}/trjs/${TEST_SET}_${MODEL}_${NOISE_SCALE}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}/run${RUN} \
    data.test_all=True \
    data.use_clash_force=False \
    data.tr_noise_scale=${NOISE_SCALE} \
    data.rot_noise_scale=${NOISE_SCALE} \
