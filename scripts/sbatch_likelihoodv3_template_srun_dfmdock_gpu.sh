#!/bin/bash
name=$1
shift
srun --job-name=diffenergy_$name --qos=qos_gpu --gres=gpu:1 --partition=a100,ica100 --account=jgray21_gpu --time=72:00:00 -o logs/$name-log.txt bash -c "module load anaconda; conda activate diffenergy; python scripts/likelihoodv3_dfmdock.py --config-name=$name $*"