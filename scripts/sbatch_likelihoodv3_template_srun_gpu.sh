#!/bin/bash
name=$1
shift
srun --job-name=diffenergy_$name --partition=a100 --account=jgray21_gpu --time=08:00:00 -o logs/$name-log.txt bash -c "module load anaconda; conda activate diffenergy; python scripts/likelihoodv3_gaussian_1d.py --config-name=$name" $@