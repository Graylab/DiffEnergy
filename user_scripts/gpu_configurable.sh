#!/bin/bash
file=$1
name=$2
shift
shift
srun --job-name=diffenergy_$name --qos=qos_gpu --gres=gpu:1 --partition=a100,ica100 --account=jgray21_gpu --time=16:00:00 -o logs/$name-log.txt bash -c "module load anaconda; conda activate diffenergy; python scripts/$file --config-name=$name $*"