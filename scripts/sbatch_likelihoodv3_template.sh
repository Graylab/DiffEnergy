#!/bin/bash
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=diffenergy_$1
#SBTACH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=jgray21_gpu
#SBATCH --time=06:00:00
#SBATCH -o logs/$1-log.txt

module load anaconda
conda activate diffenergy

python scripts/likelihoodv3_gaussian_1d.py --config-name=$1
EOT