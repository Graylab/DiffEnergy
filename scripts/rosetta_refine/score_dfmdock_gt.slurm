#!/bin/bash -l

#----------------------------------------------------
# SLURM job script to run RosettaDock MPI application 
# on MARCC.
#
# usage: from within the directory where your flag files are,
# submit to cluster using sbatch 2SIC_docking.qsub
#----------------------------------------------------

#SBATCH --job-name=rosetta_refine_gt
#SBATCH --time=72:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=slogs/%j.out
#SBATCH --account=jgray21
ml gcc

# job description
ROSETTABIN=/scratch16/jgray21/aharmal1/Rosetta/main/source/bin  
ROSETTAEXE=docking_protocol
COMPILER=mpi.linuxgccrelease
EXE=$ROSETTABIN/$ROSETTAEXE.$COMPILER
echo Starting MPI job running $EXE

mkdir -p results/rosetta/dfmdock_gt/refined_pdbs

# to give rosetta the best chance of finding a good score we make 5 structures each
date
time mpirun $EXE \
    -in:file:l scripts/rosetta_refine/dfmdock_gt_index.txt \
    -out:file:scorefile results/rosetta/dfmdock_gt/refined_score.sc \
    -out:path:pdb results/rosetta/dfmdock_gt/refined_pdbs \
    -nstruct 5 \
    @score_refine_flags
date

python scripts/rosetta_refine/score_to_csv.py results/rosetta/dfmdock_gt/refined_score.sc results/rosetta/dfmdock_gt/refined_score.csv