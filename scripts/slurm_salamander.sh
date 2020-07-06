#!/bin/bash
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 1000  # memory pool for all cores
#SBATCH -t 0-0:01 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
source /home/arreguit/farmsenv/bin/activate
cd /home/arreguit/farms/farms_amphibious/scripts/
python3 slurm_salamander_job.py --duration 10 --timestep 0.001 --fast --headless
