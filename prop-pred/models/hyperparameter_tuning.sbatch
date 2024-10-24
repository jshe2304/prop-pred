#!/bin/bash
#SBATCH --job-name=rfr-tuning
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=16:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=2000

python3 hyperparameter_tuning.py $1