#!/bin/bash

#SBATCH --job-name=jay
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 hyperparameter_search.py $1
