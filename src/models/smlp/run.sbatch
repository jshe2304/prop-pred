#!/bin/bash

#SBATCH --job-name=siamese
#SBATCH --error=siamese.err
#SBATCH --output=siamese.out
#SBATCH --account=pi-andrewferguson
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 $1 $2
