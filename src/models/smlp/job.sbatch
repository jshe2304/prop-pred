#!/bin/bash

#SBATCH --job-name=siamese_big
#SBATCH --error=siamese_big.err
#SBATCH --output=siamese_big.out
#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 siamese.py
