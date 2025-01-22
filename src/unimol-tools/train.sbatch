#!/bin/bash
#SBATCH --job-name=v1
#SBATCH --output=/dev/null
#SBATCH --error=v1.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python train.py $1