#!/bin/bash
#SBATCH --job-name=unimol
#SBATCH --output=/dev/null
#SBATCH --error=train.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python train.py $1 $2