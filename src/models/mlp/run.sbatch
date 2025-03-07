#!/bin/bash

#SBATCH --job-name=rmlpembmulti
#SBATCH --error=rmlpembmulti.err
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=03:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 $1 $2