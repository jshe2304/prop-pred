#!/bin/bash
#SBATCH --job-name=svr
#SBATCH --error=svr.err
#SBATCH --output=/dev/null

#SBATCH --account=pi-andrewferguson

#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=2000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 optimize.py $1 $2