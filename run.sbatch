#!/bin/bash
#SBATCH --job-name=fitting
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --account=pi-andrewferguson
#SBATCH --time=4:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=2000

jupyter nbconvert --execute ./prop-pred/models/svm.ipynb
jupyter nbconvert --execute ./prop-pred/models/rf.ipynb
