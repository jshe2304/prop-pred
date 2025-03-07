#!/bin/bash
#SBATCH --job-name=chemprop-train-standardize
#SBATCH --output=/dev/null
#SBATCH --error=train.err

#SBATCH --account=pi-andrewferguson
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

chemprop_train \
--dataset_type regression \
--data_path ../../data/regress_immuno/train/train.csv \
--separate_test_path ../../data/regress_immuno/test/test.csv \
--separate_val_path ../../data/regress_immuno/validation/validation.csv \
--config_path ./hyperopt.json \
--smiles_column SMILES \
--gpu 0 \
--batch_size 32 \
--epochs 1024 \
--save_dir $2 \
--features_generator rdkit_2d_normalized \
--metric r2 \
--no_features_scaling \
