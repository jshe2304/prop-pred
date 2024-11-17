#!/bin/bash
#SBATCH --job-name=chemprop-hyperopt
#SBATCH --output=/dev/null
#SBATCH --error=hyperopt.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive

chemprop_hyperopt \
--dataset_type regression \
--data_path $1/train.csv \
--separate_test_path $1/test.csv \
--separate_val_path $1/test.csv \
--save_dir $2 \
--hyperopt_checkpoint_dir ./hyperopt_log_standardize_prefix \
--config_save_path hyperopt.json \
--smiles_column SMILES \
--gpu 0 \
--batch_size 32 \
--epochs 512 \
--features_generator rdkit_2d_normalized \
--no_features_scaling \
--num_iters 64 \
