from unimol_tools import MolTrain

import pandas as pd
import sys
import os

save_path = sys.argv[1]

#######
# Model
#######

model = MolTrain(
    task='multilabel_regression', 
    data_type='molecule', 
    epochs=1024, 
    learning_rate=0.0001, 
    batch_size=32, 
    early_stopping=5, 
    metrics='mse', 
    #split='random', 
    #split_group_col='scaffold', 
    kfold=5, 
    save_path=save_path, 
    remove_hs=False, 
    smiles_col='SMILES', 
    target_col_prefix='property_', 
    target_anomaly_check='filter', 
    smiles_check='filter', 
    target_normalize='none', 
    max_norm=5.0, 
    use_cuda=True, 
    use_amp=True, 
    freeze_layers=None, 
    freeze_layers_reversed=False,
)

model.fit(data='../../data/immunomodulation/log_normalized/regression/train.csv')