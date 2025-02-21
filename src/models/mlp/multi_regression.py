import pandas as pd
import numpy as np
import json
from os import path
import sys
from sklearn.model_selection import ParameterGrid

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganGenerator
morgan = GetMorganGenerator(radius=2)
fingerprinter = lambda smiles : np.stack([morgan.GetFingerprintAsNumPy(MolFromSmiles(smile)) for smile in smiles])

import torch
import torch.nn as nn
from models import RMLPRegressor

log_file = sys.argv[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

############################################################
############################################################
# DATA 
############################################################
############################################################

data_path = '../../data/immunomodulation/log_normalized/regression'

# Unimol Embeddings

train_X = np.load(path.join(data_path, 'train/unimol_embeddings.npy'))
validation_X = np.load(path.join(data_path, 'validation/unimol_embeddings.npy'))
test_X = np.load(path.join(data_path, 'test/unimol_embeddings.npy'))

in_features = train_X.shape[1]

# Regression Targets

train_Y = pd.concat((
    pd.read_csv(path.join(data_path, 'train/y.csv')), 
    pd.read_csv(path.join(data_path, 'train/y_rdkit.csv'))
), axis=1)
validation_Y = pd.concat((
    pd.read_csv(path.join(data_path, 'validation/y.csv')), 
    pd.read_csv(path.join(data_path, 'validation/y_rdkit.csv'))
), axis=1)
test_Y = pd.concat((
    pd.read_csv(path.join(data_path, 'test/y.csv')), 
    pd.read_csv(path.join(data_path, 'test/y_rdkit.csv')), 
), axis=1)

properties = train_Y.columns

# Convert data to Tensors

train_X = torch.tensor(train_X, dtype=torch.float32).detach().to(device)
validation_X = torch.tensor(validation_X, dtype=torch.float32).detach().to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).detach().to(device)

train_Y = torch.tensor(train_Y.to_numpy(), dtype=torch.float32).detach().to(device)
validation_Y = torch.tensor(validation_Y.to_numpy(), dtype=torch.float32).detach().to(device)
test_Y = torch.tensor(test_Y.to_numpy(), dtype=torch.float32).detach().to(device)

############################################################
############################################################
# Training
############################################################
############################################################

hyperparameters = {'depth': 2, 'width': 512, 'dropout': 0.5}

with open(log_file, 'w') as f:
    f.write(','.join(properties) + '\n')

for trial in range(1024):

    model = RMLPRegressor(in_features, train_Y.shape[1], **hyperparameters).to(device)
    model.fit(
        train_X, train_Y, 
        validation_X, validation_Y, 
        lr=0.0001, epochs=512, batch_size=32, patience=0.0016
        early_stopping=True, plot=False, show_progress=False
    )
    scores = model.score(test_X, test_Y, multioutput=None)
    
    with open(log_file, 'a') as f:
        f.write(','.join(list(str(score) for score in scores)) + '\n')
