import pandas as pd
import numpy as np
import json
from os import path
from sklearn.model_selection import ParameterGrid

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganGenerator
morgan = GetMorganGenerator(radius=2)
fingerprinter = lambda smiles : np.stack([morgan.GetFingerprintAsNumPy(MolFromSmiles(smile)) for smile in smiles])

import torch
import torch.nn as nn
from models import MLP, RMLP, VMLP, RVMLP

model_list = [MLP, RMLP, VMLP, RVMLP]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

############################################################
############################################################
# DATA 
############################################################
############################################################

data_path = '../../data/immunomodulation/log_normalized/'

# Unimol Embeddings

train_X = np.load(path.join(data_path, 'train/unimol_embeddings.npy'))
validation_X = np.load(path.join(data_path, 'validation/unimol_embeddings.npy'))
test_X = np.load(path.join(data_path, 'test/unimol_embeddings.npy'))

# SMILES Fingerprints
# train_X = pd.read_csv('../../data/transformed/train/smiles.csv')
# train_X = fingerprinter(train_X.to_numpy().squeeze().tolist())
# validation_X = pd.read_csv('../../data/transformed/validation/smiles.csv')
# validation_X = fingerprinter(validation_X.to_numpy().squeeze().tolist())
# test_X = pd.read_csv('../../data/transformed/test/smiles.csv')
# test_X = fingerprinter(test_X.to_numpy().squeeze().tolist())

in_features = train_X.shape[1]

# Regression Targets

train_Y = pd.read_csv(path.join(data_path, 'train/y.csv'))
validation_Y = pd.read_csv(path.join(data_path, 'validation/y.csv'))
test_Y = pd.read_csv(path.join(data_path, 'test/y.csv'))

train_STD = pd.read_csv(path.join(data_path, 'train/std.csv'))
validation_STD = pd.read_csv(path.join(data_path, 'validation/std.csv'))
test_STD = pd.read_csv(path.join(data_path, 'test/std.csv'))

# Convert data to Tensors

train_X = torch.tensor(train_X, dtype=torch.float32).detach().to(device)
validation_X = torch.tensor(validation_X, dtype=torch.float32).detach().to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).detach().to(device)

train_Y = torch.tensor(train_Y.to_numpy(), dtype=torch.float32).detach().to(device)
validation_Y = torch.tensor(validation_Y.to_numpy(), dtype=torch.float32).detach().to(device)
test_Y = torch.tensor(test_Y.to_numpy(), dtype=torch.float32).detach().to(device)

train_Var = torch.tensor(train_STD.to_numpy(), dtype=torch.float32).detach().to(device) ** 2
validation_Var = torch.tensor(validation_STD.to_numpy(), dtype=torch.float32).detach().to(device) ** 2
test_Var = torch.tensor(test_STD.to_numpy(), dtype=torch.float32).detach().to(device) ** 2

############################################################
############################################################
# Training
############################################################
############################################################

with open('./hyperparameters/hyperparameters.json') as f:
    hyperparameter_grid = ParameterGrid(json.load(f))

n_folds = 4

print('depth,width,dropout,mlp,rmlp,vmlp,rvmlp')
for hyperparameters in hyperparameter_grid:

    mean_scores = []

    # Train models and record mean score
    for Model in model_list:
        scores = [
            Model(in_features, **hyperparameters).to(device).fit(
                train_X=train_X, train_Y=train_Y, train_Var=train_Var, 
                test_X=test_X, test_Y=test_Y, 
            ) for _ in range(n_folds)
        ]
        mean_scores.append(sum(scores)/n_folds)

    # Print hyperparameters and model scores
    print(
        ','.join(str(n) for n in list(hyperparameters.values()) + mean_scores)
    )
