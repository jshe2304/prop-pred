import pandas as pd
import numpy as np
import sys
import json

from utils.fingerprinter import Fingerprinter

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn

model_type = sys.argv[1]

##########
# Datasets
##########

train = pd.read_csv('../train_clean.csv')
test = pd.read_csv('../test_clean.csv')
properties = train.columns[1:]

fingerprinter = Fingerprinter()

train_X = fingerprinter(train['SMILES'].tolist())
test_X = fingerprinter(test['SMILES'].tolist())
if model_type == 'mlp':
    train_X = torch.tensor(train_X, dtype=torch.float32, requires_grad=False)
    test_X = torch.tensor(test_X, dtype=torch.float32, requires_grad=False)

n_features = train_X.shape[1]

################################
# Load Model and Parameter Space
################################

if model_type == 'mlp': from utils.mlp import MLP as Model
elif model_type == 'rfr': from sklearn.ensemble import RandomForestRegressor as Model
elif model_type == 'svr': from sklearn.svm import SVR as Model
elif model_type == 'xgbr': from xgboost import XGBRegressor as Model

with open(f'./hyperparameters/{model_type}_hyperparameter_space.json') as f:
    param_space = ParameterGrid(json.load(f))

######################
# Fit and Score Models
######################

tuning_dict = {
    'model': model_type, 
    'grid_len': len(param_space), 
    'properties': {}
}

for property_label in properties:

    train_Y = train[property_label]
    test_Y = test[property_label]
    if model_type == 'mlp':
        train_Y = torch.tensor(train_Y, dtype=torch.float32, requires_grad=False)
        test_Y = torch.tensor(test_Y, dtype=torch.float32, requires_grad=False)

    best_score = float('-inf')
    best_params = None
    for params in param_space:

        print(params)

        model = Model(**params)
        model.fit(train_X, train_Y)
        score = model.score(test_X, test_Y)

        if score > best_score:
            best_score = score
            best_params = params

    tuning_dict['properties'][property_label] = {
        'score': best_score, 
        'hyperparameters': best_params
    }

with open(f'./hyperparameters/{model_type}.json', 'w') as f:
    json.dump(tuning_dict, f, indent=4)