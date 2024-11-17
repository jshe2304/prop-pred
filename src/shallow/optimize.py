import pandas as pd
import numpy as np
import os
import sys
import json

from utils.fingerprinter import Fingerprinter

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn

data_path = sys.argv[1]
model_type = sys.argv[2]

##########
# Datasets
##########

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
properties = train.columns[1:]

fingerprinter = Fingerprinter()

train_X = fingerprinter(train['SMILES'].tolist())
test_X = fingerprinter(test['SMILES'].tolist())

n_features = train_X.shape[1]

################################
# Load Model and Parameter Space
################################

if model_type == 'mlp': from utils.mlp import MLP as Model
elif model_type == 'rfr': from sklearn.ensemble import RandomForestRegressor as Model
elif model_type == 'svr': from sklearn.svm import SVR as Model
elif model_type == 'xgbr': from xgboost import XGBRegressor as Model

space_fname = f'./hyperparameters/spaces/{model_type}.json'
optimized_fname = f'./hyperparameters/optimized/{model_type}.json'

with open(space_fname) as f:
    param_space = ParameterGrid(json.load(f))

######################
# Fit and Score Models
######################

try:
    with open(optimized_fname) as f:
        optimized = json.load(f)
        properties = [p for p in properties if p not in optimized['properties'].keys()]
except:
    optimized = {
        'model': model_type, 
        'grid_len': len(param_space), 
        'properties': {}
    }

for property_label in properties:

    train_Y = train[property_label]
    test_Y = test[property_label]

    best_score = float('-inf')
    best_params = None
    for params in param_space:

        if model_type == 'mlp': params['n_features'] = n_features

        model = Model(**params)
        model.fit(train_X, train_Y)
        score = model.score(test_X, test_Y)

        print(params, float(score))

        if score > best_score:
            best_score = score
            best_params = params

    optimized['properties'][property_label] = {
        'score': best_score, 
        'hyperparameters': best_params
    }

    with open(optimized_fname, 'w') as f:
        json.dump(optimized, f, indent=4)