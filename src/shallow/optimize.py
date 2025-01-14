import pandas as pd
import numpy as np
import os
import sys
import json

from utils.fingerprinter import Fingerprinter

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

import torch
import torch.nn as nn

data_path = sys.argv[1]
model_type = sys.argv[2]

##########
# Datasets
##########

train = pd.read_csv(os.path.join(data_path, 'train.csv'))

fingerprinter = Fingerprinter()
X = fingerprinter(train['SMILES'].tolist())
Y = train.iloc[:, 1:].to_numpy()

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
    param_space_dict = {('estimator__' + k) : v for k, v in json.load(f).items()}
    print(param_space_dict)

######################
# Fit and Score Models
######################

regressor = MultiOutputRegressor(Model())

print('Regressing')

regressor_grid = GridSearchCV(
    estimator=regressor, 
    param_grid=param_space_dict, 
    n_jobs = -1
)
regressor_grid.fit(X, Y)

print(regressor_grid.best_score_)
print(regressor_grid.best_params_)
print(regressor_grid.best_index_)
print(regressor_grid.cv_results_)