import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils import *

import torch

from sklearn.model_selection import ParameterGrid

from models import MLPRegressor, RMLPRegressor

model_list = [MLPRegressor, RMLPRegressor]

log_file = sys.argv[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

############################################################
############################################################
# DATA 
############################################################
############################################################

datadir = '../../data/regress_immuno'
data = to_namedtuple(to_tensor(load_data(datadir), device))

train_X = data.train.embeddings
validation_X = data.validation.embeddings
test_X = data.test.embeddings

train_Var = data.train.y_err ** 2
validation_Var = data.validation.y_err ** 2

train_Y = data.train.y
validation_Y = data.validation.y
test_Y = data.test.y

dataset = GaussianSamplingDataset(train_X, train_Y, train_Var)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

############################################################
############################################################
# Training
############################################################
############################################################

hyperparameter_grid = ParameterGrid({
    "depth": [1, 2, 3, 4, 5],
    "width": [100, 250, 400, 550, 700, 850],
    "dropout": [0.2, 0.3, 0.4, 0.5, 0.6]
})

n_folds = 4

with open(log_file, 'w') as f:
    f.write('depth,width,dropout,mlp,rmlp\n')

for hyperparameters in hyperparameter_grid:

    mean_scores = [hyperparameters['depth'], hyperparameters['width'], hyperparameters['dropout']]

    # Train models and record mean score
    for Model in model_list:
        scores = [
            Model(
                in_features=train_X.shape[1], out_features=train_Y.shape[1], 
                **hyperparameters
            ).to(device).fit(
                train_X=train_X, train_Y=train_Y, train_Var=train_Var, 
                validation_X=validation_X, validation_Y=validation_Y, validation_Var=validation_Var, 
                dataloader=dataloader, 
                patience=0.002
            ) for _ in range(n_folds)
        ]
        mean_scores.append(sum(scores)/n_folds)

    # Print hyperparameters and model scores
    with open(log_file, 'a') as f:
        f.write(','.join(
            str(score) for score in mean_scores
        ) + '\n')
