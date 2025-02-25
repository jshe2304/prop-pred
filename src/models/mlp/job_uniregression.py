import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils import *

import torch

from models import RMLPRegressor

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

train_Y = data.train.y
validation_Y = data.validation.y
test_Y = data.test.y

############################################################
############################################################
# Training
############################################################
############################################################

hyperparameters = {'depth': 2, 'width': 512, 'dropout': 0.4}

for trial in range(64):

    scores = []

    for i in range(train_Y.shape[1]):

        train_y = train_Y[:, i].unsqueeze(-1)
        validation_y = validation_Y[:, i].unsqueeze(-1)
        test_y = test_Y[:, i].unsqueeze(-1)

        model = RMLPRegressor(
            in_features=train_X.shape[1], out_features=train_y.shape[1], 
            **hyperparameters
        ).to(device)

        model.fit(
            train_X, train_y, 
            validation_X, validation_y, 
            lr=0.0001, epochs=512, batch_size=32, patience=0.002
            early_stopping=True, plot=False, show_progress=False
        )
        score = model.score(test_X, test_y)

        scores.append(float(score))

    with open(log_file, 'a') as f:
        f.write(','.join(str(score) for score in scores) + '\n')
