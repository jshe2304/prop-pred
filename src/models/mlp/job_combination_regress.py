import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils.load_immuno import *

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
data = to_namedtuple(to_tensor(load_immuno(datadir, x_type='embeddings'), device))

train_X = data.train.x
validation_X = data.validation.x
test_X = data.test.x

train_Y = data.train.y
validation_Y = data.validation.y
test_Y = data.test.y

############################################################
############################################################
# Training
############################################################
############################################################

hyperparameters = {'depth': 2, 'width': 512, 'dropout': 0.4}

with open(log_file, 'w') as f:
    f.write(','.join(
        str(i) + str(j) + ',' + str(j) + str(i)
        for i in range(len(data.properties)) 
        for j in range(i+1, len(data.properties))
    ) + '\n')

for trial in range(64):

    scores = []

    for i in range(len(data.properties)):
        for j in range(i+1, len(data.properties)):
            
            train_y = train_Y[:, (i, j)]
            validation_y = validation_Y[:, (i, j)]
            test_y = test_Y[:, (i, j)]

            model = RMLPRegressor(
                in_features=train_X.shape[1], out_features=train_y.shape[1], 
                **hyperparameters
            ).to(device)

            model.fit(
                train_X, train_y, 
                validation_X, validation_y, 
                lr=0.0001, epochs=512, batch_size=32, patience=0.002, 
                early_stopping=True, plot=False, show_progress=False
            )
            score_i, score_j = model.score(test_X, test_y, multioutput='raw_values')
            scores.append(float(score_i))
            scores.append(float(score_j))

    with open(log_file, 'a') as f:
        f.write(','.join(str(score) for score in scores) + '\n')
