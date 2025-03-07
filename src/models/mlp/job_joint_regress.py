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

train_Var = data.train.y_std ** 2
validation_Var = data.validation.y_std ** 2

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
    f.write(','.join(data.properties) + '\n')

for trial in range(128):

    model = RMLPRegressor(train_X.shape[1], train_Y.shape[1], **hyperparameters).to(device)
    model.fit(
        train_X, train_Y, 
        validation_X, validation_Y, 
        lr=0.0001, epochs=512, batch_size=32, patience=0.002, 
        early_stopping=True, plot=False, show_progress=False
    )
    scores = model.score(test_X, test_Y, multioutput='raw_values')
    
    with open(log_file, 'a') as f:
        f.write(','.join(list(str(score) for score in scores)) + '\n')
