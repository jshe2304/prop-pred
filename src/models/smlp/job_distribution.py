import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from smlp import SiameseRMLP, SiameseDataset

log_file = sys.argv[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

######
# Data
######

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

dataset = SiameseDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

############
# Train Loop
############

for i in range(64):
    
    model = SiameseRMLP(
        in_features=train_X.shape[1], out_features=train_Y.shape[1], 
        depth=2, width=512, 
        dropout=0.4, 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train Loop
    for epoch in range(1):
    
        # SGD Loop
        for batch_i, (x1, x2, y1, y2) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            # Loss
            torch.nn.functional.mse_loss(
                model.siamese_forward(x1, x2), 
                y1 - y2
            ).backward()
            
            optimizer.step()

    # Score
    y_pred_samples = []
    for i in range(512):
        
        sample_indices = torch.randint(len(train_X), (len(test_X), ))
        y_pred_samples.append(
            (model.siamese_forward(test_X, train_X[sample_indices]) + train_Y[sample_indices]).detach().cpu()
        )
    
    y_pred = torch.stack(y_pred_samples, axis=-1).mean(axis=-1)

    scores = r2_score(test_Y.cpu(), y_pred.cpu(), multioutput='raw_values')

    with open(log_file, 'a') as f:
        f.write(','.join(str(score) for score in scores) + '\n')
