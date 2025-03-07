import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils.datasets import SiameseDataset
from data_utils.load_immuno import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from smlp import DifferenceSiameseRMLP

log_file = sys.argv[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

######
# Data
######

datadir = '../../data/regress_immuno'
data = to_namedtuple(to_tensor(load_immuno(datadir, x_type='embeddings'), device))

train_X = data.train.x
validation_X = data.validation.x
test_X = data.test.x

train_Y = data.train.y
validation_Y = data.validation.y
test_Y = data.test.y

dataset = SiameseDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

############
# Train Loop
############

for i in range(64):
    
    model = DifferenceSiameseRMLP(
        in_features=train_X.shape[1], out_features=train_Y.shape[1], 
        depth=2, width=512, 
        dropout=0.4, 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train Loop
    for epoch in range(1):
    
        # SGD Loop
        for batch_i, (x, x_ref, y, y_ref) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            # Loss
            torch.nn.functional.mse_loss(
                model.siamese_forward(x, x_ref), 
                y - y_ref
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
