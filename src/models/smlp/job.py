import sys

import pandas as pd
import numpy as np

sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import SiameseDataset, SiameseRMLP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

######
# Data
######

datadir = '../../data/immunomodulation/log_normalized/regression/'
data = to_namedtuple(to_tensor(load_data(datadir), device))

train_X = data.train.embeddings
validation_X = data.validation.embeddings
test_X = data.test.embeddings

train_Y = data.train.y[:, 0].unsqueeze(-1)
validation_Y = data.validation.y[:, 0].unsqueeze(-1)
test_Y = data.test.y[:, 0].unsqueeze(-1)

dataset = SiameseDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#######
# Model
#######

model = SiameseRMLP(512, 4, 512, 0.5)

#######
# Train
#######

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train Statistics
validation_losses, train_losses, validation_scores = [], [], []

# Train Loop
for epoch in range(1):

    # SGD Loop
    for batch_i, (x1, x2, y1, y2) in tqdm(enumerate(dataloader)):

        model.train()
        
        optimizer.zero_grad()
        
        torch.nn.functional.mse_loss(
            model.siamese_forward(x1, x2), 
            y1 - y2
        ).backward()
        
        if batch_i % 32 == 0:

            model.eval()
            
            sample_1 = torch.randint(len(train_X), (64, ))
            sample_2 = torch.randint(len(train_X), (64, ))
            
            train_losses.append(float(torch.nn.functional.mse_loss(
                model.siamese_forward(
                    train_X[sample_1], train_X[sample_2]
                ), 
                train_Y[sample_1] - train_Y[sample_2]
            )))
            
            sample_3 = torch.randint(len(validation_X), (64, ))
            sample_4 = torch.randint(len(train_X), (64, ))
            
            validation_losses.append(float(torch.nn.functional.mse_loss(
                model.siamese_forward(
                    validation_X[sample_3], train_X[sample_4]
                ), 
                validation_Y[sample_3] - train_Y[sample_4]
            )))
        
        optimizer.step()

model.eval();

    torch.save(model.state_dict(), f'siamese_big_{epoch}.pt')