import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import *

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

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

train_Y = data.train.y
validation_Y = data.validation.y
test_Y = data.test.y

dataloader = DataLoader(TensorDataset(train_X, train_Y), batch_size=64, shuffle=True)

#######
# Model
#######

bnn_params = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization", 
    "moped_enable": False, 
    "moped_delta": 0.5,
}

############
# Train Loop
############

#with open(log_file, 'w') as f: f.write(','.join(properties) + '\n')

for i in range(64):

    # Model
    model = RMLP(
        in_features=train_X.shape[1], out_features=train_Y.shape[1], 
        depth=2, width=512, dropout=0.4
    ).to(device)
    dnn_to_bnn(model, bnn_params)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Loop
    validation_losses = []
    for epoch in range(256):
        
        model.train()
        # Batches
        for train_x, train_y in dataloader:
            optimizer.zero_grad()

            y_pred = model(train_x)
            kl = get_kl_loss(model)
            mse = nn.functional.mse_loss(y_pred, train_y)
            loss = mse + kl / 64
    
            loss.backward()
            optimizer.step()
    
        model.eval()
    
        # Validation Loss
        Y_pred = model(validation_X)
        mse = nn.functional.mse_loss(Y_pred, validation_Y)
        kl = get_kl_loss(model)
        validation_losses.append(float(mse + kl / validation_Y.shape[0]))
    
        # Early stopping
        stop_interval = 16
        if epoch > stop_interval // 2 and epoch % (stop_interval // 2) == 0:
            x = torch.arange(0, stop_interval) - (stop_interval / 2) + 0.5
            y = torch.tensor(validation_losses[-stop_interval:])
            if x @ y > 0: break
    
    scores = r2_score(
        test_Y.detach().cpu(), model(test_X).detach().cpu(), 
        multioutput='raw_values'
    )

    with open(log_file, 'a') as f:
        f.write(','.join(str(score) for score in scores) + '\n')

