import os
import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils.datasets import SmilesDataset
from data_utils.graphs import smiles_to_graphs

from graph_transformer import GraphTransformer

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import r2_score

log_file, weights_dir, *_ = sys.argv[1:]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(log_file, 'w') as f:
    f.write('train_mse,validation_mse,validation_r2\n')

######
# Data 
######

dataset = SmilesDataset(
    smiles='/home/jshe/prop-pred/src/data/qm9/smiles.csv', 
    y='/home/jshe/prop-pred/src/data/qm9/norm_y.csv', 
)
train_dataset, validation_dataset, _ = random_split(
    dataset, lengths=(0.8, 0.1, 0.1), 
    generator=torch.Generator().manual_seed(8)
)
del _

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=512, shuffle=True)

#######
# Model
#######

hyperparameters = dict(
    numerical_features=5, categorical_features=(9+1, 8+1, 2+1, 2+1), 
    E=64, H=4, D=4, 
    dropout=0.1, 
    out_features=dataset.n_properties, 
)

model = GraphTransformer(**hyperparameters).to(device)

#######
# Train
#######

optimizer = torch.optim.Adam(model.parameters())
mse = nn.MSELoss()

for epoch in range(64):
    for smiles, y_true in train_dataloader:
        model.train()
        optimizer.zero_grad()

        numerical_node_features, categorical_node_features, edges, padding = smiles_to_graphs(smiles, device=device)

        # Forward pass and loss
        y_pred = model(numerical_node_features.float(), categorical_node_features, edges, padding)
        loss = mse(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():

        # Validation loss
        smiles, y_true = next(iter(validation_dataloader))
        numerical_node_features, categorical_node_features, edges, padding = smiles_to_graphs(smiles, device=device)
        y_pred = model(numerical_node_features.float(), categorical_node_features, edges, padding)
        validation_loss = mse(y_pred, y_true.to(device))
        validation_score = r2_score(y_true.cpu(), y_pred.cpu())

    to_log = []
    to_log.append(float(loss))
    to_log.append(float(validation_loss))
    to_log.append(float(validation_score))

    with open(log_file, 'a') as f:
        f.write(','.join(str(n) for n in to_log) + '\n')

    if epoch > 32 and epoch % 3 == 0:
        torch.save(model.state_dict(), os.path.join(weights_dir, f'{epoch}.pt'))
