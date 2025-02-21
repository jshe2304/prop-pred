import torch
import torch.nn as nn
from sklearn.metrics import r2_score, roc_auc_score

from .base_mlp import BaseMLP

class MLPRegressor(BaseMLP):
    '''
    Generic MLP
    '''
    def __init__(self, in_features, out_features, depth, width, dropout):
        super().__init__()

        self.loss = nn.MSELoss()

        self.in_block = nn.Sequential(
            nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)
        )

        mlp_block = []
        for i in range(depth):
            mlp_block.append(nn.Linear(width, width))
            mlp_block.append(nn.ReLU())
            mlp_block.append(nn.Dropout(dropout))
        self.mlp_block = nn.Sequential(*mlp_block)

        self.out_block = nn.Linear(width, out_features)

    def scoring_function(self, y, y_pred, multioutput=None):
        return r2_score(
            y.detach().cpu(), y_pred.detach().cpu(), 
            multioutput='raw_values' if multioutput == None else 'uniform_average'
        )

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x2)

        return x3

class MLPClassifier(MLPRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.BCELoss()

    def scoring_function(self, y, y_pred, multioutput):
        return roc_auc_score(
            y.detach().cpu(), y_pred.detach().cpu(), 
            average=None if multioutput == None else 'macro'
        )
    
    def forward(self, x):
        return torch.sigmoid(super().forward(x))

class RMLPRegressor(BaseMLP):
    '''
    MLP with skip connections
    '''
    def __init__(self, in_features, out_features, depth, width, dropout):
        super().__init__()

        self.loss = nn.MSELoss()

        self.in_block = nn.Sequential(
            nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)
        )

        mlp_block = []
        for i in range(depth):
            mlp_block.append(nn.Linear(width, width))
            mlp_block.append(nn.ReLU())
            mlp_block.append(nn.Dropout(dropout))
        self.mlp_block = nn.Sequential(*mlp_block)
        
        self.out_block = nn.Linear(width, out_features)

    def scoring_function(self, y, y_pred, multioutput=None):
        return r2_score(
            y.detach().cpu(), y_pred.detach().cpu(), 
            multioutput='raw_values' if multioutput == None else 'uniform_average'
        )

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x1 + x2)

        return x3

class RMLPClassifier(RMLPRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.BCELoss()

    def scoring_function(self, y, y_pred, multioutput):
        return roc_auc_score(
            y.detach().cpu(), y_pred.detach().cpu(), 
            average=None if multioutput == None else 'macro'
        )

    def forward(self, x):
        return torch.sigmoid(super().forward(x))
