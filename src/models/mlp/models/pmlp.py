import torch
import torch.nn as nn
from .base_pmlp import BasePMLP

class PMLPRegressor(BasePMLP):
    def __init__(self, in_features, out_features, depth, width, dropout):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)
        )

        mlp_block = []
        for i in range(depth):
            mlp_block.append(nn.Linear(width, width))
            mlp_block.append(nn.ReLU())
            mlp_block.append(nn.Dropout(dropout))
        self.mlp_block = nn.Sequential(*mlp_block)

        self.mu_block = nn.Linear(width, out_features)
        self.logvar_block = nn.Linear(width, out_features)

    def forward(self, x):

        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        mu, logvar = self.mu_block(x2), self.logvar_block(x2)

        return mu, logvar

class RPMLPRegressor(BasePMLP):

    def __init__(self, in_features, out_features, depth, width, dropout):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)
        )

        mlp_block = []
        for i in range(depth):
            mlp_block.append(nn.Linear(width, width))
            mlp_block.append(nn.ReLU())
            mlp_block.append(nn.Dropout(dropout))
        self.mlp_block = nn.Sequential(*mlp_block)
        
        self.mu_block = nn.Linear(width, out_features)
        self.logvar_block = nn.Linear(width, out_features)

    def forward(self, x):

        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = x1 + x2
        mu, logvar = self.mu_block(x3), self.logvar_block(x3)

        return mu, logvar