import torch
import torch.nn as nn

class MLP(nn.Module):
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

        self.out_block = nn.Linear(width, out_features)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x2)

        return x3

class RMLP(nn.Module):
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
        
        self.out_block = nn.Linear(width, out_features)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x1 + x2)

        return x3
