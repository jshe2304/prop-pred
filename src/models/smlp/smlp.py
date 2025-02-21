import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        assert self.x.shape[0] == self.y.shape[0]
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n ** 2

    def __getitem__(self, idx):
        
        idx1 = idx % self.n # repeat
        idx2 = idx // self.n # repeat_interleave
        
        return self.x[idx1], self.x[idx2], self.y[idx1], self.y[idx2]

class SiameseRMLP(nn.Module):
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

        self.siamese_block = nn.Sequential(
            nn.Linear(width * 2, width), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(width, out_features)
        )

    def siamese_forward(self, x1, x2):
        batch_size = x1.shape[0]
        
        z = self.forward(torch.concat([x1, x2]))

        return self.siamese_block(
            torch.concat([z[:batch_size], z[batch_size:]], axis=-1)
        )

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)

        return x1 + x2