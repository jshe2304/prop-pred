import torch
import torch.nn as nn

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