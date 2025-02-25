import torch
import torch.nn as nn

import torch_geometric.nn as gnn

class GAT(nn.Module):
    '''
    GAT Block
    '''
    def __init__(self, h_channels, heads, dropout):
        super().__init__()
        
        self.gat = gnn.GATv2Conv(
            in_channels=h_channels, out_channels=h_channels // heads, 
            heads=heads, dropout=dropout
        )

        self.W = nn.Sequential(
            nn.Linear(h_channels, h_channels, bias=False), 
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(h_channels)

        self.mlp = nn.Sequential(
            nn.Linear(h_channels, h_channels * 2), 
            nn.ReLU(), 
            nn.Linear(h_channels * 2, h_channels), 
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(h_channels)

    def forward(self, x0, edges): 

        x1 = self.gat(x0, edges)
        x2 = self.W(x1) + x0
        x3 = self.norm1(x2)
        x4 = self.mlp(x3) + x3
        return self.norm2(x4)

class MultiheadAttention(nn.Module):
    def __init__(self, h_channels, heads, dropout):
        super().__init__()

        self.d = h_channels // heads

        self.QKV = nn.Linear(
            in_features=h_channels, out_features=self.d * 3, 
            bias=False
        )

        self.dropout = nn.Dropout(dropout)
        self.mask = None

    def forward(self, x0, mask):

        qkv = self.QKV(x0)
        q, k, v = qkv.chunk(3, dim=-1)

        att_logits = (q @ k.T)/self.d
        att_logits.masked_fill_(mask, -float('inf'))
        att = torch.softmax(att_logits, dim=-1)

        att = self.dropout(att)
        return att @ v

class Transformer(nn.Module):
    '''
    Graph Transformer
    '''
    def __init__(self, in_features, embed_channels, heads, out_features, dropout):
        super().__init__()

        self.embedding = nn.Linear(
            in_features=in_features, out_features=embed_channels
        )

        self.gat_block = GAT(
            h_channels=embed_channels, heads=heads, dropout=dropout
        )

        self.att_block = MultiheadAttention(
            h_channels=embed_channels, heads=heads, dropout=dropout
        )


    def forward(self, x):

        tokens = x.x.float()
        edges = x.edge_index
        
        # Make self-attention mask
        bool_mask = x.batch.unsqueeze(0) != x.batch.unsqueeze(-1)
        att_mask = bool_mask.float()
        att_mask[bool_mask] = -float('inf')

        x1 = self.embedding(tokens)
        x2 = self.gat_block(x1, edges)
        x3 = self.att_block(x2, att_mask)

        return x3
        
