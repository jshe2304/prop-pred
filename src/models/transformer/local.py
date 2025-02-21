import torch
import torch.nn as nn

import torch_geometric
import torch_geometric.nn as gnn

def NodeEmbedding(nn.Module):
    def __init__(self, node_features, embedding_dim):

        self.embed = nn.Linear(
            in_features=node_features, 
            out_features=embedding_dim, 
            bias=False
        )

    def forward(self, feature_matrix):
        return self.embed(feature_matrix)

def NeighborhoodAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):

        self.query = nn.Linear(
            in_features=embedding_dim, 
            out_features=attention_dim, 
            bias=False
        )

        self.key = nn.Linear(
            in_features=embedding_dim, 
            out_features=attention_dim, 
            bias=False
        )

        self.value = nn.Linear(
            in_features=embedding_dim, 
            out_features=attention_dim, 
            bias=False
        )

    def forward(self, node_embeddings, adjacency_matrix):

        q = self.query(node_embeddings)
        k = self.key(node_embeddings)
        v = self.value(node_embeddings)

        attention = q @ k.T
        masked_attention = 