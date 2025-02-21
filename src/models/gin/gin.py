import torch
import torch.nn as nn

import torch_geometric.nn as gnn

class GIN(torch.nn.Module):
    '''
    Graph Isomorphism Network (GIN)    
    '''
    
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        
        self.conv1 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim), 
                nn.ReLU(),
                nn.Linear(h_dim, h_dim), 
                nn.ReLU(), 
            )
        )
        self.conv2 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(h_dim, h_dim), 
                nn.BatchNorm1d(h_dim), 
                nn.ReLU(),
                nn.Linear(h_dim, h_dim), 
                nn.ReLU()
            )
        )
        self.conv3 = gnn.GINConv(
            nn.Sequential(
                nn.Linear(h_dim, h_dim), 
                nn.BatchNorm1d(h_dim), 
                nn.ReLU(),
                nn.Linear(h_dim, h_dim), 
                nn.ReLU()
            )
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(3 * h_dim, 3 * h_dim), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(3 * h_dim, out_dim)
        )

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = gnn.global_add_pool(h1, batch)
        h2 = gnn.global_add_pool(h2, batch)
        h3 = gnn.global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Feed-forward
        return self.ffn(h)
