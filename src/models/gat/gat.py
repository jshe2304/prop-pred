import torch
import torch.nn as nn

import torch_geometric.nn as gnn

class GAT(nn.Module):
    '''
    Graph Attention Network
    '''
    
    def __init__(self, in_channels, h_channels, out_channels):
        super().__init__()
        
        self.conv1 = gnn.GATv2Conv(
            in_channels=in_channels, out_channels=h_channels, 
            heads=2, 
            
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
