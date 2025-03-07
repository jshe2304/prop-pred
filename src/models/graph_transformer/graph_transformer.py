import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, E, H, dropout):
        super().__init__()

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, mask):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute and separate Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute masked attention pattern

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.masked_fill_(mask.unsqueeze(1), float('-inf'))
        attn = torch.softmax(attn, dim=-1).nan_to_num(0)

        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        
        return self.out_map(values)

class TransformerBlock(nn.Module):
    def __init__(self, E, H, dropout):
        super().__init__()
        
        self.attention = MultiheadAttention(E, H, dropout)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, padding_mask, causal_mask):

        # Attention residual block

        x1 = self.attention(x0, causal_mask)
        x1 = self.dropout(x1) 
        x2 = x1 + x0
        x2 = self.norm_1(x2)

        # MLP residual block
        
        x3 = self.mlp(x2)
        x3 = self.dropout(x3)
        x4 = x3 + x2
        x4 = x4.masked_fill(padding_mask, 0)
        x4 = self.norm_2(x4)

        return x4

class LocalGlobalTransformerBlock(nn.Module):
    def __init__(self, E, H, dropout):
        super().__init__()

        self.local_block = TransformerBlock(E, H, dropout)
        self.global_block = TransformerBlock(E, H, dropout)

    def forward(self, x, padding_mask, graph_causal_mask, padding_causal_mask):

        x = self.local_block(x, padding_mask=padding_mask, causal_mask=graph_causal_mask)
        x = self.global_block(x, padding_mask=padding_mask, causal_mask=padding_causal_mask)

        return x

class GraphTransformer(nn.Module):
    '''
    Transformer with alternating local and global masked self-attention. 
    '''
    def __init__(self, numerical_features, categorical_features, E, H, D, out_features, dropout):
        super().__init__()

        self.E, self.H = E, H

        self.numerical_embed = nn.Linear(numerical_features, E, bias=False)
        self.categorical_embeds = nn.ModuleList([
            nn.Embedding(n_categories, E, padding_idx=0) 
            for n_categories in categorical_features
        ])
        self.transformer_blocks = nn.ModuleList([
            LocalGlobalTransformerBlock(E, H, dropout)
            for _ in range(D)
        ])
        self.out_map = nn.Linear(E, out_features)

    def forward(self, numerical_node_features, categorical_node_features, adj, padding):

        B, L, _ = categorical_node_features.size()

        # Create causal and padding masks

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)
        padding_causal_mask = torch.logical_or(
            padding.unsqueeze(-2), padding.unsqueeze(-1)
        )
        graph_causal_mask = ~adj

        # Forward Pass

        x = sum(embed(categorical_node_features[:, :, i]) for i, embed in enumerate(self.categorical_embeds))
        x += self.numerical_embed(numerical_node_features)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, padding_mask, graph_causal_mask, padding_causal_mask)
        x = x.sum(dim=1) # (B, E)
        x = self.out_map(x)
        return x
