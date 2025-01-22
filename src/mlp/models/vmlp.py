import torch
import torch.nn as nn

class VMLP(nn.Module):
    def __init__(self, in_features=2048, out_features=7, depth=2, width=128, dropout=0.4):
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

    def score(self, x, y_true):
        '''
        Compute R^2 score
        '''

        y_pred = self.forward(x)[0].squeeze()

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()

        return float(1 - u/v)

    def fit(self, train_X, train_Y, train_Var, test_X, test_Y, *args, **kwargs):
        '''
        Train a VMLP
        '''
    
        self.train()
    
        # Loss
        kl_divergence = lambda mu, logvar, mu_true, var_true: (
            (((mu_true - mu) ** 2 + torch.exp(logvar))/var_true + torch.log(var_true) - logvar - 1)/2
        ).mean()

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    
        # Train Loop
        best_r2 = -1
        for epoch in range(1024):
            optimizer.zero_grad()
        
            # Train loss
            mu, logvar = self.forward(train_X)
            train_loss = kl_divergence(mu, logvar, train_Y, train_Var)
            train_loss.backward()
            train_loss = float(train_loss)
                
            # Test R2
            mu, logvar = self.forward(test_X)
            u = ((test_Y - mu) ** 2).sum()
            v = ((test_Y - mu.mean()) ** 2).sum()
            r2 = float(1 - u/v)
    
            # Update R2
            best_r2 = max(best_r2, r2)

            optimizer.step()
        
        return best_r2