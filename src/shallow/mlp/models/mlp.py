import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=2048, out_features=7, depth=2, width=128, dropout=0.4, model_type='regression'):
        super().__init__()

        self.model_type = model_type

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

        return x3 if self.model_type == 'regression' else torch.sigmoid(x3)

    def score(self, x, y_true):
        '''
        Compute R^2 score
        '''

        y_pred = self.forward(x).squeeze()

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()

        return float(1 - u/v)

    def fit(self, train_X, train_Y, test_X, test_Y, *args, **kwargs):

        self.train()

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        # Train Loop
        best_r2 = -1
        for epoch in range(512):
            optimizer.zero_grad()
        
            # Train Loss
            mu = self.forward(train_X)
            train_loss = nn.functional.mse_loss(mu, train_Y)
            train_loss.backward()
            train_loss = float(train_loss)
        
            # Test R2
            mu = self.forward(test_X)
            u = ((test_Y - mu) ** 2).sum()
            v = ((test_Y - mu.mean()) ** 2).sum()
            r2 = float(1 - u/v)

            # Update R2
            best_r2 = max(best_r2, r2)
        
            # Early Stopping
            #if epoch > 24 and (sum(validation_losses[-24:-12]) < sum(validation_losses[-12:])): break
        
            optimizer.step()

        return best_r2