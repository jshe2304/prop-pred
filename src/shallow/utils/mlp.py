import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features=2048, depth=2, width=32, dropout=0.25, lr=0.001, weight_decay=0):
        super().__init__()

        self.in_map = nn.Linear(n_features, width)

        self.mlp = []
        for i in range(depth):
            self.mlp.append(nn.Linear(width, width))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*self.mlp)
        
        self.out_map = nn.Linear(width, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):

        if type(x) != torch.Tensor: x = torch.tensor(x, dtype=torch.float32).detach()

        x1 = self.in_map(x)
        x2 = self.mlp(x1)
        x3 = self.out_map(x2)

        return x3

    def predict(self, x):

        if type(x) != torch.Tensor: x = torch.tensor(x, dtype=torch.float32).detach()

        return self.forward(x).detach()

    def score(self, x, y_true):
        '''
        Compute R^2 score
        '''

        if type(x) != torch.Tensor: x = torch.tensor(x, dtype=torch.float32).detach()
        if type(y_true) != torch.Tensor: y_true = torch.tensor(y_true, dtype=torch.float32).detach()

        y_pred = self.predict(x).squeeze()

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()

        return float(1 - u/v)

    def fit(self, x, y_true):

        if type(x) != torch.Tensor: x = torch.tensor(x, dtype=torch.float32).detach()
        if type(y_true) != torch.Tensor: y_true = torch.tensor(y_true, dtype=torch.float32).detach()

        self.train()
        for epoch in range(512):
            self.optimizer.zero_grad()
    
            y_pred = self.forward(x).squeeze()
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()

            self.optimizer.step()
    
        self.eval()

        return float(loss)