import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn

class BasePMLP(nn.Module):
    '''
    Posterior MLP
    '''
    def __init__(self):
        super().__init__()
        
    def loss(self, mu_pred, logvar_pred, mu, var): 
        return (
            (((mu - mu_pred) ** 2 + torch.exp(logvar_pred))/var + torch.log(var) - logvar_pred - 1)/2
        ).mean()

    def forward(self): None

    def scoring_function(self, y, y_pred, multioutput):
        return r2_score(
            y.detach().cpu(), y_pred.detach().cpu(), 
            multioutput=multioutput
        )

    def score(self, x, y, multioutput='uniform_average'):
        y_pred, _ = self.forward(x)
        return self.scoring_function(y, y_pred, multioutput)

    def fit(self, \
            train_X, train_Y, train_Var, \
            validation_X, validation_Y, validation_Var, \
            dataloader=None, \
            lr=0.0001, epochs=512, batch_size=32, \
            patience=None, plot=False, show_progress=False, \
            *args, **kwargs):

        # Data
        if dataloader is None:
            dataloader = DataLoader(TensorDataset(
                train_X, train_Y, train_Var
            ), batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Early Stopping
        stop_interval = int(patience//lr) if patience is not None else epochs
        
        # Train Statistics
        best_score = -1
        validation_losses = []
        if plot: train_losses, validation_scores = [], []

        # Train Loop
        progress_bar = tqdm(range(epochs)) if show_progress else range(epochs)
        for epoch in progress_bar:

            # SGD Loop
            self.train()
            for train_x, train_y, train_var in dataloader:
                optimizer.zero_grad()

                mu_pred, logvar_pred = self.forward(train_x)
                train_loss = self.loss(mu_pred, logvar_pred, train_y, train_var)
                train_loss.backward()

                optimizer.step()

            self.eval()
            
            # Train Loss
            if plot: train_losses.append(float(
                self.loss(
                    *self.forward(train_X), train_Y, train_Var
                )
            ))

            # Validation Score
            validation_score = self.score(validation_X, validation_Y, multioutput=None)
            best_score = max(best_score, validation_score.mean())
            if plot: validation_scores.append(validation_score)

            # Validation Loss
            validation_losses.append(float(self.loss(
                *self.forward(validation_X), validation_Y, validation_Var
            )))

            # Early Stopping
            if epoch > stop_interval // 2 and epoch % (stop_interval // 2) == 0:
                x = torch.arange(0, stop_interval) - (stop_interval / 2) + 0.5
                y = torch.tensor(validation_losses[-stop_interval:])
                if x @ y > 0: break

        if plot:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 5))
            ax0.plot(train_losses, label='Train')
            ax0.plot(validation_losses, label='Validation')
            ax0.legend()
            lines = ax1.plot(validation_scores)
            if type(validation_scores[0]) != float:
                ax1.legend(lines, range(validation_scores[0].shape[0]), fontsize=8)
            plt.tight_layout()

        return best_score
