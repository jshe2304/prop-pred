import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

class BaseMLP(nn.Module):
    '''
    Generic MLP Class intended for inheritance. 
    '''
    def __init__(self):
        super().__init__()

        # self.loss
        # self.scoring

    def forward(self):
        None

    def scoring_function(self):
        None

    def score(self, x, y):
        '''
        Compute R^2 score
        '''
        y_pred = self.forward(x)
        return self.scoring_function(y, y_pred)

    def fit(self, \
            train_X, train_Y, validation_X, validation_Y, \
            lr=0.0001, epochs=512, batch_size=32, early_stopping=True, plot=False, \
            *args, **kwargs):
        '''
        Fit the MLP with validation
        '''

        # Data
        dataloader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Early Stopping
        if early_stopping: stop_interval = int(self.patience//lr)
        
        # Train Statistics
        best_score = -1
        if plot: train_losses, validation_losses, validation_scores = [], [], []

        # Train Loop
        for epoch in tqdm(range(epochs)):

            # SGD Loop
            self.train()
            for train_x, train_y in dataloader:
                optimizer.zero_grad()

                self.loss(
                    self.forward(train_x), train_y
                ).backward()
                
                optimizer.step()

            self.eval()
            
            # Train Loss
            if plot: train_losses.append(
                float(self.loss(
                    self.forward(train_X), train_Y
                ))
            )

            # Validation Score
            validation_score = self.score(validation_X, validation_Y)
            best_score = max(best_score, validation_score)
            if plot: validation_scores.append(validation_score)

            # Validation Loss
            validation_loss = float(self.loss(
                self.forward(validation_X), validation_Y
            ))
            if plot: validation_losses.append(validation_loss)

            # Early Stopping
            if early_stopping and epoch >= stop_interval and epoch % stop_interval == 0:
                x = torch.arange(0, stop_interval) - (stop_interval / 2) + 0.5
                y = torch.tensor(validation_losses[-stop_interval:])
                if x @ y > 0: break

        if plot:
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 4))
            ax0.plot(train_losses, label='Train')
            ax0.plot(validation_losses, label='Validation')
            ax0.legend()
            ax1.plot(validation_scores)

        return best_score