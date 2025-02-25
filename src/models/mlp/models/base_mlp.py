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

    def score(self, x, y, multioutput='uniform_average'):
        '''
        Compute R^2 score
        '''
        y_pred = self.forward(x)
        return self.scoring_function(y, y_pred, multioutput)

    def fit(self, \
            train_X, train_Y, \
            validation_X, validation_Y, \
            dataloader=None, \
            lr=0.0001, epochs=512, batch_size=32, \
            patience=None, plot=False, show_progress=False, \
            *args, **kwargs):
        '''
        Fit the MLP with validation
        '''

        # Data
        if dataloader is None:
            dataloader = DataLoader(TensorDataset(
                train_X, train_Y
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
            for train_x, train_y, *_ in dataloader:
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
            validation_score = self.score(validation_X, validation_Y, multioutput=None)
            best_score = max(best_score, validation_score.mean())
            if plot: validation_scores.append(validation_score)

            # Validation Loss
            validation_loss = float(self.loss(
                self.forward(validation_X), validation_Y
            ))
            validation_losses.append(validation_loss)

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