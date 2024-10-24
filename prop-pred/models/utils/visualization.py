import matplotlib.pyplot as plt
import torch

def plot_regression_histograms(skmodel_dict, train_X, test_X, test, train):
    
    fig, axs = plt.subplots(len(skmodel_dict), 1, figsize=(5, 3 * len(skmodel_dict)))
    if len(skmodel_dict) == 1: axs = [axs]
    
    for ax, (property_label, skmodel) in zip(axs, skmodel_dict.items()):
    
        y_pred = skmodel.predict(test_X)

        y_true_train = torch.tensor(
            train[property_label], 
            dtype=torch.float32, 
            requires_grad=False
        )
        y_true = torch.tensor(
            test[property_label], 
            dtype=torch.float32, 
            requires_grad=False
        )
    
        train_r2 = skmodel.score(train_X, y_true_train)
        test_r2 = skmodel.score(test_X, y_true)

        txt = 'Train $R^2$ = {:.2f}\n'.format(train_r2) 
        txt += 'Test $R^2$ = {:.2f}\n'.format(test_r2) 
        ax.text(1.1, 0.9, txt, transform=ax.transAxes , fontsize=10, verticalalignment='top')
    
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_true.max() - y_true.min()
    
        if actual_range > pred_range:
            pred_bins = int(32 * pred_range / actual_range) * 2
            actual_bins = 32
        else:
            pred_bins = 32
            actual_bins = int(32 * actual_range / pred_range) * 2
    
        ax.hist(y_pred, bins=pred_bins, alpha=0.5, density=True, color='b', label='Predicted')
        ax.hist(y_true, bins=actual_bins, alpha=0.5, density=True, color='r', label='Actual')
        ax.set_xlabel(property_label)
        ax.set_yticks([])
        ax.legend();
    
    plt.tight_layout()
        