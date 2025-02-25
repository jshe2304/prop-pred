import torch
from torch.utils.data import Dataset

class GaussianSamplingDataset(Dataset):
    def __init__(self, x, y, var):
        self.x = x
        self.y = y
        self.std = torch.sqrt(var)
        
        assert self.x.shape[0] == self.y.shape[0] == self.std.shape[0]
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        std = self.std[idx]

        return self.x[idx], self.y[idx] + std * torch.randn_like(std)

class SiameseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        assert self.x.shape[0] == self.y.shape[0]
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n ** 2

    def __getitem__(self, idx):
        
        idx1 = idx % self.n # repeat
        idx2 = idx // self.n # repeat_interleave
        
        return self.x[idx1], self.x[idx2], self.y[idx1], self.y[idx2]

