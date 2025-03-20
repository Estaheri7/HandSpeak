import torch
from torch import Tensor
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, out_dim)
        )

    def forward(self, x: Tensor):
        return self.fc(x)