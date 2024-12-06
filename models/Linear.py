import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

rank = 2

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.LinearA = nn.Linear(self.seq_len, rank)
        self.LinearB = nn.Linear(rank, self.seq_len)
        # Use this line if you want to visualize the weights

        self.LinearA.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, rank])
        )
        self.LinearB.weight = nn.Parameter(torch.ones([rank, self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        Linear = self.LinearA @ self.LinearB
        x = Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]
