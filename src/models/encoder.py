import torch.nn as nn
import torch.nn.functional as F

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=256, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
