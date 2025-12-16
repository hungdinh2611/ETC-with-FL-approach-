import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)
