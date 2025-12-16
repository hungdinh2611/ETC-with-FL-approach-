import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, emb_dim=128, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        return self.fc(z)
