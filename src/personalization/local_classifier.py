import torch
import torch.nn.functional as F
from models.classifier import Classifier

def train_local_classifier(encoder, dataloader, num_classes, epochs=10):
    device = next(encoder.parameters()).device
    encoder.eval()

    clf = Classifier(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z = encoder(x)

            loss = F.cross_entropy(clf(z), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return clf
