import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, y, margin=1.0):
    dist = F.pairwise_distance(z1, z2)
    loss = y * dist.pow(2) + (1 - y) * F.relu(margin - dist).pow(2)
    return loss.mean()
