import torch

def prox_loss(model, global_model):
    loss = 0.0
    for p, p0 in zip(model.parameters(), global_model.parameters()):
        loss += torch.norm(p - p0) ** 2
    return loss
