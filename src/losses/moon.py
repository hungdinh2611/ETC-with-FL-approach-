import torch
import torch.nn.functional as F

def moon_loss(z, z_global, z_prev, temperature=0.5):
    pos = F.cosine_similarity(z, z_global)
    neg = F.cosine_similarity(z, z_prev)
    logits = torch.stack([pos, neg], dim=1) / temperature
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)
