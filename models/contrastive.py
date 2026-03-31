
import torch
import torch.nn.functional as F

def augment(x):
    # x is 2D: [batch_size, features]
    noise = 0.05 * torch.randn_like(x)
    scale = torch.rand(x.size(0), 1).to(x.device) * 0.2 + 0.9
    
    return x * scale + noise



def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Similarity matrix
    sim = torch.matmul(z, z.T) / temperature

    # Remove diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs
    positives = torch.cat([
        torch.diag(sim, batch_size),
        torch.diag(sim, -batch_size)
    ], dim=0)

    # Labels
    labels = torch.arange(2 * batch_size).to(z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, labels)
    return loss
