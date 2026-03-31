import torch
from models.encoder import Encoder
from models.autoencoder import Autoencoder
from models.contrastive import augment
import torch.nn.functional as F
import torch.nn as nn
import time

# 6 permutations for jigsaw pretext (choose a small subset of permutations)
JIGSAW_PERMUTATIONS = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [1, 3, 0, 2],
]


def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.transpose(0, 1)) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.arange(2 * batch_size).to(z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_contrastive(loader, input_dim, epochs=30):
    model = Encoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    loss_history = []
    time_history = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            x1 = augment(x)
            x2 = augment(x)

            z1 = model(x1)
            z2 = model(x2)

            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[Contrastive] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model, loss_history, time_history


def train_autoencoder(loader, input_dim, epochs=30):
    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    loss_history = []
    time_history = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            z, x_recon = model(x)
            loss = ((x - x_recon) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[Autoencoder] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model.encoder, loss_history, time_history


def masked_contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.transpose(0, 1)) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.arange(2 * batch_size).to(z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_masked(loader, input_dim, epochs=30):
    model = Encoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    loss_history = []
    time_history = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            mask = (torch.rand_like(x) > 0.1).float()
            x_masked = x * mask

            z1 = model(x_masked)
            z2 = model(x)

            loss = masked_contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[Masked] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model, loss_history, time_history


def train_mae(loader, input_dim, epochs=30, mask_ratio=0.75):
    model = Autoencoder(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    loss_history = []
    time_history = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            mask = (torch.rand_like(x) > mask_ratio).float()
            x_masked = x * mask

            z, x_recon = model(x_masked)

            masked_loss = ((x - x_recon) ** 2 * (1 - mask)).sum()
            denom = torch.sum(1 - mask) + 1e-8
            loss = masked_loss / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[MAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model.encoder, loss_history, time_history


def train_rotation(loader, input_dim, epochs=30):
    model = Encoder(input_dim)
    head = nn.Linear(64, 4)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    model.train()
    head.train()
    loss_history = []
    time_history = []

    step_size = max(1, input_dim // 4)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            labels = torch.randint(0, 4, (x.size(0),)).to(x.device)
            x_rot = x.clone()
            for i, r in enumerate(labels):
                x_rot[i] = torch.roll(x[i], shifts=int(r * step_size), dims=0)

            z = model(x_rot)
            logits = head(z)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[Rotation] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model, loss_history, time_history


def apply_jigsaw(x, perm):
    chunk_size = x.size(0) // 4
    segments = [x[i * chunk_size : (i + 1) * chunk_size] for i in range(4)]
    return torch.cat([segments[i] for i in perm], dim=0)


def train_jigsaw(loader, input_dim, epochs=30):
    model = Encoder(input_dim)
    head = nn.Linear(64, len(JIGSAW_PERMUTATIONS))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    model.train()
    head.train()
    loss_history = []
    time_history = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for x, _ in loader:
            perm_indices = torch.randint(0, len(JIGSAW_PERMUTATIONS), (x.size(0),)).to(x.device)
            x_jig = torch.zeros_like(x)
            for i, perm_idx in enumerate(perm_indices):
                x_jig[i] = apply_jigsaw(x[i], JIGSAW_PERMUTATIONS[int(perm_idx)])

            z = model(x_jig)
            logits = head(z)

            loss = F.cross_entropy(logits, perm_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        time_history.append(epoch_time)
        print(f"[Jigsaw] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    return model, loss_history, time_history