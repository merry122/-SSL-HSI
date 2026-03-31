import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.metrics import evaluate


def train_classifier(encoder, X, y, epochs=30, batch_size=256):

    encoder.eval()

    # Extract features
    with torch.no_grad():
        features = encoder(X)

    num_classes = len(torch.unique(y))

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.3, random_state=42
    )

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Classifier model
    classifier = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    classifier.train()

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in train_loader:
            outputs = classifier(xb)
            loss = loss_fn(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Classifier] Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Evaluation
    classifier.eval()

    with torch.no_grad():
        preds = classifier(X_test).argmax(dim=1)

    acc, f1, cm = evaluate(y_test.numpy(), preds.numpy())

    return acc, f1, cm