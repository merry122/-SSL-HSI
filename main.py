import torch
from utils.data_loader import load_data
from train_ssl import (
    train_contrastive,
    train_autoencoder,
    train_masked,
    train_mae,
    train_rotation,
    train_jigsaw,
)
from train_classifier import train_classifier
from utils.visualize import plot_results, plot_confusion, plot_tsne


def _convergence_epoch(loss_history, threshold=1e-4):
    for i in range(1, len(loss_history)):
        if abs(loss_history[i] - loss_history[i - 1]) < threshold:
            return i + 1
    return len(loss_history)


def main():
    X, y, loader = load_data()
    input_dim = X.shape[1]

    results = {}

    # Contrastive
    enc, c_loss, c_time = train_contrastive(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["Contrastive"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(c_loss),
        "total_time": sum(c_time),
    }
    plot_confusion(cm, "Contrastive")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "Contrastive")

    # Autoencoder
    enc, ae_loss, ae_time = train_autoencoder(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["Autoencoder"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(ae_loss),
        "total_time": sum(ae_time),
    }
    plot_confusion(cm, "Autoencoder")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "Autoencoder")

    # Masked Contrastive
    enc, m_loss, m_time = train_masked(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["MaskedContrastive"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(m_loss),
        "total_time": sum(m_time),
    }
    plot_confusion(cm, "MaskedContrastive")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "MaskedContrastive")

    # MAE
    enc, mae_loss, mae_time = train_mae(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["MAE"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(mae_loss),
        "total_time": sum(mae_time),
    }
    plot_confusion(cm, "MAE")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "MAE")

    # Rotation prediction
    enc, rot_loss, rot_time = train_rotation(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["Rotation"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(rot_loss),
        "total_time": sum(rot_time),
    }
    plot_confusion(cm, "Rotation")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "Rotation")

    # Jigsaw
    enc, jig_loss, jig_time = train_jigsaw(loader, input_dim)
    acc, f1, cm = train_classifier(enc, X, y)
    results["Jigsaw"] = {
        "acc": acc,
        "f1": f1,
        "convergence_epoch": _convergence_epoch(jig_loss),
        "total_time": sum(jig_time),
    }
    plot_confusion(cm, "Jigsaw")
    with torch.no_grad():
        embeddings = enc(X).numpy()
    plot_tsne(embeddings, y.numpy(), "Jigsaw")

    plot_results(results)

    print(results)


if __name__ == "__main__":
    main()