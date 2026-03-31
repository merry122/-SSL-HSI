import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_results(results):
    methods = list(results.keys())
    accs = [results[m]["acc"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, accs)
    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy")

    for label in ax.get_xticklabels():
        label.set_fontsize(7)
        label.set_rotation(30)
        label.set_ha("right")

    plt.tight_layout()
    plt.savefig("accuracy.png")
    plt.close()

def plot_confusion(cm, title):
    plt.figure()
    sns.heatmap(cm, annot=False)
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()


def plot_tsne(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"t-SNE: {title}")
    plt.savefig(f"tsne_{title}.png")
    plt.close()