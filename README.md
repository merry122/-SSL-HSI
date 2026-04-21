# Self-Supervised Learning on Salinas HSI

This project benchmarks multiple self-supervised learning (SSL) methods for hyperspectral image (HSI) representation learning on the **Salinas** dataset, then evaluates learned embeddings using a downstream classifier.

## What This Project Does

- Loads and preprocesses Salinas spectral data.
- Trains 6 SSL pretext tasks to learn feature embeddings:
  - Contrastive
  - Autoencoder
  - Masked Contrastive
  - MAE (Masked Autoencoder-style)
  - Rotation prediction
  - Jigsaw prediction
- Trains a supervised MLP classifier on top of frozen embeddings.
- Reports metrics (accuracy, weighted F1, confusion matrix) per method.
- Saves comparison and visualization plots.

Main orchestration happens in `main.py`.

## Project Structure

- `main.py` - full experiment pipeline (train SSL -> train classifier -> visualize -> print results)
- `train_ssl.py` - SSL training loops and losses
- `train_classifier.py` - downstream classifier training/evaluation
- `models/`
  - `encoder.py` - embedding network
  - `autoencoder.py` - autoencoder used by AE/MAE variants
  - `contrastive.py` - augmentation utilities
- `utils/`
  - `data_loader.py` - data loading and preprocessing
  - `metrics.py` - evaluation metrics
  - `visualize.py` - plotting results, confusion matrices, and t-SNE
- `data/` - place dataset files here

## Data Requirements

Expected files:

- `data/salinas.mat`
- `data/salinas_gt.mat`

The loader expects these MATLAB keys:

- `salinas_corrected` in `salinas.mat`
- `salinas_gt` in `salinas_gt.mat`

## Preprocessing Pipeline

In `utils/data_loader.py`, default `load_data()` behavior:

1. Flatten HSI cube into per-pixel spectral vectors.
2. Remove unlabeled pixels (`label == 0`).
3. Shift labels to zero-based indexing.
4. Standardize each feature (z-score normalization).
5. Convert arrays to PyTorch tensors and wrap in `DataLoader`.

Optional patch mode (`use_patches=True`) builds flattened local spatial patches.

## Setup

Use Python 3.9+ (recommended), then install dependencies:

```bash
pip install torch scipy numpy scikit-learn matplotlib seaborn
```

## Run

From repository root:

```bash
python main.py
```

The script will sequentially run all SSL methods and evaluate each with the same downstream classifier routine.

## Outputs

Generated image files include:

- `accuracy.png` - bar chart comparing method accuracies
- `{Method}.png` - confusion matrix for each method
- `tsne_{Method}.png` - 2D t-SNE embedding visualization per method

Console output includes:

- per-epoch SSL loss and epoch time
- per-epoch classifier loss
- final `results` dictionary with:
  - `acc`
  - `f1`
  - `convergence_epoch` (simple threshold-based estimate)
  - `total_time` (sum of SSL epoch times)

## Notes

- No checkpoint saving is implemented by default.
- Hyperparameters (epochs, batch size, mask ratio) are defined in function arguments in source files.
- The pipeline currently runs on default PyTorch device behavior (CPU unless you move tensors/models to CUDA manually).


