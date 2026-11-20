"""Entrenamiento de la CNN baseline (elíptica vs. espiral).

Funciones reutilizables desde notebooks o desde este mismo script
(`python -m src.train`) — no se reimplementa la lógica de entrenamiento en el
notebook (ver README.md, regla de oro).
"""

import copy
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CNN_BATCH_SIZE,
    CNN_DROPOUT,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    CNN_NUM_WORKERS,
    LABEL_TO_INT,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.data.augmentation import get_eval_transforms, get_train_transforms
from src.data.dataset import load_splits_from_csv
from src.data.torch_dataset import InMemoryGalaxyDataset
from src.evaluate import evaluate_binary_classifier
from src.models.baseline_cnn import BaselineCNN
from src.utils import set_seed


def compute_class_weights(train_df, device: torch.device) -> torch.Tensor:
    """Pesos de clase balanceados (ver README.md, sección Manejo de desbalance de clases) para `CrossEntropyLoss`.

    Args:
        train_df: DataFrame de entrenamiento con columna `label` codificable
            vía `src.config.LABEL_TO_INT` (0=elíptica, 1=espiral).
        device: dispositivo donde vivirá el tensor de pesos.

    Returns:
        Tensor `[peso_elíptica, peso_espiral]`.
    """
    y = train_df["label"].map(LABEL_TO_INT).to_numpy()
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Corre una época de entrenamiento y devuelve la pérdida promedio."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa el modelo sobre `loader`.

    Returns:
        `(avg_loss, y_true, y_pred, y_proba)`, con `y_proba` = probabilidad
        de la clase positiva (1 = espiral), listos para
        `src.evaluate.evaluate_binary_classifier`.
    """
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_proba = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        proba = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = CNN_EPOCHS,
    lr: float = CNN_LEARNING_RATE,
) -> Dict[str, List[float]]:
    """Entrena `model`, evaluando en `val_loader` al final de cada época.

    Se queda con los pesos de la época de mejor ROC-AUC en validación
    (`model` termina cargado con esos pesos al retornar).

    Args:
        model: red a entrenar (modificada in-place).
        train_loader: DataLoader del split de entrenamiento (con augmentation).
        val_loader: DataLoader del split de validación (sin augmentation).
        class_weights: pesos de clase para `CrossEntropyLoss` (ver
            `compute_class_weights`).
        device: `torch.device` de entrenamiento.
        epochs: número de épocas.
        lr: learning rate para Adam.

    Returns:
        Historial por época: `train_loss`, `val_loss`, `val_accuracy`,
        `val_roc_auc`.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_roc_auc": []}
    best_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred, y_proba = evaluate(model, val_loader, criterion, device)
        metrics = evaluate_binary_classifier(y_true, y_pred, y_proba)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_roc_auc"].append(metrics["roc_auc"])

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{epoch:02d}/{epochs}] train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={metrics['accuracy']:.4f}  "
            f"val_auc={metrics['roc_auc']:.4f}  ({time.time() - t0:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df, _test_df = load_splits_from_csv()

    train_ds = InMemoryGalaxyDataset(train_df, transform=get_train_transforms())
    val_ds = InMemoryGalaxyDataset(val_df, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=CNN_NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=CNN_NUM_WORKERS
    )

    class_weights = compute_class_weights(train_df, device)
    model = BaselineCNN(dropout=CNN_DROPOUT).to(device)

    fit(model, train_loader, val_loader, class_weights, device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "best_cnn_baseline.pt")
    print(f"Modelo guardado en {MODELS_DIR / 'best_cnn_baseline.pt'}")


if __name__ == "__main__":
    main()

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
