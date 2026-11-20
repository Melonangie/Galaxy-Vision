"""Linear probe: backbone preentrenado congelado + regresión logística.

Herramienta para **comparar variantes de imágenes**, no el clasificador final
(ese es el transfer learning con fine-tuning de README.md, sección
Arquitectura del modelo). Al congelar el backbone, la única diferencia entre
dos corridas es el dataset: no hay hiperparámetros de entrenamiento que
favorezcan a una variante sobre otra, y entrena en segundos incluso con pocas
galaxias. Ver README.md, sección Elegir la variante de imágenes.

Flujo:
    backbone = build_backbone()
    X = embed_cutouts(names, "ls-dr11", "grz", backbone)          # (n, 512)
    predictability_test(make_logistic_probe(), X, y)              # src/evaluate.py
"""

from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF

from src.config import (
    LEGACY_BANDS,
    LEGACY_LAYER,
    PROBE_BACKBONE,
    PROBE_BATCH_SIZE,
    PROBE_IMAGE_SIZE,
    PROBE_LOGREG_C,
    RANDOM_SEED,
)
from src.data.cutout_io import load_jpg_cutout

# Normalización con la que se entrenaron los pesos de ImageNet de torchvision.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name: str = PROBE_BACKBONE, device: Optional[str] = None) -> nn.Module:
    """Backbone de torchvision con pesos de ImageNet, sin la capa de clasificación.

    Args:
        name: arquitectura de `torchvision.models` (familia ResNet, p. ej.
            `"resnet34"`); se usan sus pesos `DEFAULT`.
        device: `"cuda"`/`"cpu"`; `None` elige GPU si está disponible.

    Returns:
        Modelo en modo `eval`, congelado, que devuelve el vector de features
        del global average pooling (512 dimensiones para ResNet-18/34).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.get_model(name, weights="DEFAULT")
    backbone.fc = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone.eval().to(device)


def _to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    """RGB `uint8` `(H, W, 3)` -> tensor normalizado `(3, S, S)`."""
    tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    tensor = TF.resize(tensor, [image_size, image_size], antialias=True)
    return TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)


@torch.no_grad()
def embed_images(
    images: Iterable[np.ndarray],
    backbone: nn.Module,
    image_size: int = PROBE_IMAGE_SIZE,
    batch_size: int = PROBE_BATCH_SIZE,
) -> np.ndarray:
    """Features del backbone para una colección de imágenes RGB.

    Args:
        images: arreglos `(H, W, 3)` `uint8`.
        backbone: salida de `build_backbone`.
        image_size: lado al que se reescala cada imagen.
        batch_size: imágenes por lote en la GPU/CPU.

    Returns:
        Matriz `(n_imágenes, n_features)` de `float32`.
    """
    return embed_tensors([_to_tensor(image, image_size) for image in images], backbone, batch_size)


@torch.no_grad()
def embed_tensors(
    tensors: List[torch.Tensor],
    backbone: nn.Module,
    batch_size: int = PROBE_BATCH_SIZE,
) -> np.ndarray:
    """Features del backbone para tensores ya preparados (mismo tamaño y normalización).

    Args:
        tensors: tensores `(3, S, S)` listos para el backbone.
        backbone: salida de `build_backbone`.
        batch_size: tensores por lote en la GPU/CPU.

    Returns:
        Matriz `(n_tensores, n_features)` de `float32`.
    """
    device = next(backbone.parameters()).device
    batches = [
        backbone(torch.stack(tensors[i:i + batch_size]).to(device)).cpu()
        for i in range(0, len(tensors), batch_size)
    ]
    return torch.cat(batches).numpy().astype(np.float32)


def embed_cutouts(
    names: Iterable[str],
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    backbone: Optional[nn.Module] = None,
    masked: bool = False,
) -> np.ndarray:
    """Features del backbone para los cutouts `.jpg` de una variante.

    Args:
        names: identificadores de las galaxias, en el orden deseado.
        layer: capa del survey de la variante.
        bands: bandas de la variante.
        backbone: salida de `build_backbone`; `None` construye uno.
        masked: `True` usa los cutouts enmascarados.

    Returns:
        Matriz `(n_galaxias, n_features)`, en el orden de `names`.
    """
    backbone = backbone or build_backbone()
    images = [load_jpg_cutout(name, layer, bands, masked=masked) for name in names]
    return embed_images(images, backbone)


def make_logistic_probe(c: float = PROBE_LOGREG_C, seed: int = RANDOM_SEED) -> Pipeline:
    """Clasificador del linear probe: escalado + regresión logística regularizada.

    Args:
        c: inverso de la regularización L2 (menor = más regularizado).
        seed: semilla del solver.

    Returns:
        `Pipeline` de sklearn listo para `predictability_test` /
        `out_of_fold_proba` (`src/evaluate.py`).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed)),
    ])

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
