"""Extracción de features de píxeles planos para los baselines de sklearn.

Los modelos clásicos de sklearn no consumen imágenes directamente: cada imagen
se reescala a `FEATURE_IMAGE_SIZE`x`FEATURE_IMAGE_SIZE`, se normaliza a [0, 1]
y se aplana a un vector 1D (ver README.md, sección Preprocesamiento de
imágenes). Esta es intencionalmente una representación simple (sin recorte
adaptativo ni extracción de features tipo HOG): sirve de piso de comparación
antes de pasar a la CNN/ResNet de la fase 1 (ver README.md, sección
Arquitectura del modelo).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.config import FEATURE_IMAGE_SIZE, LABEL_TO_INT, TRAIN_IMAGES_DIR


def _load_image_as_vector(path: Path, image_size: int) -> np.ndarray:
    """Carga una imagen, la reescala, normaliza a [0, 1] y aplana."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def extract_pixel_features(
    df: pd.DataFrame,
    images_dir: Path = TRAIN_IMAGES_DIR,
    image_size: int = FEATURE_IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae `X` (píxeles aplanados) e `y` (etiqueta binaria numérica).

    Args:
        df: DataFrame con columnas `GalaxyID` y `label` (`elliptical`/`spiral`).
        images_dir: carpeta con las imágenes `<GalaxyID>.jpg`.
        image_size: lado (px) al que se reescala cada imagen antes de aplanar.

    Returns:
        `(X, y)` con `X` de forma `(n_samples, image_size * image_size * 3)`
        y `y` con codificación entera (`LABEL_TO_INT`).
    """
    vectors = [
        _load_image_as_vector(images_dir / f"{galaxy_id}.jpg", image_size)
        for galaxy_id in df["GalaxyID"]
    ]
    X = np.stack(vectors)
    y = df["label"].map(LABEL_TO_INT).to_numpy()
    return X, y

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
