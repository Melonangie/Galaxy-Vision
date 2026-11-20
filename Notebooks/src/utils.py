"""Utilidades transversales del proyecto."""

import random

import numpy as np
import torch

from src.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fija la semilla de `random`, `numpy` y `torch` (CPU y CUDA).

    Crítico para reproducibilidad (ver README.md, sección Convenciones de
    código): varios de los papers de referencia reportan variabilidad de
    resultados por inicialización aleatoria.

    Args:
        seed: semilla a usar en los tres generadores.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
