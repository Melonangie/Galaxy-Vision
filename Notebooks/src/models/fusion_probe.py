"""Linear probe de fusión: imagen `.jpg` + cubo `.fits` + fotometría.

Responde si el `.fits` aporta algo al clasificador temprana/tardía que no esté
ya en el `.jpg` (ver README.md, sección Imágenes + .fits: ¿aporta el .fits al
clasificador?). Es la misma herramienta de comparación que
`src/models/linear_probe.py`, extendida a varias entradas:

- **imagen:** features de la ResNet congelada sobre el `.jpg` del visor.
- **fits:** features de la **misma** ResNet congelada sobre el cubo `.fits`
  normalizado (`src/data/fits_normalization.py`), con las bandas agrupadas en
  los canales R, G y B (`FUSION_FITS_RGB_CHANNELS`).
- **fotometria:** unos pocos números medidos en el `.fits`
  (`FUSION_PHOTOMETRY_FEATURES`, de `src/data/fits_photometry.py`).

La "red combinada" es una fusión tardía: cada entrada pasa por su propia rama
congelada, los vectores se concatenan y una regresión logística clasifica.
Con decenas de galaxias no se puede entrenar una red de dos ramas de punta a
punta sin sobreajustar; congelar las ramas deja como única diferencia entre
corridas las entradas.

Cada comparación es por variante (capa y bandas, `FUSION_VARIANTS`): el `.jpg`
y el `.fits` de la misma variante.

Flujo:
    raw = load_fits_sample(names, petro_th90, layer, bands)
    stats = fit_fits_input_stats(raw)
    X_fits = embed_fits_cubes(raw, backbone, stats, bands)          # (n, 512)
    X = np.hstack([X_img, X_fits])
    out_of_fold_proba(make_fusion_probe([512, 512]), X, y)          # src/evaluate.py
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from torch import nn
from torchvision.transforms import functional as TF

from src.config import (
    FITS_NORM_SCHEME,
    FITS_NORM_SCHEME_GLOBAL,
    FUSION_FITS_RGB_CHANNELS,
    PROBE_BATCH_SIZE,
    PROBE_IMAGE_SIZE,
    PROBE_LOGREG_C,
    RANDOM_SEED,
    SHORTCUT_CV_FOLDS,
)
from src.data.cutout_io import fits_pixscale, load_fits_cutout
from src.data.fits_normalization import (
    GlobalNormReference,
    channel_stats,
    fit_global_reference,
    normalize_fits_cube,
    standardize,
)
from src.models.linear_probe import embed_tensors, make_logistic_probe

# Cubo crudo, escala (arcsec/px) y PETRO_TH90 (arcsec) de una galaxia.
RawFits = Tuple[np.ndarray, float, float]


@dataclass(frozen=True)
class FitsInputStats:
    """Constantes de normalización del `.fits`, medidas en un set de entrenamiento.

    Attributes:
        reference: constantes del esquema global (`None` con el por imagen).
        mean: media por banda de los cubos normalizados.
        std: desviación por banda de los cubos normalizados.
    """

    reference: Optional[GlobalNormReference]
    mean: np.ndarray
    std: np.ndarray


def load_fits_sample(
    names: Sequence[str],
    petro_th90: Sequence[float],
    layer: str,
    bands: str,
) -> List[RawFits]:
    """Carga los cubos `.fits` crudos de una lista de galaxias.

    Args:
        names: identificadores de las galaxias, en el orden deseado.
        petro_th90: PETRO_TH90 (arcsec) de cada galaxia, mismo orden.
        layer: capa del survey de los `.fits`.
        bands: bandas de los `.fits`.

    Returns:
        Lista de `(cubo, pixscale, petro_th90)` en el orden de `names`.
    """
    return [
        (load_fits_cutout(name, layer, bands), fits_pixscale(name, layer, bands), float(radius))
        for name, radius in zip(names, petro_th90)
    ]


def _normalized_cube(raw: RawFits, reference: Optional[GlobalNormReference]) -> np.ndarray:
    """Cubo normalizado (`normalize_fits_cube`) con los píxeles sin cobertura en 0 (cielo)."""
    cube, pixscale, petro_th90 = raw
    normalized = normalize_fits_cube(cube, pixscale, petro_th90, reference=reference)
    return np.nan_to_num(normalized, nan=0.0)


def fit_fits_input_stats(raws: Sequence[RawFits]) -> FitsInputStats:
    """Mide las constantes de normalización del `.fits` con un set de entrenamiento.

    Son los pasos 3 y 5 del pipeline de README.md, sección Normalización de los
    .fits: escala global (si `FITS_NORM_SCHEME` es el global) y media y
    desviación por banda. No usan las etiquetas.

    Args:
        raws: salida de `load_fits_sample` (solo galaxias de entrenamiento).

    Returns:
        `FitsInputStats` para `embed_fits_cubes`.
    """
    reference = fit_global_reference(raws) if FITS_NORM_SCHEME == FITS_NORM_SCHEME_GLOBAL else None
    mean, std = channel_stats(_normalized_cube(raw, reference) for raw in raws)
    return FitsInputStats(reference=reference, mean=mean, std=std)


def fits_rgb_channels(
    cube: np.ndarray,
    bands: str,
    rgb_channels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Agrupa las bandas de un cubo en los 3 canales R, G y B que recibe la red.

    Args:
        cube: cubo `(n_bandas, H, W)` en el orden de `bands`.
        bands: bandas del cubo, en orden.
        rgb_channels: bandas de cada canal R, G y B (un canal con varias bandas
            es su promedio); `None` usa `FUSION_FITS_RGB_CHANNELS[bands]`.

    Returns:
        Arreglo `(3, H, W)` `float32` contiguo.
    """
    channels = rgb_channels or FUSION_FITS_RGB_CHANNELS[bands]
    planes = [cube[[bands.index(band) for band in channel]].mean(axis=0) for channel in channels]
    return np.ascontiguousarray(np.stack(planes), dtype=np.float32)


def fits_network_input(
    raw: RawFits,
    stats: FitsInputStats,
    bands: str,
    rgb_channels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Cubo `.fits` listo para la red: normalizado, estandarizado y en canales RGB.

    Args:
        raw: `(cubo, pixscale, petro_th90)` de una galaxia.
        stats: salida de `fit_fits_input_stats`.
        bands: bandas del cubo, en orden.
        rgb_channels: ver `fits_rgb_channels`.

    Returns:
        Arreglo `(3, H, W)` `float32` contiguo, orientado como el `.jpg`.
    """
    cube = standardize(_normalized_cube(raw, stats.reference), stats.mean, stats.std)
    return fits_rgb_channels(cube, bands, rgb_channels)


def embed_fits_cubes(
    raws: Sequence[RawFits],
    backbone: nn.Module,
    stats: FitsInputStats,
    bands: str,
    image_size: int = PROBE_IMAGE_SIZE,
    batch_size: int = PROBE_BATCH_SIZE,
) -> np.ndarray:
    """Features del backbone congelado para cubos `.fits`.

    El cubo ya llega estandarizado por banda (`fits_network_input`), así que no
    se aplica la normalización de ImageNet que sí lleva el `.jpg`.

    Args:
        raws: salida de `load_fits_sample`.
        backbone: salida de `build_backbone` (el mismo que para el `.jpg`).
        stats: salida de `fit_fits_input_stats`.
        bands: bandas de los cubos, en orden.
        image_size: lado al que se reescala cada cubo.
        batch_size: cubos por lote en la GPU/CPU.

    Returns:
        Matriz `(n_galaxias, n_features)` de `float32`, en el orden de `raws`.
    """
    tensors = [
        TF.resize(
            torch.from_numpy(fits_network_input(raw, stats, bands)), [image_size, image_size], antialias=True
        )
        for raw in raws
    ]
    return embed_tensors(tensors, backbone, batch_size)


def block_weights(block_sizes: Sequence[int]) -> np.ndarray:
    """Peso por columna para que cada bloque de features pese lo mismo en la regularización.

    Con L2, multiplicar las columnas de un bloque por `w` equivale a dividir su
    penalización entre `w²`. Con `w_b = sqrt(D / (B · d_b))` (D columnas en
    total, B bloques, d_b columnas del bloque b), todos los bloques reciben la
    misma "varianza a priori" total `d_b · w_b² = D / B`: un bloque de 3
    números de fotometría no queda ahogado por 512 features de la ResNet. Con
    un solo bloque, o bloques del mismo tamaño, todos los pesos valen 1 y el
    probe es idéntico a `make_logistic_probe`.

    Args:
        block_sizes: número de columnas de cada bloque, en el orden de la concatenación.

    Returns:
        Arreglo de `sum(block_sizes)` pesos.
    """
    total, n_blocks = sum(block_sizes), len(block_sizes)
    return np.concatenate([np.full(size, np.sqrt(total / (n_blocks * size))) for size in block_sizes])


def _scale_columns(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiplica cada columna de `X` por su peso (función de módulo: se puede serializar)."""
    return X * weights


def make_fusion_probe(
    block_sizes: Sequence[int],
    c: float = PROBE_LOGREG_C,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Clasificador de la fusión: escalado + peso por bloque + regresión logística.

    Es `make_logistic_probe` con un paso intermedio que iguala el peso de cada
    bloque (`block_weights`); misma regularización, mismo solver.

    Args:
        block_sizes: número de columnas de cada bloque concatenado.
        c: inverso de la regularización L2.
        seed: semilla del solver.

    Returns:
        `Pipeline` de sklearn listo para `src/evaluate.py`.
    """
    probe = make_logistic_probe(c, seed)
    weighting = FunctionTransformer(_scale_columns, kw_args={"weights": block_weights(block_sizes)})
    probe.steps.insert(1, ("bloques", weighting))
    return probe


def fits_out_of_fold_proba_foldwise(
    raws: Sequence[RawFits],
    y: np.ndarray,
    backbone: nn.Module,
    bands: str,
    other_blocks: Sequence[np.ndarray] = (),
    n_splits: int = SHORTCUT_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Predicciones out-of-fold midiendo la normalización del `.fits` solo con cada set de entrenamiento.

    Control de fuga: el flujo principal mide `fit_fits_input_stats` una vez con
    todas las galaxias (sin etiquetas). Aquí se repite dentro de cada fold, como
    exige README.md (sección Normalización de los .fits); si el AUC no cambia,
    la versión rápida es válida. Usa las mismas particiones que
    `out_of_fold_proba` (`src/evaluate.py`).

    Args:
        raws: salida de `load_fits_sample`.
        y: etiquetas 0/1.
        backbone: salida de `build_backbone`.
        bands: bandas de los cubos, en orden.
        other_blocks: matrices de features que no dependen de la normalización
            (p. ej. la imagen), que se concatenan antes del `.fits`.
        n_splits: folds de la validación cruzada.
        seed: semilla de las particiones.

    Returns:
        Arreglo `(n_muestras,)` con la probabilidad de `y == 1`.
    """
    proba = np.empty(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(np.zeros((len(y), 1)), y):
        stats = fit_fits_input_stats([raws[i] for i in train])
        blocks = [*other_blocks, embed_fits_cubes(raws, backbone, stats, bands)]
        X = np.hstack(blocks)
        probe = make_fusion_probe([block.shape[1] for block in blocks]).fit(X[train], y[train])
        proba[test] = probe.predict_proba(X[test])[:, 1]
    return proba

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
