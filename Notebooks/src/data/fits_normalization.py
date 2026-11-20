"""Normalización de los cubos `.fits` para usarlos como entrada de la red (Fase 2).

Un cubo `.fits` crudo no se puede pasar tal cual a una CNN: su rango cambia en
dos órdenes de magnitud entre galaxias, el cielo no está en cero y la mayor
parte de la información morfológica vive en flujos ~100 veces menores que el
núcleo. Pasos (ver README.md, sección Normalización de los .fits, y
notebook/normalization_fits.ipynb):

1. **Cielo:** se resta la mediana del cielo de cada banda (`subtract_sky`).
2. **Estiramiento asinh tipo Lupton et al. (2004)** sobre la intensidad media
   de las bandas: cada banda se multiplica por el mismo factor por píxel, así
   que los cocientes entre bandas (el color) se conservan (`lupton_asinh`).
3. **Escala:** se divide para que un flujo de referencia valga 1, con
   constantes globales (iguales para todo el dataset) o por imagen.
4. **Recorte** a `[FITS_NORM_CLIP_MIN, FITS_NORM_CLIP_MAX]` y orientación de
   `.jpg` (norte arriba), para que el cubo se alinee con el cutout del visor.

La estandarización por canal (media y desviación del set de entrenamiento) se
hace después, con `channel_stats` y `standardize`.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from src.config import (
    FITS_NORM_CLIP_MAX,
    FITS_NORM_CLIP_MIN,
    FITS_NORM_GLOBAL_REF_PERCENTILE,
    FITS_NORM_REF_PERCENTILE,
    FITS_NORM_SCHEME,
    FITS_NORM_SCHEME_GLOBAL,
    FITS_NORM_SCHEME_PER_IMAGE,
    FITS_NORM_SOFTENING_SIGMA,
    FITS_SKY_EXCLUDE_FACTOR,
)
from src.data.cutout_io import fits_to_image_orientation
from src.data.fits_photometry import radius_grid, sky_background


@dataclass(frozen=True)
class GlobalNormReference:
    """Constantes del esquema global, medidas en el set de entrenamiento.

    Attributes:
        noise: ruido típico (mediana entre galaxias) del cielo de la
            intensidad media, en nanomaggies por píxel nativo.
        ref_flux: flujo que se mapea a 1 (nanomaggies por píxel nativo).
    """

    noise: float
    ref_flux: float


def subtract_sky(
    cube: np.ndarray,
    pixscale: float,
    petro_th90: float,
    exclude_factor: float = FITS_SKY_EXCLUDE_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resta la mediana del cielo de cada banda.

    Args:
        cube: arreglo `(n_bandas, H, W)` de `load_fits_cutout`.
        pixscale: escala del cutout (arcsec/px).
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).
        exclude_factor: el cielo se mide fuera de este múltiplo de `petro_th90`.

    Returns:
        Tupla `(cubo sin cielo en float64, ruido del cielo por banda)`.
    """
    radius = radius_grid(cube.shape[1:], pixscale)
    clean = np.empty(cube.shape, dtype=np.float64)
    noise = np.empty(cube.shape[0])
    for band in range(cube.shape[0]):
        sky, noise[band] = sky_background(cube[band].astype(np.float64), radius, exclude_factor * petro_th90)
        clean[band] = cube[band] - sky
    return clean, noise


def intensity_noise(clean: np.ndarray, pixscale: float, petro_th90: float) -> float:
    """Ruido del cielo de la intensidad media de las bandas (ya sin cielo).

    Args:
        clean: cubo sin cielo de `subtract_sky`.
        pixscale: escala del cutout (arcsec/px).
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).

    Returns:
        Desviación estándar con recorte sigma, en nanomaggies por píxel nativo.
    """
    radius = radius_grid(clean.shape[1:], pixscale)
    return sky_background(clean.mean(axis=0), radius, FITS_SKY_EXCLUDE_FACTOR * petro_th90)[1]


def lupton_asinh(clean: np.ndarray, softening: float) -> np.ndarray:
    """Estiramiento asinh sobre la intensidad, conservando el color.

    Cada banda se multiplica por `asinh(I/β) / (I/β)`, con `I` la intensidad
    media de las bandas y `β = softening`: lineal para `I << β`, logarítmico
    para `I >> β`. Como el factor es el mismo en todas las bandas de un píxel,
    los cocientes entre bandas no cambian.

    Args:
        clean: cubo sin cielo `(n_bandas, H, W)`.
        softening: β, en las unidades del cubo.

    Returns:
        Cubo estirado en unidades de β (la intensidad estirada es `asinh(I/β)`).
    """
    scaled = clean / softening
    intensity = scaled.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(np.abs(intensity) > 1e-6, np.arcsinh(intensity) / intensity, 1.0)
    return scaled * factor


def reference_flux(
    clean: np.ndarray,
    pixscale: float,
    petro_th90: float,
    percentile: float = FITS_NORM_REF_PERCENTILE,
) -> float:
    """Flujo de referencia de una galaxia: percentil de la intensidad en PETRO_TH90.

    Un percentil alto (y no el máximo) evita que una estrella o un píxel
    caliente dentro de la galaxia fije la escala.

    Args:
        clean: cubo sin cielo.
        pixscale: escala del cutout (arcsec/px).
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).
        percentile: percentil de la intensidad dentro de `petro_th90`.

    Returns:
        Flujo en las unidades del cubo.
    """
    inside = radius_grid(clean.shape[1:], pixscale) <= petro_th90
    return float(np.percentile(clean.mean(axis=0)[inside], percentile))


def fit_global_reference(
    cubes: Iterable[Tuple[np.ndarray, float, float]],
    ref_percentile: float = FITS_NORM_REF_PERCENTILE,
    global_percentile: float = FITS_NORM_GLOBAL_REF_PERCENTILE,
) -> GlobalNormReference:
    """Mide las constantes del esquema global (solo con el set de entrenamiento).

    El suavizado β se calcula después, en `normalize_fits_cube`, como
    `softening_sigma × noise`.

    Args:
        cubes: iterable de `(cubo crudo, pixscale, petro_th90)`.
        ref_percentile: percentil de la intensidad en PETRO_TH90 por galaxia.
        global_percentile: percentil, entre galaxias, de esos flujos de
            referencia que se mapea a 1 (con 90, ~10% de las galaxias tendrán
            el núcleo por encima de 1).

    Returns:
        `GlobalNormReference` con el ruido típico y el flujo de referencia.
    """
    noises, refs = [], []
    for cube, pixscale, petro_th90 in cubes:
        clean, _ = subtract_sky(cube, pixscale, petro_th90)
        noises.append(intensity_noise(clean, pixscale, petro_th90))
        refs.append(reference_flux(clean, pixscale, petro_th90, ref_percentile))
    return GlobalNormReference(
        noise=float(np.median(noises)), ref_flux=float(np.percentile(refs, global_percentile))
    )


def normalize_fits_cube(
    cube: np.ndarray,
    pixscale: float,
    petro_th90: float,
    scheme: str = FITS_NORM_SCHEME,
    reference: Optional[GlobalNormReference] = None,
    softening_sigma: float = FITS_NORM_SOFTENING_SIGMA,
    ref_percentile: float = FITS_NORM_REF_PERCENTILE,
    clip: Tuple[float, float] = (FITS_NORM_CLIP_MIN, FITS_NORM_CLIP_MAX),
    image_orientation: bool = True,
) -> np.ndarray:
    """Normaliza un cubo `.fits` crudo para usarlo como entrada de la red.

    Args:
        cube: arreglo `(n_bandas, H, W)` de `load_fits_cutout`.
        pixscale: escala del cutout (arcsec/px).
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).
        scheme: `FITS_NORM_SCHEME_GLOBAL` o `FITS_NORM_SCHEME_PER_IMAGE`.
        reference: constantes de `fit_global_reference` (obligatorias con el
            esquema global).
        softening_sigma: β en múltiplos del ruido de la intensidad (el ruido
            global con el esquema global, el de la imagen con el por imagen).
        ref_percentile: percentil del flujo de referencia por imagen.
        clip: recorte final `(mínimo, máximo)`.
        image_orientation: `True` voltea las filas para alinearlo con el `.jpg`.

    Returns:
        Cubo `(n_bandas, H, W)` `float32`; el flujo de referencia vale ~1 en
        la intensidad media y el cielo queda alrededor de 0.

    Raises:
        ValueError: si el esquema no existe o falta `reference` con el global.
    """
    clean, _ = subtract_sky(cube, pixscale, petro_th90)
    if scheme == FITS_NORM_SCHEME_GLOBAL:
        if reference is None:
            raise ValueError("El esquema global necesita `reference` (fit_global_reference).")
        noise, ref_flux = reference.noise, reference.ref_flux
    elif scheme == FITS_NORM_SCHEME_PER_IMAGE:
        noise = intensity_noise(clean, pixscale, petro_th90)
        ref_flux = reference_flux(clean, pixscale, petro_th90, ref_percentile)
    else:
        raise ValueError(f"Esquema desconocido: {scheme!r}.")

    softening = softening_sigma * noise
    stretched = lupton_asinh(clean, softening) / np.arcsinh(ref_flux / softening)
    normalized = np.clip(stretched, *clip).astype(np.float32)
    return fits_to_image_orientation(normalized) if image_orientation else normalized


def channel_stats(cubes: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Media y desviación por canal de cubos ya normalizados (set de entrenamiento).

    Args:
        cubes: iterable de cubos `(n_bandas, H, W)` de `normalize_fits_cube`.

    Returns:
        Tupla `(media, desviación)`, arreglos de `n_bandas`.
    """
    total, total_sq, count = 0.0, 0.0, 0
    for cube in cubes:
        values = cube.reshape(cube.shape[0], -1).astype(np.float64)
        total = total + values.sum(axis=1)
        total_sq = total_sq + (values ** 2).sum(axis=1)
        count += values.shape[1]
    mean = total / count
    return mean, np.sqrt(total_sq / count - mean ** 2)


def standardize(cube: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Estandariza cada canal con la media y desviación del entrenamiento.

    Args:
        cube: cubo normalizado `(n_bandas, H, W)`.
        mean: media por canal de `channel_stats`.
        std: desviación por canal de `channel_stats`.

    Returns:
        Cubo `float32` con media ~0 y desviación ~1 por canal.
    """
    return ((cube - mean[:, None, None]) / std[:, None, None]).astype(np.float32)
