"""Tratamientos de imagen alternativos a la máscara dura de `galaxy_mask.py`.

La máscara de color con fondo negro empeora al clasificador temprana/tardía
(notebook/variant-classification-for-images-only.ipynb). Este módulo reúne las
alternativas que se probaron para limpiar o realzar la galaxia sin ese costo, y
que se comparan con el linear probe en notebook/image-curation.ipynb, sección 7
(ver README.md, sección Limpieza de imágenes):

- **Fondo de cielo en vez de negro** (`fill_background_with_sky`): lo que queda
  fuera de la máscara se rellena con ruido del color y la dispersión del cielo
  de la propia imagen, para no crear un borde duro sobre negro.
- **Bordes suaves** (`soft_color_mask`, `soft_aperture`): la transición galaxia
  -> cielo es gradual (caída gaussiana), con más margen que la máscara actual o
  con una apertura circular de radio ~PETRO_TH90.
- **Solo quitar estrellas** (`remove_stars`): se conserva toda la imagen y solo
  se reemplazan por cielo las fuentes compactas que no tocan la galaxia central.
- **Realce** (`gamma_stretch`): aclara el disco externo y el halo débil.

`IMAGE_TREATMENTS` mapea cada nombre a una función `(imagen, rng) -> imagen`
para recorrerlos en un experimento.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from src.config import (
    CUTOUT_FOV_FACTOR,
    GALAXY_MASK_BG_PERCENTILE,
    GALAXY_MASK_BORDER_PX,
    SOFT_APERTURE_EDGE_SIGMA_PX,
    SOFT_APERTURE_R90_FACTOR,
    SOFT_MASK_EDGE_SIGMA_PX,
    SOFT_MASK_MARGIN_PX,
    STAR_DETECT_SIGMA,
    STAR_GROW_PX,
    STAR_PROTECT_MARGIN_PX,
    TREATMENT_GAMMA,
)
from src.data.galaxy_mask import _add_margin, _border_mask, apply_mask, galaxy_color_mask

Treatment = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def sky_color_stats(
    image: np.ndarray, exclude: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Mediana y desviación RGB del cielo, medidas en el marco perimetral.

    Args:
        image: arreglo `(H, W, 3)` RGB.
        exclude: máscara booleana `(H, W)` de píxeles a ignorar (p. ej. la
            galaxia cuando llega al borde).

    Returns:
        Tupla `(mediana, desviación)`, cada una de forma `(3,)`. Se descarta el
        `100 - GALAXY_MASK_BG_PERCENTILE` % más brillante del marco (estrellas).
    """
    border = _border_mask(image.shape[:2], GALAXY_MASK_BORDER_PX)
    if exclude is not None:
        border &= ~exclude
    sky = image[border].astype(np.float64)
    brightness = sky.mean(axis=1)
    sky = sky[brightness < np.percentile(brightness, GALAXY_MASK_BG_PERCENTILE)]
    return np.median(sky, axis=0), sky.std(axis=0)


def sky_noise(
    image: np.ndarray, rng: np.random.Generator, exclude: Optional[np.ndarray] = None
) -> np.ndarray:
    """Imagen de ruido gaussiano con el color y la dispersión del cielo de `image`.

    Args:
        image: arreglo `(H, W, 3)` RGB de referencia.
        rng: generador de números aleatorios (fijar su semilla).
        exclude: ver `sky_color_stats`.

    Returns:
        Arreglo `(H, W, 3)` `float64` en [0, 255].
    """
    median, std = sky_color_stats(image, exclude)
    return np.clip(rng.normal(median, std, image.shape), 0, 255)


def blend_with_sky(image: np.ndarray, weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mezcla `image` con ruido de cielo según un peso por píxel.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        weight: arreglo `(H, W)` en [0, 1]; 1 conserva la imagen, 0 la
            reemplaza por cielo.
        rng: generador de números aleatorios.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    w = weight[..., None]
    mixed = w * image + (1 - w) * sky_noise(image, rng)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def hard_mask(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Máscara actual de `galaxy_mask.py`: galaxia central sobre fondo negro."""
    return apply_mask(image, galaxy_color_mask(image))


def fill_background_with_sky(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Máscara actual, pero con el fondo relleno de ruido de cielo en vez de negro.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: generador de números aleatorios.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    mask = galaxy_color_mask(image)
    return blend_with_sky(image, mask.astype(np.float64), rng)


def wide_mask(
    image: np.ndarray, rng: np.random.Generator, margin_px: int = SOFT_MASK_MARGIN_PX
) -> np.ndarray:
    """Máscara actual con `margin_px` píxeles más de margen y fondo negro.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: generador de números aleatorios (no se usa; firma común).
        margin_px: margen adicional (px) sobre el de `galaxy_color_mask`.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    return apply_mask(image, _add_margin(galaxy_color_mask(image), margin_px))


def soft_color_mask(
    image: np.ndarray,
    rng: np.random.Generator,
    margin_px: int = SOFT_MASK_MARGIN_PX,
    edge_sigma_px: float = SOFT_MASK_EDGE_SIGMA_PX,
) -> np.ndarray:
    """Máscara de color con más margen y borde suavizado hacia el ruido de cielo.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: generador de números aleatorios.
        margin_px: margen adicional (px) sobre el de `galaxy_color_mask`.
        edge_sigma_px: sigma (px) de la caída gaussiana fuera de la máscara.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    mask = _add_margin(galaxy_color_mask(image), margin_px)
    distance = ndi.distance_transform_edt(~mask)
    return blend_with_sky(image, np.exp(-0.5 * (distance / edge_sigma_px) ** 2), rng)


def soft_aperture(
    image: np.ndarray,
    rng: np.random.Generator,
    r90_factor: float = SOFT_APERTURE_R90_FACTOR,
    edge_sigma_px: float = SOFT_APERTURE_EDGE_SIGMA_PX,
) -> np.ndarray:
    """Apertura circular centrada de radio `r90_factor` × PETRO_TH90, con borde suave.

    No depende del color: con el encuadre de `CUTOUT_FOV_FACTOR`, PETRO_TH90
    mide `lado / CUTOUT_FOV_FACTOR` px en toda la muestra.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: generador de números aleatorios.
        r90_factor: radio de la apertura en múltiplos de PETRO_TH90.
        edge_sigma_px: sigma (px) de la caída gaussiana fuera del radio.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    h, w = image.shape[:2]
    radius_px = r90_factor * min(h, w) / CUTOUT_FOV_FACTOR
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    weight = np.where(r <= radius_px, 1.0, np.exp(-0.5 * ((r - radius_px) / edge_sigma_px) ** 2))
    return blend_with_sky(image, weight, rng)


def star_mask(
    image: np.ndarray,
    detect_sigma: float = STAR_DETECT_SIGMA,
    grow_px: int = STAR_GROW_PX,
    protect_margin_px: int = STAR_PROTECT_MARGIN_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuentes brillantes que no tocan la galaxia central (estrellas, vecinas separadas).

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        detect_sigma: umbral de detección en desviaciones robustas del cielo.
        grow_px: dilatación (px) de cada fuente para cubrir su halo.
        protect_margin_px: margen (px) alrededor de la galaxia central que no
            se toca.

    Returns:
        Tupla `(estrellas, galaxia_protegida)`, máscaras booleanas `(H, W)`.
    """
    protected = _add_margin(galaxy_color_mask(image), protect_margin_px)
    brightness = ndi.gaussian_filter(image.astype(np.float64).mean(axis=2), 1.0)
    sky = brightness[_border_mask(brightness.shape, GALAXY_MASK_BORDER_PX)]
    median = np.median(sky)
    sigma = max(1.4826 * np.median(np.abs(sky - median)), 1.0)
    labels, _ = ndi.label(brightness > median + detect_sigma * sigma)
    touching = np.unique(labels[protected & (labels > 0)])
    stars = (labels > 0) & ~np.isin(labels, touching)
    return ndi.binary_dilation(stars, iterations=grow_px), protected


def remove_stars(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reemplaza por ruido de cielo solo las fuentes de `star_mask`; conserva lo demás.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: generador de números aleatorios.

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    stars, protected = star_mask(image)
    cleaned = np.where(stars[..., None], sky_noise(image, rng, exclude=protected), image)
    return cleaned.astype(np.uint8)


def gamma_stretch(
    image: np.ndarray, rng: Optional[np.random.Generator] = None, gamma: float = TREATMENT_GAMMA
) -> np.ndarray:
    """Corrección gamma: con `gamma < 1` aclara el disco externo y el halo débil.

    Args:
        image: arreglo `(H, W, 3)` `uint8`.
        rng: no se usa; firma común de `IMAGE_TREATMENTS`.
        gamma: exponente aplicado a la intensidad normalizada a [0, 1].

    Returns:
        Arreglo `(H, W, 3)` `uint8`.
    """
    return (255 * (image / 255.0) ** gamma).astype(np.uint8)


def remove_stars_and_gamma(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`remove_stars` seguido de `gamma_stretch`."""
    return gamma_stretch(remove_stars(image, rng))


def background_only(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Control diagnóstico: la galaxia central en negro y el resto intacto.

    Mide cuánta información útil para el clasificador queda **fuera** de la
    máscara (luz débil de la galaxia, vecinas, entorno). No es un tratamiento.
    """
    return apply_mask(image, ~galaxy_color_mask(image))


IMAGE_TREATMENTS: Dict[str, Treatment] = {
    "mascara_actual": hard_mask,
    "mascara_fondo_cielo": fill_background_with_sky,
    "mascara_margen_ancho": wide_mask,
    "mascara_suave": soft_color_mask,
    "apertura_suave_R90": soft_aperture,
    "sin_estrellas": remove_stars,
    "gamma": gamma_stretch,
    "sin_estrellas_gamma": remove_stars_and_gamma,
    "control_solo_fondo": background_only,
}

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
