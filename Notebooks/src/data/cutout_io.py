"""Lectura de los cutouts descargados (.jpg compuesto y .fits por banda).

Dos formatos, dos librerías y dos usos distintos (ver README.md, sección
Formato de las imágenes):

- `.jpg`  -> Pillow. RGB de 8 bits ya estirado por el visor; es el formato del
  set de entrenamiento, se enchufa directo a `torchvision`.
- `.fits` -> `astropy.io.fits`. Cubo `(n_bandas, H, W)` float32 con el flujo
  calibrado de cada banda (g, r, z con la variante por defecto); es el formato
  para medir (fotometría, índices de color, perfiles de brillo), no para
  alimentar la CNN tal cual.

Todas las funciones de carga reciben `layer` y `bands` para elegir la variante
(subcarpeta `<layer>_<bands>`, ver `src.data.legacy_cutouts.cutout_path`); por
defecto usan la variante activa de `src/config.py`.

`fits_cube_to_rgb` permite pasar del segundo al primero, para comparar el
render propio contra el del visor.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy import ndimage as ndi

from src.config import (
    CUTOUT_FORMAT_FITS,
    CUTOUT_FORMAT_JPG,
    CUTOUTS_MASKED_JPG_DIR,
    FITS_RGB_ASINH_Q,
    FITS_RGB_ASINH_STRETCH,
    LEGACY_BANDS,
    LEGACY_LAYER,
    NO_COVERAGE_SATURATION,
    NO_COVERAGE_VAR_MAX,
    NO_COVERAGE_WINDOW_PX,
)
from src.data.legacy_cutouts import cutout_path


def load_jpg_file(path: Path) -> np.ndarray:
    """Carga un `.jpg` cualquiera como arreglo RGB de 8 bits.

    Args:
        path: ruta del archivo.

    Returns:
        Arreglo `(H, W, 3)` de `uint8`.
    """
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def load_jpg_cutout(
    name: str,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    masked: bool = False,
) -> np.ndarray:
    """Carga el cutout `.jpg` de una galaxia como arreglo RGB de 8 bits.

    Args:
        name: identificador de la galaxia (columna `name`).
        layer: capa del survey de la variante.
        bands: bandas de la variante.
        masked: `True` lee la versión enmascarada (`CUTOUTS_MASKED_JPG_DIR`).

    Returns:
        Arreglo `(H, W, 3)` de `uint8`.
    """
    root = CUTOUTS_MASKED_JPG_DIR if masked else None
    return load_jpg_file(cutout_path(name, CUTOUT_FORMAT_JPG, layer, bands, root))


def no_coverage_fraction(
    image: np.ndarray,
    window_px: int = NO_COVERAGE_WINDOW_PX,
    var_max: float = NO_COVERAGE_VAR_MAX,
    saturation: int = NO_COVERAGE_SATURATION,
) -> float:
    """Fracción de la imagen sin cobertura del survey (color plano, sin ruido).

    El visor rellena las zonas sin datos con un color uniforme, mientras que el
    cielo real siempre trae ruido; un píxel se considera sin cobertura si la
    varianza local de los tres canales en una ventana de `window_px` es menor
    que `var_max` y ningún canal está saturado (un núcleo saturado también es
    plano, pero sí tiene cobertura).

    Args:
        image: arreglo `(H, W, 3)` RGB.
        window_px: lado de la ventana donde se mide la varianza local.
        var_max: varianza máxima (intensidad^2) para considerar plano un píxel.
        saturation: intensidad a partir de la cual un canal se considera saturado.

    Returns:
        Fracción entre 0 (cobertura completa) y 1 (imagen vacía).
    """
    pixels = image.astype(np.float64)
    size = (window_px, window_px, 1)
    mean = ndi.uniform_filter(pixels, size=size)
    var = ndi.uniform_filter(pixels ** 2, size=size) - mean ** 2
    flat = var.max(axis=-1) < var_max
    saturated = image.max(axis=-1) >= saturation
    return float((flat & ~saturated).mean())


def load_fits_cutout(
    name: str, layer: str = LEGACY_LAYER, bands: str = LEGACY_BANDS
) -> np.ndarray:
    """Carga el cutout `.fits` de una galaxia como cubo de bandas.

    Args:
        name: identificador de la galaxia (columna `name`).
        layer: capa del survey de la variante.
        bands: bandas de la variante.

    Returns:
        Arreglo `(n_bandas, H, W)` de `float32` con el flujo de cada banda, en
        el orden de `bands`. Los píxeles sin cobertura vienen como `NaN`.
    """
    with fits.open(cutout_path(name, CUTOUT_FORMAT_FITS, layer, bands)) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def fits_pixscale(
    name: str, layer: str = LEGACY_LAYER, bands: str = LEGACY_BANDS
) -> float:
    """Escala de píxel (arcsec/px) de un cutout `.fits`, leída de su WCS.

    Cada galaxia se descarga con su propio zoom (ver `compute_pixscale` en
    `src/data/legacy_cutouts.py`); el encabezado lo trae en `CD2_2` (grados/px).

    Args:
        name: identificador de la galaxia (columna `name`).
        layer: capa del survey de la variante.
        bands: bandas de la variante.

    Returns:
        Arcosegundos por píxel.
    """
    header = fits.getheader(cutout_path(name, CUTOUT_FORMAT_FITS, layer, bands))
    return abs(float(header["CD2_2"])) * 3600.0


def fits_cube_to_rgb(
    cube: np.ndarray,
    bands: str = LEGACY_BANDS,
    q: float = FITS_RGB_ASINH_Q,
    stretch: float = FITS_RGB_ASINH_STRETCH,
    image_orientation: bool = True,
) -> np.ndarray:
    """Convierte un cubo con g, r y z en una imagen RGB de 8 bits (asinh).

    Mapeo de canales: R = z (banda más roja), G = r, B = g, que es la
    convención del visor de Legacy Survey. Si el cubo trae más bandas (p. ej.
    `griz`), las demás se ignoran.

    Args:
        cube: arreglo `(n_bandas, H, W)` devuelto por `load_fits_cutout`.
        bands: bandas del cubo, en orden (las de la variante descargada).
        q: parámetro de suavidad del asinh; valores altos comprimen más el
            núcleo brillante.
        stretch: escala de flujo a partir de la cual el estiramiento satura.
        image_orientation: `True` voltea las filas para que coincida píxel a
            píxel con el `.jpg` (ver `fits_to_image_orientation`).

    Returns:
        Arreglo `(H, W, 3)` de `uint8`.

    Raises:
        ValueError: si el cubo no coincide con `bands` o le falta g, r o z.
    """
    if cube.ndim != 3 or cube.shape[0] != len(bands):
        raise ValueError(
            f"Se esperaba un cubo ({len(bands)}, H, W) con bandas {bands!r}, "
            f"se recibió {cube.shape}."
        )
    band_g, band_r, band_z = (
        np.nan_to_num(band_image(cube, band, bands), nan=0.0) for band in "grz"
    )
    rgb = np.stack([band_z, band_r, band_g], axis=-1)

    intensity = np.maximum(rgb.mean(axis=-1, keepdims=True), 1e-10)
    scaled = rgb * np.arcsinh(q * intensity / stretch) / (q * intensity)
    image = (np.clip(scaled, 0.0, 1.0) * 255).round().astype(np.uint8)
    return fits_to_image_orientation(image) if image_orientation else image


def fits_to_image_orientation(array: np.ndarray) -> np.ndarray:
    """Voltea filas de un arreglo `.fits` para alinearlo con el `.jpg`.

    En FITS la fila 0 es el borde inferior (sur) y en un `.jpg` es el superior
    (norte); sin voltear, el `.fits` sale espejado de arriba abajo respecto al
    cutout del visor (verificado en notebook/exploration_fits.ipynb).

    Args:
        array: plano `(H, W)`, cubo `(n_bandas, H, W)` o imagen `(H, W, 3)`.
            Un arreglo 3D cuyo primer eje es corto (<= 4) se trata como cubo.

    Returns:
        Vista del arreglo con las filas en orden de imagen (norte arriba).
    """
    if array.ndim == 3 and array.shape[0] <= 4:
        return array[:, ::-1, :]
    return array[::-1]


def band_image(cube: np.ndarray, band: str, bands: str = LEGACY_BANDS) -> np.ndarray:
    """Extrae una banda del cubo por su letra.

    Args:
        cube: arreglo `(n_bandas, H, W)` devuelto por `load_fits_cutout`.
        band: letra de la banda, p. ej. `"g"`, `"r"` o `"z"`.
        bands: bandas del cubo, en orden.

    Returns:
        Arreglo `(H, W)` de `float32` con el flujo de esa banda.

    Raises:
        ValueError: si la banda no está entre las descargadas.
    """
    if band not in bands:
        raise ValueError(f"Banda {band!r} no está en {bands!r}.")
    return cube[bands.index(band)]


def cutout_stats(
    cube: np.ndarray, name: Optional[str] = None, bands: str = LEGACY_BANDS
) -> dict:
    """Estadísticas rápidas de un cubo `.fits`, para control de calidad.

    Útil para detectar cutouts vacíos o con cobertura parcial antes de
    entrenar (ver README.md, sección Formato de las imágenes).

    Args:
        cube: arreglo `(n_bandas, H, W)` devuelto por `load_fits_cutout`.
        name: identificador de la galaxia, si se quiere en el resultado.
        bands: bandas del cubo, en orden.

    Returns:
        Diccionario con `name`, `shape`, `frac_finita` (proporción de píxeles
        con dato) y el flujo mediano y máximo por banda.
    """
    stats = {
        "name": name,
        "shape": tuple(cube.shape),
        "frac_finita": float(np.isfinite(cube).mean()),
    }
    for band in bands:
        plane = band_image(cube, band, bands)
        stats[f"mediana_{band}"] = float(np.nanmedian(plane))
        stats[f"max_{band}"] = float(np.nanmax(plane))
    return stats

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
