"""Fotometría básica sobre los cutouts `.fits` de legacysurvey.org (Fase 2).

El `.fits` es el formato para medir (ver README.md, secciones Formato de las
imágenes y Exploración de los .fits). Dos detalles del servicio que hay que
respetar para que las medidas tengan unidades físicas:

- **Unidades:** cada píxel está en nanomaggies (AB, punto cero 22.5) **por
  píxel nativo** de `LEGACY_NATIVE_PIXSCALE` (0.262"), aunque el cutout se haya
  pedido con otro zoom: el servicio remuestrea conservando el brillo
  superficial, no el flujo. Verificado pidiendo la misma galaxia a 0.262 y
  0.524 "/px: la suma en una apertura baja 4x y el ruido del cielo no cambia.
  Para obtener flujo hay que multiplicar la suma por
  `(pixscale / LEGACY_NATIVE_PIXSCALE) ** 2` (`native_flux_factor`).
- **Zoom por galaxia:** cada cutout tiene su propio `pixscale` (ver
  `fits_pixscale`), así que los radios se manejan en arcsec, no en píxeles.

Exploración y validación en `notebook/exploration_fits.ipynb`.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats

from src.config import (
    CONCENTRATION_INNER_FRAC,
    CONCENTRATION_OUTER_FRAC,
    FITS_APERTURE_FACTOR,
    FITS_CONCENTRATION_BAND,
    FITS_PROFILE_MAX_FACTOR,
    FITS_PROFILE_N_BINS,
    FITS_SKY_EXCLUDE_FACTOR,
    FITS_SKY_MIN_FRAC,
    FITS_SKY_SIGMA_CLIP,
    LEGACY_BANDS,
    LEGACY_LAYER,
    LEGACY_NANOMAGGY_ZEROPOINT,
    LEGACY_NATIVE_PIXSCALE,
)
from src.data.cutout_io import band_image, fits_pixscale, load_fits_cutout


def native_flux_factor(
    pixscale: float, native_pixscale: float = LEGACY_NATIVE_PIXSCALE
) -> float:
    """Factor que convierte una suma de píxeles remuestreados en flujo.

    Args:
        pixscale: escala del cutout (arcsec/px).
        native_pixscale: escala nativa del survey (arcsec/px).

    Returns:
        `(pixscale / native_pixscale) ** 2`.
    """
    return (pixscale / native_pixscale) ** 2


def nanomaggies_to_mag(
    flux: np.ndarray, zeropoint: float = LEGACY_NANOMAGGY_ZEROPOINT
) -> np.ndarray:
    """Convierte flujo en nanomaggies a magnitud AB (NaN si el flujo es <= 0).

    Args:
        flux: flujo en nanomaggies (escalar o arreglo).
        zeropoint: punto cero de la escala (22.5 para nanomaggies).

    Returns:
        Magnitud AB, mismo tamaño que `flux`.
    """
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, zeropoint - 2.5 * np.log10(flux), np.nan)


def surface_brightness_mag(
    value: np.ndarray,
    native_pixscale: float = LEGACY_NATIVE_PIXSCALE,
    zeropoint: float = LEGACY_NANOMAGGY_ZEROPOINT,
) -> np.ndarray:
    """Convierte valores de píxel (nanomaggies/px nativo) a mag/arcsec².

    Args:
        value: valor de píxel ya sin cielo (escalar o arreglo).
        native_pixscale: escala nativa del survey (arcsec/px).
        zeropoint: punto cero de la escala.

    Returns:
        Brillo superficial en mag/arcsec² (NaN donde `value` <= 0).
    """
    return nanomaggies_to_mag(np.asarray(value) / native_pixscale ** 2, zeropoint)


def radius_grid(shape: Tuple[int, int], pixscale: float) -> np.ndarray:
    """Distancia de cada píxel al centro del cutout, en arcsec.

    Args:
        shape: `(H, W)` del plano.
        pixscale: escala del cutout (arcsec/px).

    Returns:
        Arreglo `(H, W)` con el radio en arcsec.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    return np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2) * pixscale


def sky_background(
    plane: np.ndarray,
    radius: np.ndarray,
    exclude_radius: float,
    sigma: float = FITS_SKY_SIGMA_CLIP,
    min_frac: float = FITS_SKY_MIN_FRAC,
) -> Tuple[float, float]:
    """Mediana y desviación estándar del cielo con recorte sigma.

    Args:
        plane: plano `(H, W)` de una banda.
        radius: radio de cada píxel en arcsec (`radius_grid`).
        exclude_radius: radio (arcsec) dentro del cual no se mide el cielo.
        sigma: umbral del recorte iterativo.
        min_frac: fracción mínima de la imagen que debe quedar fuera de
            `exclude_radius`; si queda menos, se usa la imagen completa.

    Returns:
        Tupla `(mediana, desviación)` en las unidades del plano.
    """
    outside = (radius > exclude_radius) & np.isfinite(plane)
    pixels = plane[outside] if outside.mean() >= min_frac else plane[np.isfinite(plane)]
    _, median, std = sigma_clipped_stats(pixels, sigma=sigma)
    return float(median), float(std)


def light_radius(
    plane: np.ndarray, radius: np.ndarray, aperture: float, fraction: float
) -> float:
    """Radio que encierra `fraction` del flujo dentro de `aperture` (arcsec).

    Se ordenan los píxeles por radio y se interpola la curva de crecimiento.

    Args:
        plane: plano `(H, W)` ya sin cielo.
        radius: radio de cada píxel en arcsec.
        aperture: radio de la apertura que define el 100% (arcsec).
        fraction: fracción de luz, entre 0 y 1.

    Returns:
        Radio en arcsec (NaN si el flujo de la apertura no es positivo).
    """
    inside = (radius <= aperture) & np.isfinite(plane)
    order = np.argsort(radius[inside])
    sorted_radius = radius[inside][order]
    growth = np.cumsum(plane[inside][order])
    if growth[-1] <= 0:
        return float("nan")
    return float(np.interp(fraction * growth[-1], growth, sorted_radius))


def concentration_index(
    plane: np.ndarray,
    radius: np.ndarray,
    aperture: float,
    inner: float = CONCENTRATION_INNER_FRAC,
    outer: float = CONCENTRATION_OUTER_FRAC,
) -> float:
    """Índice de concentración `C = 5 log10(r_outer / r_inner)`.

    Args:
        plane: plano `(H, W)` ya sin cielo.
        radius: radio de cada píxel en arcsec.
        aperture: radio de la apertura que define el 100% (arcsec).
        inner: fracción de luz del radio interior (0.2 en la definición usual).
        outer: fracción de luz del radio exterior (0.8).

    Returns:
        Índice de concentración (adimensional).
    """
    r_inner = light_radius(plane, radius, aperture, inner)
    r_outer = light_radius(plane, radius, aperture, outer)
    return float(5 * np.log10(r_outer / r_inner))


def radial_profile(
    plane: np.ndarray, radius: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Valor medio del plano en anillos concéntricos.

    Args:
        plane: plano `(H, W)` ya sin cielo.
        radius: radio de cada píxel en arcsec.
        edges: bordes de los anillos en arcsec (`n + 1` valores).

    Returns:
        Arreglo de `n` valores medios (NaN en anillos sin píxeles).
    """
    bins = np.digitize(radius.ravel(), edges) - 1
    values = plane.ravel()
    valid = (bins >= 0) & (bins < len(edges) - 1) & np.isfinite(values)
    sums = np.bincount(bins[valid], weights=values[valid], minlength=len(edges) - 1)
    counts = np.bincount(bins[valid], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return sums / np.where(counts > 0, counts, np.nan)


def galaxy_photometry(
    cube: np.ndarray,
    pixscale: float,
    petro_th90: float,
    bands: str = LEGACY_BANDS,
    aperture_factor: float = FITS_APERTURE_FACTOR,
    sky_exclude_factor: float = FITS_SKY_EXCLUDE_FACTOR,
    concentration_band: str = FITS_CONCENTRATION_BAND,
) -> Dict[str, float]:
    """Fotometría de apertura de la galaxia central de un cubo `.fits`.

    Args:
        cube: arreglo `(n_bandas, H, W)` de `load_fits_cutout`.
        pixscale: escala del cutout (arcsec/px).
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).
        bands: bandas del cubo, en orden.
        aperture_factor: apertura en múltiplos de `petro_th90`; se recorta al
            radio inscrito del cutout.
        sky_exclude_factor: el cielo se mide fuera de este múltiplo de
            `petro_th90`.
        concentration_band: banda donde se miden r20, r50, r80, r90 y C.

    Returns:
        Diccionario con, por banda, `cielo_<b>`, `ruido_<b>` (nanomaggies/px
        nativo), `flujo_<b>` (nanomaggies) y `mag_<b>`; los colores `g_r` y
        `r_z` (si están las bandas), `apertura_arcsec`, `apertura_recortada`, y
        en la banda de concentración `r20`, `r50`, `r80`, `r90` (arcsec) y
        `concentracion`.
    """
    radius = radius_grid(cube.shape[1:], pixscale)
    max_radius = min(cube.shape[1:]) / 2 * pixscale
    aperture = min(aperture_factor * petro_th90, max_radius)
    factor = native_flux_factor(pixscale)

    result: Dict[str, float] = {
        "apertura_arcsec": aperture,
        "apertura_recortada": aperture < aperture_factor * petro_th90,
    }
    clean = {}
    for band in bands:
        plane = band_image(cube, band, bands).astype(np.float64)
        sky, noise = sky_background(plane, radius, sky_exclude_factor * petro_th90)
        clean[band] = plane - sky
        flux = float(np.nansum(clean[band][radius <= aperture])) * factor
        result.update(
            {
                f"cielo_{band}": sky,
                f"ruido_{band}": noise,
                f"flujo_{band}": flux,
                f"mag_{band}": float(nanomaggies_to_mag(flux)),
            }
        )
    for blue, red in (("g", "r"), ("r", "z")):
        if blue in bands and red in bands:
            result[f"{blue}_{red}"] = result[f"mag_{blue}"] - result[f"mag_{red}"]

    plane = clean[concentration_band]
    for fraction in (0.2, 0.5, 0.8, 0.9):
        result[f"r{int(fraction * 100)}"] = light_radius(plane, radius, aperture, fraction)
    result["concentracion"] = concentration_index(plane, radius, aperture)
    return result


def build_fits_photometry_table(
    galaxies: pd.DataFrame,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
) -> pd.DataFrame:
    """Fotometría de todas las galaxias de un DataFrame con `.fits` en disco.

    Args:
        galaxies: DataFrame con al menos `name` y `PETRO_TH90`; el resto de
            columnas (p. ej. `label`, `objdec`) se conserva.
        layer: capa del survey de la variante.
        bands: bandas de la variante.

    Returns:
        `galaxies` con `pixscale` (leído del WCS) y las columnas de
        `galaxy_photometry`, una fila por galaxia.
    """
    records = []
    for row in galaxies.to_dict("records"):
        pixscale = fits_pixscale(row["name"], layer, bands)
        cube = load_fits_cutout(row["name"], layer, bands)
        row["pixscale"] = pixscale
        row.update(galaxy_photometry(cube, pixscale, row["PETRO_TH90"], bands))
        records.append(row)
    return pd.DataFrame.from_records(records)


def profile_edges(
    petro_th90: float,
    max_factor: float = FITS_PROFILE_MAX_FACTOR,
    n_bins: int = FITS_PROFILE_N_BINS,
) -> np.ndarray:
    """Bordes de anillos para un perfil radial normalizado por PETRO_TH90.

    Los anillos que caen fuera del cutout quedan vacíos (NaN en
    `radial_profile`).

    Args:
        petro_th90: radio PETRO_TH90 de la galaxia (arcsec).
        max_factor: radio máximo del perfil en múltiplos de `petro_th90`.
        n_bins: número de anillos.

    Returns:
        Arreglo de `n_bins + 1` bordes en arcsec.
    """
    return np.linspace(0.0, max_factor * petro_th90, n_bins + 1)
