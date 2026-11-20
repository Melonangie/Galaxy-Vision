"""Transformaciones RGB globales y reproducibles para cutouts astronómicos."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULT_TRANSFORM_PARAMETERS: dict[str, dict[str, float]] = {
    "gamma": {"gamma": 0.85},
    "grayscale": {},
    "percentile": {"low_percentile": 1.0, "high_percentile": 99.5},
}


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    """Valida una imagen RGB y la devuelve como arreglo ``uint8``."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Se esperaba una imagen RGB de forma (alto, ancho, 3), se recibió {rgb.shape}.")
    if not np.isfinite(rgb).all():
        raise ValueError("La imagen contiene valores no finitos.")
    return np.clip(rgb, 0, 255).astype(np.uint8, copy=False)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Calcula luminancia lineal Rec. 709 para un RGB normalizado."""
    return np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))


def apply_gamma(image: np.ndarray, *, gamma: float = 0.85) -> np.ndarray:
    """Aplica corrección gamma global sobre valores RGB normalizados a [0, 1]."""
    if gamma <= 0:
        raise ValueError("gamma debe ser mayor que cero.")
    rgb = _validate_rgb(image).astype(np.float32) / 255.0
    return np.rint(np.power(rgb, gamma) * 255).astype(np.uint8)


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte a luminancia y la replica en los tres canales RGB."""
    rgb = _validate_rgb(image).astype(np.float32) / 255.0
    gray = np.rint(_luminance(rgb) * 255).astype(np.uint8)
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def _percentile_bounds(
    luminance: np.ndarray, low_percentile: float, high_percentile: float
) -> tuple[float, float]:
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Los percentiles deben cumplir 0 <= bajo < alto <= 100.")
    return tuple(float(value) for value in np.percentile(luminance, (low_percentile, high_percentile)))


def apply_percentile_normalization(
    image: np.ndarray, *, low_percentile: float = 1.0, high_percentile: float = 99.5
) -> np.ndarray:
    """Normaliza luminancia entre percentiles conservando las proporciones RGB.

    Si el rango percentilar es nulo, devuelve la imagen sin cambios para no
    inventar contraste en un recorte uniforme.
    """
    rgb = _validate_rgb(image).astype(np.float32) / 255.0
    luminance = _luminance(rgb)
    low, high = _percentile_bounds(luminance, low_percentile, high_percentile)
    if np.isclose(low, high):
        return _validate_rgb(image).copy()
    normalized_luminance = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    scale = normalized_luminance / np.maximum(luminance, np.finfo(np.float32).eps)
    normalized = np.clip(rgb * scale[..., np.newaxis], 0.0, 1.0)
    return np.rint(normalized * 255).astype(np.uint8)


def percentile_saturation_statistics(
    image: np.ndarray, *, low_percentile: float = 1.0, high_percentile: float = 99.5
) -> dict[str, float | int | bool]:
    """Resume los píxeles saturados por la normalización percentilar."""
    rgb = _validate_rgb(image).astype(np.float32) / 255.0
    luminance = _luminance(rgb)
    low, high = _percentile_bounds(luminance, low_percentile, high_percentile)
    pixel_count = int(luminance.size)
    if np.isclose(low, high):
        return {
            "low_luminance_threshold": low,
            "high_luminance_threshold": high,
            "lower_saturated_pixels": 0,
            "upper_saturated_pixels": 0,
            "lower_saturated_fraction": 0.0,
            "upper_saturated_fraction": 0.0,
            "degenerate_range": True,
            "pixel_count": pixel_count,
        }
    lower_mask = luminance <= low
    upper_mask = luminance >= high
    return {
        "low_luminance_threshold": low,
        "high_luminance_threshold": high,
        "lower_saturated_pixels": int(lower_mask.sum()),
        "upper_saturated_pixels": int(upper_mask.sum()),
        "lower_saturated_fraction": float(lower_mask.mean()),
        "upper_saturated_fraction": float(upper_mask.mean()),
        "degenerate_range": False,
        "pixel_count": pixel_count,
    }


TRANSFORMS = {
    "gamma": apply_gamma,
    "grayscale": apply_grayscale,
    "percentile": apply_percentile_normalization,
}


def apply_transform(
    method: str, image: np.ndarray, parameters: Mapping[str, object] | None = None
) -> np.ndarray:
    """Aplica una transformación registrada con parámetros configurables."""
    if method not in TRANSFORMS:
        raise ValueError(f"Método no válido: {method}. Métodos: {sorted(TRANSFORMS)}.")
    resolved = dict(DEFAULT_TRANSFORM_PARAMETERS[method])
    if parameters is not None:
        unknown = set(parameters) - set(resolved)
        if unknown:
            raise ValueError(f"Parámetros desconocidos para {method}: {sorted(unknown)}.")
        resolved.update(parameters)
    result = TRANSFORMS[method](image, **resolved)
    return _validate_rgb(result)
