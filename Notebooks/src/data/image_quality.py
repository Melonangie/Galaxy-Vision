"""Filtro de calidad y de atajos por variante de capa/bandas (Fase 2).

Antes de comparar variantes entrenando un modelo, se descartan las que tienen
problemas objetivos (ver README.md, sección Elegir la variante de imágenes):

- **Calidad:** cobertura incompleta, máscaras vacías o que tocan el borde
  (columnas ya medidas en `cutouts_index.csv` por `src.data.download_cutouts`).
- **Homogeneidad de color y fuga norte/sur:** Legacy Surveys combina dos
  telescopios según la posición en el cielo (`legacy_region`). Si una variante
  pinta distinto cada región (p. ej. `ls-dr10_grz`, morada en el sur), una red
  puede aprender la región en vez de la morfología. `color_features` resume el
  color del cielo y de la galaxia y `region_color_gap` mide cuánto cambia
  entre regiones.

La prueba estadística de "¿esta representación predice la región?" vive en
`src/evaluate.py` (`predictability_test`); este módulo solo construye las
tablas de features. Exploración en `notebook/image-curation.ipynb`, sección 6.

La segunda parte del módulo ("Artefactos en cutouts RGB") reúne heurísticas
explicables sobre JPEGs RGB normalizados (no flujo fotométrico calibrado):
líneas brillantes, saturación y componentes externos. Su punto de entrada es
`analyze_image_quality`, reexportado en `src.data`.
"""

from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from astropy import units as u
from astropy.coordinates import SkyCoord

from src.config import (
    COLOR_GAP_MAX_FACTOR,
    CUTOUT_FORMAT_JPG,
    LEGACY_NORTH_DEC_MIN,
    LEGACY_REGION_NORTH,
    LEGACY_REGION_SOUTH,
    LEGACY_VARIANTS,
)
from src.data.galaxy_mask import mask_cutout, mask_shape_features
from src.data.legacy_cutouts import variant_name

# Columnas de color que produce `color_features`, en orden estable.
COLOR_FEATURES = [
    "cielo_R", "cielo_G", "cielo_B", "cielo_r", "cielo_g",
    "galaxia_R", "galaxia_G", "galaxia_B", "galaxia_r", "galaxia_g",
]
SHAPE_FEATURES = ["mask_frac", "elongacion", "compacidad", "solidez"]


def legacy_region(ra: Iterable[float], dec: Iterable[float]) -> np.ndarray:
    """Región de Legacy Surveys de cada galaxia: `"norte"` (BASS/MzLS) o `"sur"` (DECam).

    Es norte solo si está en el casquete galáctico norte (latitud galáctica
    b > 0) y además `dec >= LEGACY_NORTH_DEC_MIN`.

    Args:
        ra: ascensión recta en grados.
        dec: declinación en grados.

    Returns:
        Arreglo de cadenas `LEGACY_REGION_NORTH` / `LEGACY_REGION_SOUTH`.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    galactic_b = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic.b.deg
    north = (galactic_b > 0) & (dec >= LEGACY_NORTH_DEC_MIN)
    return np.where(north, LEGACY_REGION_NORTH, LEGACY_REGION_SOUTH)


def color_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Color mediano del cielo (fuera de la máscara) y de la galaxia (dentro).

    Además de la mediana por canal se guarda la cromaticidad `r = R/(R+G+B)`
    y `g = G/(R+G+B)`, que mide el tono independientemente del brillo y,
    a diferencia de cocientes como G/R, no se dispara cuando un canal es casi
    0 (variantes `gri`). Ahí se nota un render de color distinto entre
    regiones (p. ej. el morado de `ls-dr10_grz` tiene `g` bajo).

    Args:
        image: arreglo `(H, W, 3)` RGB.
        mask: máscara booleana `(H, W)` de la galaxia central.

    Returns:
        Diccionario con las columnas de `COLOR_FEATURES` (NaN en la parte de
        la galaxia si la máscara está vacía).
    """
    features = {}
    for region, selector in (("cielo", ~mask), ("galaxia", mask)):
        pixels = image[selector].astype(np.float64)
        rgb = np.median(pixels, axis=0) if len(pixels) else np.full(3, np.nan)
        features.update({f"{region}_{c}": float(v) for c, v in zip("RGB", rgb)})
        total = max(float(rgb.sum()), 1.0)
        features[f"{region}_r"] = float(rgb[0] / total)
        features[f"{region}_g"] = float(rgb[1] / total)
    return features


def build_variant_feature_table(
    index: pd.DataFrame,
    variants: Optional[Iterable[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Tabla larga de features por (galaxia, variante) a partir del índice.

    Para cada cutout `.jpg` descargado sin error recalcula la máscara (es
    determinista) y mide color (`color_features`) y forma
    (`mask_shape_features`). Añade la región del survey y conserva las
    columnas de calidad del índice.

    Args:
        index: `cutouts_index.csv` leído con pandas.
        variants: tuplas `(layer, bands)`; `None` usa `LEGACY_VARIANTS`.

    Returns:
        DataFrame con `name`, `label`, `variante`, `layer`, `bands`, `objra`,
        `objdec`, `region`, las columnas de calidad del índice
        (`frac_sin_cobertura`, `mask_toca_borde`), `COLOR_FEATURES` y
        `SHAPE_FEATURES`.
    """
    variants = list(LEGACY_VARIANTS if variants is None else variants)
    wanted = {variant_name(layer, bands) for layer, bands in variants}
    rows = index[
        (index["format"] == CUTOUT_FORMAT_JPG)
        & (index["status"] != "error")
        & index["variante"].isin(wanted)
    ]

    records = []
    for row in rows.to_dict("records"):
        image, mask = mask_cutout(row["name"], row["layer"], row["bands"])
        record = {
            key: row.get(key)
            for key in ("name", "label", "variante", "layer", "bands", "objra", "objdec",
                        "frac_sin_cobertura", "mask_toca_borde")
        }
        record.update(color_features(image, mask))
        record.update(mask_shape_features(mask))
        records.append(record)

    table = pd.DataFrame.from_records(records)
    table["region"] = legacy_region(table["objra"], table["objdec"])
    order = [variant_name(layer, bands) for layer, bands in variants]
    table["variante"] = pd.Categorical(table["variante"], categories=order, ordered=True)
    return table.sort_values(["variante", "name"]).reset_index(drop=True)


def region_color_gap(
    table: pd.DataFrame, max_factor: float = COLOR_GAP_MAX_FACTOR
) -> pd.DataFrame:
    """Diferencia de tono entre regiones norte y sur, por variante.

    Distancia euclidiana entre la cromaticidad mediana `(r, g)` del norte y la
    del sur, para el cielo y para la galaxia. Todas las variantes tienen algo
    de diferencia (son dos telescopios distintos); se marca como heterogénea
    la variante que supera `max_factor` veces la mediana entre variantes.

    Args:
        table: salida de `build_variant_feature_table` (con ambas regiones).
        max_factor: múltiplo de la mediana a partir del cual se marca.

    Returns:
        DataFrame indexado por variante con `gap_cielo`, `gap_galaxia` y
        `color_heterogeneo` (bool).
    """
    medians = table.groupby(["variante", "region"], observed=True)[
        ["cielo_r", "cielo_g", "galaxia_r", "galaxia_g"]
    ].median()
    north = medians.xs(LEGACY_REGION_NORTH, level="region")
    south = medians.xs(LEGACY_REGION_SOUTH, level="region")
    gaps = pd.DataFrame({
        f"gap_{part}": np.hypot(north[f"{part}_r"] - south[f"{part}_r"],
                                north[f"{part}_g"] - south[f"{part}_g"])
        for part in ("cielo", "galaxia")
    })
    limits = max_factor * gaps.median()
    gaps["color_heterogeneo"] = (gaps > limits).any(axis=1)
    return gaps


def quality_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Resumen de calidad por variante (cuántas galaxias tienen cada problema).

    Args:
        table: salida de `build_variant_feature_table`.

    Returns:
        DataFrame indexado por variante con `n`, `con_zona_sin_cobertura`,
        `mascara_vacia` y `mascara_toca_borde`.
    """
    return table.assign(
        con_zona_sin_cobertura=table["frac_sin_cobertura"].fillna(0) > 0,
        mascara_vacia=table["mask_frac"] == 0,
        mascara_toca_borde=table["mask_toca_borde"].astype(bool),
    ).groupby("variante", observed=True).agg(
        n=("name", "size"),
        con_zona_sin_cobertura=("con_zona_sin_cobertura", "sum"),
        mascara_vacia=("mascara_vacia", "sum"),
        mascara_toca_borde=("mascara_toca_borde", "sum"),
    )


# ---------------------------------------------------------------------------
# Artefactos en cutouts RGB (heurísticas sobre JPEG normalizado)
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_QUALITY_CONFIG = {
    "bright_quantile": 0.99,
    "bright_quantile_line": 0.97,
    "chroma_threshold": 0.5,
    "saturation_threshold": 0.98,
    "line_span_threshold": 0.3,
    "line_aspect_threshold": 5.0,
    "line_z_threshold": 8.0,
    "outer_radius": 0.3,
    "outer_component_threshold": 0.01,
}


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Valida y convierte una imagen RGB normalizada a ``float64``."""
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image debe tener forma (alto, ancho, 3).")
    if not np.isfinite(rgb).all():
        raise ValueError("image debe contener solo valores finitos.")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("image debe estar normalizada al intervalo [0, 1].")
    return rgb


def _resolve_config(config: Mapping[str, float] | None) -> dict[str, float]:
    """Combina configuración parcial con los umbrales predeterminados."""
    resolved = DEFAULT_IMAGE_QUALITY_CONFIG.copy()
    if config is not None:
        unknown_keys = set(config) - set(resolved)
        if unknown_keys:
            raise ValueError(f"Claves de configuración desconocidas: {sorted(unknown_keys)}")
        resolved.update(config)
    return resolved


def intensity_map(image: np.ndarray) -> np.ndarray:
    """Devuelve la intensidad máxima por píxel en RGB normalizado."""
    return _validate_image(image).max(axis=2)


def chroma_map(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Devuelve la cromaticidad ``(max(RGB) - min(RGB)) / (max(RGB) + eps)``."""
    rgb = _validate_image(image)
    maximum = rgb.max(axis=2)
    return (maximum - rgb.min(axis=2)) / (maximum + eps)


def robust_zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calcula z-scores robustos centrados en la mediana."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if scale < eps:
        scale = values.std()
    if scale < eps:
        return np.zeros_like(values)
    return (values - median) / scale


def masks_from_image(
    image: np.ndarray, config: Mapping[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye mapas y máscaras de brillo, color y saturación."""
    parameters = _resolve_config(config)
    intensity = intensity_map(image)
    chroma = chroma_map(image)
    bright = intensity >= np.quantile(intensity, parameters["bright_quantile"])
    colorful = chroma >= parameters["chroma_threshold"]
    saturated = intensity >= parameters["saturation_threshold"]
    return intensity, chroma, bright, colorful, bright & colorful, saturated


def line_mask_from_image(
    image: np.ndarray, config: Mapping[str, float] | None = None
) -> np.ndarray:
    """Devuelve la máscara brillante menos restrictiva usada para líneas."""
    parameters = _resolve_config(config)
    intensity = intensity_map(image)
    return intensity >= np.quantile(intensity, parameters["bright_quantile_line"])


def longest_true_run(mask_1d: np.ndarray) -> int:
    """Calcula la longitud de la secuencia consecutiva más larga de ``True``."""
    longest = current = 0
    for value in np.asarray(mask_1d, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def profile_metrics(
    image: np.ndarray, config: Mapping[str, float] | None = None
) -> dict[str, np.ndarray]:
    """Resume perfiles por filas y columnas de estructuras brillantes."""
    parameters = _resolve_config(config)
    intensity, chroma, _, colorful, _, _ = masks_from_image(image, parameters)
    bright = line_mask_from_image(image, parameters)
    bright_color = bright & (chroma >= parameters["chroma_threshold"])
    return {
        "row_mean_intensity": intensity.mean(axis=1),
        "col_mean_intensity": intensity.mean(axis=0),
        "row_max_intensity": intensity.max(axis=1),
        "col_max_intensity": intensity.max(axis=0),
        "row_bright_fraction": bright.mean(axis=1),
        "col_bright_fraction": bright.mean(axis=0),
        "row_color_fraction": colorful.mean(axis=1),
        "col_color_fraction": colorful.mean(axis=0),
        "row_bright_color_fraction": bright_color.mean(axis=1),
        "col_bright_color_fraction": bright_color.mean(axis=0),
        "row_longest_bright_run": np.array(
            [longest_true_run(row) / bright.shape[1] for row in bright]
        ),
        "col_longest_bright_run": np.array(
            [longest_true_run(column) / bright.shape[0] for column in bright.T]
        ),
    }


def distance_map(shape: tuple[int, ...]) -> np.ndarray:
    """Calcula distancia radial al centro, normalizada a la esquina más lejana."""
    height, width = shape[:2]
    yy, xx = np.indices((height, width))
    radius = np.hypot(yy - (height - 1) / 2, xx - (width - 1) / 2)
    return radius / radius.max()


def component_table(mask: np.ndarray, intensity: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Mide los componentes conexos de una máscara booleana."""
    labels, n_components = ndimage.label(mask)
    height, width = mask.shape
    records = []
    for component_id in range(1, n_components + 1):
        ys, xs = np.where(labels == component_id)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        component_height, component_width = y1 - y0, x1 - x0
        short_axis = min(component_width, component_height)
        centroid_y, centroid_x = ys.mean(), xs.mean()
        center_distance = np.hypot(
            (centroid_y - (height - 1) / 2) / height,
            (centroid_x - (width - 1) / 2) / width,
        ) / np.sqrt(0.5**2 + 0.5**2)
        records.append(
            {
                "component": component_id,
                "area": len(ys),
                "area_fraction": len(ys) / mask.size,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "width": component_width,
                "height": component_height,
                "aspect_ratio": max(component_width, component_height) / max(short_axis, 1),
                "span": max(component_width / width, component_height / height),
                "thickness_fraction": short_axis / min(height, width),
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "distance_to_center": center_distance,
                "mean_intensity": float(intensity[labels == component_id].mean()),
                "max_intensity": float(intensity[labels == component_id].max()),
            }
        )
    return pd.DataFrame(records), labels


def extract_quality_metrics(
    image: np.ndarray, config: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Extrae métricas visuales explicables de un cutout RGB normalizado."""
    parameters = _resolve_config(config)
    intensity, _, bright, _, _, saturated = masks_from_image(image, parameters)
    profiles = profile_metrics(image, parameters)
    bright_components, _ = component_table(bright, intensity)
    line_components, _ = component_table(line_mask_from_image(image, parameters), intensity)
    saturated_components, _ = component_table(saturated, intensity)
    radius = distance_map(image.shape)
    outer = radius >= parameters["outer_radius"]

    metrics = {
        "max_row_bright_fraction": float(profiles["row_bright_fraction"].max()),
        "max_col_bright_fraction": float(profiles["col_bright_fraction"].max()),
        "max_row_color_fraction": float(profiles["row_color_fraction"].max()),
        "max_col_color_fraction": float(profiles["col_color_fraction"].max()),
        "max_row_bright_color_fraction": float(profiles["row_bright_color_fraction"].max()),
        "max_col_bright_color_fraction": float(profiles["col_bright_color_fraction"].max()),
        "max_row_bright_z": float(robust_zscore(profiles["row_bright_fraction"]).max()),
        "max_col_bright_z": float(robust_zscore(profiles["col_bright_fraction"]).max()),
        "max_horizontal_run_fraction": float(profiles["row_longest_bright_run"].max()),
        "max_vertical_run_fraction": float(profiles["col_longest_bright_run"].max()),
        "saturated_fraction": float(saturated.mean()),
        "bright_fraction_center": float(bright[radius < 0.25].mean()),
        "bright_fraction_outer": float(bright[outer].mean()),
        "n_bright_components": float(len(bright_components)),
        "largest_bright_component_fraction": 0.0,
        "largest_component_aspect_ratio": 0.0,
        "largest_component_span": 0.0,
        "largest_outer_component_fraction": 0.0,
        "largest_saturated_component_fraction": 0.0,
    }

    if not bright_components.empty:
        metrics["largest_bright_component_fraction"] = float(
            bright_components["area_fraction"].max()
        )
        outer_components = bright_components[
            bright_components["distance_to_center"] >= parameters["outer_radius"]
        ]
        if not outer_components.empty:
            metrics["largest_outer_component_fraction"] = float(
                outer_components["area_fraction"].max()
            )
    if not line_components.empty:
        metrics["largest_component_aspect_ratio"] = float(
            line_components["aspect_ratio"].max()
        )
        metrics["largest_component_span"] = float(line_components["span"].max())
    if not saturated_components.empty:
        metrics["largest_saturated_component_fraction"] = float(
            saturated_components["area_fraction"].max()
        )
    return metrics


def _clipped_ratio(value: float, threshold: float) -> float:
    return float(np.clip(value / max(threshold, 1e-8), 0, 1))


def analyze_image_quality(
    image: np.ndarray, config: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Analiza calidad visual y devuelve métricas, banderas y razones.

    Args:
        image: RGB de forma ``(alto, ancho, 3)`` normalizado a ``[0, 1]``.
        config: Sobrescrituras parciales de ``DEFAULT_IMAGE_QUALITY_CONFIG``.

    Returns:
        Diccionario con banderas de líneas, contaminación brillante y
        saturación, sus scores heurísticos, métricas y razones legibles.
    """
    parameters = _resolve_config(config)
    metrics = extract_quality_metrics(image, parameters)
    horizontal_evidence = max(
        _clipped_ratio(metrics["max_horizontal_run_fraction"], parameters["line_span_threshold"]),
        _clipped_ratio(metrics["max_row_bright_z"], parameters["line_z_threshold"]),
    )
    vertical_evidence = max(
        _clipped_ratio(metrics["max_vertical_run_fraction"], parameters["line_span_threshold"]),
        _clipped_ratio(metrics["max_col_bright_z"], parameters["line_z_threshold"]),
    )
    line_geometry = (
        metrics["largest_component_aspect_ratio"] >= parameters["line_aspect_threshold"]
        and metrics["largest_component_span"] >= parameters["line_span_threshold"]
    )
    line_score = max(horizontal_evidence, vertical_evidence) * (1.0 if line_geometry else 0.5)
    bright_score = max(
        _clipped_ratio(
            metrics["largest_outer_component_fraction"], parameters["outer_component_threshold"]
        ),
        _clipped_ratio(metrics["bright_fraction_outer"], parameters["outer_component_threshold"]),
    )
    saturation_score = max(
        _clipped_ratio(metrics["saturated_fraction"], 0.01),
        _clipped_ratio(metrics["largest_saturated_component_fraction"], 0.005),
    )

    reasons = []
    if metrics["max_horizontal_run_fraction"] >= parameters["line_span_threshold"]:
        reasons.append(
            f"Estructura horizontal brillante continua: "
            f"{metrics['max_horizontal_run_fraction']:.0%} del ancho."
        )
    if metrics["max_vertical_run_fraction"] >= parameters["line_span_threshold"]:
        reasons.append(
            f"Estructura vertical brillante continua: "
            f"{metrics['max_vertical_run_fraction']:.0%} de la altura."
        )
    if line_geometry:
        reasons.append(
            f"Componente alargado: aspect={metrics['largest_component_aspect_ratio']:.1f}, "
            f"span={metrics['largest_component_span']:.0%}."
        )
    if metrics["largest_outer_component_fraction"] >= parameters["outer_component_threshold"]:
        reasons.append(
            f"Componente brillante externo grande: "
            f"{metrics['largest_outer_component_fraction']:.1%} del frame."
        )
    if metrics["saturated_fraction"] >= 0.01:
        reasons.append(f"Fracción saturada RGB elevada: {metrics['saturated_fraction']:.1%} del frame.")
    if metrics["largest_saturated_component_fraction"] >= 0.005:
        reasons.append(
            f"Región saturada RGB grande: "
            f"{metrics['largest_saturated_component_fraction']:.1%} del frame."
        )
    if not reasons:
        reasons.append("No hay evidencia fuerte con los thresholds exploratorios actuales.")

    line_artifact = bool(line_geometry and max(horizontal_evidence, vertical_evidence) >= 1.0)
    bright_contamination = bool(bright_score >= 1.0)
    saturation = bool(saturation_score >= 1.0)
    return {
        "is_suspicious": line_artifact or bright_contamination,
        "line_artifact": line_artifact,
        "line_score": line_score,
        "bright_contamination": bright_contamination,
        "bright_score": bright_score,
        "saturation": saturation,
        "saturation_score": saturation_score,
        "metrics": metrics,
        "reasons": reasons,
    }

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
