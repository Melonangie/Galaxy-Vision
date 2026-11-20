"""Máscara de color para aislar la galaxia central de un cutout `.jpg`.

Implementa la limpieza de "contaminación por estrellas" pedida en la junta con
el IA-UNAM (ver README.md, sección Limpieza de imágenes). Los cutouts vienen
centrados en la galaxia objetivo, así que su tono se estima con los píxeles del
centro de la imagen, y el del cielo con el marco perimetral. Como el tono de
fondo y de galaxia cambia de imagen a imagen (distinto estiramiento del visor),
ambos modelos de color se ajustan por imagen, no con umbrales fijos.

Pasos de `galaxy_color_mask`:

1. Suavizado gaussiano leve para reducir el ruido de cielo.
2. Modelo de color (mediana + covarianza RGB) del disco central -> galaxia, y
   del marco perimetral sin su 5% más brillante -> fondo.
3. Un píxel es candidato si su color está más cerca (distancia de Mahalanobis)
   del modelo de la galaxia que del de fondo.
4. Apertura morfológica para cortar puentes finos con estrellas vecinas, y se
   conserva solo la componente conexa que contiene el centro. Las estrellas
   lejos de la galaxia quedan en otras componentes y se eliminan; las que caen
   encima del disco quedan dentro (no se pueden separar por conectividad).
5. Cierre y relleno de huecos para obtener una máscara compacta.
6. Dilatación circular de `margin_px` píxeles para dejar un margen de cielo
   alrededor de la galaxia y que no quede recortada justo en su contorno.

Limitación conocida: una galaxia vecina que se superpone a la central (pares en
interacción, grupos) queda en la misma componente y no se separa.

Las funciones que trabajan por nombre de galaxia reciben `layer` y `bands` para
elegir la variante; la salida enmascarada se guarda con la misma estructura que
la entrada: `CUTOUTS_MASKED_JPG_DIR/<layer>_<bands>/<name>.jpg`.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from src.config import (
    CUTOUT_FORMAT_JPG,
    CUTOUTS_MASKED_JPG_DIR,
    GALAXY_MASK_BG_PERCENTILE,
    GALAXY_MASK_BORDER_PX,
    GALAXY_MASK_CENTER_FRAC,
    GALAXY_MASK_CLOSE_ITER,
    GALAXY_MASK_COV_REG,
    GALAXY_MASK_MARGIN_PX,
    GALAXY_MASK_OPEN_ITER,
    GALAXY_MASK_SEED_HALF,
    GALAXY_MASK_SMOOTH_SIGMA,
    LEGACY_BANDS,
    LEGACY_LAYER,
    MASK_SILHOUETTE_SIZE,
)
from src.data.cutout_io import load_jpg_cutout, load_jpg_file
from src.data.legacy_cutouts import cutout_path

# Conectividad 4 para las operaciones morfológicas: más conservadora que la 8
# al decidir si una estrella "toca" la galaxia.
_STRUCTURE = ndi.generate_binary_structure(2, 1)


def _color_model(pixels: np.ndarray, cov_reg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Ajusta un modelo de color robusto (mediana, covarianza inversa) en RGB."""
    center = np.median(pixels, axis=0)
    cov = np.cov(pixels.T) + np.eye(3) * cov_reg
    return center, np.linalg.inv(cov)


def _mahalanobis_sq(image: np.ndarray, model: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Distancia de Mahalanobis al cuadrado de cada píxel `(H, W, 3)` al modelo."""
    center, inv_cov = model
    diff = image - center
    return np.einsum("...i,ij,...j->...", diff, inv_cov, diff)


def _border_mask(shape: Tuple[int, int], border_px: int) -> np.ndarray:
    """Máscara booleana del marco perimetral de `border_px` píxeles de ancho."""
    mask = np.zeros(shape, dtype=bool)
    mask[:border_px] = mask[-border_px:] = True
    mask[:, :border_px] = mask[:, -border_px:] = True
    return mask


def _central_component(
    candidates: np.ndarray, open_iter: int, close_iter: int, seed_half: int
) -> np.ndarray:
    """Conserva solo la componente conexa de `candidates` que contiene el centro."""
    h, w = candidates.shape
    opened = ndi.binary_opening(candidates, structure=_STRUCTURE, iterations=open_iter)
    labels, _ = ndi.label(opened, structure=_STRUCTURE)

    seed = labels[h // 2 - seed_half:h // 2 + seed_half + 1, w // 2 - seed_half:w // 2 + seed_half + 1]
    seed = seed[seed > 0]
    if seed.size == 0:
        return np.zeros_like(candidates)
    component = labels == np.bincount(seed).argmax()

    # Recupera el borde que la apertura erosionó, sin volver a unir estrellas:
    # solo se añaden píxeles candidatos adyacentes a la componente elegida.
    grown = ndi.binary_dilation(component, structure=_STRUCTURE, iterations=open_iter)
    component |= grown & candidates
    component = ndi.binary_closing(component, structure=_STRUCTURE, iterations=close_iter)
    return ndi.binary_fill_holes(component)


def _add_margin(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Dilata `mask` un disco de radio `margin_px` (borde redondeado, no en rombo)."""
    if margin_px <= 0 or not mask.any():
        return mask
    return ndi.distance_transform_edt(~mask) <= margin_px


def galaxy_color_mask(
    image: np.ndarray,
    center_frac: float = GALAXY_MASK_CENTER_FRAC,
    border_px: int = GALAXY_MASK_BORDER_PX,
    smooth_sigma: float = GALAXY_MASK_SMOOTH_SIGMA,
    bg_percentile: float = GALAXY_MASK_BG_PERCENTILE,
    cov_reg: float = GALAXY_MASK_COV_REG,
    open_iter: int = GALAXY_MASK_OPEN_ITER,
    close_iter: int = GALAXY_MASK_CLOSE_ITER,
    seed_half: int = GALAXY_MASK_SEED_HALF,
    margin_px: int = GALAXY_MASK_MARGIN_PX,
) -> np.ndarray:
    """Calcula la máscara de la galaxia central de un cutout RGB.

    Args:
        image: arreglo `(H, W, 3)` RGB (p. ej. de `load_jpg_cutout`).
        center_frac: radio del disco central de muestreo del tono de la
            galaxia, como fracción del lado menor de la imagen.
        border_px: ancho del marco de donde se estima el color del cielo.
        smooth_sigma: sigma (px) del suavizado gaussiano previo.
        bg_percentile: percentil de brillo por encima del cual los píxeles del
            marco se descartan del modelo de fondo (estrellas en el borde).
        cov_reg: regularización sumada a la diagonal de las covarianzas.
        open_iter: iteraciones de apertura para separar estrellas vecinas.
        close_iter: iteraciones de cierre final.
        seed_half: semilanchura del recuadro central de búsqueda de la galaxia.
        margin_px: margen (px) que se añade alrededor de la máscara final; 0
            la deja pegada al contorno de la galaxia.

    Returns:
        Máscara booleana `(H, W)`; `True` en los píxeles de la galaxia central.
        Queda vacía si ningún píxel del centro se clasifica como galaxia.
    """
    h, w, _ = image.shape
    smoothed = ndi.gaussian_filter(image.astype(np.float64), sigma=(smooth_sigma, smooth_sigma, 0))

    yy, xx = np.mgrid[:h, :w]
    radius = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    galaxy_pixels = smoothed[radius <= center_frac * min(h, w)]

    sky_pixels = smoothed[_border_mask((h, w), border_px)]
    sky_brightness = sky_pixels.mean(axis=1)
    sky_pixels = sky_pixels[sky_brightness < np.percentile(sky_brightness, bg_percentile)]

    dist_galaxy = _mahalanobis_sq(smoothed, _color_model(galaxy_pixels, cov_reg))
    dist_sky = _mahalanobis_sq(smoothed, _color_model(sky_pixels, cov_reg))
    candidates = dist_galaxy < dist_sky

    mask = _central_component(candidates, open_iter, close_iter, seed_half)
    return _add_margin(mask, margin_px)


def apply_mask(image: np.ndarray, mask: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Devuelve una copia de `image` con los píxeles fuera de `mask` a `fill_value`.

    Args:
        image: arreglo `(H, W, 3)`.
        mask: máscara booleana `(H, W)`.
        fill_value: valor para el fondo (0 = negro).

    Returns:
        Arreglo `(H, W, 3)` del mismo dtype que `image`.
    """
    return np.where(mask[..., None], image, np.asarray(fill_value, dtype=image.dtype))


def mask_cutout(
    name: str, layer: str = LEGACY_LAYER, bands: str = LEGACY_BANDS
) -> Tuple[np.ndarray, np.ndarray]:
    """Carga el cutout `.jpg` de una galaxia y calcula su máscara central.

    Args:
        name: identificador de la galaxia (columna `name`).
        layer: capa del survey de la variante.
        bands: bandas de la variante.

    Returns:
        Tupla `(imagen, máscara)`: RGB `uint8` `(H, W, 3)` y booleana `(H, W)`.
    """
    image = load_jpg_cutout(name, layer, bands)
    return image, galaxy_color_mask(image)


def mask_stats(mask: np.ndarray) -> Dict[str, float]:
    """Resumen de una máscara para revisar un lote sin abrir las imágenes.

    Args:
        mask: máscara booleana `(H, W)` de `galaxy_color_mask`.

    Returns:
        `mask_frac` (fracción de la imagen cubierta; 0 = máscara vacía, la
        galaxia no se detectó) y `mask_toca_borde` (la máscara llega al borde:
        la galaxia no cabe en el encuadre o se unió a algo del fondo).
    """
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return {"mask_frac": float(mask.mean()), "mask_toca_borde": bool(edge.any())}


def mask_shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Descriptores de forma del contorno de la máscara (sin mirar la galaxia).

    Sirven para el "atajo de la máscara": si estos números por sí solos ya
    predicen temprana/tardía, un modelo entrenado con imágenes enmascaradas
    puede aprender el contorno recortado en vez de la estructura interna.

    Args:
        mask: máscara booleana `(H, W)` de `galaxy_color_mask`.

    Returns:
        `mask_frac` (área relativa), `elongacion` (1 - eje menor / eje mayor,
        por momentos de segundo orden; 0 = circular), `compacidad`
        (4·pi·área / perímetro², ~1 = disco, baja con contornos dentados) y
        `solidez` (área / área de la envolvente convexa, baja con entrantes,
        p. ej. brazos recortados). Todo en 0 si la máscara está vacía.
    """
    area = float(mask.sum())
    if area == 0:
        return {"mask_frac": 0.0, "elongacion": 0.0, "compacidad": 0.0, "solidez": 0.0}

    yy, xx = np.nonzero(mask)
    eigenvalues = np.linalg.eigvalsh(np.cov(np.stack([xx, yy])) + np.eye(2) * 1e-6)
    minor, major = np.sqrt(eigenvalues)

    # Perímetro: aristas entre píxeles dentro/fuera de la máscara. Esa longitud
    # "en escalera" sobreestima un contorno suave en 4/pi en promedio sobre
    # orientaciones, de ahí la corrección.
    padded = np.pad(mask, 1).astype(np.int8)
    edges = np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum()
    perimeter = float(edges) * np.pi / 4

    hull_area = ConvexHull(np.column_stack([xx, yy])).volume if area >= 3 else area
    return {
        "mask_frac": area / mask.size,
        "elongacion": float(1 - minor / major),
        "compacidad": float(4 * np.pi * area / max(perimeter, 1.0) ** 2),
        "solidez": float(min(area / max(hull_area, 1.0), 1.0)),
    }


def mask_silhouette(mask: np.ndarray, size: int = MASK_SILHOUETTE_SIZE) -> np.ndarray:
    """Silueta de la máscara reducida a `size` x `size` y aplanada.

    Args:
        mask: máscara booleana `(H, W)`.
        size: lado de la silueta reducida.

    Returns:
        Vector `(size * size,)` de `float32` en [0, 1] (fracción de cada celda
        cubierta por la máscara).
    """
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32).reshape(-1) / 255.0


def mask_jpg_file(source: Path, destination: Path) -> Dict[str, float]:
    """Enmascara un `.jpg` cualquiera y guarda el resultado (fondo negro).

    Args:
        source: cutout `.jpg` de entrada.
        destination: ruta de salida (se crea el directorio padre si falta).

    Returns:
        Estadísticas de la máscara (ver `mask_stats`).
    """
    image = load_jpg_file(source)
    mask = galaxy_color_mask(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(apply_mask(image, mask)).save(destination, quality=95)
    return mask_stats(mask)


def save_masked_cutout(
    name: str, layer: str = LEGACY_LAYER, bands: str = LEGACY_BANDS
) -> Dict[str, float]:
    """Guarda el cutout con el fondo enmascarado en `CUTOUTS_MASKED_JPG_DIR`.

    Args:
        name: identificador de la galaxia (columna `name`).
        layer: capa del survey de la variante.
        bands: bandas de la variante.

    Returns:
        Estadísticas de la máscara (ver `mask_stats`); `mask_frac == 0` indica
        una máscara vacía, útil para detectar fallos al revisar el lote.
    """
    return mask_jpg_file(
        cutout_path(name, CUTOUT_FORMAT_JPG, layer, bands),
        cutout_path(name, CUTOUT_FORMAT_JPG, layer, bands, root=CUTOUTS_MASKED_JPG_DIR),
    )
