"""Descarga de imágenes (cutouts) de galaxias desde legacysurvey.org.

Implementa en `src/` el script de referencia descrito en el correo de
J. A. Vázquez Mata (`archives/email_1.pdf`), con dos diferencias deliberadas:

1. El zoom no es fijo: se calcula por galaxia a partir de `PETRO_TH90`, que es
   justamente lo que sugiere el correo (usar el radio aproximado de la galaxia
   para variar la "scale").
2. El `.fits` se pide con todas las bandas en una sola petición (el servicio
   devuelve un cubo de un plano por banda), en vez de una descarga por banda.

Cada combinación capa/bandas ("variante") vive en su propia subcarpeta:
`data/legacy_cutouts/<formato>/<layer>_<bands>/<name>.<ext>` (ver
`cutout_path`). Ver README.md, secciones Imágenes: legacysurvey.org y
Variantes de capa y bandas.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config import (
    CUTOUT_FORMAT,
    CUTOUT_FORMAT_FITS,
    CUTOUT_FORMAT_JPG,
    CUTOUT_FOV_FACTOR,
    CUTOUT_MAX_RETRIES,
    CUTOUT_PAUSE_S,
    CUTOUT_PIXSCALE_MAX,
    CUTOUT_PIXSCALE_MIN,
    CUTOUT_RETRY_BACKOFF_S,
    CUTOUT_SIZE_PX,
    CUTOUT_TIMEOUT_S,
    CUTOUTS_DIR,
    CUTOUTS_FITS_DIR,
    CUTOUTS_JPG_DIR,
    LEGACY_BANDS,
    LEGACY_CUTOUT_BASE_URL,
    LEGACY_LAYER,
    LEGACY_NATIVE_PIXSCALE,
    PROJECT_ROOT,
)

# Endpoint del visor, extensión y carpeta destino de cada formato soportado.
_FORMAT_SPEC = {
    CUTOUT_FORMAT_JPG: ("jpeg-cutout", ".jpg", CUTOUTS_JPG_DIR),
    CUTOUT_FORMAT_FITS: ("fits-cutout", ".fits", CUTOUTS_FITS_DIR),
}


def compute_pixscale(
    petro_th90: Optional[float],
    size_px: int = CUTOUT_SIZE_PX,
    fov_factor: float = CUTOUT_FOV_FACTOR,
    pixscale_min: float = CUTOUT_PIXSCALE_MIN,
    pixscale_max: float = CUTOUT_PIXSCALE_MAX,
) -> float:
    """Calcula el zoom (arcsec/px) para encuadrar una galaxia por su tamaño.

    El campo de visión pedido es `fov_factor * PETRO_TH90` arcsec de lado, de
    modo que galaxias grandes y pequeñas lleguen a la red con un tamaño
    aparente comparable. Si falta `PETRO_TH90` se cae a la escala nativa del
    survey, que es el comportamiento del script original.

    Args:
        petro_th90: radio de Petrosian al 90% de luz (arcsec); `None`/NaN
            activa el fallback a `LEGACY_NATIVE_PIXSCALE`.
        size_px: lado del cutout en píxeles.
        fov_factor: campo de visión en múltiplos de `petro_th90`.
        pixscale_min: cota inferior del zoom (arcsec/px).
        pixscale_max: cota superior del zoom (arcsec/px).

    Returns:
        Escala de píxel en arcsec/px, acotada a `[pixscale_min, pixscale_max]`.
    """
    if petro_th90 is None or not np.isfinite(petro_th90) or petro_th90 <= 0:
        return LEGACY_NATIVE_PIXSCALE
    return float(np.clip(fov_factor * petro_th90 / size_px, pixscale_min, pixscale_max))


def compute_fov_pixscale(
    petro_th90: float,
    fov_factor: float,
    size_px: int = CUTOUT_SIZE_PX,
    clamp_pixscale: bool = True,
    pixscale_min: float = CUTOUT_PIXSCALE_MIN,
    pixscale_max: float = CUTOUT_PIXSCALE_MAX,
) -> float:
    """Calcula la escala para un FOV relativo a ``PETRO_TH90``.

    A diferencia de :func:`compute_pixscale`, esta variante requiere un tamaño
    válido y permite desactivar los límites. Es útil para experimentos donde el
    único parámetro que debe variar es ``fov_factor``.
    """
    if not np.isfinite(petro_th90) or petro_th90 <= 0:
        raise ValueError("PETRO_TH90 debe ser un valor positivo y finito.")
    if not np.isfinite(fov_factor) or fov_factor <= 0:
        raise ValueError("fov_factor debe ser un valor positivo y finito.")
    if size_px <= 0:
        raise ValueError("size_px debe ser positivo.")

    pixscale = float(fov_factor * petro_th90 / size_px)
    if not clamp_pixscale:
        return pixscale
    if pixscale_min <= 0 or pixscale_min > pixscale_max:
        raise ValueError("Los límites de pixscale no son válidos.")
    return float(np.clip(pixscale, pixscale_min, pixscale_max))


def build_cutout_url(
    ra: float,
    dec: float,
    pixscale: float,
    image_format: str = CUTOUT_FORMAT,
    size_px: int = CUTOUT_SIZE_PX,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
) -> str:
    """Construye la URL del cutout service de legacysurvey.org.

    Args:
        ra: ascensión recta en grados (`objra`).
        dec: declinación en grados (`objdec`).
        pixscale: escala de píxel en arcsec/px (ver `compute_pixscale`).
        image_format: `"jpg"` (RGB compuesto) o `"fits"` (cubo por banda).
        size_px: lado del cutout en píxeles.
        layer: capa del survey (p. ej. `ls-dr10`).
        bands: bandas fotométricas a componer/descargar.

    Returns:
        URL completa lista para `requests.get`.

    Raises:
        ValueError: si `image_format` no es un formato soportado.
    """
    if image_format not in _FORMAT_SPEC:
        raise ValueError(
            f"Formato no soportado: {image_format!r}. "
            f"Use uno de {sorted(_FORMAT_SPEC)}."
        )
    endpoint = _FORMAT_SPEC[image_format][0]
    return (
        f"{LEGACY_CUTOUT_BASE_URL}/{endpoint}"
        f"?ra={ra}&dec={dec}&size={size_px}&layer={layer}"
        f"&pixscale={pixscale:.4f}&bands={bands}"
    )


def variant_name(layer: str = LEGACY_LAYER, bands: str = LEGACY_BANDS) -> str:
    """Nombre de carpeta de una variante de capa/bandas: `<layer>_<bands>`.

    Args:
        layer: capa del survey (p. ej. `ls-dr10`).
        bands: bandas pedidas al visor (p. ej. `grz`).

    Returns:
        Cadena como `"ls-dr10_grz"`.
    """
    return f"{layer}_{bands}"


def parse_variant_name(variant: str) -> Tuple[str, str]:
    """Inversa de `variant_name`: `"ls-dr10_grz"` -> `("ls-dr10", "grz")`.

    Args:
        variant: nombre de carpeta `<layer>_<bands>`.

    Returns:
        Tupla `(layer, bands)`.

    Raises:
        ValueError: si el nombre no tiene la forma `<layer>_<bands>`.
    """
    layer, sep, bands = variant.rpartition("_")
    if not sep or not layer or not bands:
        raise ValueError(f"Variante mal formada: {variant!r}; se espera '<layer>_<bands>'.")
    return layer, bands


def variant_dir(
    image_format: str = CUTOUT_FORMAT,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    root: Optional[Path] = None,
) -> Path:
    """Carpeta de una variante para un formato.

    Args:
        image_format: `"jpg"` o `"fits"`.
        layer: capa del survey.
        bands: bandas pedidas al visor.
        root: carpeta raíz alternativa (p. ej. `CUTOUTS_MASKED_JPG_DIR`);
            `None` usa la del formato.

    Returns:
        Ruta `<root>/<layer>_<bands>/`.
    """
    base = _FORMAT_SPEC[image_format][2] if root is None else root
    return base / variant_name(layer, bands)


def cutout_path(
    name: str,
    image_format: str = CUTOUT_FORMAT,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    root: Optional[Path] = None,
) -> Path:
    """Ruta en disco del cutout de una galaxia.

    Args:
        name: identificador de la galaxia (columna `name`).
        image_format: `"jpg"` o `"fits"`.
        layer: capa del survey (por defecto, la variante activa de config).
        bands: bandas pedidas al visor.
        root: carpeta raíz alternativa (p. ej. `CUTOUTS_MASKED_JPG_DIR` para
            el cutout enmascarado); `None` usa la del formato.

    Returns:
        Ruta `data/legacy_cutouts/<formato>/<layer>_<bands>/<name>.<ext>`.
    """
    extension = _FORMAT_SPEC[image_format][1]
    return variant_dir(image_format, layer, bands, root) / f"{name}{extension}"


def list_cutout_names(
    image_format: str = CUTOUT_FORMAT,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    root: Optional[Path] = None,
) -> List[str]:
    """Galaxias con cutout en disco para una variante, ordenadas por nombre.

    Args:
        image_format: `"jpg"` o `"fits"`.
        layer: capa del survey.
        bands: bandas pedidas al visor.
        root: carpeta raíz alternativa (ver `cutout_path`).

    Returns:
        Lista de identificadores (`name`), vacía si la variante no existe.
    """
    extension = _FORMAT_SPEC[image_format][1]
    directory = variant_dir(image_format, layer, bands, root)
    return sorted(p.stem for p in directory.glob(f"*{extension}"))


def build_fov_jobs(
    df: pd.DataFrame,
    fov_factors: list[float],
    output_dir: Path,
    size_px: int = CUTOUT_SIZE_PX,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    clamp_pixscale: bool = True,
    pixscale_min: float = CUTOUT_PIXSCALE_MIN,
    pixscale_max: float = CUTOUT_PIXSCALE_MAX,
) -> list[dict]:
    """Construye un job JPEG por combinación de galaxia y factor FOV.

    La ruta resultante incorpora tanto el FOV como el ``pixscale`` calculado.
    Esto evita reutilizar una imagen de caché con una escala distinta cuando se
    cambian los límites de ``pixscale``.
    """
    required_columns = {"name", "objra", "objdec", "PETRO_TH90"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
    if not fov_factors:
        raise ValueError("fov_factors no puede estar vacío.")

    output_dir = Path(output_dir)
    jobs = []
    for row in df.to_dict("records"):
        petro_th90 = float(row["PETRO_TH90"])
        for fov_factor in fov_factors:
            pixscale = compute_fov_pixscale(
                petro_th90,
                fov_factor=float(fov_factor),
                size_px=size_px,
                clamp_pixscale=clamp_pixscale,
                pixscale_min=pixscale_min,
                pixscale_max=pixscale_max,
            )
            fov_tag = str(fov_factor).replace(".", "p")
            pixscale_tag = f"{pixscale:.5f}".replace(".", "p")
            local_path = output_dir / str(row["name"]) / (
                f"fov_{fov_tag}_pixscale_{pixscale_tag}.jpg"
            )
            requested_fov_arcsec = float(fov_factor) * petro_th90
            jobs.append(
                {
                    "name": row["name"],
                    "label": row.get("label"),
                    "Ttype": row.get("Ttype"),
                    "objra": float(row["objra"]),
                    "objdec": float(row["objdec"]),
                    "PETRO_TH90": petro_th90,
                    "layer": layer,
                    "bands": bands,
                    "fov_factor": float(fov_factor),
                    "fov_requested_arcsec": round(requested_fov_arcsec, 4),
                    "fov_effective_arcsec": round(pixscale * size_px, 4),
                    "pixscale": round(pixscale, 6),
                    "pixscale_clamped": not np.isclose(
                        requested_fov_arcsec / size_px, pixscale
                    ),
                    "size_px": size_px,
                    "url": build_cutout_url(
                        float(row["objra"]),
                        float(row["objdec"]),
                        pixscale,
                        image_format=CUTOUT_FORMAT_JPG,
                        size_px=size_px,
                        layer=layer,
                        bands=bands,
                    ),
                    "local_path": str(local_path),
                }
            )
    return jobs


def download_cutout(
    url: str,
    destination: Path,
    max_retries: int = CUTOUT_MAX_RETRIES,
    timeout_s: int = CUTOUT_TIMEOUT_S,
    backoff_s: float = CUTOUT_RETRY_BACKOFF_S,
) -> int:
    """Descarga un cutout y lo escribe en disco, con reintentos.

    Args:
        url: URL construida por `build_cutout_url`.
        destination: ruta destino (se crea el directorio padre si falta).
        max_retries: número máximo de intentos antes de propagar el error.
        timeout_s: timeout por intento, en segundos.
        backoff_s: espera base entre reintentos (crece linealmente).

    Returns:
        Bytes escritos en disco.

    Raises:
        requests.RequestException: si fallan todos los intentos.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[requests.RequestException] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout_s)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return len(response.content)
        except requests.RequestException as error:  # red, 5xx o timeout
            last_error = error
            if attempt < max_retries:
                time.sleep(backoff_s * attempt)

    raise last_error


def download_cutouts(
    df: pd.DataFrame,
    image_format: str = CUTOUT_FORMAT,
    size_px: int = CUTOUT_SIZE_PX,
    fov_factor: float = CUTOUT_FOV_FACTOR,
    overwrite: bool = False,
    pause_s: float = CUTOUT_PAUSE_S,
    verbose: bool = True,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    tag: Optional[str] = None,
) -> pd.DataFrame:
    """Descarga los cutouts de un DataFrame de galaxias y devuelve el índice.

    La descarga es reanudable: los archivos ya presentes se marcan como
    `"cached"` y no se vuelven a pedir (salvo `overwrite=True`).

    Args:
        df: DataFrame con columnas `name`, `objra`, `objdec` y `PETRO_TH90`
            (si trae `label`, se copia al índice).
        image_format: `"jpg"` o `"fits"`.
        size_px: lado del cutout en píxeles.
        fov_factor: campo de visión en múltiplos de `PETRO_TH90`.
        overwrite: vuelve a descargar aunque el archivo ya exista.
        pause_s: pausa entre peticiones, por cortesía con el servicio público.
        verbose: imprime una línea por galaxia fallida.
        layer: capa del survey (variante a descargar).
        bands: bandas pedidas al visor.

    Returns:
        DataFrame índice con una fila por galaxia: `name`, `label`, `objra`,
        `objdec`, `PETRO_TH90`, `pixscale`, `fov_arcsec`, `format`,
        `variante`, `layer`, `bands`, `path`, `bytes`, `status`
        (`"downloaded"`, `"cached"` o `"error"`) y `error`.
    """
    records = []

    for row in df.to_dict("records"):
        petro_th90 = row.get("PETRO_TH90", np.nan)
        pixscale = compute_pixscale(petro_th90, size_px=size_px, fov_factor=fov_factor)
        destination = cutout_path(row["name"], image_format, layer, bands)

        if tag:
            # add _tag suffix to the second folder in path
            destination = destination.parent.parent / f"{destination.parent.name}_{tag}" / destination.name
            # create dir
            destination.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "name": row["name"],
            "label": row.get("label"),
            "objra": row["objra"],
            "objdec": row["objdec"],
            "PETRO_TH90": petro_th90,
            "pixscale": round(pixscale, 4),
            "fov_arcsec": round(pixscale * size_px, 2),
            "format": image_format,
            "variante": variant_name(layer, bands),
            "layer": layer,
            "bands": bands,
            "path": destination.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": None,
            "status": None,
            "error": None,
        }

        if destination.exists() and not overwrite:
            record.update(bytes=destination.stat().st_size, status="cached")
            records.append(record)
            continue

        url = build_cutout_url(
            row["objra"], row["objdec"], pixscale, image_format,
            size_px=size_px, layer=layer, bands=bands,
        )
        try:
            record.update(bytes=download_cutout(url, destination), status="downloaded")
        except requests.RequestException as error:
            record.update(status="error", error=str(error))
            if verbose:
                print(f"  ! {row['name']}: {error}")
        if pause_s:
            time.sleep(pause_s)
        records.append(record)

    return pd.DataFrame.from_records(records)


def download_fov_jobs(
    jobs: list[dict],
    overwrite: bool = False,
    pause_s: float = CUTOUT_PAUSE_S,
    max_retries: int = CUTOUT_MAX_RETRIES,
    timeout_s: int = CUTOUT_TIMEOUT_S,
    backoff_s: float = CUTOUT_RETRY_BACKOFF_S,
    verbose: bool = True,
) -> pd.DataFrame:
    """Descarga una grilla de jobs FOV y devuelve un índice auditable.

    Los jobs se generan con :func:`build_fov_jobs`. Cada fila conserva los
    parámetros de encuadre, URL y ruta local, junto con el estado de descarga.
    """
    records = []
    for job in jobs:
        record = dict(job)
        destination = Path(job["local_path"])
        started_at = time.monotonic()

        if destination.exists() and not overwrite:
            record.update(
                status="cached",
                bytes=destination.stat().st_size,
                error=None,
                download_time_s=0.0,
            )
        else:
            try:
                bytes_written = download_cutout(
                    job["url"],
                    destination,
                    max_retries=max_retries,
                    timeout_s=timeout_s,
                    backoff_s=backoff_s,
                )
                record.update(
                    status="downloaded",
                    bytes=bytes_written,
                    error=None,
                    download_time_s=round(time.monotonic() - started_at, 3),
                )
            except requests.RequestException as error:
                record.update(
                    status="error",
                    bytes=None,
                    error=str(error),
                    download_time_s=round(time.monotonic() - started_at, 3),
                )
                if verbose:
                    print(f"  ! {job['name']} (FOV={job['fov_factor']}): {error}")

        records.append(record)
        if pause_s:
            time.sleep(pause_s)

    return pd.DataFrame.from_records(records)


def save_fov_index(index: pd.DataFrame, index_csv: Path) -> None:
    """Guarda el índice de una descarga FOV, creando el directorio si falta."""
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_csv, index=False)


# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
