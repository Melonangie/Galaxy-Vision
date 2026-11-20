"""Descarga robusta y reanudable de cutouts JPEG de Legacy Survey.

Este módulo concentra el mecanismo validado en los experimentos de FOV: jobs
explícitos, concurrencia acotada, límite global de peticiones, reintentos y
validación de los JPEG escritos. A diferencia de ``legacy_cutouts``, no decide
la muestra ni el destino: el llamador construye ambos de manera auditable.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image

from src.config import (
    CUTOUT_MAX_RETRIES,
    CUTOUT_PIXSCALE_MAX,
    CUTOUT_PIXSCALE_MIN,
    CUTOUT_RETRY_BACKOFF_S,
    CUTOUT_SIZE_PX,
    CUTOUT_TIMEOUT_S,
    LEGACY_BANDS,
    LEGACY_CUTOUT_BASE_URL,
    LEGACY_LAYER,
)


class RateLimiter:
    """Espacia las peticiones compartidas entre los hilos de una descarga."""

    def __init__(self, min_interval_s: float) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s no puede ser negativo.")
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self) -> None:
        """Espera lo necesario antes de iniciar la siguiente petición."""
        with self._lock:
            wait_s = self._last_request_at + self.min_interval_s - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_at = time.monotonic()


def compute_fov_pixscale(
    petro_th90: float,
    fov_factor: float,
    size_px: int = CUTOUT_SIZE_PX,
    pixscale_min: float = CUTOUT_PIXSCALE_MIN,
    pixscale_max: float = CUTOUT_PIXSCALE_MAX,
    ignore_limits: bool = False,
) -> float:
    """Calcula el pixscale limitado para un FOV relativo a ``PETRO_TH90``."""
    if not np.isfinite(petro_th90) or petro_th90 <= 0:
        raise ValueError("PETRO_TH90 debe ser positivo y finito.")
    if not np.isfinite(fov_factor) or fov_factor <= 0:
        raise ValueError("fov_factor debe ser positivo y finito.")
    if size_px <= 0:
        raise ValueError("size_px debe ser positivo.")
    if pixscale_min <= 0 or pixscale_min > pixscale_max:
        raise ValueError("Los limites de pixscale no son validos.")
    if ignore_limits:
        return float(fov_factor * petro_th90 / size_px)
    return float(np.clip(fov_factor * petro_th90 / size_px, pixscale_min, pixscale_max))


def build_cutout_url(
    ra: float,
    dec: float,
    pixscale: float,
    size_px: int = CUTOUT_SIZE_PX,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
) -> str:
    """Construye la URL del endpoint JPEG de Legacy Survey."""
    return (
        f"{LEGACY_CUTOUT_BASE_URL}/jpeg-cutout"
        f"?ra={ra}&dec={dec}&size={size_px}&layer={layer}"
        f"&pixscale={pixscale:.6f}&bands={bands}"
    )


def build_fov_jobs(
    sample: pd.DataFrame,
    output_dir: Path,
    fov_factor: float,
    size_px: int = CUTOUT_SIZE_PX,
    layer: str = LEGACY_LAYER,
    bands: str = LEGACY_BANDS,
    pixscale_min: float = CUTOUT_PIXSCALE_MIN,
    pixscale_max: float = CUTOUT_PIXSCALE_MAX,
    ignore_limits: bool = False,
    fov_estimation: str = None,
) -> list[dict]:
    """Construye un job JPEG por fila de la muestra para un único FOV."""
    required_columns = {"name", "objra", "objdec", "PETRO_TH90"}
    missing_columns = sorted(required_columns - set(sample.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
    if fov_estimation is not None:
        assert fov_estimation in sample.columns, f"fov_estimation column '{fov_estimation}' not found in sample"

    output_dir = Path(output_dir)
    jobs = []
    for row in sample.to_dict("records"):
        petro_th90 = float(row["PETRO_TH90"])
        if fov_estimation is not None:
            fov_factor_job = float(row[fov_estimation])
        else:
            fov_factor_job = float(fov_factor)

        pixscale = compute_fov_pixscale(
            petro_th90,
            fov_factor_job,
            size_px=size_px,
            pixscale_min=pixscale_min,
            pixscale_max=pixscale_max,
            ignore_limits=ignore_limits,
        )
        requested_pixscale = fov_factor_job * petro_th90 / size_px
        
        jobs.append(
            {
                "name": row["name"],
                "label": row.get("label"),
                "Ttype": row.get("Ttype"),
                "objra": float(row["objra"]),
                "objdec": float(row["objdec"]),
                "PETRO_TH90": petro_th90,
                "fov_factor": fov_factor_job,
                "fov_requested_arcsec": round(fov_factor_job * petro_th90, 4),
                "fov_effective_arcsec": round(pixscale * size_px, 4),
                "pixscale": round(pixscale, 6),
                "pixscale_clamped": not np.isclose(requested_pixscale, pixscale),
                "size_px": size_px,
                "layer": layer,
                "bands": bands,
                "url": build_cutout_url(
                    float(row["objra"]),
                    float(row["objdec"]),
                    pixscale,
                    size_px=size_px,
                    layer=layer,
                    bands=bands,
                ),
                "local_path": str(output_dir / f"{row['name']}.jpg"),
            }
        )
    return jobs


def _download_one(
    job: dict,
    rate_limiter: RateLimiter,
    overwrite: bool,
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    backoff_429_s: float,
) -> dict:
    """Descarga o recupera de cache un job sin propagar errores al lote."""
    record = dict(job)
    destination = Path(record["local_path"])
    if destination.exists() and not overwrite:
        record.update(
            status="cached",
            http_status=None,
            bytes=destination.stat().st_size,
            error=None,
            download_time_s=0.0,
        )
        return record

    destination.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    last_error: Optional[Exception] = None
    http_status: Optional[int] = None
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait()
        try:
            response = requests.get(record["url"], timeout=timeout_s)
            http_status = response.status_code
            if http_status == 429:
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else backoff_429_s * attempt
                raise requests.HTTPError(
                    f"429 Too Many Requests (retry en {wait_s:.1f}s)",
                    response=response,
                )
            response.raise_for_status()
            if not response.content:
                raise ValueError("Respuesta vacia.")
            destination.write_bytes(response.content)
            with Image.open(destination) as image:
                image.verify()
            record.update(
                status="downloaded",
                http_status=http_status,
                bytes=len(response.content),
                error=None,
                download_time_s=round(time.monotonic() - started_at, 3),
            )
            return record
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                retry_after = getattr(getattr(error, "response", None), "headers", {}).get(
                    "Retry-After"
                )
                wait_s = float(retry_after) if retry_after else backoff_s * attempt
                time.sleep(wait_s)

    record.update(
        status="error",
        http_status=http_status,
        bytes=None,
        error=str(last_error),
        download_time_s=round(time.monotonic() - started_at, 3),
    )
    return record


def download_batch(
    jobs: Iterable[dict],
    max_concurrent: int = 3,
    min_request_interval_s: float = 1.0,
    overwrite: bool = False,
    timeout_s: int = CUTOUT_TIMEOUT_S,
    max_retries: int = CUTOUT_MAX_RETRIES,
    backoff_s: float = CUTOUT_RETRY_BACKOFF_S,
    backoff_429_s: float = 10.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Descarga jobs con concurrencia acotada, ritmo limitado y reintentos."""
    if max_concurrent <= 0:
        raise ValueError("max_concurrent debe ser positivo.")
    if max_retries <= 0:
        raise ValueError("max_retries debe ser positivo.")

    jobs = list(jobs)
    rate_limiter = RateLimiter(min_request_interval_s)
    records = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(
                _download_one,
                job,
                rate_limiter,
                overwrite,
                timeout_s,
                max_retries,
                backoff_s,
                backoff_429_s,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if verbose and record["status"] == "error":
                print(f"  ! {record['name']}: {record['error']}")

    index = pd.DataFrame.from_records(records)
    if verbose:
        counts = index["status"].value_counts() if not index.empty else pd.Series(dtype=int)
        print(
            f"Lote completo: {len(index)} jobs -> "
            f"{counts.get('downloaded', 0)} descargadas, "
            f"{counts.get('cached', 0)} cacheadas, "
            f"{counts.get('error', 0)} con error"
        )
    return index