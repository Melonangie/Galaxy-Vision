"""Funciones de descarga y visualización del playground de cutouts de Legacy Survey.

Extraídas de `legacy_cutouts_playground.ipynb` para poder reutilizarlas desde otros
notebooks/experimentos sin copiar/pegar celdas. Sigue siendo código de playground (no
pipeline de producción): la configuración (tamaños, capas, límites de concurrencia) se
recibe siempre como argumento, nunca como variable global de módulo, para que cada
notebook decida sus propios valores.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image


def compute_pixscale(petro_th90, fov_factor, size_px=256, pixscale_min=0.1, pixscale_max=0.6):
    """Calcula pixscale (arcsec/px) a partir de PETRO_TH90 y el fov_factor.

    FOV = fov_factor * PETRO_TH90 (arcsec); pixscale = FOV / size_px, acotado a
    [pixscale_min, pixscale_max] para no pedir zooms absurdos.
    """
    fov_arcsec = fov_factor * petro_th90
    pixscale = fov_arcsec / size_px
    return float(np.clip(pixscale, pixscale_min, pixscale_max))


def build_cutout_url(ra, dec, pixscale, size_px=256, layer="ls-dr11", bands=None):
    """Construye la URL del jpeg-cutout service de legacysurvey.org.

    `bands=None` omite el parámetro `bands`: el Viewer usa su render de color
    por defecto.
    """
    url = (
        f"https://www.legacysurvey.org/viewer/jpeg-cutout"
        f"?ra={ra}&dec={dec}&size={size_px}&layer={layer}&pixscale={pixscale:.4f}"
    )
    if bands:
        url += f"&bands={bands}"
    return url


def playground_path(playground_dir, experiment, subfolder, name):
    """Ruta destino dentro de `playground_dir`, organizada por experimento/subcarpeta."""
    return Path(playground_dir) / experiment / subfolder / f"{name}.jpg"


def prepare_jobs(
    sample_df,
    playground_dir,
    experiment,
    subfolder,
    layer,
    bands,
    fov_factor,
    size_px=256,
    pixscale_min=0.1,
    pixscale_max=0.6,
):
    """Arma la lista de "jobs" (metadatos + URL + ruta local) para un lote de descarga.

    Una fila por galaxia en `sample_df` (columnas name/label/Ttype/objra/objdec/
    PETRO_TH90, y opcionalmente size_group).
    """
    jobs = []
    for row in sample_df.to_dict("records"):
        pixscale = compute_pixscale(
            row["PETRO_TH90"], fov_factor=fov_factor, size_px=size_px,
            pixscale_min=pixscale_min, pixscale_max=pixscale_max,
        )
        url = build_cutout_url(
            row["objra"], row["objdec"], pixscale, size_px=size_px, layer=layer, bands=bands
        )
        local_path = playground_path(playground_dir, experiment, subfolder, row["name"])
        jobs.append(
            {
                "name": row["name"],
                "label": row["label"],
                "Ttype": row["Ttype"],
                "objra": row["objra"],
                "objdec": row["objdec"],
                "PETRO_TH90": row["PETRO_TH90"],
                "size_group": row.get("size_group"),
                "experiment": experiment,
                "layer": layer,
                "bands_or_color_mode": bands if bands else "default",
                "fov_factor": fov_factor,
                "fov_arcsec": round(pixscale * size_px, 2),
                "pixscale": round(pixscale, 4),
                "size_px": size_px,
                "url": url,
                "local_path": str(local_path),
            }
        )
    return jobs


class RateLimiter:
    """Fuerza un espacio mínimo entre requests, compartido por todos los hilos.

    Un límite de concurrencia (p. ej. `ThreadPoolExecutor(max_workers=N)`) acota
    cuántas peticiones están en vuelo a la vez, pero no cuántas por segundo. Este
    limitador espacia cada request al menos `min_interval_s`, para evitar 429 (Too
    Many Requests) del servicio público.
    """

    def __init__(self, min_interval_s):
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            wait_s = self._last_request_at + self.min_interval_s - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request_at = time.time()


def download_one_cutout(
    job,
    rate_limiter=None,
    overwrite=False,
    timeout_s=60,
    max_retries=3,
    backoff_s=2.0,
    backoff_429_s=10.0,
):
    """Descarga (o recupera de caché) un único cutout y valida la imagen.

    Devuelve una fila de índice completa, nunca lanza excepción: los errores
    quedan en `status="error"` para no interrumpir el resto del lote.
    """
    dest = Path(job["local_path"])
    record = dict(job)

    if dest.exists() and not overwrite:
        record.update(status="cached", http_status=None, bytes=dest.stat().st_size, error=None, download_time_s=0.0)
        return record

    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last_error = None
    http_status = None

    for attempt in range(1, max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            response = requests.get(job["url"], timeout=timeout_s)
            http_status = response.status_code
            if http_status == 429:
                # Respetar Retry-After si el servidor lo manda; si no, backoff largo.
                retry_after = response.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else backoff_429_s * attempt
                raise requests.HTTPError(f"429 Too Many Requests (retry en {wait_s:.1f}s)", response=response)
            response.raise_for_status()
            content = response.content
            if not content:
                raise ValueError("respuesta vacía")
            dest.write_bytes(content)
            with Image.open(dest) as im:
                im.verify()  # valida que sea una imagen legible
            record.update(
                status="downloaded",
                http_status=http_status,
                bytes=len(content),
                error=None,
                download_time_s=round(time.time() - t0, 3),
            )
            return record
        except Exception as error:  # red, HTTP, timeout o imagen corrupta
            last_error = error
            if attempt < max_retries:
                if http_status == 429:
                    retry_after = getattr(getattr(error, "response", None), "headers", {}).get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else backoff_429_s * attempt)
                else:
                    time.sleep(backoff_s * attempt)

    record.update(
        status="error",
        http_status=http_status,
        bytes=None,
        error=str(last_error),
        download_time_s=round(time.time() - t0, 3),
    )
    return record


def download_batch(
    jobs,
    max_concurrent=3,
    min_request_interval_s=1.0,
    overwrite=False,
    timeout_s=60,
    max_retries=3,
    backoff_s=2.0,
    backoff_429_s=10.0,
    verbose=True,
):
    """Descarga un lote de jobs con concurrencia acotada y ritmo limitado.

    `max_concurrent` limita cuántas peticiones están en vuelo a la vez;
    `min_request_interval_s` (vía `RateLimiter`) espacia el ritmo entre ellas,
    compartido por todos los hilos, para no disparar 429.
    """
    rate_limiter = RateLimiter(min_request_interval_s)
    records = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(
                download_one_cutout,
                job,
                rate_limiter=rate_limiter,
                overwrite=overwrite,
                timeout_s=timeout_s,
                max_retries=max_retries,
                backoff_s=backoff_s,
                backoff_429_s=backoff_429_s,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if verbose and record["status"] == "error":
                print(f"  ! {record['name']} ({record['experiment']}): {record['error']}")
    index = pd.DataFrame.from_records(records)
    if verbose:
        print(
            f"Lote completo: {len(index)} jobs -> "
            f"{(index['status'] == 'downloaded').sum()} descargadas, "
            f"{(index['status'] == 'cached').sum()} cacheadas, "
            f"{(index['status'] == 'error').sum()} con error"
        )
    return index


def plot_comparison_grid(index_df, col_field, col_values, col_titles=None,
                          galaxies_per_grid=6, suptitle=""):
    """Cuadrícula galaxia (fila) x configuración (columna), paginada.

    `index_df` debe traer una fila por (galaxia, valor de `col_field`), con
    columnas name/label/Ttype/PETRO_TH90/local_path/status.
    """
    import matplotlib.pyplot as plt  # import perezoso: solo hace falta al graficar

    col_titles = col_titles or [str(v) for v in col_values]
    names = index_df["name"].drop_duplicates().tolist()
    pages = [names[i : i + galaxies_per_grid] for i in range(0, len(names), galaxies_per_grid)]

    for page_i, page_names in enumerate(pages, start=1):
        n_rows, n_cols = len(page_names), len(col_values)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows), squeeze=False)

        for r, name in enumerate(page_names):
            galaxy_rows = index_df[index_df["name"] == name]
            meta = galaxy_rows.iloc[0]
            for c, col_value in enumerate(col_values):
                ax = axes[r][c]
                match = galaxy_rows[galaxy_rows[col_field] == col_value]
                if len(match) and match.iloc[0]["status"] != "error" and Path(match.iloc[0]["local_path"]).exists():
                    ax.imshow(Image.open(match.iloc[0]["local_path"]))
                else:
                    ax.text(0.5, 0.5, "sin imagen", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(str(col_titles[c]), fontsize=11)
                if c == 0:
                    label_txt = (
                        f"{meta['name']}\n{meta['label']} (T={meta['Ttype']})\n"
                        f"PETRO_TH90={meta['PETRO_TH90']:.1f}\""
                    )
                    ax.set_ylabel(label_txt, fontsize=8, rotation=0, ha="right", va="center")

        fig.suptitle(f"{suptitle} (página {page_i}/{len(pages)})", fontsize=13)
        plt.tight_layout()
        plt.show()


def make_review_table(index_df, extra_group_field, extra_group_values):
    """DataFrame vacío de revisión manual: `review_score` y `notes` en blanco."""
    rows = []
    for name, group in index_df.groupby("name"):
        meta = group.iloc[0]
        for value in extra_group_values:
            rows.append(
                {
                    "name": name,
                    "label": meta["label"],
                    "size_group": meta["size_group"],
                    extra_group_field: value,
                    "review_score": pd.NA,  # -2 .. +2 (demasiado cerca .. demasiado lejos)
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def describe_image_quality(local_path):
    """Métricas descriptivas simples de un JPEG: solo apoyo visual, no una métrica de calidad final."""
    with Image.open(local_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return {
        "mean_brightness": float(arr.mean()),
        "std_brightness": float(arr.std()),
        "pct_dark_px": float((arr < 10).mean() * 100),
        "pct_saturated_px": float((arr > 245).mean() * 100),
    }
