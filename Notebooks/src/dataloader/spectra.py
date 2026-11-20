"""Lectura, preprocesamiento y caché reproducible de espectros MaNGA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from astropy.io import fits


AVAILABLE_CHANNELS = ("FLUX_1RE", "FLUX_FoV")
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SpectrumPreprocessConfig:
    """Parámetros que definen completamente una representación espectral."""

    channels: tuple[str, ...] = AVAILABLE_CHANNELS
    rest_wave_min: float = 3600.0
    rest_wave_max: float = 9000.0
    length: int = 4096
    continuum_window: tuple[float, float] = (5400.0, 5500.0)
    normalization: str = "per_channel"
    min_continuum_samples: int = 5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> SpectrumPreprocessConfig:
        values = values or {}
        channels = tuple(values.get("channels", AVAILABLE_CHANNELS))
        window = tuple(values.get("continuum_window", (5400.0, 5500.0)))
        config = cls(
            channels=channels,
            rest_wave_min=float(values.get("rest_wave_min", 3600.0)),
            rest_wave_max=float(values.get("rest_wave_max", 9000.0)),
            length=int(values.get("length", 4096)),
            continuum_window=(float(window[0]), float(window[1])),
            normalization=str(values.get("normalization", "per_channel")),
            min_continuum_samples=int(values.get("min_continuum_samples", 5)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("spectrum.channels debe contener canales únicos.")
        unsupported = set(self.channels) - set(AVAILABLE_CHANNELS)
        if unsupported:
            raise ValueError(f"Canales espectrales no soportados: {sorted(unsupported)}")
        if self.rest_wave_min >= self.rest_wave_max or self.length < 2:
            raise ValueError("La malla espectral debe tener rango creciente y al menos dos muestras.")
        if not (self.rest_wave_min <= self.continuum_window[0] < self.continuum_window[1] <= self.rest_wave_max):
            raise ValueError("continuum_window debe estar contenida en la malla de reposo.")
        if self.normalization not in {"per_channel", "shared"}:
            raise ValueError("normalization debe ser 'per_channel' o 'shared'.")
        if self.min_continuum_samples < 1:
            raise ValueError("min_continuum_samples debe ser positivo.")

    @property
    def wave_grid(self) -> np.ndarray:
        return np.linspace(self.rest_wave_min, self.rest_wave_max, self.length, dtype=np.float32)

    def fingerprint(self) -> str:
        payload = {"format_version": CACHE_FORMAT_VERSION, **asdict(self)}
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def spectrum_file_for_name(name: str, fits_dir: str | Path) -> Path:
    """Resuelve el FITS comprimido o sin comprimir de una galaxia MaNGA."""
    directory = Path(fits_dir)
    for suffix in (".fits", ".fits.gz"):
        candidate = directory / f"{name}-1D{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No se encontró FITS para {name} en {directory}.")


def _records_fingerprint(records: pd.DataFrame) -> str:
    selected = records.loc[:, ["name", "nsa_z"]].sort_values("name")
    payload = selected.to_csv(index=False, float_format="%.12g")
    return sha256(payload.encode()).hexdigest()


def read_fits_spectrum(path: str | Path, channels: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Lee WAVE y los flujos solicitados de la extensión ``Spectra``."""
    requested = tuple(channels)
    required = {"WAVE", *requested}
    with fits.open(path, memmap=False) as hdul:
        if "Spectra" not in hdul:
            raise KeyError(f"{path}: no existe la extensión Spectra.")
        hdu = hdul["Spectra"]
        available = set(hdu.columns.names or [])
        missing = required - available
        if missing:
            raise ValueError(f"{path}: faltan columnas {sorted(missing)}.")
        wave = np.asarray(hdu.data["WAVE"], dtype=np.float64).copy()
        fluxes = {channel: np.asarray(hdu.data[channel], dtype=np.float64).copy() for channel in requested}
    if wave.ndim != 1 or not all(flux.shape == wave.shape for flux in fluxes.values()):
        raise ValueError(f"{path}: WAVE y flujos deben ser vectores 1D con la misma forma.")
    return wave, fluxes


def _continuum_scale(wave: np.ndarray, flux: np.ndarray, config: SpectrumPreprocessConfig) -> float:
    valid = np.isfinite(wave) & np.isfinite(flux)
    continuum = valid & (wave >= config.continuum_window[0]) & (wave <= config.continuum_window[1])
    if int(continuum.sum()) < config.min_continuum_samples:
        raise ValueError("Cobertura insuficiente en la ventana de continuo.")
    scale = float(np.median(flux[continuum]))
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        raise ValueError("Escala de continuo no finita o aproximadamente nula.")
    return scale


def preprocess_spectrum(
    wave_observed: np.ndarray,
    fluxes: Mapping[str, np.ndarray],
    redshift: float,
    config: SpectrumPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte a reposo, normaliza e interpola sin extrapolar sobre la malla común."""
    config.validate()
    if not np.isfinite(redshift) or redshift <= -1:
        raise ValueError(f"Redshift inválido: {redshift}.")
    wave_rest = np.asarray(wave_observed, dtype=np.float64) / (1.0 + float(redshift))
    if wave_rest.ndim != 1:
        raise ValueError("WAVE debe ser un vector 1D.")

    channel_fluxes = {channel: np.asarray(fluxes[channel], dtype=np.float64) for channel in config.channels}
    if not all(flux.shape == wave_rest.shape for flux in channel_fluxes.values()):
        raise ValueError("Cada canal de flujo debe tener la misma forma que WAVE.")

    channel_scales = {channel: _continuum_scale(wave_rest, flux, config) for channel, flux in channel_fluxes.items()}
    if config.normalization == "shared":
        shared_scale = float(np.median(list(channel_scales.values())))
        if not np.isfinite(shared_scale) or np.isclose(shared_scale, 0.0):
            raise ValueError("Escala compartida de continuo inválida.")
        channel_scales = {channel: shared_scale for channel in config.channels}

    grid = config.wave_grid.astype(np.float64)
    interpolated_channels = []
    channel_masks = []
    for channel in config.channels:
        flux = channel_fluxes[channel] / channel_scales[channel]
        valid = np.isfinite(wave_rest) & np.isfinite(flux)
        valid_wave = wave_rest[valid]
        valid_flux = flux[valid]
        order = np.argsort(valid_wave)
        valid_wave = valid_wave[order]
        valid_flux = valid_flux[order]
        valid_wave, unique_indices = np.unique(valid_wave, return_index=True)
        valid_flux = valid_flux[unique_indices]
        if len(valid_wave) < 2:
            raise ValueError(f"{channel}: menos de dos puntos finitos para interpolar.")
        covered = (grid >= valid_wave[0]) & (grid <= valid_wave[-1])
        values = np.zeros_like(grid, dtype=np.float32)
        values[covered] = np.interp(grid[covered], valid_wave, valid_flux).astype(np.float32)
        interpolated_channels.append(values)
        channel_masks.append(covered)

    mask = np.logical_and.reduce(channel_masks)
    spectrum = np.stack(interpolated_channels).astype(np.float32)
    spectrum[:, ~mask] = 0.0
    scales = np.asarray([channel_scales[channel] for channel in config.channels], dtype=np.float32)
    return spectrum, mask.astype(bool), scales


def build_spectrum_cache(
    records: pd.DataFrame,
    fits_dir: str | Path,
    output_dir: str | Path,
    config: SpectrumPreprocessConfig,
    overwrite: bool = False,
) -> Path:
    """Preprocesa FITS y guarda tensores, máscaras e índice auditables por ID."""
    required = {"name", "nsa_z"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"No se pueden crear cachés: faltan columnas {sorted(missing)}.")
    if records["name"].duplicated().any():
        raise ValueError("No se pueden crear cachés para IDs duplicados.")

    config.validate()
    records_hash = _records_fingerprint(records)
    cache_dir = Path(output_dir) / f"spectra_{config.fingerprint()[:8]}_{records_hash[:8]}"
    metadata_path = cache_dir / "metadata.json"
    index_path = cache_dir / "index.csv"
    if index_path.is_file() and not overwrite:
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tensors").mkdir(exist_ok=True)
    (cache_dir / "masks").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    for record in records.loc[:, ["name", "nsa_z"]].itertuples(index=False):
        row: dict[str, Any] = {"name": record.name, "status": "error", "error": ""}
        try:
            source_path = spectrum_file_for_name(record.name, fits_dir)
            wave, fluxes = read_fits_spectrum(source_path, config.channels)
            spectrum, mask, scales = preprocess_spectrum(wave, fluxes, float(record.nsa_z), config)
            tensor_path = Path("tensors") / f"{record.name}.npy"
            mask_path = Path("masks") / f"{record.name}.npy"
            np.save(cache_dir / tensor_path, spectrum)
            np.save(cache_dir / mask_path, mask)
            row.update(
                {
                    "status": "ready",
                    "source_path": str(source_path),
                    "tensor_path": str(tensor_path),
                    "mask_path": str(mask_path),
                    "valid_fraction": float(mask.mean()),
                    **{f"scale_{channel}": float(scale) for channel, scale in zip(config.channels, scales, strict=True)},
                }
            )
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)

    pd.DataFrame(rows).to_csv(index_path, index=False)
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "fingerprint": config.fingerprint(),
        "records_fingerprint": records_hash,
        "config": asdict(config),
        "n_requested": len(records),
        "n_ready": sum(row["status"] == "ready" for row in rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return cache_dir


class SpectrumCache:
    """Acceso perezoso a tensores espectrales preprocesados y versionados."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        index_path = self.cache_dir / "index.csv"
        if not metadata_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Caché espectral incompleta: {self.cache_dir}")
        self.metadata = json.loads(metadata_path.read_text())
        self.config = SpectrumPreprocessConfig.from_mapping(self.metadata["config"])
        index = pd.read_csv(index_path)
        ready = index.loc[index["status"].eq("ready")].copy()
        if ready["name"].duplicated().any():
            raise ValueError("La caché espectral contiene IDs listos duplicados.")
        self._records = ready.set_index("name").to_dict(orient="index")

    def has(self, name: str) -> bool:
        return name in self._records

    def get(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in self._records:
            raise KeyError(f"{name} no está listo en la caché espectral.")
        record = self._records[name]
        spectrum = np.load(self.cache_dir / record["tensor_path"])
        mask = np.load(self.cache_dir / record["mask_path"])
        expected_shape = (len(self.config.channels), self.config.length)
        if spectrum.shape != expected_shape or mask.shape != (self.config.length,):
            raise ValueError(f"{name}: tensor de caché incompatible con su configuración.")
        return torch.as_tensor(spectrum, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.bool)