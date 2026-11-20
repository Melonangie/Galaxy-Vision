"""Dataset y DataLoaders configurables para el manifest MaNGA multimodal."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataloader.multimodal_transforms import build_image_transform
from src.dataloader.spectra import SpectrumCache
from src.dataloader.starlight import PopulationStore, load_properties_table, validate_property_features


LABEL_TO_TARGET = {"temprana": 0.0, "tardia": 1.0}
SUPPORTED_MODALITIES = frozenset({"image", "spectrum", "metadata", "properties", "populations"})
PROHIBITED_METADATA_FEATURES = frozenset({"label", "Ttype", "Type", "name", "objra", "objdec"})
SPLIT_NAMES = ("train", "validation", "test")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} debe ser un mapping.")
    return value


def _parse_modalities(value: object) -> frozenset[str]:
    if not isinstance(value, Collection) or isinstance(value, str):
        raise TypeError("data.modalities debe ser una secuencia de modalidades.")
    modalities = frozenset(str(item) for item in value)
    if not modalities:
        raise ValueError("Se debe solicitar al menos una modalidad.")
    unsupported = modalities - SUPPORTED_MODALITIES
    if unsupported:
        raise ValueError(f"Modalidades no soportadas: {sorted(unsupported)}")
    return modalities


@dataclass(frozen=True)
class MetadataPreprocessor:
    """Estado de imputación mediana y estandarización ajustado en train."""

    features: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Sequence[str]) -> MetadataPreprocessor:
        feature_names = tuple(features)
        if not feature_names:
            raise ValueError("metadata.features no puede estar vacío.")
        forbidden = set(feature_names) & PROHIBITED_METADATA_FEATURES
        if forbidden:
            raise ValueError(f"Features prohibidas por leakage: {sorted(forbidden)}")
        missing = set(feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Features ausentes del manifest: {sorted(missing)}")

        values = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any(axis=0).all():
            invalid_features = [feature_names[index] for index, has_value in enumerate(valid.any(axis=0)) if not has_value]
            raise ValueError(f"Features sin valores numéricos en train: {invalid_features}")

        medians = np.nanmedian(np.where(valid, values, np.nan), axis=0)
        imputed = np.where(valid, values, medians)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        return cls(feature_names, medians, means, scales)

    def transform(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Features ausentes del frame: {sorted(missing)}")
        values = frame.loc[:, self.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        mask = np.isfinite(values)
        imputed = np.where(mask, values, self.medians)
        standardized = (imputed - self.means) / self.scales
        return (
            torch.as_tensor(standardized, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool),
        )


class GalaxyDataset(Dataset[dict[str, Any]]):
    """Carga perezosa de modalidades activadas en una partición del manifest."""

    def __init__(
        self,
        frame: pd.DataFrame,
        modalities: Sequence[str],
        project_root: str | Path | None = None,
        image_fov_column: str | None = None,
        image_transform: Any | None = None,
        metadata_preprocessor: MetadataPreprocessor | None = None,
        spectrum_cache: SpectrumCache | None = None,
        properties: tuple[torch.Tensor, torch.Tensor] | None = None,
        populations: torch.Tensor | None = None,
        random_image_fov_columns: Sequence[str] | None = None,
    ) -> None:
        self.modalities = _parse_modalities(modalities)
        self.project_root = Path(project_root) if project_root is not None else _project_root()
        self.frame = frame.reset_index(drop=True).copy()
        self._validate_base_frame()

        self.image_path_column: str | None = None
        self.random_image_path_columns: tuple[str, ...] = ()
        self.image_transform = image_transform
        if "image" in self.modalities:
            if not image_fov_column:
                raise ValueError("image_fov_column es obligatorio cuando se solicita image.")
            self.image_path_column = f"{image_fov_column}_path"
            self._validate_image_paths(image_fov_column)
            if random_image_fov_columns:
                fov_columns = tuple(dict.fromkeys(random_image_fov_columns))
                for fov_column in fov_columns:
                    self._validate_image_paths(fov_column)
                # Cada acceso elige uno de estos FOV al azar; el FOV fijo se conserva para evaluación.
                self.random_image_path_columns = tuple(f"{fov_column}_path" for fov_column in fov_columns)

        self.metadata_values: torch.Tensor | None = None
        self.metadata_mask: torch.Tensor | None = None
        if "metadata" in self.modalities:
            if metadata_preprocessor is None:
                raise ValueError("metadata_preprocessor es obligatorio cuando se solicita metadata.")
            self.metadata_values, self.metadata_mask = metadata_preprocessor.transform(self.frame)

        self.properties_values: torch.Tensor | None = None
        self.properties_mask: torch.Tensor | None = None
        if "properties" in self.modalities:
            if properties is None or len(properties[0]) != len(self.frame):
                raise ValueError("properties debe estar alineado con el frame cuando se solicita properties.")
            self.properties_values, self.properties_mask = properties

        self.population_values = populations
        if "populations" in self.modalities and (populations is None or len(populations) != len(self.frame)):
            raise ValueError("populations debe estar alineado con el frame cuando se solicita populations.")

        self.spectrum_cache = spectrum_cache
        if "spectrum" in self.modalities:
            if spectrum_cache is None:
                raise ValueError("spectrum_cache es obligatorio cuando se solicita spectrum.")
            missing_spectra = [name for name in self.frame["name"] if not spectrum_cache.has(name)]
            if missing_spectra:
                raise ValueError(f"{len(missing_spectra)} IDs no están listos en la caché espectral.")

    def _validate_base_frame(self) -> None:
        required = {"name", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"El dataset no contiene: {sorted(missing)}")
        if self.frame["name"].duplicated().any():
            raise ValueError("El dataset contiene IDs duplicados.")
        unknown_labels = set(self.frame["label"]) - set(LABEL_TO_TARGET)
        if unknown_labels:
            raise ValueError(f"Etiquetas no soportadas: {sorted(unknown_labels)}")

    def _validate_image_paths(self, fov_column: str) -> None:
        path_column = f"{fov_column}_path"
        required = {fov_column, path_column}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"El manifest no contiene: {sorted(missing)}")
        unavailable = ~self.frame[fov_column].eq(True)
        if unavailable.any():
            raise ValueError(f"{unavailable.sum()} filas no tienen disponible {fov_column}.")
        empty_paths = self.frame[path_column].isna() | self.frame[path_column].eq("")
        if empty_paths.any():
            raise ValueError(f"{empty_paths.sum()} filas no tienen ruta para {fov_column}.")
        missing_paths = [
            path_value
            for path_value in self.frame[path_column]
            if not _resolve_path(path_value, self.project_root).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(f"No existe el JPEG registrado: {missing_paths[0]}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample: dict[str, Any] = {
            "id": row["name"],
            "label": torch.tensor(LABEL_TO_TARGET[row["label"]], dtype=torch.float32),
        }
        if "image" in self.modalities:
            path_column = (
                random.choice(self.random_image_path_columns)
                if self.random_image_path_columns
                else self.image_path_column
            )
            image_path = _resolve_path(row[path_column], self.project_root)
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
                sample["image"] = self.image_transform(image) if self.image_transform else image
        if "metadata" in self.modalities:
            sample["metadata"] = self.metadata_values[index]
            sample["metadata_mask"] = self.metadata_mask[index]
        if "properties" in self.modalities:
            sample["properties"] = self.properties_values[index]
            sample["properties_mask"] = self.properties_mask[index]
        if "populations" in self.modalities:
            sample["populations"] = self.population_values[index]
        if "spectrum" in self.modalities:
            spectrum, spectrum_mask = self.spectrum_cache.get(row["name"])
            sample["spectrum"] = spectrum
            sample["spectrum_mask"] = spectrum_mask
        return sample


def _load_joined_data(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path, Mapping[str, Any], Mapping[str, Any]]:
    data_config = _as_mapping(config.get("data"), "data")
    training_config = _as_mapping(config.get("training"), "training")
    root = Path(data_config.get("project_root", _project_root()))
    manifest_path = _resolve_path(data_config.get("manifest", "data/galaxy_manifest.csv"), root)
    splits_path = _resolve_path(
        data_config.get("splits", "data/processed/splits_seed42_obj-label.csv"), root
    )
    manifest = pd.read_csv(manifest_path)
    splits = pd.read_csv(splits_path)
    required_splits = {"name", "label", "split"}
    missing_splits = required_splits - set(splits.columns)
    if missing_splits:
        raise ValueError(f"El archivo de splits no contiene: {sorted(missing_splits)}")
    if splits["name"].duplicated().any():
        raise ValueError("El archivo de splits contiene IDs duplicados.")

    joined = splits.merge(manifest, on="name", how="left", validate="one_to_one", suffixes=("_split", ""))
    if joined["label"].isna().any():
        raise ValueError("Hay IDs del split ausentes del manifest.")
    if not joined["label_split"].eq(joined["label"]).all():
        raise ValueError("Las etiquetas del split y del manifest no coinciden.")
    return joined.drop(columns=["label_split"]), root, data_config, training_config


def build_dataloaders(config: Mapping[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construye DataLoaders train/validation/test para modalidades configuradas."""
    joined, root, data_config, training_config = _load_joined_data(config)
    modalities = _parse_modalities(data_config.get("modalities"))

    image_config = _as_mapping(data_config.get("image"), "data.image")
    metadata_config = _as_mapping(data_config.get("metadata"), "data.metadata")
    image_fov_column = str(image_config.get("fov_column", "fov_0"))
    image_size = int(image_config.get("size", 256))
    normalization = str(image_config.get("normalization", "none"))
    train_fov_columns = image_config.get("train_fov_columns") or None
    if train_fov_columns is not None:
        if isinstance(train_fov_columns, str) or not isinstance(train_fov_columns, Sequence):
            raise ValueError("data.image.train_fov_columns debe ser una lista de columnas FOV.")
        train_fov_columns = [str(column) for column in train_fov_columns]
    spectrum_config = _as_mapping(data_config.get("spectrum"), "data.spectrum")
    spectrum_cache = None
    if "spectrum" in modalities:
        cache_dir = spectrum_config.get("cache_dir")
        if not cache_dir:
            raise ValueError("data.spectrum.cache_dir es obligatorio cuando se solicita spectrum.")
        spectrum_cache = SpectrumCache(_resolve_path(cache_dir, root))

    frames = {split: joined.loc[joined["split"].eq(split)].copy() for split in SPLIT_NAMES}
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Cada split debe contener al menos una galaxia.")

    metadata_preprocessor = None
    if "metadata" in modalities:
        metadata_preprocessor = MetadataPreprocessor.fit(
            frames["train"], metadata_config.get("features", ())
        )

    properties_by_split: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = dict.fromkeys(SPLIT_NAMES)
    if "properties" in modalities:
        properties_config = _as_mapping(data_config.get("properties"), "data.properties")
        features = tuple(properties_config.get("features", ()))
        validate_property_features(features, bool(properties_config.get("allow_unverified_mass", False)))
        properties_dir = _resolve_path(
            properties_config.get("dir", "data/processed/starlight/propiedades"), root
        )
        table = load_properties_table(properties_dir, joined["name"]).set_index("name")
        properties_preprocessor = MetadataPreprocessor.fit(
            table.loc[frames["train"]["name"]].reset_index(), features
        )
        for split, frame in frames.items():
            properties_by_split[split] = properties_preprocessor.transform(table.loc[frame["name"]].reset_index())

    populations_by_split: dict[str, torch.Tensor | None] = dict.fromkeys(SPLIT_NAMES)
    if "populations" in modalities:
        populations_config = _as_mapping(data_config.get("populations"), "data.populations")
        populations_dir = _resolve_path(
            populations_config.get("dir", "data/processed/starlight/poblaciones_estelares"), root
        )
        for split, frame in frames.items():
            populations_by_split[split] = PopulationStore(
                populations_dir,
                frame["name"],
                channels=populations_config.get("channels", ("luz_pct", "masa_actual_pct")),
                renormalize=bool(populations_config.get("renormalize", True)),
                flatten=bool(populations_config.get("flatten", False)),
            ).values

    datasets = {
        "train": GalaxyDataset(
            frames["train"],
            modalities,
            root,
            image_fov_column,
            build_image_transform(image_size, train=True, normalization=normalization),
            metadata_preprocessor,
            spectrum_cache,
            properties_by_split["train"],
            populations_by_split["train"],
            train_fov_columns if "image" in modalities else None,
        ),
        "validation": GalaxyDataset(
            frames["validation"],
            modalities,
            root,
            image_fov_column,
            build_image_transform(image_size, train=False, normalization=normalization),
            metadata_preprocessor,
            spectrum_cache,
            properties_by_split["validation"],
            populations_by_split["validation"],
        ),
        "test": GalaxyDataset(
            frames["test"],
            modalities,
            root,
            image_fov_column,
            build_image_transform(image_size, train=False, normalization=normalization),
            metadata_preprocessor,
            spectrum_cache,
            properties_by_split["test"],
            populations_by_split["test"],
        ),
    }

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size debe ser positivo y num_workers no negativo.")
    pin_memory = bool(training_config.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(training_config.get("persistent_workers", num_workers > 0))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    return (
        DataLoader(datasets["train"], shuffle=True, **loader_options),
        DataLoader(datasets["validation"], shuffle=False, **loader_options),
        DataLoader(datasets["test"], shuffle=False, **loader_options),
    )