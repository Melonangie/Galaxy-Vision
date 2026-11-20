"""Carga de poblaciones estelares y propiedades derivadas de STARLIGHT."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

POPULATION_CHANNELS = ("luz_pct", "masa_actual_pct")
POPULATION_BASE_FILE = "base_poblaciones.npz"
PROPERTIES_INDEX_FILE = "indice.csv"
UNVERIFIED_MASS_PROPERTIES = frozenset({"mini_tot", "mcor_tot"})
NON_FEATURE_PROPERTIES = frozenset({"name", "source_file", "flux_units", "mass_units_verified"})
# Describen el procedimiento de ajuste, no la galaxia (guía 11.2).
FIT_PROCEDURE_PROPERTIES = frozenset({"chi2_nl_eff", "adev_pct", "fobs_norm"})
FIT_PROCEDURE_PREFIXES = ("sn_", "n_")


def load_properties_table(properties_dir: Path, names: Sequence[str]) -> pd.DataFrame:
    """Devuelve las propiedades alineadas con ``names``; falla si falta alguna galaxia."""
    index_path = properties_dir / PROPERTIES_INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"No existe el índice de propiedades: {index_path}")
    table = pd.read_csv(index_path)
    if "name" not in table.columns:
        raise ValueError(f"{index_path} no contiene la columna 'name'.")
    if table["name"].duplicated().any():
        raise ValueError("El índice de propiedades contiene IDs duplicados.")
    aligned = pd.DataFrame({"name": list(names)}).merge(table, on="name", how="left", validate="one_to_one", indicator=True)
    missing = aligned.loc[aligned["_merge"].ne("both"), "name"]
    if not missing.empty:
        raise ValueError(f"{len(missing)} galaxias no tienen propiedades STARLIGHT (p. ej. {missing.iloc[0]}).")
    return aligned.drop(columns="_merge")


def validate_property_features(features: Sequence[str], allow_unverified_mass: bool) -> None:
    forbidden = set(features) & NON_FEATURE_PROPERTIES
    if forbidden:
        raise ValueError(f"Propiedades no numéricas o identificadoras: {sorted(forbidden)}")
    procedure = sorted(
        feature
        for feature in features
        if feature in FIT_PROCEDURE_PROPERTIES or feature.startswith(FIT_PROCEDURE_PREFIXES)
    )
    if procedure:
        raise ValueError(f"Propiedades de calidad/normalización del ajuste excluidas de las features: {procedure}")
    unverified = set(features) & UNVERIFIED_MASS_PROPERTIES
    if unverified and not allow_unverified_mass:
        raise ValueError(
            f"{sorted(unverified)} no tienen unidades verificadas; use allow_unverified_mass=true bajo su responsabilidad."
        )


class PopulationStore:
    """Matrices de poblaciones [C, 45] validadas contra la base común y renormalizadas."""

    def __init__(
        self,
        populations_dir: Path,
        names: Sequence[str],
        channels: Sequence[str] = POPULATION_CHANNELS,
        renormalize: bool = True,
        flatten: bool = False,
    ) -> None:
        channels = tuple(channels)
        unknown = set(channels) - set(POPULATION_CHANNELS)
        if not channels or unknown:
            raise ValueError(f"Canales de poblaciones inválidos: {sorted(unknown) or 'vacío'}; use {POPULATION_CHANNELS}.")
        base_path = populations_dir / POPULATION_BASE_FILE
        if not base_path.is_file():
            raise FileNotFoundError(f"No existe la base de poblaciones: {base_path}")
        with np.load(base_path) as base:
            base_ages, base_metallicities = base["edades_yr"], base["metalicidades_z"]

        rows = []
        for name in names:
            path = populations_dir / f"{name}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"No existe el archivo de poblaciones: {path}")
            with np.load(path) as data:
                stored_channels = tuple(str(c) for c in data["canales"])
                matrix = np.asarray(data["matriz"], dtype=np.float64)
                if stored_channels != POPULATION_CHANNELS or matrix.shape != (len(POPULATION_CHANNELS), base_ages.size):
                    raise ValueError(f"{name}: formato de poblaciones inesperado {stored_channels} {matrix.shape}.")
                if not (np.array_equal(data["edades_yr"], base_ages) and np.array_equal(data["metalicidades_z"], base_metallicities)):
                    raise ValueError(f"{name}: la cuadrícula edad/Z difiere de la base común.")
            if not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(f"{name}: poblaciones no finitas o negativas.")
            rows.append(matrix[[POPULATION_CHANNELS.index(c) for c in channels]])

        values = np.stack(rows) / 100.0
        sums = values.sum(axis=2, keepdims=True)
        if (sums <= 0).any():
            raise ValueError("Hay galaxias con suma de población nula; no se pueden renormalizar.")
        if renormalize:
            values = values / sums
        if flatten:
            values = values.reshape(len(rows), -1)
        self.channels = channels
        self.values = torch.as_tensor(values, dtype=torch.float32)
