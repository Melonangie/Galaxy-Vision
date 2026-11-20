"""Carga del subconjunto limpio (elíptica/espiral) y split train/val/test.

Migra a `src/` la lógica de `notebook/exploration.ipynb` (regla de oro, ver
README.md: lo validado en el notebook no debe quedarse solo ahí).
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_LABEL_THRESHOLD,
    ELLIPTICAL_LABEL,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SPIRAL_LABEL,
    TEST_FRAC,
    TRAIN_FRAC,
    TRAIN_SOLUTIONS_CSV,
    VAL_FRAC,
)
from src.data.labeling import derive_binary_label

# Columnas que se persisten en los CSV de split (índice + lo necesario para
# reproducir el etiquetado sin tener que rederivarlo).
SPLIT_INDEX_COLUMNS = ["GalaxyID", "Class1.1", "Class1.2", "label"]


def load_clean_dataset(
    csv_path=TRAIN_SOLUTIONS_CSV, threshold: float = CLEAN_LABEL_THRESHOLD
) -> pd.DataFrame:
    """Carga `training_solutions_rev1.csv` y filtra al subconjunto limpio.

    Args:
        csv_path: ruta al CSV de vote fractions.
        threshold: umbral de vote fraction confiable (ver `derive_binary_label`).

    Returns:
        DataFrame con solo las galaxias etiquetadas `elliptical` o `spiral`
        (se descartan las `ambigua`), índice reiniciado.
    """
    df = pd.read_csv(csv_path)
    df = derive_binary_label(df, threshold=threshold)
    df_clean = df[df["label"].isin([ELLIPTICAL_LABEL, SPIRAL_LABEL])]
    return df_clean.reset_index(drop=True)


def stratified_split(
    df_clean: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split train/val/test estratificado por `label`, a nivel de `GalaxyID`.

    Args:
        df_clean: DataFrame con columna `label` (`elliptical`/`spiral`).
        train_frac: fracción destinada a entrenamiento.
        val_frac: fracción destinada a validación.
        test_frac: fracción destinada a test (no usada en entrenamiento).
        seed: semilla para reproducibilidad.

    Returns:
        Tupla `(train_df, val_df, test_df)`.
    """
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=(val_frac + test_frac),
        stratify=df_clean["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits_to_csv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Guarda los tres splits como `train.csv`, `val.csv`, `test.csv`.

    Persiste solo el índice (`GalaxyID` + `label` + vote fractions de Task 1),
    no las imágenes: son estas rutas las que deben leerse en notebooks/scripts
    de entrenamiento en vez de rederivar el split cada vez (evita divergencias
    entre corridas si cambia el orden de ejecución o la versión de sklearn).

    Args:
        train_df: split de entrenamiento (de `stratified_split`).
        val_df: split de validación.
        test_df: split de test.
        processed_dir: carpeta de salida.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split[SPLIT_INDEX_COLUMNS].to_csv(processed_dir / f"{name}.csv", index=False)


def load_splits_from_csv(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga `train.csv`, `val.csv`, `test.csv` desde `processed_dir`.

    Args:
        processed_dir: carpeta donde `save_splits_to_csv` guardó los splits.

    Returns:
        Tupla `(train_df, val_df, test_df)`.

    Raises:
        FileNotFoundError: si los CSV no existen todavía (correr primero
        `python -m src.data.make_splits`).
    """
    return tuple(
        pd.read_csv(processed_dir / f"{name}.csv") for name in ("train", "val", "test")
    )

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
