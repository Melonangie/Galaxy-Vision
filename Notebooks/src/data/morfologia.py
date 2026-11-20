"""Carga y etiquetado binario del dataset MaNGA (`data/morfologia.csv`).

Migra a `src/` el criterio validado en `notebook/exploration_morfologia.ipynb`
(regla de oro, ver README.md): la etiqueta temprana/tardía se deriva de `Ttype`
con el umbral comunicado por J. A. Vázquez Mata (UNAM), no con un criterio de
la literatura. Ver README.md, sección Dataset — MaNGA + Legacy Survey.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from src.config import (
    EARLY_TYPE_LABEL,
    INCL_SUSPECT_VALUE,
    LATE_TYPE_LABEL,
    LOGMASS_MAX,
    LOGMASS_MIN,
    MORFOLOGIA_CSV,
    MORFOLOGIA_CURATED_CSV,
    NSA_Z_MAX,
    NSA_Z_MIN,
    NSA_Z_SENTINEL,
    RANDOM_SEED,
    TTYPE_EARLY_MAX,
    TTYPE_LATE_MIN,
    TTYPE_OUT_OF_RANGE_LABEL,
)

# Rango superior de "tardía" declarado en el correo. Los 67 registros con
# Ttype == 11 quedan fuera a propósito (confirmado en la junta del 2026-09-23:
# Ttype > 10 se ignora).
TTYPE_LATE_MAX = 10


def load_morfologia(csv_path: Path = MORFOLOGIA_CSV) -> pd.DataFrame:
    """Carga `morfologia.csv` tal cual, sin filtrar ni imputar.

    Args:
        csv_path: ruta al CSV de la tabla MaNGA.

    Returns:
        DataFrame con las columnas originales (`name`, `objra`, `objdec`,
        `Type`, `Ttype`, `nsa_z`, `LogMass`, `incl`, `PETRO_TH90`).
    """
    return pd.read_csv(csv_path)


def derive_ttype_label(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna `label` con la etiqueta binaria temprana/tardía.

    Reglas (criterio de J. A. Vázquez, ver README.md):
        temprana        si Ttype <= TTYPE_EARLY_MAX
        tardía          si TTYPE_LATE_MIN <= Ttype <= TTYPE_LATE_MAX
        fuera_de_rango  en cualquier otro caso (p. ej. Ttype == 11)

    Args:
        df: DataFrame con al menos la columna `Ttype`.

    Returns:
        Copia de `df` con la columna adicional `label`.
    """
    out = df.copy()
    out["label"] = TTYPE_OUT_OF_RANGE_LABEL
    out.loc[out["Ttype"] <= TTYPE_EARLY_MAX, "label"] = EARLY_TYPE_LABEL
    out.loc[
        out["Ttype"].between(TTYPE_LATE_MIN, TTYPE_LATE_MAX), "label"
    ] = LATE_TYPE_LABEL
    return out


# Cortes acordados en la junta del 2026-09-23, en el orden en que se aplican.
AGREED_QUALITY_CUTS = ("ttype", "nsa_z", "logmass")
# Cortes aún pendientes de revisión manual (se miden, no se aplican).
CANDIDATE_QUALITY_CUTS = ("incl_cero",)


def quality_cut_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Máscaras booleanas de las filas que **elimina** cada corte de calidad.

    Criterios de la junta con el IA-UNAM (ver README.md, sección
    Retroalimentación de la junta con el IA-UNAM). Cada máscara se evalúa
    sobre el DataFrame completo, de forma independiente a las demás:

        ttype      Ttype > TTYPE_LATE_MAX
        nsa_z      nsa_z fuera de [NSA_Z_MIN, NSA_Z_MAX] (incluye el
                   centinela NSA_Z_SENTINEL y los NaN)
        logmass    LogMass fuera de [LOGMASS_MIN, LOGMASS_MAX] (incluye NaN)
        incl_cero  incl == INCL_SUSPECT_VALUE (candidato, no acordado)

    `PETRO_TH90` no tiene corte: sus faltantes se resuelven con un `scale`
    fijo al descargar la imagen.

    Args:
        df: DataFrame con las columnas originales de `morfologia.csv`.

    Returns:
        Diccionario `nombre_del_corte -> Serie booleana` (True = se elimina).
    """
    nsa_z = df["nsa_z"].replace(NSA_Z_SENTINEL, float("nan"))
    return {
        "ttype": df["Ttype"] > TTYPE_LATE_MAX,
        "nsa_z": ~nsa_z.between(NSA_Z_MIN, NSA_Z_MAX),
        "logmass": ~df["LogMass"].between(LOGMASS_MIN, LOGMASS_MAX),
        "incl_cero": df["incl"] == INCL_SUSPECT_VALUE,
    }


def quality_cut_report(
    df: pd.DataFrame,
    cuts: Sequence[str] = AGREED_QUALITY_CUTS,
) -> pd.DataFrame:
    """Cuánto del dataset elimina cada corte de calidad, sin aplicarlo.

    Para cada corte (en el orden de `cuts`) reporta tres vistas, todas en
    número de filas y en % del total original:

        individual  lo que quita el corte aplicado solo, sobre el total.
        exclusivo   lo que quita solo ese corte y ningún otro de `cuts`
                    (el costo real de mantenerlo).
        secuencial  lo que quita al aplicarlo después de los anteriores,
                    más el acumulado y las filas que quedan.

    Además desglosa el % individual por clase temprana/tardía, para detectar
    si un corte desbalancea las clases.

    Args:
        df: DataFrame con las columnas originales de `morfologia.csv`.
        cuts: nombres de corte de `quality_cut_masks`, en orden de aplicación.

    Returns:
        DataFrame con una fila por corte, indexado por nombre de corte.
    """
    labeled = derive_ttype_label(df) if "label" not in df.columns else df
    masks = quality_cut_masks(labeled)
    n_total = len(labeled)
    is_early = labeled["label"] == EARLY_TYPE_LABEL
    is_late = labeled["label"] == LATE_TYPE_LABEL

    rows = []
    removed_so_far = pd.Series(False, index=labeled.index)
    for cut in cuts:
        mask = masks[cut]
        others = pd.Series(False, index=labeled.index)
        for other in cuts:
            if other != cut:
                others |= masks[other]
        new = mask & ~removed_so_far
        removed_so_far |= mask
        rows.append({
            "corte": cut,
            "n_individual": int(mask.sum()),
            "pct_individual": 100 * mask.mean(),
            "n_exclusivo": int((mask & ~others).sum()),
            "pct_exclusivo": 100 * (mask & ~others).mean(),
            "n_secuencial": int(new.sum()),
            "pct_secuencial": 100 * new.mean(),
            "pct_acumulado": 100 * removed_so_far.mean(),
            "n_restantes": int(n_total - removed_so_far.sum()),
            "pct_temprana": 100 * mask[is_early].mean(),
            "pct_tardia": 100 * mask[is_late].mean(),
        })
    return pd.DataFrame(rows).set_index("corte")


def apply_quality_cuts(
    df: pd.DataFrame,
    cuts: Sequence[str] = AGREED_QUALITY_CUTS,
) -> pd.DataFrame:
    """Elimina las filas que no pasan los cortes de calidad indicados.

    Por defecto aplica solo los cortes acordados (`AGREED_QUALITY_CUTS`); los
    candidatos (p. ej. `incl_cero`) hay que pedirlos explícitamente en `cuts`.
    El impacto de cada corte se mide antes con `quality_cut_report`.

    Args:
        df: DataFrame con las columnas originales de `morfologia.csv`.
        cuts: nombres de corte de `quality_cut_masks`.

    Returns:
        Copia filtrada de `df` con el índice reiniciado.
    """
    masks = quality_cut_masks(df)
    removed = pd.Series(False, index=df.index)
    for cut in cuts:
        removed |= masks[cut]
    return df.loc[~removed].reset_index(drop=True)


def build_curated_morfologia(
    csv_path: Path = MORFOLOGIA_CSV,
    out_path: Optional[Path] = MORFOLOGIA_CURATED_CSV,
    cuts: Sequence[str] = AGREED_QUALITY_CUTS,
) -> pd.DataFrame:
    """Genera el dataset curado: carga, aplica los cortes y etiqueta.

    Tras los cortes acordados no queda ninguna fila `fuera_de_rango` ni el
    centinela de `nsa_z`, así que la columna `label` solo toma los valores
    temprana/tardía.

    Args:
        csv_path: ruta al `morfologia.csv` crudo.
        out_path: dónde escribir el CSV curado; `None` para no escribirlo.
        cuts: nombres de corte de `quality_cut_masks`.

    Returns:
        DataFrame curado, con las columnas originales más `label`.
    """
    curated = derive_ttype_label(apply_quality_cuts(load_morfologia(csv_path), cuts))
    if out_path is not None:
        curated.to_csv(out_path, index=False)
    return curated


def sample_balanced(
    df: pd.DataFrame,
    n: int,
    seed: int = RANDOM_SEED,
    require_petro_th90: bool = True,
) -> pd.DataFrame:
    """Muestra balanceada temprana/tardía para descargar imágenes de prueba.

    Descarta las filas `fuera_de_rango` y, por defecto, las que no traen
    `PETRO_TH90` (sin ese radio no se puede ajustar el zoom del cutout por
    galaxia; en la muestra actual son 33 filas).

    Las muestras son anidadas para una misma semilla: la de `n=10` está
    contenida en la de `n=50`. Así, ampliar la muestra más adelante solo
    descarga imágenes nuevas en vez de cambiar el conjunto entero.

    Args:
        df: DataFrame ya etiquetado por `derive_ttype_label`.
        n: número total de galaxias a muestrear (se reparte mitad y mitad).
        seed: semilla para que la muestra sea reproducible.
        require_petro_th90: si `True`, exige `PETRO_TH90` no nulo.

    Returns:
        DataFrame con hasta `n` filas, ordenado aleatoriamente y con el índice
        reiniciado.
    """
    pool = df[df["label"].isin([EARLY_TYPE_LABEL, LATE_TYPE_LABEL])]
    if require_petro_th90:
        pool = pool.dropna(subset=["PETRO_TH90"])

    per_class = n // 2
    parts = [
        group.sample(frac=1.0, random_state=seed).head(per_class)
        for _, group in pool.groupby("label", sort=True)
    ]
    sample = pd.concat(parts)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_labeled_sample(
    n: Optional[int] = None,
    csv_path: Path = MORFOLOGIA_CSV,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Atajo: carga el CSV, deriva la etiqueta y (opcionalmente) muestrea.

    Args:
        n: tamaño de la muestra balanceada; `None` devuelve el dataset completo
            ya etiquetado, sin filtrar.
        csv_path: ruta al CSV de la tabla MaNGA.
        seed: semilla de muestreo.

    Returns:
        DataFrame etiquetado (completo o muestreado).
    """
    labeled = derive_ttype_label(load_morfologia(csv_path))
    return labeled if n is None else sample_balanced(labeled, n, seed=seed)

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
