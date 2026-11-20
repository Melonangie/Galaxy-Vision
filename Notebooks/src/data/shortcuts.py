"""Atajos: atributos que podrían predecir la clase sin ser morfología (Fase 2).

Un atajo necesita dos cosas: que el atributo esté **correlacionado con la
etiqueta** en los datos y que sea **visible en la imagen**. Se distinguen dos
tipos de atributo (ver README.md, sección Atajos: qué podría aprender el
modelo en lugar de la morfología):

- **Observacionales** (no deberían informar la clase): redshift, región del
  survey, zoom del cutout, latitud galáctica. Si predicen la etiqueta, una red
  puede usarlos en vez de la forma.
- **Físicos** (correlatos legítimos de la morfología): masa estelar, tamaño en
  kpc. No son trampa, pero confunden: hay que saber qué parte de la señal de un
  atributo observacional pasa a través de ellos (`matching_weights`).

También detecta repeticiones y vecinas dentro del encuadre (`galaxy_groups`),
que deben ir juntas en el mismo split para no filtrar imágenes entre train y
test. Exploración en `notebook/shortcuts.ipynb`.
"""

from typing import Iterable, List, Optional

import astropy.cosmology
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

from src.config import (
    COSMOLOGY,
    CUTOUT_PIXSCALE_MAX,
    CUTOUT_PIXSCALE_MIN,
    CUTOUT_SIZE_PX,
    DUPLICATE_MAX_SEP_ARCSEC,
    LEGACY_NATIVE_PIXSCALE,
    LEGACY_REGION_SOUTH,
    MORFOLOGIA_LABEL_TO_INT,
    PROBE_BOOTSTRAP_N,
    RANDOM_SEED,
    SHORTCUT_MATCH_N_BINS,
)
from src.data.image_quality import legacy_region
from src.data.legacy_cutouts import compute_pixscale
from src.evaluate import bootstrap_auc_ci

# Columnas que añade `observational_attributes`, por tipo.
OBSERVATIONAL_COLUMNS = [
    "nsa_z", "sur", "lat_galactica_abs", "PETRO_TH90", "pixscale", "ampliada", "zoom_topado",
]
PHYSICAL_COLUMNS = ["LogMass", "r90_kpc", "incl"]


def observational_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Añade al catálogo los atributos de observación de cada galaxia.

    Args:
        df: DataFrame con `objra`, `objdec`, `nsa_z` y `PETRO_TH90`.

    Returns:
        Copia de `df` con `region`, `sur` (1 = DECam), `lat_galactica_abs`
        (|b| en grados), `pixscale` (el zoom con que se descarga el cutout),
        `ampliada` (1 si el zoom es más fino que el píxel nativo: la imagen es
        interpolada), `zoom_topado` (1 si el zoom quedó en un límite o sin
        `PETRO_TH90`), `kpc_por_arcsec` y `r90_kpc`.
    """
    out = df.copy()
    coords = SkyCoord(ra=out["objra"].to_numpy() * u.deg, dec=out["objdec"].to_numpy() * u.deg)
    out["region"] = legacy_region(out["objra"], out["objdec"])
    out["sur"] = (out["region"] == LEGACY_REGION_SOUTH).astype(int)
    out["lat_galactica_abs"] = np.abs(coords.galactic.b.deg)
    out["pixscale"] = [compute_pixscale(value) for value in out["PETRO_TH90"]]
    out["ampliada"] = (out["pixscale"] < LEGACY_NATIVE_PIXSCALE).astype(int)
    out["zoom_topado"] = (
        out["PETRO_TH90"].isna()
        | np.isclose(out["pixscale"], CUTOUT_PIXSCALE_MIN)
        | np.isclose(out["pixscale"], CUTOUT_PIXSCALE_MAX)
    ).astype(int)
    cosmology = getattr(astropy.cosmology, COSMOLOGY)
    redshift = out["nsa_z"].clip(lower=1e-4).to_numpy()
    out["kpc_por_arcsec"] = cosmology.kpc_proper_per_arcmin(redshift).value / 60.0
    out["r90_kpc"] = out["PETRO_TH90"] * out["kpc_por_arcsec"]
    return out


def label_to_int(labels: pd.Series) -> np.ndarray:
    """Etiqueta temprana/tardía a 0/1 (tardía = 1, `MORFOLOGIA_LABEL_TO_INT`)."""
    return labels.map(MORFOLOGIA_LABEL_TO_INT).to_numpy()


def auc_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    label_col: str = "label",
    n_boot: int = PROBE_BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """AUC de cada columna usada sola como puntaje de la etiqueta.

    El AUC se orienta para que >= 0.5 signifique "separa" (la dirección se
    reporta aparte). No se ajusta nada, así que no hace falta validación
    cruzada. Las filas con NaN en la columna se ignoran.

    Args:
        df: DataFrame con las columnas y la etiqueta.
        columns: columnas numéricas a evaluar.
        label_col: columna de la etiqueta temprana/tardía.
        n_boot: remuestreos bootstrap del intervalo de confianza.
        seed: semilla del bootstrap.

    Returns:
        DataFrame indexado por columna con `auc`, `ic_inf`, `ic_sup`, `n` y
        `direccion`, ordenado de mayor a menor AUC.
    """
    rows = []
    for column in columns:
        valid = df[column].notna()
        y = label_to_int(df.loc[valid, label_col])
        score = df.loc[valid, column].to_numpy(dtype=float)
        sign = 1.0 if roc_auc_score(y, score) >= 0.5 else -1.0
        result = bootstrap_auc_ci(y, sign * score, n_boot=n_boot, seed=seed)
        result.update(
            n=int(valid.sum()),
            direccion="mayor = tardía" if sign > 0 else "menor = tardía",
        )
        rows.append(pd.Series(result, name=column))
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def matching_weights(
    df: pd.DataFrame,
    on: List[str],
    label_col: str = "label",
    n_bins: int = SHORTCUT_MATCH_N_BINS,
) -> pd.Series:
    """Pesos que igualan la distribución de `on` entre las dos clases.

    Cada variable de `on` se parte en `n_bins` cuantiles; en cada celda se
    pondera cada clase para que su peso total sea proporcional al de la celda
    en la muestra completa. Con estos pesos, la clase ya no se puede predecir a
    partir de `on`, y cualquier AUC que quede en otra variable es señal que no
    pasa por `on` (p. ej. el redshift a masa fija). Las celdas con una sola
    clase quedan con peso 0 (fuera del soporte común). Se conserva el número
    total de galaxias de cada clase.

    Args:
        df: DataFrame con las columnas de `on` y la etiqueta.
        on: variables a igualar entre clases.
        label_col: columna de la etiqueta.
        n_bins: cuantiles por variable.

    Returns:
        Serie de pesos alineada con `df` (NaN en `on` -> peso 0).
    """
    bins = pd.concat(
        [pd.qcut(df[column], n_bins, labels=False, duplicates="drop") for column in on], axis=1
    )
    valid = bins.notna().all(axis=1)
    # Celda = combinación de bins, codificada como un entero en base n_bins.
    cell = (bins.fillna(0).astype(int) * (n_bins ** np.arange(len(on)))).sum(axis=1)

    counts = pd.crosstab(cell[valid], df.loc[valid, label_col])
    common = counts[(counts > 0).all(axis=1)]
    cell_share = common.sum(axis=1) / common.to_numpy().sum()

    weights = pd.Series(0.0, index=df.index)
    for label in common.columns:
        rows = valid & (df[label_col] == label) & cell.isin(common.index)
        raw = cell_share.loc[cell[rows]].to_numpy() / common.loc[cell[rows], label].to_numpy()
        weights.loc[rows] = raw * (df[label_col] == label).sum() / raw.sum()
    return weights


def weighted_auc(
    df: pd.DataFrame,
    column: str,
    weights: pd.Series,
    label_col: str = "label",
) -> float:
    """AUC (sin orientar) de una columna como puntaje, con pesos por galaxia.

    Args:
        df: DataFrame con la columna y la etiqueta.
        column: columna usada como puntaje.
        weights: pesos de `matching_weights`.
        label_col: columna de la etiqueta.

    Returns:
        AUC con `sample_weight`; > 0.5 = valores altos van con tardía.
    """
    keep = (weights > 0) & df[column].notna()
    return float(roc_auc_score(
        label_to_int(df.loc[keep, label_col]), df.loc[keep, column], sample_weight=weights[keep]
    ))


def galaxy_groups(
    df: pd.DataFrame,
    size_px: int = CUTOUT_SIZE_PX,
    duplicate_sep_arcsec: float = DUPLICATE_MAX_SEP_ARCSEC,
    pixscale: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Agrupa galaxias que no pueden separarse entre train y test.

    Dos entradas quedan en el mismo grupo si una cae dentro del cutout de la
    otra (separación menor que medio lado del encuadre de cualquiera de las
    dos) o si son la misma galaxia repetida (separación menor que
    `duplicate_sep_arcsec`). Los grupos son las componentes conexas de esa
    relación.

    Args:
        df: DataFrame con `objra`, `objdec` y `PETRO_TH90`.
        size_px: lado del cutout en píxeles.
        duplicate_sep_arcsec: separación máxima para considerar repetición.
        pixscale: zoom de cada galaxia; si es `None` se calcula con
            `compute_pixscale`, igual que al descargar.

    Returns:
        DataFrame alineado con `df` con `grupo` (entero), `tam_grupo`,
        `repetida` (comparte grupo con una entrada a < `duplicate_sep_arcsec`)
        y `vecina_en_encuadre`.
    """
    if pixscale is None:
        pixscale = pd.Series([compute_pixscale(v) for v in df["PETRO_TH90"]], index=df.index)
    half_fov = pixscale.to_numpy() * size_px / 2
    coords = SkyCoord(ra=df["objra"].to_numpy() * u.deg, dec=df["objdec"].to_numpy() * u.deg)
    first, second, sep, _ = coords.search_around_sky(coords, half_fov.max() * u.arcsec)
    keep = first < second
    first, second, sep = first[keep], second[keep], sep.arcsec[keep]
    linked = sep < np.maximum(half_fov[first], half_fov[second])
    duplicate = sep < duplicate_sep_arcsec

    n = len(df)
    graph = coo_matrix((np.ones(linked.sum()), (first[linked], second[linked])), shape=(n, n))
    _, group = connected_components(graph, directed=False)

    repeated = np.zeros(n, dtype=bool)
    repeated[first[duplicate]] = repeated[second[duplicate]] = True
    neighbor = np.zeros(n, dtype=bool)
    neighbor[first[linked & ~duplicate]] = neighbor[second[linked & ~duplicate]] = True
    sizes = np.bincount(group)
    return pd.DataFrame(
        {"grupo": group, "tam_grupo": sizes[group], "repetida": repeated, "vecina_en_encuadre": neighbor},
        index=df.index,
    )
