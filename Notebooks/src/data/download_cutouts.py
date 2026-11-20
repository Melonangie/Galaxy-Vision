"""Descarga una muestra de imágenes de galaxias desde legacysurvey.org.

Cada variante de capa/bandas se guarda en su propia subcarpeta y, para `.jpg`,
se le aplica además la máscara de color de `src/data/galaxy_mask.py` (ver
README.md, secciones Imágenes: legacysurvey.org y Variantes de capa y bandas):

    data/legacy_cutouts/jpg/<layer>_<bands>/<name>.jpg
    data/legacy_cutouts/jpg_masked/<layer>_<bands>/<name>.jpg
    data/legacy_cutouts/fits/<layer>_<bands>/<name>.fits

El índice `cutouts_index.csv` trae una fila por (galaxia, formato, variante)
con el zoom aplicado, el estado de la descarga y, para `.jpg`, la fracción sin
cobertura del survey y las estadísticas de la máscara. La descarga es
reanudable: volver a ejecutar el comando solo pide lo que falte. Las máscaras
se recalculan siempre (son deterministas y baratas), así el índice refleja los
parámetros `GALAXY_MASK_*` vigentes.

Uso:
    python -m src.data.download_cutouts                          # 50 galaxias, variante activa, .jpg
    python -m src.data.download_cutouts --variants all           # todas las de LEGACY_VARIANTS
    python -m src.data.download_cutouts --variants ls-dr9_grz ls-dr11_grz
    python -m src.data.download_cutouts --n 10 --format fits     # subconjunto .fits
    python -m src.data.download_cutouts --n 0                    # dataset completo
"""

import argparse
from typing import List, Optional, Tuple

import pandas as pd

from src.config import (
    ALL_VARIANTS,
    CUTOUT_FORMAT,
    CUTOUT_FORMAT_FITS,
    CUTOUT_FORMAT_JPG,
    CUTOUT_FOV_FACTOR,
    CUTOUT_SAMPLE_SIZE,
    CUTOUT_SIZE_PX,
    CUTOUTS_INDEX_CSV,
    CUTOUTS_MASKED_JPG_DIR,
    LEGACY_BANDS,
    LEGACY_LAYER,
    LEGACY_VARIANTS,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.data.cutout_io import load_jpg_cutout, no_coverage_fraction
from src.data.galaxy_mask import save_masked_cutout
from src.data.legacy_cutouts import (
    cutout_path,
    download_cutouts,
    parse_variant_name,
    variant_name,
)
from src.data.morfologia import derive_ttype_label, load_morfologia, sample_balanced

# Columnas que identifican una fila del índice.
INDEX_KEY = ["name", "format", "variante"]


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=CUTOUT_SAMPLE_SIZE,
        help="galaxias a descargar (muestra balanceada); 0 = dataset completo",
    )
    parser.add_argument(
        "--format",
        choices=[CUTOUT_FORMAT_JPG, CUTOUT_FORMAT_FITS],
        default=CUTOUT_FORMAT,
        help="formato del cutout",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[variant_name(LEGACY_LAYER, LEGACY_BANDS)],
        metavar="LAYER_BANDS",
        help=(
            "variantes <layer>_<bands> a descargar (por defecto la activa en "
            f"config); '{ALL_VARIANTS}' = todas las de LEGACY_VARIANTS"
        ),
    )
    parser.add_argument(
        "--size", type=int, default=CUTOUT_SIZE_PX, help="lado del cutout en px"
    )
    parser.add_argument(
        "--fov-factor",
        type=float,
        default=CUTOUT_FOV_FACTOR,
        help="campo de visión en múltiplos de PETRO_TH90",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="semilla del muestreo"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="vuelve a descargar las imágenes ya presentes en disco",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="no aplica la máscara de color a los .jpg descargados",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="sufijo para identificar la corrida, por ejemplo fov4, fov5, fov6",
    )
    return parser.parse_args()


def select_variants(names: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Traduce nombres `<layer>_<bands>` a tuplas `(layer, bands)`.

    Args:
        names: nombres pedidos; `None` o `[ALL_VARIANTS]` devuelve todas las
            de `LEGACY_VARIANTS`. Se aceptan también variantes fuera de esa
            lista (p. ej. para probar una capa nueva).

    Returns:
        Lista de tuplas `(layer, bands)`, sin repetidos y en el orden pedido.

    Raises:
        ValueError: si algún nombre no tiene la forma `<layer>_<bands>`.
    """
    if names is None or names == [ALL_VARIANTS]:
        return list(LEGACY_VARIANTS)
    return list(dict.fromkeys(parse_variant_name(name) for name in names))


def curate_variant(index: pd.DataFrame, layer: str, bands: str) -> pd.DataFrame:
    """Mide cobertura y aplica la máscara a los `.jpg` descargados de una variante.

    Args:
        index: índice de `download_cutouts` para esa variante (formato `.jpg`).
        layer: capa del survey.
        bands: bandas pedidas al visor.

    Returns:
        El índice con las columnas `frac_sin_cobertura`, `mask_frac`,
        `mask_toca_borde` y `masked_path`.
    """
    records = []
    for row in index.to_dict("records"):
        if row["status"] != "error":
            image = load_jpg_cutout(row["name"], layer, bands)
            row["frac_sin_cobertura"] = round(no_coverage_fraction(image), 4)
            row.update(save_masked_cutout(row["name"], layer, bands))
            masked = cutout_path(
                row["name"], CUTOUT_FORMAT_JPG, layer, bands, root=CUTOUTS_MASKED_JPG_DIR
            )
            row["masked_path"] = masked.relative_to(PROJECT_ROOT).as_posix()
        records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_variants(index: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por variante para comparar de un vistazo.

    Args:
        index: índice de cutouts (`cutouts_index.csv`), una o varias variantes.

    Returns:
        DataFrame con una fila por (formato, variante): galaxias OK, errores,
        MB en disco y, si hay columnas de máscara, cutouts con zona sin
        cobertura, máscaras vacías, máscaras que tocan el borde y fracción
        media cubierta por la máscara.
    """
    ok = index["status"] != "error"
    columns = {"ok": ok, "errores": ~ok, "mb": index["bytes"].where(ok, 0) / 1024 / 1024}
    if "mask_frac" in index:
        columns.update(
            con_zona_sin_cobertura=index["frac_sin_cobertura"].fillna(0) > 0,
            mascara_vacia=ok & (index["mask_frac"] == 0),
            mascara_toca_borde=index["mask_toca_borde"].astype("boolean").fillna(False),
            mask_frac_media=index["mask_frac"],
        )
    aggregations = {name: "sum" for name in columns}
    if "mask_frac_media" in columns:
        aggregations["mask_frac_media"] = "mean"
    return (
        pd.DataFrame(columns)
        .groupby([index["format"], index["variante"]], sort=False)
        .agg(aggregations)
        .round(3)
    )


def update_index(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Fusiona filas nuevas en `cutouts_index.csv` y lo guarda.

    Solo se reemplazan las filas con la misma clave (`name`, `format`,
    `variante`), para no perder lo registrado en corridas previas con otras
    variantes, formatos o muestras.

    Args:
        new_rows: filas recién procesadas.

    Returns:
        El índice completo tal como quedó en disco.
    """
    index = new_rows
    if CUTOUTS_INDEX_CSV.exists():
        previous = pd.read_csv(CUTOUTS_INDEX_CSV)
        keys = set(new_rows[INDEX_KEY].itertuples(index=False, name=None))
        keep = [key not in keys for key in previous[INDEX_KEY].itertuples(index=False, name=None)]
        index = pd.concat([previous[keep], new_rows], ignore_index=True)
    CUTOUTS_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(CUTOUTS_INDEX_CSV, index=False)
    return index


def main() -> None:
    """Descarga (y enmascara) la muestra pedida y actualiza el índice CSV."""
    args = parse_args()
    variants = select_variants(args.variants)
    apply_masks = args.format == CUTOUT_FORMAT_JPG and not args.no_mask

    labeled = derive_ttype_label(load_morfologia())
    if args.n > 0:
        galaxies = sample_balanced(labeled, args.n, seed=args.seed)
    else:
        galaxies = labeled.dropna(subset=["PETRO_TH90"])

    print(
        f"{len(galaxies)} galaxias x {len(variants)} variantes, .{args.format} "
        f"({args.size}x{args.size} px, FOV = {args.fov_factor} x PETRO_TH90)"
    )
    frames = []
    for layer, bands in variants:
        print(f"\n== {variant_name(layer, bands)}")
        index = download_cutouts(
            galaxies,
            image_format=args.format,
            size_px=args.size,
            fov_factor=args.fov_factor,
            overwrite=args.overwrite,
            layer=layer,
            bands=bands,
            tag=args.tag,
        )
        frames.append(curate_variant(index, layer, bands) if apply_masks else index)

    current = pd.concat(frames, ignore_index=True)
    update_index(current)
    print("\n" + summarize_variants(current).to_string())
    print(current.drop_duplicates("name")["label"].value_counts().to_string())
    print(f"\nÍndice actualizado en {CUTOUTS_INDEX_CSV}")


if __name__ == "__main__":
    main()

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
