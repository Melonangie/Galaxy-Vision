"""CLI para generar la caché espectral reproducible de una cohorte MaNGA."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataloader.spectra import SpectrumPreprocessConfig, build_spectrum_cache


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/galaxy_manifest.csv")
    parser.add_argument(
        "--splits", type=Path, default=root / "data/processed/splits_seed42_obj-label.csv"
    )
    parser.add_argument("--fits-dir", type=Path, default=root / "data/MPL-11")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/spectra")
    parser.add_argument("--all-manifest", action="store_true")
    parser.add_argument("--channels", nargs="+", default=["FLUX_1RE", "FLUX_FoV"])
    parser.add_argument("--rest-wave-min", type=float, default=3600.0)
    parser.add_argument("--rest-wave-max", type=float, default=9000.0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--continuum-min", type=float, default=5400.0)
    parser.add_argument("--continuum-max", type=float, default=5500.0)
    parser.add_argument("--normalization", choices=["per_channel", "shared"], default="per_channel")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.all_manifest:
        records = manifest
    else:
        splits = pd.read_csv(args.splits)
        records = splits.loc[:, ["name"]].merge(manifest, on="name", how="left", validate="one_to_one")
        if records["nsa_z"].isna().any():
            raise ValueError("El split contiene IDs ausentes del manifest o sin nsa_z.")
    config = SpectrumPreprocessConfig(
        channels=tuple(args.channels),
        rest_wave_min=args.rest_wave_min,
        rest_wave_max=args.rest_wave_max,
        length=args.length,
        continuum_window=(args.continuum_min, args.continuum_max),
        normalization=args.normalization,
    )
    cache_dir = build_spectrum_cache(
        records, args.fits_dir, args.output_dir, config, overwrite=args.overwrite
    )
    index = pd.read_csv(cache_dir / "index.csv")
    print(f"Caché: {cache_dir}")
    print(f"Listos: {(index['status'] == 'ready').sum()}/{len(index)}")
    print(f"Errores: {(index['status'] == 'error').sum()}")


if __name__ == "__main__":
    main()