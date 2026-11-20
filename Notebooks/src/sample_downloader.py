import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cutout_downloader import build_fov_jobs, download_batch
from src.data.morfologia import load_labeled_sample


def main(output_dir: Path, fov_factor: float| str, max_concurrent: int, layer: str, bands: str, n: int = 20, seed: int = 42, path:Path=None, ignore_limits: bool = True, fov_estimation: str = None):
  assert path is not None, "Path to the CSV file must be provided"
  
  sample = load_labeled_sample(n=n, seed=seed, csv_path=path)
  sample["PETRO_TH90"] = pd.to_numeric(sample["PETRO_TH90"], errors="coerce")
  valid_petro = sample["PETRO_TH90"].gt(0) & np.isfinite(sample["PETRO_TH90"])
  sample = sample.loc[valid_petro].copy()

  if fov_estimation is not None:
      print(f"Usando FOV individual desde la columna: {fov_estimation}")

  jobs = build_fov_jobs(
      sample,
      output_dir=output_dir,
      fov_factor=fov_factor,
      layer=layer,
      bands=bands,
      ignore_limits=ignore_limits,
      fov_estimation=fov_estimation,
  )

  download_index = download_batch(
      jobs,
      max_concurrent=max_concurrent,
      min_request_interval_s=1.0,
      overwrite=False,
      timeout_s=60,
      max_retries=3,
  )

  print(f"Downloaded cutouts index: {download_index}")
  print(f"Output directory: {output_dir}")

  # Save the download index to a file for future reference
  index_file = output_dir / "download_index.csv"
  download_index.to_csv(index_file, index=False)
  print(f"Saved download index to: {index_file}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("data/mi_experimento/fov8"),
      help="Directory to save the downloaded cutouts",
  )
  parser.add_argument(
      "--fov-factor",
      type=float,
      default=8.0,
      help="Field of view factor for the cutouts",
  )
  parser.add_argument(
      "--max-concurrent",
      type=int,
      default=2,
      help="Maximum number of concurrent download requests",
  )
  parser.add_argument(
    "--layer",
    type=str,
    default="ls-dr11",
    help="Layer to use for the cutouts",
  )
  parser.add_argument(
      "--bands",
      type=str,
      default="grz",
      help="Bands to use for the cutouts",
  )
  parser.add_argument(
      "--n",
      type=int,
      default=None,
      help="Number of samples to load if no sample is provided",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for loading the sample",
  )
  parser.add_argument(
      "--path",
      type=Path,
      default=None,
      help="Path to the CSV file containing the labeled sample",
  )
  parser.add_argument(
      "--fov-estimation",
      type=str,
      default=None,
      help="Column name to use for FOV estimation",
  )
  args = parser.parse_args()

  main(
      output_dir=args.output_dir,
      fov_factor=args.fov_factor,
      max_concurrent=args.max_concurrent,
      layer=args.layer,
      bands=args.bands,
      n=args.n,
      seed=args.seed,
      path=args.path,
      ignore_limits=True,
      fov_estimation=args.fov_estimation,
  )


