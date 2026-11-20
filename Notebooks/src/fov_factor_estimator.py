from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.config import *
from src.data.galaxy_mask import galaxy_color_mask


def centered_crop_box(width, height, base_fov_factor, target_fov_factor):
    if target_fov_factor <= 0 or target_fov_factor > base_fov_factor:
        raise ValueError("El FOV objetivo debe ser positivo y no mayor al FOV base.")

    crop_fraction = target_fov_factor / base_fov_factor
    crop_side_px = max(1, int(round(min(width, height) * crop_fraction)))
    left = (width - crop_side_px) // 2
    top = (height - crop_side_px) // 2

    return left, top, crop_side_px


def simulate_fov_from_base(image, base_fov_factor, target_fov_factor):
    source = image.convert("RGB")
    left, top, crop_side_px = centered_crop_box(
        *source.size,
        base_fov_factor,
        target_fov_factor,
    )
    crop = source.crop((left, top, left + crop_side_px, top + crop_side_px))
    simulated = crop.resize(
        (CUTOUT_SIZE_PX, CUTOUT_SIZE_PX),
        Image.Resampling.LANCZOS,
    )
    return simulated, crop_side_px


def simulate_mask_from_base(mask, base_fov_factor, target_fov_factor):
    height, width = mask.shape
    left, top, crop_side_px = centered_crop_box(
        width,
        height,
        base_fov_factor,
        target_fov_factor,
    )

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    crop = mask_image.crop((left, top, left + crop_side_px, top + crop_side_px))
    simulated_mask = crop.resize(
        (CUTOUT_SIZE_PX, CUTOUT_SIZE_PX),
        Image.Resampling.NEAREST,
    )
    return np.asarray(simulated_mask, dtype=bool)


def angular_ring_coverage(mask):
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width]
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2

    radius = np.hypot(yy - center_y, xx - center_x)
    ring = np.abs(radius - TARGET_RADIUS_PX) <= RING_HALF_WIDTH_PX

    angles = (
        np.arctan2(yy - center_y, xx - center_x) + 2 * np.pi
    ) % (2 * np.pi)
    sectors = np.floor(angles * N_SECTORS / (2 * np.pi)).astype(int)

    occupied_sectors = np.unique(sectors[ring & mask])
    return len(occupied_sectors) / N_SECTORS


def main(args):
  INPUT_DIR = args.input_dir
  OUTPUT_DIR = args.output_dir
  DOWNLOAD_INDEX_CSV = args.download_index_csv
  SIMULATION_INDEX_CSV = args.simulation_index_csv

  CENTER_FRAC = CENTER_RADIUS_PX / CUTOUT_SIZE_PX
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  download_index = pd.read_csv(DOWNLOAD_INDEX_CSV)
  download_index = download_index.drop_duplicates("name").set_index("name")

  simulation_records = []

  for source_path in tqdm(sorted(INPUT_DIR.glob("*.jpg"))):
      name = source_path.stem

      if name not in download_index.index:
          print(f"Sin metadata en download_index.csv: {name}")
          continue

      metadata = download_index.loc[name]
      base_fov_factor = float(metadata["fov_factor"])

      if not np.isfinite(base_fov_factor) or base_fov_factor <= 0:
          print(f"FOV base inválido para {name}: {base_fov_factor}")
          continue

      with Image.open(source_path) as source_image:
          base_image = source_image.convert("RGB")

      base_mask = galaxy_color_mask(
          np.asarray(base_image),
          center_frac=CENTER_FRAC,
      )

      for fov_factor in ESTIMATION_FOVS:
          simulated, crop_side_px = simulate_fov_from_base(
              base_image,
              base_fov_factor,
              fov_factor,
          )
          simulated_mask = simulate_mask_from_base(
              base_mask,
              base_fov_factor,
              fov_factor,
          )

          coverage = angular_ring_coverage(simulated_mask)
          reaches_target = coverage >= MIN_ANGULAR_COVERAGE

          if reaches_target:
            simulation_records.append(
                {
                    "name": name,
                    "source_path": str(source_path),
                    #"simulated_path": str(simulated_path),
                    "base_fov_factor": base_fov_factor,
                    "fov_factor": fov_factor,
                    "crop_side_px": crop_side_px,
                    "output_size_px": CUTOUT_SIZE_PX,
                    "center_radius_px": CENTER_RADIUS_PX,
                    "target_radius_px": TARGET_RADIUS_PX,
                    "ring_half_width_px": RING_HALF_WIDTH_PX,
                    "n_sectors": N_SECTORS,
                    "min_angular_coverage": MIN_ANGULAR_COVERAGE,
                    "angular_coverage": coverage,
                    "reaches_target": reaches_target,
                    "label": metadata.get("label"),
                    "PETRO_TH90": metadata.get("PETRO_TH90"),
                    "layer": metadata.get("layer"),
                    "bands_or_color_mode": metadata.get("bands_or_color_mode"),
                    "base_pixscale": metadata.get("pixscale"),
                    "base_fov_requested_arcsec": metadata.get(
                        "fov_requested_arcsec"
                    ),
                    "base_fov_effective_arcsec": metadata.get(
                        "fov_effective_arcsec"
                    ),
                    "base_status": metadata.get("status"),
                }
            )
            break

  simulation_index = pd.DataFrame(simulation_records)
  simulation_index.to_csv(SIMULATION_INDEX_CSV, index=False)

  download_index_updated = pd.merge(
      download_index,
      simulation_index[["name", "fov_factor", "angular_coverage", "reaches_target"]],
      on="name",
      how="left",
  )
  estimated_index_path = DOWNLOAD_INDEX_CSV.with_name(
    f"{DOWNLOAD_INDEX_CSV.stem}_fov_estimated.csv"
    )
  download_index_updated.to_csv(estimated_index_path, index=False)

  print(f"Simulaciones guardadas en: {OUTPUT_DIR}")
  print(f"Índice guardado en: {SIMULATION_INDEX_CSV}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download_index_csv", type=Path, required=True)
    parser.add_argument("--simulation_index_csv", type=Path, required=True)
    args = parser.parse_args()

    main(args)