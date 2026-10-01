#!/usr/bin/env bash

set -u

INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-csv_output}"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
files=("$INPUT_DIR"/*.fits "$INPUT_DIR"/*.fits.gz)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .fits or .fits.gz files found in $INPUT_DIR"
    exit 1
fi

for fits_file in "${files[@]}"; do
    python - "$fits_file" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

fits_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

# Remove .fits.gz or .fits from the filename
name = fits_path.name
if name.endswith(".fits.gz"):
    name = name[:-8]
elif name.endswith(".fits"):
    name = name[:-5]

csv_path = output_dir / f"{name}.csv"

try:
    with fits.open(fits_path, memmap=False) as hdul:
        # Find the first FITS table HDU
        table_hdu = next(
            (hdu for hdu in hdul if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))),
            None,
        )

        if table_hdu is not None:
            table = Table(table_hdu.data)
            table.write(csv_path, format="ascii.csv", overwrite=True)
        else:
            # Convert an image HDU to rows and columns
            image_hdu = next(
                (hdu for hdu in hdul if hdu.data is not None),
                None,
            )

            if image_hdu is None:
                raise ValueError("No data found")

            data = np.asarray(image_hdu.data)
            np.savetxt(csv_path, np.atleast_2d(data), delimiter=",", fmt="%.18g")

    print(f"Converted: {fits_path} -> {csv_path}")

except Exception as error:
    print(f"ERROR: {fits_path}: {error}", file=sys.stderr)
PY
done
