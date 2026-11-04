%pip -q install astropy dust_extinction pyneb scipy pandas matplotlib seaborn scikit-learn tqdm

from pathlib import Path
from collections import defaultdict
from itertools import combinations
import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.stats import mad_std
from scipy.optimize import least_squares
from scipy.stats import kruskal, mannwhitneyu, false_discovery_control
from dust_extinction.parameter_averages import CCM89
from tqdm.auto import tqdm
import pyneb as pn

from google.colab import drive
drive.mount("/content/drive")

# EDIT THESE PATHS.
SPECTRA_DIR = Path("/content/drive/MyDrive/galaxies/spectra")
STARLIGHT_DIR = Path("/content/drive/MyDrive/galaxies/starlight")
CATALOG_PATH = Path("/content/drive/MyDrive/galaxies/morfologia.fits")
OUTPUT_DIR = Path("/content/drive/MyDrive/galaxies/eda_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "spectra_cache").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

# None means find a table HDU containing the required columns.
# Set an integer/name if your FITS files contain multiple suitable tables.
SPECTRUM_HDU = None
CATALOG_HDU = None

FITS_FRAME = "observed"
STARLIGHT_FRAME = "rest"

# REQUIRED: determine this from the extraction pipeline/FITS documentation.
# The line table below is in AIR wavelengths.
# Set both independently; a redshift does not convert vacuum to air.
FITS_WAVELENGTH_MEDIUM = "unknown"       # "air" or "vacuum"
STARLIGHT_WAVELENGTH_MEDIUM = "unknown"  # "air" or "vacuum"

# Used only if the FITS WAVE column has no unit.
WAVE_UNIT_IF_MISSING = u.AA

# These measurements remain in the original flux units unless you verify
# the FITS flux calibration. Do not label them erg/s/cm² automatically.
FLUX_UNIT_DESCRIPTION = "input flux-density units"

DETECTION_SNR = 3.0
NORMALIZATION_BAND = (5450.0, 5550.0)  # Rest-frame, approximately line-free.
GRID = np.arange(3500.0, 9001.0, 1.0)

# Reject large gaps rather than interpolating through them.
MAX_GAP_FACTOR = 3.0
MAX_CONTINUUM_MISMATCH = 0.15  # Exploratory configurable QC threshold.

# Start with a small run, inspect the QC, then use None for all files.
MAX_GALAXIES = 50

# Physical assumptions for optional nebular calculations.
GAS_TEMPERATURE = 10_000.0  # K; assumed, not measured.
GAS_DENSITY = 100.0        # cm^-3; assumed for Case B.
RV = 3.1