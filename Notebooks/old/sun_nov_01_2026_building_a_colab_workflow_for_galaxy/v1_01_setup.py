%pip -q install astropy dust_extinction scipy pandas matplotlib seaborn scikit-learn tqdm

from pathlib import Path
import re
import json
import warnings
import importlib.metadata as metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from dust_extinction.parameter_averages import CCM89
from scipy.optimize import least_squares
from scipy.stats import kruskal
from tqdm.auto import tqdm

from google.colab import drive
drive.mount("/content/drive")

# CHANGE THESE PATHS.
SPECTRA_DIR = Path("/content/drive/MyDrive/galaxies/spectra")
STARLIGHT_DIR = Path("/content/drive/MyDrive/galaxies/starlight")
CATALOG_PATH = Path("/content/drive/MyDrive/galaxies/morfologia.fits")
OUTPUT_DIR = Path("/content/drive/MyDrive/galaxies/eda_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FITS layouts cannot be established from CSV exports alone.
# Leave these as None only when there is one unambiguous matching table.
SPECTRUM_HDU = None
CATALOG_HDU = None

# Confirm these against the original FITS and STARLIGHT preprocessing.
WAVE_UNIT = u.AA
WAVELENGTH_CONVENTION = "vacuum"  # Typical MaNGA convention.
STARLIGHT_WAVELENGTH_CONVENTION = "vacuum"

# The supplied STARLIGHT examples appear already rest-frame.
STARLIGHT_FRAME = "rest"

# Assumes STARLIGHT was supplied the original FLUX_FoV values after
# shifting the wavelength, WITHOUT multiplying the flux by (1 + z).
# Set True if its input preparation already applied that multiplication.
STARLIGHT_INPUT_HAS_FLUX_JACOBIAN = False

# We apply the Jacobian to preserve integrated flux in rest coordinates:
# lambda_rest = lambda_observed / (1 + z)
# f_lambda_rest = (1 + z) * f_lambda_observed
# This is NOT a conversion to luminosity.
NORMALIZATION_BAND = (5450.0, 5550.0)
REST_GRID = np.arange(3500.0, 9001.0, 2.0)

MIN_ESTIMATED_LINE_SN = 3.0
MAX_ALIGNMENT_FRACTIONAL_DIFFERENCE = 0.15  # Exploratory, inspect visually.
RANDOM_SEED = 42

# For a small initial run, use 20; use None for all galaxies.
MAX_GALAXIES = 20

sns.set_theme(style="whitegrid")