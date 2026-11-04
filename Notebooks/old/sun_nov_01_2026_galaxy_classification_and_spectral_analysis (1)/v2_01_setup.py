# ------------------------------------------------------------------
# Libraries
#   astropy -> the standard astronomy library: reads FITS / FITS.GZ files,
#              tables and does robust (sigma-clipped) statistics
#   numpy / pandas -> numbers and tables
#   scipy / sklearn -> statistics and feature ranking
#   matplotlib / seaborn -> plots
# ------------------------------------------------------------------
!pip -q install astropy seaborn tqdm

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy.integrate import trapezoid
from scipy.stats import kruskal, spearmanr

warnings.filterwarnings("ignore")   # astropy prints many harmless unit warnings
sns.set_theme(style="whitegrid", context="notebook")

# Give Colab access to your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# >>>>> EDIT THESE PATHS <<<<<
FITS_DIR   = "/content/drive/MyDrive/galaxy/fits"        # manga-XXXX-YYYY-1D.fits.gz
OUT_DIR    = "/content/drive/MyDrive/galaxy/starlight"   # manga-XXXX-YYYY-1D-out.txt / .out
MORPH_FILE = "/content/drive/MyDrive/galaxy/morfologia.fits"
SAVE_DIR   = "/content/drive/MyDrive/galaxy/eda_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# astropy opens .fits.gz directly (no need to unzip)
fits_files = sorted(glob.glob(f"{FITS_DIR}/*.fits.gz") + glob.glob(f"{FITS_DIR}/*.fits"))
out_files  = sorted(glob.glob(f"{OUT_DIR}/*-out.txt") + glob.glob(f"{OUT_DIR}/*.out"))
print(len(fits_files), "spectra |", len(out_files), "STARLIGHT outputs")

def galaxy_name(path):
    """'manga-10001-1901-1D.fits.gz' / 'manga-10001-1901-1D-out.txt' -> 'manga-10001-1901'.
    This is the key used to join spectra, STARLIGHT outputs and the catalogue."""
    return os.path.basename(path).split("-1D")[0]