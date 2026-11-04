# ------------------------------------------------------------------
# Libraries
#   astropy  -> the standard astronomy library: reads FITS files and tables
#   numpy / pandas -> numbers and tables
#   scipy / sklearn -> statistics and machine learning (feature ranking)
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

warnings.filterwarnings("ignore")          # astropy prints many harmless unit warnings
sns.set_theme(style="whitegrid", context="notebook")

# Mount Google Drive so Colab can see your files
from google.colab import drive
drive.mount('/content/drive')

# >>>>> EDIT THESE PATHS <<<<<
FITS_DIR   = "/content/drive/MyDrive/galaxy/fits"        # one 1D .fits spectrum per galaxy
OUT_DIR    = "/content/drive/MyDrive/galaxy/starlight"   # STARLIGHT outputs
MORPH_FILE = "/content/drive/MyDrive/galaxy/morfologia.fits"
SAVE_DIR   = "/content/drive/MyDrive/galaxy/eda_results" # tables & figures are saved here
os.makedirs(SAVE_DIR, exist_ok=True)

fits_files = sorted(glob.glob(f"{FITS_DIR}/*.fits"))
# STARLIGHT outputs may be called *.out or *-out.txt -> accept both
out_files  = sorted(glob.glob(f"{OUT_DIR}/*.out") + glob.glob(f"{OUT_DIR}/*-out.txt"))
print(len(fits_files), "FITS spectra |", len(out_files), "STARLIGHT outputs")

def galaxy_name(path):
    """'manga-10001-1901-1D.fits' or 'manga-10001-1901-1D-out.txt' -> 'manga-10001-1901'
    so that spectra, STARLIGHT files and the catalogue can be joined on the same key."""
    return os.path.basename(path).split("-1D")[0]