# Copilot Chat Conversation Export: Galaxy classification and spectral analysis

**User:** @maria-rabelero-mx_nestle
**Thread URL:** https://github.com/copilot/c/56a77ffb-25b2-4b81-892f-444c9c255e7a

## @maria-rabelero-mx_nestle

Do EDA to thousands of .fits and .out files to find information and features  that will help me classify galaxies by morphological type and find which features best separate the types

Do spectral line diagnostic (BPT, balmer, emission and absorption lines, etc.) per morphological class

Graph all the spectrums group by morphological class

Graph the mean spectrum of all morphological classes in single graph

Each .fits file is of a galaxy. The galaxies are of all morphology ttype. These are some samples of the .fits files, they have Gaps in the spectra but the .fits files do not.  Use wave and flux_fov, ignore ERROR_FoV is incorrect. Wave needs to be converted to rest-frame wavelength
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D.csv

These are some of the .out files. They were made using http://www.starlight.ufsc.br/downloads/ and the .fits files, wave and flux_fov.
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D-out.txt

I also have a supporting .fits file that I don't want to do EDA on. Use its information to complement the EDA. This is a sample of the supporting .fits file
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/morfologia.csv

I will use google colab, the .fits and .out files are in google drive

Use astronomy libraries

Document as much as you can in galaxy morphology context with reference

make the code as simple as possible and human readable as possible, explain what the code is doing and why, using inline comments and codeblocks. 


## Copilot



*System: User retried message*

## Copilot

Below is a complete Google Colab notebook. You can paste each block into its own cell and run them in order. It reads the galaxy spectra and the STARLIGHT files, joins them to the morphology catalogue, and produces:

1. **EDA**: data quality checks, how many galaxies each class has, and the catalogue properties per class.
2. **Features**: stellar population results from STARLIGHT, emission lines (from the STARLIGHT residual), absorption indices (Dn4000 and Lick indices) and spectral PCA.
3. **Feature ranking**: three methods (Kruskal–Wallis effect size, mutual information, random-forest permutation importance) to find which features best separate the morphological types.
4. **Spectral diagnostics per class**: BPT (three diagrams), WHAN, Balmer decrement, emission and absorption lines, Dn4000 vs HδA.
5. **Plots**: every spectrum grouped by class, and the mean spectra of all classes in one figure.

I haven't run the notebook. I wrote it from the five sample CSVs, the `.out` files and `morfologia.csv`. Run it on a few hundred files first and check the paths and column names. Things I checked or assumed:

* **The STARLIGHT `.out` spectra are already in the rest frame.** In `manga-10001-1901` (z = 0.02575), the first observed wavelength is 3622 Å. Divided by (1+z) that gives 3531 Å, which is exactly where `f_obs` stops being 0 in the `.out` file. So I only shift the FITS wavelengths (`WAVE / (1+z)`), never the `.out` ones.
* STARLIGHT divides `f_obs` and `f_syn` by `fobs_norm`, the median flux at 4010–4060 Å. Line ratios and equivalent widths don't depend on that, so they are safe to use.
* Some catalogue rows have `nsa_z = -9999` (no redshift). Those galaxies are dropped, because they can't be shifted to the rest frame. Rows with `Ttype = 11` (Type "S", spiral with no subtype) are also dropped.
* I couldn't confirm the source of the morphology catalogue. Labels like `Edc` and `SABbc` look like the MaNGA visual morphology catalogue of Vázquez‑Mata et al. (2022), but please check that before citing it.

---

## Cell 1 – Setup

```python name=01_setup.py
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
```

## Cell 2 – Morphology catalogue (supporting file, no EDA on it)

**Background:** The Hubble sequence (Hubble 1926) runs from ellipticals (E) through lenticulars (S0) to spirals (Sa→Sd) and irregulars. The numerical T‑type of de Vaucouleurs (1959; RC3, de Vaucouleurs et al. 1991) turns this into a number from −5 (E) to 10 (Irr). Many physical properties change smoothly along this sequence: gas fraction, star formation rate, colour, age, mass and bulge-to-total ratio (Roberts & Haynes 1994, ARA&A 32, 115; Kennicutt 1998, ARA&A 36, 189; Blanton & Moustakas 2009, ARA&A 47, 159). The spectra also change in a systematic way along it (Kennicutt 1992, ApJS 79, 255). That is why spectra can be used to classify morphology.

```python name=02_catalog.py
# Read the supporting catalogue (FITS table) and keep only what we need
cat = Table.read(MORPH_FILE).to_pandas()

# FITS strings sometimes arrive as bytes (b'manga-...'): convert to normal text
for col in ["name", "Type"]:
    cat[col] = cat[col].apply(lambda s: s.decode() if isinstance(s, bytes) else s).str.strip()

# nsa_z = -9999 means "no redshift" -> we cannot go to the rest frame, drop them
cat = cat[cat["nsa_z"] > 0].copy()
# Ttype 11 = "S" (spiral with unknown subtype) is not on the -5..10 scale -> drop it
cat = cat[cat["Ttype"].between(-5, 10)].copy()

# Group the T-types into broader classes so every class has enough galaxies.
# The boundaries follow the usual early-type / early-spiral / late-spiral split
# (e.g. Nair & Abraham 2010, ApJS 186, 427).
def morph_class(t):
    if t <= -3: return "E"
    if t <= -1: return "S0"
    if t <=  2: return "S0a-Sab"
    if t <=  4: return "Sb-Sbc"
    if t <=  7: return "Sc-Sd"
    return "Sdm-Irr"

CLASS_ORDER = ["E", "S0", "S0a-Sab", "Sb-Sbc", "Sc-Sd", "Sdm-Irr"]
cat["morph_class"] = cat["Ttype"].apply(morph_class)

# Extra information we can get from the Type text: is there a bar? (SB = strong bar, SAB = weak bar)
cat["bar"] = np.select([cat["Type"].str.startswith("SB"), cat["Type"].str.startswith("SAB")],
                       ["strong", "weak"], default="none")

# Colours from red (early type) to blue (late type), to match the real galaxy colours
CLASS_COLORS = dict(zip(CLASS_ORDER, sns.color_palette("RdYlBu", len(CLASS_ORDER))))
cat = cat.set_index("name")
print(cat["morph_class"].value_counts().reindex(CLASS_ORDER))
```

## Cell 3 – Read the FITS spectra, shift to the rest frame, normalise

Each 1D spectrum is observed at redshift z. Dividing the wavelength by (1+z) puts every galaxy on the same physical wavelength scale, so the same line (for example Hα 6563 Å) lands at the same pixel for every galaxy. This is needed before stacking spectra or comparing them. Flux density per Å is multiplied by (1+z) to keep the energy correct. Each spectrum is then divided by its median flux at 5450–5550 Å, a line-free window. Galaxies have very different brightnesses, and this way we compare spectral *shapes*, not brightness.

```python name=03_read_spectra.py
# Common rest-frame wavelength grid.
# MaNGA covers ~3600-10300 A observed. With z up to ~0.15 the range that ALL galaxies
# share in the rest frame is about 3650-8900 A (same range STARLIGHT used: 3400-8900).
GRID = np.arange(3650.0, 8900.0, 1.0)   # 1 A steps
NORM_WINDOW = (5450, 5550)              # line-free window used for normalisation

def get_col(tab, wanted):
    """Find a column without caring about upper/lower case (FLUX_FoV vs flux_fov)."""
    for c in tab.colnames:
        if c.lower() == wanted.lower():
            return np.asarray(tab[c], dtype=float)
    raise KeyError(wanted)

def read_rest_spectrum(path, z):
    tab  = Table.read(path)                 # astropy reads the binary table of the 1D FITS
    wave = get_col(tab, "WAVE")             # observed wavelength [A]
    flux = get_col(tab, "FLUX_FoV")         # whole field-of-view flux (ERROR_FoV is NOT used)

    wave_rest = wave / (1 + z)              # rest-frame wavelength
    flux_rest = flux * (1 + z)              # keeps the flux per unit wavelength correct

    # Put the spectrum on the common grid (NaN where this galaxy has no data)
    f = np.interp(GRID, wave_rest, flux_rest, left=np.nan, right=np.nan)

    # Normalise by the median flux in the line-free window
    m = (GRID >= NORM_WINDOW[0]) & (GRID <= NORM_WINDOW[1])
    norm = np.nanmedian(f[m])
    return f / norm, norm

names, spectra, norms = [], [], []
for p in tqdm(fits_files, desc="FITS"):
    name = galaxy_name(p)
    if name not in cat.index:               # no morphology or no redshift -> skip
        continue
    try:
        f, n = read_rest_spectrum(p, cat.loc[name, "nsa_z"])
        if not np.isfinite(n) or n <= 0:    # broken or empty spectrum -> skip
            continue
        names.append(name); spectra.append(f.astype(np.float32)); norms.append(n)
    except Exception as e:
        print("problem with", name, e)

SPEC = np.vstack(spectra)                    # shape: (n_galaxies, n_wavelengths)
spec_index = pd.Index(names, name="name")
np.savez_compressed(f"{SAVE_DIR}/rest_spectra.npz", spec=SPEC, grid=GRID, names=names)
print(SPEC.shape)
```

## Cell 4 – Read the STARLIGHT `.out` files

STARLIGHT (Cid Fernandes et al. 2005, MNRAS 358, 363) fits each spectrum as a mix of simple stellar populations (Bruzual & Charlot 2003, MNRAS 344, 1000), here 45 ages × metallicities from `Base.BC03.N`. From each file we take:

* **Population vector `x_j`** (fraction of light from each age and metallicity) and **`Mcor_j`** (fraction of mass). From these come the light- and mass-weighted mean age and metallicity, and the young / intermediate / old light fractions (xY < 10⁸ yr, xI = 10⁸–1.4×10⁹ yr, xO > 1.4×10⁹ yr; Cid Fernandes et al. 2005).
* **`AV`** (stellar dust attenuation, CCM law, Cardelli, Clayton & Mathis 1989), **`vd`** (stellar velocity dispersion, closely linked to bulge mass and early type), and **χ²** and **adev** (fit quality).
* **Model spectrum `f_syn`**. Observed minus model is the **pure emission-line spectrum** (the stellar absorption is removed). This is the standard way to measure Balmer emission correctly (e.g. Asari et al. 2007; Cid Fernandes et al. 2010).

```python name=04_read_starlight.py
def read_starlight(path):
    lines = open(path).read().splitlines()

    def value(tag):
        """Return the number at the start of the first line that contains `tag`."""
        for ln in lines:
            if tag in ln:
                return float(ln.split()[0])
        return np.nan

    res = dict(
        sl_chi2   = value("[chi2/Nl_eff]"),
        sl_adev   = value("[adev (%)]"),
        sl_SN     = value("[S/N in S/N window]"),
        sl_v0     = value("[v0_min"),           # first match = best model
        sl_vd     = value("[vd_min"),           # velocity dispersion [km/s]
        sl_AV     = value("[AV_min"),           # stellar extinction [mag]
        fobs_norm = value("[fobs_norm"),
    )
    n_base = int(value("[N_base]"))

    # --- population table: j, x_j, Mini_j, Mcor_j, age_j, Z_j ...
    start = next(i for i, ln in enumerate(lines) if ln.startswith("# j") and "Mini_j" in ln) + 1
    pop = np.array([ln.split()[:6] for ln in lines[start:start + n_base]], dtype=float)
    x, mcor, age, Z = pop[:, 1], pop[:, 3], pop[:, 4], pop[:, 5]

    # light-weighted (x) and mass-weighted (Mcor) mean log age and metallicity
    res["logage_L"] = np.sum(x * np.log10(age)) / np.sum(x)
    res["logage_M"] = np.sum(mcor * np.log10(age)) / np.sum(mcor)
    res["logZ_L"]   = np.sum(x * np.log10(Z / 0.02)) / np.sum(x)       # relative to solar (Z=0.02)
    res["logZ_M"]   = np.sum(mcor * np.log10(Z / 0.02)) / np.sum(mcor)
    # young / intermediate / old light fractions (Cid Fernandes et al. 2005)
    res["xY"] = x[age < 1e8].sum()   / x.sum()
    res["xI"] = x[(age >= 1e8) & (age <= 1.4e9)].sum() / x.sum()
    res["xO"] = x[age > 1.4e9].sum() / x.sum()

    # --- synthetic spectrum block: l_obs  f_obs  f_syn  wei  (already in the rest frame)
    s = next(i for i, ln in enumerate(lines) if "## Synthetic spectrum" in ln)
    n = int(lines[s + 1].split()[0])
    spec = np.array([ln.split()[:4] for ln in lines[s + 2:s + 2 + n]], dtype=float)
    return res, spec[:, 0], spec[:, 1], spec[:, 2]       # info, wavelength, f_obs, f_syn
```

## Cell 5 – Emission lines from the STARLIGHT residual

Each line flux is the integral of (f_obs − f_syn) over a ±7 Å window. The noise comes from the residual near the line, after masking every known line. Lines are used only with **S/N ≥ 3**, the usual cut for BPT work (e.g. Kauffmann et al. 2003). Equivalent width is line flux divided by the model continuum, positive for emission.

```python name=05_emission_lines.py
# Rest-frame air wavelengths [A]
EMISSION = {
    "OII3727": 3727.4, "Hd": 4101.7, "Hg": 4340.5, "Hb": 4861.3,
    "OIII4959": 4958.9, "OIII5007": 5006.8, "OI6300": 6300.3,
    "NII6548": 6548.1, "Ha": 6562.8, "NII6584": 6583.5,
    "SII6717": 6716.4, "SII6731": 6730.8,
}
HALF_WIDTH = {"OII3727": 9}          # [OII] is a doublet -> slightly wider window; others use 7 A

def measure_lines(wl, f_obs, f_syn):
    good = f_obs > 0                              # STARLIGHT writes 0 where there is no data
    wl, f_obs, f_syn = wl[good], f_obs[good], f_syn[good]
    resid = f_obs - f_syn                         # emission-only spectrum

    # mask all lines (+-20 A) so the noise estimate only uses line-free pixels
    line_free = np.ones_like(wl, bool)
    for c in EMISSION.values():
        line_free &= np.abs(wl - c) > 20

    out = {}
    for name, c in EMISSION.items():
        hw  = HALF_WIDTH.get(name, 7)
        win = np.abs(wl - c) <= hw
        if win.sum() < 3:                          # line outside the covered range
            out.update({f"F_{name}": np.nan, f"SN_{name}": np.nan, f"EW_{name}": np.nan}); continue
        flux = trapezoid(resid[win], wl[win])
        # local noise: sigma-clipped std of the line-free residual within +-150 A
        near = line_free & (np.abs(wl - c) < 150)
        _, _, sig = sigma_clipped_stats(resid[near]) if near.sum() > 10 else (0, 0, np.nan)
        err  = sig * np.sqrt(win.sum()) * np.median(np.diff(wl))
        cont = np.median(f_syn[win])
        out[f"F_{name}"]  = flux
        out[f"SN_{name}"] = flux / err if err > 0 else np.nan
        out[f"EW_{name}"] = flux / cont            # [A], positive = emission
    return out
```

## Cell 6 – Absorption indices from the rest-frame FITS spectra

* **Dn4000** (Balogh et al. 1999): ratio of ⟨F_ν⟩ at 4000–4100 to 3850–3950 Å. It is the classic age indicator: old, metal-rich populations (E/S0) have a strong 4000 Å break (Bruzual 1983; Kauffmann et al. 2003a, MNRAS 341, 33).
* **Lick indices** (Worthey 1994; Worthey & Ottaviani 1997): HδA, HγA and Hβ trace younger or intermediate-age stars. Mgb, Fe5270 and Fe5335 trace metallicity and α/Fe. NaD traces old stars and the cold interstellar medium.

They are measured at MaNGA resolution, not the original Lick resolution. The values are fine for comparing classes with each other but can't be compared directly with published Lick tables. Hβ and Hδ can be partly filled in by emission. If that matters, measure the same indices on `f_syn` instead.

```python name=06_absorption_indices.py
# (blue continuum, index band, red continuum) in A
LICK = {
    "HdA":    ((4041.600, 4079.750), (4083.500, 4122.250), (4128.500, 4161.000)),
    "HgA":    ((4283.500, 4319.750), (4319.750, 4363.500), (4367.250, 4419.750)),
    "Hb_abs": ((4827.875, 4847.875), (4847.875, 4876.625), (4876.625, 4891.625)),
    "Mgb":    ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
    "Fe5270": ((5233.150, 5248.150), (5245.650, 5285.650), (5285.650, 5318.150)),
    "Fe5335": ((5304.625, 5315.875), (5312.125, 5352.125), (5353.375, 5363.375)),
    "NaD":    ((5860.625, 5875.625), (5876.875, 5909.375), (5922.125, 5948.125)),
}

def band_mean(wl, fl, band):
    m = (wl >= band[0]) & (wl <= band[1])
    return np.nanmean(fl[m])

def lick_index(wl, fl, blue, center, red):
    """Equivalent width [A]: integral of (1 - F/F_continuum), the continuum being a
    straight line through the mean flux of the blue and red side bands."""
    fb, fr = band_mean(wl, fl, blue), band_mean(wl, fl, red)
    xb, xr = np.mean(blue), np.mean(red)
    m = (wl >= center[0]) & (wl <= center[1])
    cont = fb + (fr - fb) * (wl[m] - xb) / (xr - xb)
    return trapezoid(1 - fl[m] / cont, wl[m])

def dn4000(wl, fl):
    fnu = fl * wl**2                       # F_lambda -> F_nu (constant factor cancels)
    return band_mean(wl, fnu, (4000, 4100)) / band_mean(wl, fnu, (3850, 3950))

def absorption_features(fl):
    out = {"Dn4000": dn4000(GRID, fl)}
    for k, bands in LICK.items():
        out[k] = lick_index(GRID, fl, *bands)
    # simple continuum colour: blue/red flux ratio (late types are bluer, Strateva et al. 2001)
    out["color_4500_6500"] = band_mean(GRID, fl, (4450, 4550)) / band_mean(GRID, fl, (6450, 6550))
    return out
```

## Cell 7 – Build the feature table

```python name=07_build_features.py
sl_lookup = {galaxy_name(p): p for p in out_files}
rows = []
for i, name in enumerate(tqdm(spec_index, desc="features")):
    row = {"name": name, **absorption_features(SPEC[i])}
    if name in sl_lookup:
        try:
            info, wl, fo, fs = read_starlight(sl_lookup[name])
            row.update(info)
            row.update(measure_lines(wl, fo, fs))
        except Exception as e:
            print("STARLIGHT problem:", name, e)
    rows.append(row)

feat = pd.DataFrame(rows).set_index("name").join(cat, how="inner")

# ---------- derived diagnostics ----------
ok = lambda *ls: np.all([feat[f"SN_{l}"] >= 3 for l in ls], axis=0)   # S/N >= 3 in all lines

feat["log_NII_Ha"]  = np.where(ok("NII6584", "Ha"),  np.log10(feat.F_NII6584 / feat.F_Ha), np.nan)
feat["log_OIII_Hb"] = np.where(ok("OIII5007", "Hb"), np.log10(feat.F_OIII5007 / feat.F_Hb), np.nan)
feat["log_SII_Ha"]  = np.where(ok("SII6717", "SII6731", "Ha"),
                               np.log10((feat.F_SII6717 + feat.F_SII6731) / feat.F_Ha), np.nan)
feat["log_OI_Ha"]   = np.where(ok("OI6300", "Ha"), np.log10(feat.F_OI6300 / feat.F_Ha), np.nan)

# Balmer decrement: intrinsic Ha/Hb = 2.86 for Case B, T=10^4 K (Osterbrock & Ferland 2006).
# E(B-V)_gas = 2.5/(k_Hb - k_Ha) * log10((Ha/Hb)/2.86), with k_Hb=3.61, k_Ha=2.53 (Cardelli et al. 1989).
feat["Ha_Hb"]   = np.where(ok("Ha", "Hb"), feat.F_Ha / feat.F_Hb, np.nan)
feat["AV_gas"]  = 3.1 * 2.5 / (3.61 - 2.53) * np.log10(np.clip(feat.Ha_Hb, 2.86, None) / 2.86)
feat["SII_ratio"] = np.where(ok("SII6717", "SII6731"), feat.F_SII6717 / feat.F_SII6731, np.nan)  # electron density
feat["log_EW_Ha"] = np.log10(feat.EW_Ha.clip(lower=0.01))

feat.to_csv(f"{SAVE_DIR}/features.csv")
print(feat.shape)
```

## Cell 8 – Data quality and catalogue properties per class

```python name=08_eda_overview.py
# 1) How many galaxies per class? (class imbalance changes how classifiers must be trained)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
feat["morph_class"].value_counts().reindex(CLASS_ORDER).plot.bar(ax=ax[0], color=[CLASS_COLORS[c] for c in CLASS_ORDER])
ax[0].set_title("Galaxies per morphological class")
feat["Ttype"].value_counts().sort_index().plot.bar(ax=ax[1], color="gray"); ax[1].set_title("Galaxies per T-type")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/counts.png", dpi=120); plt.show()

# 2) Missing values: which features exist for which galaxies
miss = feat.isna().mean().sort_values(ascending=False)
print("Fraction missing (top 20):\n", miss.head(20))

# 3) STARLIGHT fit quality: discard bad fits from the analysis
print(feat[["sl_chi2", "sl_adev", "sl_SN"]].describe())
feat["good_fit"] = (feat.sl_adev < 10) & (feat.sl_SN > 5)   # adjust if needed

# 4) Catalogue properties by class.
# Mass rises toward early types (Roberts & Haynes 1994). Redshift and size show selection effects.
# Inclination matters: edge-on disks are redder and dustier, so spectra of late types change with inclination.
cat_cols = ["nsa_z", "LogMass", "incl", "PETRO_TH90"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, c in zip(axes, cat_cols):
    sns.boxplot(data=feat, x="morph_class", y=c, order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax, showfliers=False)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/catalog_props.png", dpi=120); plt.show()

# 5) Bars by class (barred fraction peaks in early/intermediate spirals; e.g. Masters et al. 2011)
pd.crosstab(feat.morph_class, feat.bar, normalize="index").reindex(CLASS_ORDER).plot.bar(stacked=True, figsize=(8, 4))
plt.title("Bar fraction per class"); plt.show()
```

## Cell 9 – Feature distributions per class

```python name=09_feature_distributions.py
FEATURES = [
    # stellar populations (STARLIGHT)
    "logage_L", "logage_M", "logZ_L", "logZ_M", "xY", "xI", "xO", "sl_AV", "sl_vd",
    # absorption features
    "Dn4000", "HdA", "HgA", "Hb_abs", "Mgb", "Fe5270", "Fe5335", "NaD", "color_4500_6500",
    # emission features
    "log_EW_Ha", "EW_OII3727", "EW_Hb", "log_NII_Ha", "log_OIII_Hb", "log_SII_Ha", "Ha_Hb", "AV_gas",
    # catalogue
    "LogMass",
]
d = feat[feat.good_fit]

n = len(FEATURES); ncol = 4
fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(20, 3.2 * np.ceil(n / ncol)))
for ax, f in zip(axes.flat, FEATURES):
    sns.boxplot(data=d, x="morph_class", y=f, order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax, showfliers=False)
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=45)
for ax in axes.flat[n:]: ax.remove()
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/feature_boxplots.png", dpi=110); plt.show()

# Correlations between features: strongly correlated features carry the same information
plt.figure(figsize=(14, 12))
sns.heatmap(d[FEATURES + ["Ttype"]].corr(method="spearman"), cmap="coolwarm", center=0, annot=False)
plt.title("Spearman correlation"); plt.show()
```

## Cell 10 – Which features best separate the types?

Three methods are used together, because each one alone can mislead:
* **Kruskal–Wallis ε²**: a non-parametric effect size telling how differently the classes are distributed. It doesn't assume the data are normal, which suits skewed quantities like EWs.
* **Spearman ρ with T‑type**: whether the feature changes steadily along the Hubble sequence.
* **Mutual information** and **random-forest permutation importance**: these also catch non-linear effects and features that only matter in combination with others.

```python name=10_feature_ranking.py
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

rank = []
for f in FEATURES:
    groups = [g[f].dropna().values for _, g in d.groupby("morph_class") if g[f].notna().sum() > 5]
    H, p = kruskal(*groups)
    n_obs = sum(len(g) for g in groups)
    rho, _ = spearmanr(d[f], d["Ttype"], nan_policy="omit")
    rank.append({"feature": f, "epsilon2": H / (n_obs - 1), "p_value": p, "spearman_Ttype": rho,
                 "coverage": d[f].notna().mean()})
rank = pd.DataFrame(rank).set_index("feature")

# Machine-learning ranking (missing values -> median; balanced classes)
X = d[FEATURES]; y = d["morph_class"]
Xi = SimpleImputer(strategy="median").fit_transform(X)
rank["mutual_info"] = mutual_info_classif(Xi, y, random_state=0)

model = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1))
print("5-fold balanced accuracy:", cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean().round(3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model.fit(Xtr, ytr)
pi = permutation_importance(model, Xte, yte, n_repeats=10, scoring="balanced_accuracy", random_state=0)
rank["perm_importance"] = pi.importances_mean

rank = rank.sort_values("epsilon2", ascending=False)
rank.to_csv(f"{SAVE_DIR}/feature_ranking.csv")
display(rank.round(3))

rank[["epsilon2", "mutual_info", "perm_importance"]].plot.barh(subplots=True, layout=(1, 3), figsize=(18, 8),
                                                               sharey=True, legend=False)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
```

**What to expect (from the literature):** Dn4000, light-weighted age, xY, EW(Hα), HδA, σ (`sl_vd`), stellar mass and continuum colour usually separate early from late types best (Kauffmann et al. 2003a; Kennicutt 1992). Telling neighbouring spiral types apart (Sb vs Sc) from one integrated spectrum is much harder. Expect the confusion to be mostly between adjacent classes.

## Cell 11 – Spectral PCA (data-driven features)

```python name=11_pca.py
from sklearn.decomposition import PCA

# PCA needs complete rows -> fill the few NaNs with the mean spectrum value at that wavelength
S = SPEC[spec_index.isin(d.index)]
S = np.where(np.isnan(S), np.nanmean(S, axis=0), S)
lab = d.loc[spec_index[spec_index.isin(d.index)], "morph_class"].values

pca = PCA(n_components=10).fit(S)
P = pca.transform(S)
print("explained variance:", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for c in CLASS_ORDER:
    m = lab == c
    ax[0].scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(markerscale=3)
for k in range(3):
    ax[1].plot(GRID, pca.components_[k] + 0.05 * k, lw=0.7, label=f"PC{k+1}")
ax[1].legend(); ax[1].set_xlabel("Rest wavelength [Å]"); ax[1].set_title("Eigenspectra")
plt.show()
# PC1 usually follows the young-vs-old continuum shape plus emission-line strength (Yip et al. 2004, AJ 128, 585)
```

## Cell 12 – BPT, WHAN and Balmer diagnostics per class

* **BPT** (Baldwin, Phillips & Terlevich 1981; Veilleux & Osterbrock 1987): [OIII]/Hβ vs [NII]/Hα, [SII]/Hα and [OI]/Hα. The dividing lines are Kauffmann et al. 2003 (pure star-forming), Kewley et al. 2001 (maximum starburst), Kewley et al. 2006 and Schawinski et al. 2007 (Seyfert vs LINER).
* **WHAN** (Cid Fernandes et al. 2011, MNRAS 413, 1687): needs only Hα and [NII]. It recovers weak-line early types, many of which are "retired" galaxies whose gas is ionised by old hot stars, not by an AGN (Stasińska et al. 2008).

```python name=12_bpt_whan.py
x = np.linspace(-2, 1, 300)
kauff = lambda x: 0.61 / (x - 0.05) + 1.30          # Kauffmann+03
kew01 = lambda x: 0.61 / (x - 0.47) + 1.19          # Kewley+01

def bpt_class(nii, oiii):
    if np.isnan(nii) or np.isnan(oiii): return "No/weak lines"
    if nii < 0.05 and oiii < kauff(nii): return "Star-forming"
    if nii < 0.47 and oiii < kew01(nii): return "Composite"
    return "Seyfert" if oiii > 1.05 * nii + 0.45 else "LINER"   # Schawinski+07

def whan_class(nii, ew_ha, ew_nii):
    if ew_ha < 0.5 and ew_nii < 0.5: return "Passive"
    if ew_ha < 3:                    return "Retired"
    if np.isnan(nii):                return "Unclassified"
    if nii < -0.4:                   return "SF"
    return "sAGN" if ew_ha > 6 else "wAGN"

feat["BPT"]  = [bpt_class(a, b) for a, b in zip(feat.log_NII_Ha, feat.log_OIII_Hb)]
feat["WHAN"] = [whan_class(a, b, c) for a, b, c in zip(feat.log_NII_Ha, feat.EW_Ha, feat.EW_NII6584)]
d = feat[feat.good_fit]

# --- [NII] BPT, one panel per class
fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
for ax, c in zip(axes.flat, CLASS_ORDER):
    g = d[d.morph_class == c]
    ax.scatter(g.log_NII_Ha, g.log_OIII_Hb, s=8, alpha=0.6, color=CLASS_COLORS[c])
    ax.plot(x[x < 0.05], kauff(x[x < 0.05]), "k--", lw=1, label="Kauffmann+03")
    ax.plot(x[x < 0.47], kew01(x[x < 0.47]), "k-",  lw=1, label="Kewley+01")
    ax.plot(np.linspace(-0.18, 1, 50), 1.05 * np.linspace(-0.18, 1, 50) + 0.45, "k:", lw=1, label="Schawinski+07")
    ax.set_title(f"{c}  (N with lines = {g.log_OIII_Hb.notna().sum()}/{len(g)})")
    ax.set_xlim(-1.8, 0.8); ax.set_ylim(-1.5, 1.5)
axes[0, 0].legend(fontsize=8)
fig.supxlabel("log [NII]6584/Hα"); fig.supylabel("log [OIII]5007/Hβ")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/BPT_NII_by_class.png", dpi=120); plt.show()

# --- [SII] and [OI] BPT (Kewley+06 lines), all classes coloured
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for c in CLASS_ORDER:
    g = d[d.morph_class == c]
    ax[0].scatter(g.log_SII_Ha, g.log_OIII_Hb, s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
    ax[1].scatter(g.log_OI_Ha,  g.log_OIII_Hb, s=6, alpha=0.5, color=CLASS_COLORS[c])
xs = np.linspace(-1.5, 0.3, 100); ax[0].plot(xs, 0.72 / (xs - 0.32) + 1.30, "k-")
ax[0].plot(np.linspace(-0.31, 0.5), 1.89 * np.linspace(-0.31, 0.5) + 0.76, "k:")
xo = np.linspace(-2.5, -0.6, 100); ax[1].plot(xo, 0.73 / (xo + 0.59) + 1.33, "k-")
ax[1].plot(np.linspace(-1.1, 0.5), 1.18 * np.linspace(-1.1, 0.5) + 1.30, "k:")
ax[0].set_xlabel("log [SII]/Hα"); ax[1].set_xlabel("log [OI]/Hα"); ax[0].set_ylabel("log [OIII]/Hβ")
for a in ax: a.set_ylim(-1.5, 1.5)
ax[0].legend(markerscale=3); plt.show()

# --- WHAN diagram
plt.figure(figsize=(8, 6))
for c in CLASS_ORDER:
    g = d[d.morph_class == c]
    plt.scatter(g.log_NII_Ha, g.log_EW_Ha, s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
plt.axhline(np.log10(3), c="k"); plt.axhline(np.log10(6), c="k", ls=":", xmin=0.5); plt.axvline(-0.4, c="k")
plt.xlabel("log [NII]/Hα"); plt.ylabel("log EW(Hα) [Å]"); plt.legend(markerscale=3); plt.title("WHAN"); plt.show()

# --- How the BPT/WHAN classes are spread within each morphology (main result table)
for col in ["BPT", "WHAN"]:
    ct = pd.crosstab(d.morph_class, d[col], normalize="index").reindex(CLASS_ORDER)
    display((100 * ct).round(1))
    ct.plot.bar(stacked=True, figsize=(10, 4), colormap="tab10"); plt.title(f"{col} class fraction per morphology")
    plt.legend(bbox_to_anchor=(1, 1)); plt.show()

# --- Balmer decrement, dust and age-sensitive absorption per class
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=d, x="morph_class", y="Ha_Hb", order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax[0], showfliers=False)
ax[0].axhline(2.86, c="k", ls="--", label="Case B (2.86)"); ax[0].legend()
sns.boxplot(data=d, x="morph_class", y="AV_gas", order=CLASS_ORDER, palette=CLASS_COLORS, ax=ax[1], showfliers=False)
for c in CLASS_ORDER:                               # Dn4000-HδA plane (Kauffmann et al. 2003a)
    g = d[d.morph_class == c]
    ax[2].scatter(g.Dn4000, g.HdA, s=5, alpha=0.5, color=CLASS_COLORS[c], label=c)
ax[2].set_xlabel("Dn4000"); ax[2].set_ylabel("HδA [Å]"); ax[2].legend(markerscale=3)
plt.tight_layout(); plt.show()

# --- Summary table of line diagnostics per class (medians)
diag = ["EW_Ha", "EW_Hb", "EW_OII3727", "EW_OIII5007", "Ha_Hb", "AV_gas", "SII_ratio",
        "Dn4000", "HdA", "HgA", "Hb_abs", "Mgb", "Fe5270", "NaD"]
display(d.groupby("morph_class")[diag].median().reindex(CLASS_ORDER).round(2))
feat.to_csv(f"{SAVE_DIR}/features_with_diagnostics.csv")
```

## Cell 13 – All spectra grouped by morphological class

```python name=13_plot_all_spectra.py
# Main lines to mark on the plots (emission = blue, absorption = red)
MARK_EM  = {"[OII]": 3727, "Hβ": 4861, "[OIII]": 5007, "[OI]": 6300, "Hα+[NII]": 6563, "[SII]": 6724}
MARK_ABS = {"CaII K": 3934, "CaII H": 3969, "G band": 4304, "Mgb": 5175, "NaD": 5893}

def mark_lines(ax, ymax):
    for lab, w in MARK_EM.items():
        ax.axvline(w, c="tab:blue", lw=0.5, ls=":"); ax.text(w, ymax, lab, fontsize=7, rotation=90, va="top", color="tab:blue")
    for lab, w in MARK_ABS.items():
        ax.axvline(w, c="tab:red", lw=0.5, ls=":"); ax.text(w, ymax, lab, fontsize=7, rotation=90, va="top", color="tab:red")

cls_of = feat["morph_class"].reindex(spec_index).values
fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(16, 3.2 * len(CLASS_ORDER)), sharex=True)
for ax, c in zip(axes, CLASS_ORDER):
    S_c = SPEC[cls_of == c]
    for s in S_c:                                    # every galaxy: thin, transparent line
        ax.plot(GRID, s, color=CLASS_COLORS[c], lw=0.3, alpha=0.05, rasterized=True)
    ax.plot(GRID, np.nanmedian(S_c, axis=0), "k", lw=1, label="median")
    ax.set_ylim(0, 3); ax.set_ylabel("F / F(5500)")
    ax.set_title(f"{c}  (N = {len(S_c)})", loc="left"); mark_lines(ax, 2.95)
axes[-1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/all_spectra_by_class.png", dpi=110); plt.show()
```

## Cell 14 – Mean spectrum of every class in one graph

```python name=14_mean_spectra.py
fig, ax = plt.subplots(2, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
mean_spec = {}
for c in CLASS_ORDER:
    S_c = SPEC[cls_of == c]
    mean_spec[c] = np.nanmean(S_c, axis=0)
    ax[0].plot(GRID, mean_spec[c], color=CLASS_COLORS[c], lw=1, label=f"{c} (N={len(S_c)})")
ax[0].set_ylabel("Mean F / F(5500)"); ax[0].legend(); mark_lines(ax[0], ax[0].get_ylim()[1])

# Bottom panel: each class divided by the elliptical mean -> shows only the differences
for c in CLASS_ORDER[1:]:
    ax[1].plot(GRID, mean_spec[c] / mean_spec["E"], color=CLASS_COLORS[c], lw=0.8, label=f"{c} / E")
ax[1].axhline(1, c="k", lw=0.5); ax[1].set_ylabel("Ratio to E"); ax[1].legend(ncol=5)
ax[1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/mean_spectra.png", dpi=130); plt.show()

pd.DataFrame(mean_spec, index=pd.Index(GRID, name="wave_rest")).to_csv(f"{SAVE_DIR}/mean_spectra.csv")
```

**What to look for in the mean spectra** (Kennicutt 1992): going from E to late spirals, the continuum gets bluer, the 4000 Å break and the Ca II H+K, G‑band, Mgb and NaD absorptions get weaker, and the Balmer absorption (Hδ, Hγ) and the emission lines ([OII], Hβ, [OIII], Hα, [NII], [SII]) get stronger. S0 galaxies sit between E and Sa: they have old continua like ellipticals, with weak LINER-like or "retired" emission.

---

### Notes
* **Speed:** for thousands of files, Cells 3 and 7 are the slow part. The results are saved to Drive (`rest_spectra.npz`, `features.csv`), so you can reload them instead of re-reading everything.
* **Changing the grouping:** if you want finer classes, change `morph_class()` or use `Ttype` directly. Classes with fewer than about 30 galaxies give noisy results.
* **Biases to keep in mind:**
  * `FLUX_FoV` integrates the whole bundle. Early types are bulge-dominated, so their spectra are dominated by the old centre.
  * The bundle covers different physical sizes at different redshifts (aperture effects).
  * For late types, inclination increases dust attenuation and reddening.
  * Check that the redshift and mass distributions are similar across classes before reading the feature ranking as a morphology effect.

### References
Asari et al. 2007, MNRAS 381, 263 · Baldwin, Phillips & Terlevich 1981, PASP 93, 5 · Balogh et al. 1999, ApJ 527, 54 · Blanton & Moustakas 2009, ARA&A 47, 159 · Bruzual 1983, ApJ 273, 105 · Bruzual & Charlot 2003, MNRAS 344, 1000 · Bundy et al. 2015, ApJ 798, 7 (MaNGA) · Cardelli, Clayton & Mathis 1989, ApJ 345, 245 · Cid Fernandes et al. 2005, MNRAS 358, 363; 2010, MNRAS 403, 1036; 2011, MNRAS 413, 1687 · de Vaucouleurs 1959, Handbuch der Physik 53, 275; de Vaucouleurs et al. 1991 (RC3) · Hubble 1926, ApJ 64, 321 · Kauffmann et al. 2003a, MNRAS 341, 33; 2003b, MNRAS 346, 1055 · Kennicutt 1992, ApJS 79, 255; 1998, ARA&A 36, 189 · Kewley et al. 2001, ApJ 556, 121; 2006, MNRAS 372, 961 · Masters et al. 2011, MNRAS 411, 2026 · Nair & Abraham 2010, ApJS 186, 427 · Osterbrock & Ferland 2006, *Astrophysics of Gaseous Nebulae and AGN* · Roberts & Haynes 1994, ARA&A 32, 115 · Schawinski et al. 2007, MNRAS 382, 1415 · Stasińska et al. 2008, MNRAS 391, L29 · Strateva et al. 2001, AJ 122, 1861 · Vázquez‑Mata et al. 2022, MNRAS 512, 2222 · Veilleux & Osterbrock 1987, ApJS 63, 295 · Worthey 1994, ApJS 95, 107 · Worthey & Ottaviani 1997, ApJS 111, 377 · Yip et al. 2004, AJ 128, 585.

*System: User retried message*

## @maria-rabelero-mx_nestle

Do EDA to thousands of .fits.gz and .out files to find information and features  that will help me classify galaxies by morphological type and which features best separate the types

Do spectral line diagnostic (BPT, balmer, emission and absorption lines, etc.) per ttype

Use the Emission lines : OII
NeIII
Hδ
Hγ
OIII
Hβ
OIII
HeI
OI
NII
Hα
NII
SII
ArIII

Use this avsotion lines: K
H
G
Mg
NaD
CaII

Graph all the spectrums group by ttype

Graph the mean spectrum of all ttypes in single graph

Each .fits.gz file is of a galaxy. The galaxies are of all morphology types. The .fits.gz files have the same name pattern and columns as these csv files, these csv have Gaps in the spectra but the .fits.gz files do not.  Use wave and flux_fov, ignore ERROR_FoV is incorrect. Wave needs to be converted to rest-frame wavelength
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D.csv

These are some of the .out files. They were obtained using http://www.starlight.ufsc.br/downloads/, and the wave and flux_fov from the .fits.gz files.
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D-out.txt

I also have a supporting .fits file that I don't want to do EDA on. Use its information to complement the EDA. The supporting .fits file has the same columns as this csv file.
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/morfologia.csv

I will use google colab, the .fits and .out files are in google drive

Use astronomy libraries

Document as much as you can about galaxy morphology with reference

make the code as simple as possible and human readable as possible, explain what the code is doing and why, using inline comments and codeblocks. 


## Copilot

I've rewritten the notebook for your new setup. It reads `.fits.gz` files and works per T‑type instead of grouped classes. It measures only the emission and absorption lines you listed, and makes all the requested plots. Paste each block into its own Colab cell and run them in order.

I haven't run it. I wrote it from your five sample CSVs, the `.out` files and `morfologia.csv`. Run it on a few hundred galaxies first and check the paths. Things I checked or assumed:

* **Rest frame.** The FITS wavelengths are divided by (1+z) using `nsa_z`. The STARLIGHT `.out` spectra are already in the rest frame: for `manga-10001-1901` (z = 0.02575), the first observed point at 3622 Å becomes 3531 Å, exactly where `f_obs` stops being 0. So the `.out` wavelengths are not shifted again.
* **Galaxies dropped.** Rows with `nsa_z = -9999` have no redshift, so they can't be shifted to the rest frame. Rows with `Ttype = 11` ("S", spiral of unknown subtype) are off the −5…10 scale and are skipped too.
* **Two OIII entries.** You listed OIII twice, so I used **[OIII] 4363** (temperature-sensitive) and **[OIII] 5007** (used in BPT). I also measure **[OIII] 4959**, because it is cheap and checks the 5007 measurement (the 5007/4959 ratio should be about 3).
* **"CaII" absorption.** K and H are already Ca II lines, so I read this as the **Ca II near‑infrared triplet** (8498, 8542, 8662 Å).
* **Line overlaps.**
  * Ca II H (3968 Å) overlaps Hε and [NeIII] 3967, so it is measured on the STARLIGHT stellar model (`f_syn`). The model contains no emission lines.
  * HeI 5876 sits next to NaD (5890/5896). HeI is measured on the emission residual, and NaD uses Lick bands that leave HeI out.

---

## Cell 1 – Setup

```python name=01_setup.py
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
```

## Cell 2 – Supporting catalogue (used only as labels and extra information, no EDA on it)

**Background: the T‑type.** Hubble (1926) ordered galaxies into ellipticals (E), lenticulars (S0) and spirals (Sa→Sd). de Vaucouleurs (1959) extended this into the numerical **T‑type**: −5 E, −3/−2 S0, 0 S0/a, 1 Sa, 2 Sab, 3 Sb, 4 Sbc, 5 Sc, 6 Scd, 7 Sd, 8 Sdm, 9 Sm, 10 Irr (RC3; de Vaucouleurs et al. 1991). Going along the sequence, bulge-to-total ratio, stellar mass, mean stellar age and metallicity go down. Gas fraction, specific star formation rate and blue colour go up (Roberts & Haynes 1994; Kennicutt 1998; Blanton & Moustakas 2009). The integrated spectra change in step with this (Morgan & Mayall 1957; Kennicutt 1992), which is why spectra carry morphological information.

```python name=02_catalog.py
cat = Table.read(MORPH_FILE).to_pandas()

# FITS text columns sometimes come as bytes (b'manga-...'): turn them into normal strings
for col in ["name", "Type"]:
    cat[col] = cat[col].apply(lambda s: s.decode() if isinstance(s, bytes) else s).str.strip()

cat = cat[cat["nsa_z"] > 0]                      # -9999 = no redshift -> cannot de-redshift
cat = cat[cat["Ttype"].between(-5, 10)].copy()   # 11 = 'S' (no subtype) is not on the scale
cat["Ttype"] = cat["Ttype"].astype(int)

# Bar information from the Type text: SB = strong bar, SAB = weak bar
cat["bar"] = np.select([cat.Type.str.startswith("SB"), cat.Type.str.startswith("SAB")],
                       ["strong", "weak"], default="none")
cat = cat.set_index("name")

# One colour per T-type: red (early, old, red galaxies) -> blue (late, young, blue galaxies)
TTYPES = sorted(cat["Ttype"].unique())
TT_COLORS = dict(zip(TTYPES, sns.color_palette("RdYlBu", len(TTYPES))))
TT_NAMES = {-5:"E", -4:"E+", -3:"S0-", -2:"S0", -1:"S0+", 0:"S0a", 1:"Sa", 2:"Sab", 3:"Sb",
            4:"Sbc", 5:"Sc", 6:"Scd", 7:"Sd", 8:"Sdm", 9:"Sm", 10:"Irr"}
print(cat["Ttype"].value_counts().sort_index())
```

## Cell 3 – Read the `.fits.gz` spectra, convert to rest frame, normalise

* **Rest frame:** λ_rest = λ_obs/(1+z). After this, each line sits at the same wavelength in every galaxy, so spectra can be stacked and compared. F_λ is multiplied by (1+z) to keep the flux per unit wavelength correct.
* **Normalisation:** each spectrum is divided by its median flux at 5450–5550 Å, a window with no strong lines. This removes differences in distance and brightness, so only the spectral *shape* is compared.

```python name=03_read_spectra.py
# Common rest-frame grid. MaNGA covers ~3600-10300 A observed; for z <= 0.15 the
# range shared by all galaxies in the rest frame is ~3650-8900 A (STARLIGHT used 3400-8900).
GRID = np.arange(3650.0, 8900.0, 1.0)
NORM_WINDOW = (5450, 5550)

def get_col(tab, wanted):
    """Find a column ignoring upper/lower case (FLUX_FoV vs flux_fov)."""
    for c in tab.colnames:
        if c.lower() == wanted.lower():
            return np.asarray(tab[c], dtype=float)
    raise KeyError(wanted)

def read_rest_spectrum(path, z):
    tab  = Table.read(path)                  # works for .fits and .fits.gz
    wave = get_col(tab, "WAVE")              # observed wavelength [A]
    flux = get_col(tab, "FLUX_FoV")          # full field-of-view flux; ERROR_FoV is NOT used
    wave_rest = wave / (1 + z)
    flux_rest = flux * (1 + z)
    f = np.interp(GRID, wave_rest, flux_rest, left=np.nan, right=np.nan)   # same grid for all
    m = (GRID >= NORM_WINDOW[0]) & (GRID <= NORM_WINDOW[1])
    norm = np.nanmedian(f[m])
    return f / norm, norm

names, spectra = [], []
for p in tqdm(fits_files, desc="reading spectra"):
    name = galaxy_name(p)
    if name not in cat.index:
        continue
    try:
        f, n = read_rest_spectrum(p, cat.loc[name, "nsa_z"])
        if np.isfinite(n) and n > 0:
            names.append(name); spectra.append(f.astype(np.float32))
    except Exception as e:
        print("problem with", name, e)

SPEC = np.vstack(spectra)                    # (n_galaxies, n_wavelengths)
spec_index = pd.Index(names, name="name")
np.savez_compressed(f"{SAVE_DIR}/rest_spectra.npz", spec=SPEC, grid=GRID, names=names)
print(SPEC.shape)
```

## Cell 4 – Read the STARLIGHT `.out` files

STARLIGHT (Cid Fernandes et al. 2005) fits each spectrum as a sum of 45 simple stellar populations (Bruzual & Charlot 2003), with dust following the CCM law (Cardelli et al. 1989). From each file we take:

* **Stellar population summary:** the light-weighted (`x_j`) and mass-weighted (`Mcor_j`) mean age and metallicity, and the young/intermediate/old light fractions (xY < 10⁸ yr, xI = 10⁸–1.4×10⁹ yr, xO > 1.4×10⁹ yr).
* **AV** (stellar dust), **v_d** (stellar velocity dispersion, a proxy for bulge or dynamical mass), and the fit-quality numbers **χ²**, **adev** and **S/N**.
* **f_syn**, the pure stellar model with no emission lines. Observed minus f_syn gives the **emission-only spectrum**. This matters for Hβ, Hγ and Hδ, which sit inside stellar absorption lines and would be underestimated otherwise (Asari et al. 2007).

```python name=04_read_starlight.py
def read_starlight(path):
    lines = open(path).read().splitlines()

    def value(tag):
        """Number at the start of the first line containing `tag`."""
        for ln in lines:
            if tag in ln:
                return float(ln.split()[0])
        return np.nan

    info = dict(sl_chi2=value("[chi2/Nl_eff]"), sl_adev=value("[adev (%)]"),
                sl_SN=value("[S/N in S/N window]"), sl_vd=value("[vd_min"),
                sl_AV=value("[AV_min"))
    n_base = int(value("[N_base]"))

    # Population table: j  x_j  Mini_j  Mcor_j  age_j  Z_j ...
    i0 = next(i for i, ln in enumerate(lines) if ln.startswith("# j") and "Mini_j" in ln) + 1
    pop = np.array([ln.split()[:6] for ln in lines[i0:i0 + n_base]], dtype=float)
    x, mcor, age, Z = pop[:, 1], pop[:, 3], pop[:, 4], pop[:, 5]
    info["logage_L"] = np.sum(x * np.log10(age)) / x.sum()
    info["logage_M"] = np.sum(mcor * np.log10(age)) / mcor.sum()
    info["logZ_L"]   = np.sum(x * np.log10(Z / 0.02)) / x.sum()      # relative to solar
    info["logZ_M"]   = np.sum(mcor * np.log10(Z / 0.02)) / mcor.sum()
    info["xY"] = x[age < 1e8].sum() / x.sum()
    info["xI"] = x[(age >= 1e8) & (age <= 1.4e9)].sum() / x.sum()
    info["xO"] = x[age > 1.4e9].sum() / x.sum()

    # Spectrum block: l_obs  f_obs  f_syn  wei   (rest frame, normalised by fobs_norm)
    s = next(i for i, ln in enumerate(lines) if "## Synthetic spectrum" in ln)
    n = int(lines[s + 1].split()[0])
    sp = np.array([ln.split()[:4] for ln in lines[s + 2:s + 2 + n]], dtype=float)
    return info, sp[:, 0], sp[:, 1], sp[:, 2]
```

## Cell 5 – Emission lines you asked for

| Line | λ (Å) | What it tells us |
|---|---|---|
| [OII] | 3727 | star formation, also strong in LINERs (Kennicutt 1998) |
| [NeIII] | 3869 | high ionisation (AGN, low-metallicity H II regions) |
| Hδ, Hγ, Hβ, Hα | 4102, 4340, 4861, 6563 | recombination lines: star formation rate and dust (Balmer decrement) |
| [OIII] | 4363 | electron temperature (from the 4363/5007 ratio) |
| [OIII] | 5007 | ionisation level, used in BPT |
| HeI | 5876 | hard ionising photons from young, massive stars |
| [OI] | 6300 | partly ionised zones and shocks; LINER/AGN indicator |
| [NII] | 6548, 6584 | metallicity, used in BPT and WHAN |
| [SII] | 6717, 6731 | electron density (the 6717/6731 ratio), used in BPT |
| [ArIII] | 7136 | ionisation; temperature-based abundances |

```python name=05_emission_lines.py
EMISSION = {   # rest-frame air wavelengths [A]
    "OII3727": 3727.4, "NeIII3869": 3868.8, "Hd": 4101.7, "Hg": 4340.5,
    "OIII4363": 4363.2, "Hb": 4861.3, "OIII4959": 4958.9, "OIII5007": 5006.8,
    "HeI5876": 5875.6, "OI6300": 6300.3, "NII6548": 6548.1, "Ha": 6562.8,
    "NII6584": 6583.5, "SII6717": 6716.4, "SII6731": 6730.8, "ArIII7136": 7135.8,
}
HALF_WIDTH = {"OII3727": 9}     # [OII] is an unresolved doublet -> wider window; others +-7 A

def measure_emission(wl, f_obs, f_syn):
    good = f_obs > 0                          # STARLIGHT writes 0 where there is no data
    wl, f_obs, f_syn = wl[good], f_obs[good], f_syn[good]
    resid = f_obs - f_syn                     # stars removed -> only gas emission left

    # pixels far (>20 A) from every line, used to estimate the noise
    line_free = np.ones_like(wl, bool)
    for c in EMISSION.values():
        line_free &= np.abs(wl - c) > 20

    out = {}
    for name, c in EMISSION.items():
        win = np.abs(wl - c) <= HALF_WIDTH.get(name, 7)
        near = line_free & (np.abs(wl - c) < 150)
        if win.sum() < 3 or near.sum() < 10:      # line outside the covered range
            out.update({f"F_{name}": np.nan, f"SN_{name}": np.nan, f"EW_{name}": np.nan}); continue
        flux = trapezoid(resid[win], wl[win])     # line flux = area under the residual
        _, _, sig = sigma_clipped_stats(resid[near])
        err = sig * np.sqrt(win.sum()) * np.median(np.diff(wl))
        out[f"F_{name}"]  = flux
        out[f"SN_{name}"] = flux / err if err > 0 else np.nan
        out[f"EW_{name}"] = flux / np.median(f_syn[win])   # [A], positive = emission
    return out
```

## Cell 6 – Absorption lines you asked for

Absorption lines come from stellar atmospheres. Old, cool, metal-rich stars produce strong metal lines, and these dominate early types (Worthey 1994; Trager et al. 1998).

| Line | λ (Å) | Meaning | Band definition |
|---|---|---|---|
| Ca II K | 3933.7 | old/metal-rich stars; K/H ratio shows A stars (Rose 1985) | measured on stellar model |
| Ca II H | 3968.5 | blends with Hε and [NeIII] 3967 emission | measured on **f_syn** (stellar model) |
| G band (CH) | 4300 | cool giants (G/K types), old populations | Lick G4300 (Worthey et al. 1994) |
| Mg b | 5175 | metallicity and α/Fe, strong in massive E | Lick Mgb |
| NaD | 5890/5896 | old cool stars plus interstellar gas (outflows) | Lick NaD |
| Ca II triplet | 8498/8542/8662 | red giants, metallicity, IMF | Diaz, Terlevich & Terlevich (1989) |

**Dn4000** (Balogh et al. 1999) is added because the 4000 Å break comes from Ca II H+K and the metal lines next to it. It is one of the best single age indicators (Kauffmann et al. 2003a).

```python name=06_absorption_lines.py
# (blue continuum, line band, red continuum) [A]
# H and K: simple windows chosen here (continuum on both sides of each line).
# Other bands: Lick (Worthey et al. 1994) and DTT89 for the Ca triplet.
ABSORPTION = {
    "CaII_K":  ((3900.0, 3920.0), (3923.7, 3943.7), (3943.7, 3953.7)),
    "CaII_H":  ((3943.7, 3953.7), (3958.5, 3978.5), (3990.0, 4010.0)),
    "G4300":   ((4266.375, 4282.625), (4281.375, 4316.375), (4318.875, 4335.125)),
    "Mgb":     ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
    "NaD":     ((5860.625, 5875.625), (5876.875, 5909.375), (5922.125, 5948.125)),
}
CAT_LINES = [(8483, 8513), (8527, 8557), (8647, 8677)]    # Ca II triplet line bands
CAT_CONT  = [(8447.5, 8462.5), (8842.5, 8857.5)]          # continuum (DTT89)

def band_mean(wl, fl, band):
    m = (wl >= band[0]) & (wl <= band[1])
    return np.nanmean(fl[m])

def ew_index(wl, fl, blue, center, red):
    """Equivalent width [A] = integral of (1 - F/continuum). The continuum is a
    straight line between the mean flux in the blue and red side bands."""
    fb, fr = band_mean(wl, fl, blue), band_mean(wl, fl, red)
    xb, xr = np.mean(blue), np.mean(red)
    m = (wl >= center[0]) & (wl <= center[1])
    cont = fb + (fr - fb) * (wl[m] - xb) / (xr - xb)
    return trapezoid(1 - fl[m] / cont, wl[m])

def absorption_features(wl, fl, fl_model=None):
    out = {}
    for k, bands in ABSORPTION.items():
        # H and K: use the stellar model if available so gas emission does not fill them in
        use = fl_model if (k.startswith("CaII") and fl_model is not None) else fl
        out[k] = ew_index(wl, use, *bands)
    # Ca II triplet: sum of the three lines, continuum is a line through the two side bands
    out["CaT"] = sum(ew_index(wl, fl, CAT_CONT[0], b, CAT_CONT[1]) for b in CAT_LINES)
    # 4000 A break and a simple continuum colour
    fnu = fl * wl**2
    out["Dn4000"] = band_mean(wl, fnu, (4000, 4100)) / band_mean(wl, fnu, (3850, 3950))
    out["color_4500_6500"] = band_mean(wl, fl, (4450, 4550)) / band_mean(wl, fl, (6450, 6550))
    return out
```

## Cell 7 – Build the feature table

```python name=07_build_features.py
sl_lookup = {galaxy_name(p): p for p in out_files}
rows = []
for i, name in enumerate(tqdm(spec_index, desc="features")):
    row = {"name": name}
    model_on_grid = None
    if name in sl_lookup:
        try:
            info, wl, fo, fs = read_starlight(sl_lookup[name])
            row.update(info)
            row.update(measure_emission(wl, fo, fs))
            model_on_grid = np.interp(GRID, wl, fs)          # stellar model for Ca II H/K
        except Exception as e:
            print("STARLIGHT problem:", name, e)
    row.update(absorption_features(GRID, SPEC[i], model_on_grid))
    rows.append(row)

feat = pd.DataFrame(rows).set_index("name").join(cat, how="inner")

# -------- derived diagnostics (only when all lines have S/N >= 3) --------
ok = lambda *ls: np.all([feat[f"SN_{l}"] >= 3 for l in ls], axis=0)
safe_log = lambda a, b, cond: np.where(cond, np.log10(a / b), np.nan)

feat["log_NII_Ha"]   = safe_log(feat.F_NII6584, feat.F_Ha, ok("NII6584", "Ha"))
feat["log_OIII_Hb"]  = safe_log(feat.F_OIII5007, feat.F_Hb, ok("OIII5007", "Hb"))
feat["log_SII_Ha"]   = safe_log(feat.F_SII6717 + feat.F_SII6731, feat.F_Ha, ok("SII6717", "SII6731", "Ha"))
feat["log_OI_Ha"]    = safe_log(feat.F_OI6300, feat.F_Ha, ok("OI6300", "Ha"))
feat["log_NeIII_OII"] = safe_log(feat.F_NeIII3869, feat.F_OII3727, ok("NeIII3869", "OII3727"))  # ionisation
feat["log_ArIII_OIII"] = safe_log(feat.F_ArIII7136, feat.F_OIII5007, ok("ArIII7136", "OIII5007"))
feat["OIII_4363_5007"] = np.where(ok("OIII4363", "OIII5007"), feat.F_OIII4363 / feat.F_OIII5007, np.nan)  # T_e
feat["SII_6717_6731"]  = np.where(ok("SII6717", "SII6731"), feat.F_SII6717 / feat.F_SII6731, np.nan)     # n_e
feat["log_EW_Ha"]      = np.log10(feat.EW_Ha.clip(lower=0.01))

# Balmer decrements vs Case B, T = 10^4 K (Osterbrock & Ferland 2006):
#   Ha/Hb = 2.86, Hg/Hb = 0.468, Hd/Hb = 0.259. Dust raises Ha/Hb and lowers Hg/Hb, Hd/Hb.
feat["Ha_Hb"] = np.where(ok("Ha", "Hb"), feat.F_Ha / feat.F_Hb, np.nan)
feat["Hg_Hb"] = np.where(ok("Hg", "Hb"), feat.F_Hg / feat.F_Hb, np.nan)
feat["Hd_Hb"] = np.where(ok("Hd", "Hb"), feat.F_Hd / feat.F_Hb, np.nan)
# Gas extinction with the CCM law: k(Hb)=3.61, k(Ha)=2.53, R_V=3.1
feat["AV_gas"] = 3.1 * 2.5 / (3.61 - 2.53) * np.log10(np.clip(feat.Ha_Hb, 2.86, None) / 2.86)

# Good STARLIGHT fits only (adjust the thresholds after looking at Cell 8)
feat["good_fit"] = (feat.sl_adev < 10) & (feat.sl_SN > 5)
feat.to_csv(f"{SAVE_DIR}/features.csv")
print(feat.shape)
```

## Cell 8 – EDA: data quality and catalogue properties per T‑type

```python name=08_eda_overview.py
# 1) Galaxies per T-type: some T-types will be rare (Sdm/Irr in MaNGA) -> statistics are noisier
ax = feat.Ttype.value_counts().sort_index().plot.bar(color=[TT_COLORS[t] for t in sorted(feat.Ttype.unique())],
                                                       figsize=(10, 4))
ax.set_xticklabels([f"{t}\n{TT_NAMES.get(t,'')}" for t in sorted(feat.Ttype.unique())], rotation=0)
plt.title("Galaxies per T-type"); plt.savefig(f"{SAVE_DIR}/counts.png", dpi=120); plt.show()

# 2) Missing values per feature (emission lines are often missing in E/S0 because they are weak)
print(feat.isna().mean().sort_values(ascending=False).head(25))

# 3) STARLIGHT fit quality
print(feat[["sl_chi2", "sl_adev", "sl_SN"]].describe())
print("good fits:", feat.good_fit.mean().round(3))

# 4) Catalogue properties per T-type.
#    LogMass: early types are more massive (Kauffmann et al. 2003b).
#    incl: edge-on disks look redder/dustier -> keep this in mind for late types.
#    nsa_z, PETRO_TH90: different T-types may be observed at different distances and sizes.
fig, axes = plt.subplots(1, 4, figsize=(22, 4))
for ax, c in zip(axes, ["nsa_z", "LogMass", "incl", "PETRO_TH90"]):
    sns.boxplot(data=feat, x="Ttype", y=c, palette=TT_COLORS, ax=ax, showfliers=False)
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/catalog_props.png", dpi=120); plt.show()

# 5) Bar fraction per T-type (bars are most common in early/intermediate spirals; Masters et al. 2011)
pd.crosstab(feat.Ttype, feat.bar, normalize="index").plot.bar(stacked=True, figsize=(10, 4))
plt.title("Bar fraction per T-type"); plt.show()
```

## Cell 9 – Feature distributions per T‑type

```python name=09_feature_distributions.py
EMI_FEATS = [f"EW_{k}" for k in EMISSION]                         # all emission EWs
ABS_FEATS = ["CaII_K", "CaII_H", "G4300", "Mgb", "NaD", "CaT", "Dn4000", "color_4500_6500"]
POP_FEATS = ["logage_L", "logage_M", "logZ_L", "logZ_M", "xY", "xI", "xO", "sl_AV", "sl_vd"]
RATIOS    = ["log_NII_Ha", "log_OIII_Hb", "log_SII_Ha", "log_OI_Ha", "log_NeIII_OII", "Ha_Hb", "AV_gas"]
FEATURES  = POP_FEATS + ABS_FEATS + EMI_FEATS + RATIOS + ["LogMass"]

d = feat[feat.good_fit]

def boxgrid(cols, fname, ncol=4):
    n = len(cols)
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(22, 3.3 * np.ceil(n / ncol)))
    for ax, f in zip(axes.flat, cols):
        sns.boxplot(data=d, x="Ttype", y=f, palette=TT_COLORS, ax=ax, showfliers=False)
        ax.set_xlabel("")
    for ax in axes.flat[n:]: ax.remove()
    plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/{fname}", dpi=110); plt.show()

boxgrid(POP_FEATS, "box_populations.png")
boxgrid(ABS_FEATS, "box_absorption.png")
boxgrid(EMI_FEATS, "box_emission.png")
boxgrid(RATIOS,    "box_ratios.png")

# Feature-feature correlation (pairs with very high |rho| carry the same information)
plt.figure(figsize=(16, 14))
sns.heatmap(d[FEATURES + ["Ttype"]].corr(method="spearman"), cmap="coolwarm", center=0)
plt.title("Spearman correlation"); plt.show()
```

## Cell 10 – Which features best separate the T‑types?

The ranking combines four views:

* **Kruskal–Wallis ε²:** how different the distributions are across T‑types. It makes no normality assumption.
* **Spearman ρ with T‑type:** does the feature change steadily along the Hubble sequence?
* **Mutual information:** catches any non-linear dependence on T‑type.
* **Random-forest permutation importance:** how much the classifier depends on each feature, alongside the others.

```python name=10_feature_ranking.py
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

rank = []
for f in FEATURES:
    groups = [g[f].dropna().values for _, g in d.groupby("Ttype") if g[f].notna().sum() >= 5]
    if len(groups) < 2: continue
    H, p = kruskal(*groups)
    rho, _ = spearmanr(d[f], d.Ttype, nan_policy="omit")
    rank.append({"feature": f, "epsilon2": H / (sum(map(len, groups)) - 1), "p_value": p,
                 "spearman_Ttype": rho, "coverage": d[f].notna().mean()})
rank = pd.DataFrame(rank).set_index("feature")
feats_ok = list(rank.index)

# Keep T-types with at least 10 galaxies for the classifier
keep = d.Ttype.map(d.Ttype.value_counts()) >= 10
X, y = d.loc[keep, feats_ok], d.loc[keep, "Ttype"]
rank["mutual_info"] = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(X), y, random_state=0)

model = make_pipeline(SimpleImputer(strategy="median"),
                      RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                             random_state=0, n_jobs=-1))
print("5-fold balanced accuracy:", cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy").mean().round(3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model.fit(Xtr, ytr)
rank["perm_importance"] = permutation_importance(model, Xte, yte, n_repeats=10,
                                                 scoring="balanced_accuracy", random_state=0).importances_mean
rank = rank.sort_values("epsilon2", ascending=False)
rank.to_csv(f"{SAVE_DIR}/feature_ranking.csv"); display(rank.round(3))

rank[["epsilon2", "mutual_info", "perm_importance"]].plot.barh(subplots=True, layout=(1, 3),
                                                               figsize=(18, 10), sharey=True, legend=False)
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()

# Confusion matrix: mistakes are usually between neighbouring T-types
ConfusionMatrixDisplay.from_estimator(model, Xte, yte, normalize="true", cmap="Blues", values_format=".2f")
plt.title("Normalised confusion matrix"); plt.show()
```

**What to expect:** The strongest separators are usually Dn4000, light-weighted age, xY, EW(Hα), EW([OII]), the Ca II H/K and G‑band strength, σ (`sl_vd`), stellar mass and colour. These separate early from late types well (Kennicutt 1992; Kauffmann et al. 2003a; Strateva et al. 2001). Neighbouring T‑types (for example 3 vs 4) are hard to tell apart from integrated spectra alone, so expect the confusion matrix to be strongest along the diagonal and its neighbours.

## Cell 11 – Spectral PCA (data-driven features)

```python name=11_pca.py
from sklearn.decomposition import PCA

sel = spec_index.isin(d.index)
S = SPEC[sel]
S = np.where(np.isnan(S), np.nanmean(S, axis=0), S)     # fill rare NaNs with the mean spectrum
tt = d.loc[spec_index[sel], "Ttype"].values

pca = PCA(n_components=10).fit(S); P = pca.transform(S)
print("explained variance:", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 2, figsize=(17, 6))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=tt, cmap="RdYlBu", s=6, alpha=0.6)
plt.colorbar(sc, ax=ax[0], label="T-type"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
for k in range(3):
    ax[1].plot(GRID, pca.components_[k] + 0.05 * k, lw=0.7, label=f"PC{k+1}")
ax[1].legend(); ax[1].set_xlabel("Rest wavelength [Å]"); ax[1].set_title("Eigenspectra")
plt.show()
# PC1 normally tracks the old-vs-young continuum shape plus line strength (Connolly et al. 1995; Yip et al. 2004)
```

## Cell 12 – Spectral diagnostics per T‑type (BPT, WHAN, Balmer, lines)

* **BPT diagrams** (Baldwin, Phillips & Terlevich 1981; Veilleux & Osterbrock 1987) separate star-forming galaxies, composites, Seyferts and LINERs. Dividing lines: Kauffmann et al. 2003, Kewley et al. 2001 and 2006, Schawinski et al. 2007.
* **WHAN** (Cid Fernandes et al. 2011) uses EW(Hα) to separate true AGN from "retired" galaxies, whose gas is ionised by old hot stars (Stasińska et al. 2008). This matters for E/S0, which often look like LINERs on the BPT.
* **Balmer decrement:** Hα/Hβ above 2.86 means there is dust in the ionised gas (Osterbrock & Ferland 2006). Late spirals usually show more dust than early types.

```python name=12_diagnostics.py
kauff = lambda x: 0.61 / (x - 0.05) + 1.30       # Kauffmann+03
kew01 = lambda x: 0.61 / (x - 0.47) + 1.19       # Kewley+01

def bpt_class(n2, o3):
    if np.isnan(n2) or np.isnan(o3): return "No/weak lines"
    if n2 < 0.05 and o3 < kauff(n2):  return "Star-forming"
    if n2 < 0.47 and o3 < kew01(n2):  return "Composite"
    return "Seyfert" if o3 > 1.05 * n2 + 0.45 else "LINER"

def whan_class(n2, ew_ha, ew_n2):
    if ew_ha < 0.5 and ew_n2 < 0.5: return "Passive"
    if ew_ha < 3:                   return "Retired"
    if np.isnan(n2):                return "Unclassified"
    if n2 < -0.4:                   return "SF"
    return "sAGN" if ew_ha > 6 else "wAGN"

feat["BPT"]  = [bpt_class(a, b) for a, b in zip(feat.log_NII_Ha, feat.log_OIII_Hb)]
feat["WHAN"] = [whan_class(a, b, c) for a, b, c in zip(feat.log_NII_Ha, feat.EW_Ha, feat.EW_NII6584)]
d = feat[feat.good_fit]
x = np.linspace(-2, 1, 300)

# ---- [NII] BPT: one panel per T-type
tts = sorted(d.Ttype.unique()); ncol = 4
fig, axes = plt.subplots(int(np.ceil(len(tts) / ncol)), ncol, figsize=(20, 4.5 * np.ceil(len(tts) / ncol)),
                         sharex=True, sharey=True)
for ax, t in zip(axes.flat, tts):
    g = d[d.Ttype == t]
    ax.scatter(g.log_NII_Ha, g.log_OIII_Hb, s=8, alpha=0.6, color=TT_COLORS[t])
    ax.plot(x[x < 0.05], kauff(x[x < 0.05]), "k--", lw=1)
    ax.plot(x[x < 0.47], kew01(x[x < 0.47]), "k-", lw=1)
    xs = np.linspace(-0.18, 1, 50); ax.plot(xs, 1.05 * xs + 0.45, "k:", lw=1)
    ax.set_title(f"T={t} {TT_NAMES.get(t,'')}  ({g.log_OIII_Hb.notna().sum()}/{len(g)} with lines)")
    ax.set_xlim(-1.8, 0.8); ax.set_ylim(-1.5, 1.5)
for ax in axes.flat[len(tts):]: ax.remove()
fig.supxlabel("log [NII]6584/Hα"); fig.supylabel("log [OIII]5007/Hβ")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/BPT_NII_by_ttype.png", dpi=110); plt.show()

# ---- [SII] and [OI] BPT, coloured by T-type (Kewley+06 lines)
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
for a, xcol in zip(ax, ["log_SII_Ha", "log_OI_Ha"]):
    sc = a.scatter(d[xcol], d.log_OIII_Hb, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6)
    a.set_xlabel(xcol); a.set_ylim(-1.5, 1.5)
xs = np.linspace(-1.5, 0.3, 100); ax[0].plot(xs, 0.72 / (xs - 0.32) + 1.30, "k-")
xs = np.linspace(-0.31, 0.5); ax[0].plot(xs, 1.89 * xs + 0.76, "k:")
xo = np.linspace(-2.5, -0.6, 100); ax[1].plot(xo, 0.73 / (xo + 0.59) + 1.33, "k-")
xo = np.linspace(-1.1, 0.5); ax[1].plot(xo, 1.18 * xo + 1.30, "k:")
ax[0].set_ylabel("log [OIII]/Hβ"); plt.colorbar(sc, ax=ax, label="T-type"); plt.show()

# ---- WHAN
plt.figure(figsize=(8, 6))
sc = plt.scatter(d.log_NII_Ha, d.log_EW_Ha, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6)
plt.axhline(np.log10(3), c="k"); plt.axhline(np.log10(6), c="k", ls=":"); plt.axvline(-0.4, c="k")
plt.colorbar(sc, label="T-type"); plt.xlabel("log [NII]/Hα"); plt.ylabel("log EW(Hα) [Å]"); plt.title("WHAN"); plt.show()

# ---- Fraction of BPT / WHAN classes in each T-type
for col in ["BPT", "WHAN"]:
    ct = pd.crosstab(d.Ttype, d[col], normalize="index")
    display((100 * ct).round(1))
    ct.plot.bar(stacked=True, figsize=(11, 4), colormap="tab10")
    plt.title(f"{col} class per T-type"); plt.legend(bbox_to_anchor=(1, 1)); plt.show()

# ---- Balmer decrements per T-type, with Case B values
fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
for a, col, caseB in zip(ax, ["Ha_Hb", "Hg_Hb", "Hd_Hb", "AV_gas"], [2.86, 0.468, 0.259, 0]):
    sns.boxplot(data=d, x="Ttype", y=col, palette=TT_COLORS, ax=a, showfliers=False)
    a.axhline(caseB, c="k", ls="--")
plt.tight_layout(); plt.show()

# ---- Age-sensitive plane: Dn4000 vs Ca II K and vs EW(Ha)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].scatter(d.Dn4000, d.CaII_K, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6); ax[0].set_ylabel("Ca II K EW [Å]")
sc = ax[1].scatter(d.Dn4000, d.log_EW_Ha, c=d.Ttype, cmap="RdYlBu", s=6, alpha=0.6); ax[1].set_ylabel("log EW(Hα)")
for a in ax: a.set_xlabel("Dn4000")
plt.colorbar(sc, ax=ax, label="T-type"); plt.show()

# ---- Summary tables: median value per T-type + detection rate (S/N >= 3) of each emission line
display(d.groupby("Ttype")[ABS_FEATS + EMI_FEATS + RATIOS].median().round(2).T)
detect = pd.DataFrame({k: d.groupby("Ttype")[f"SN_{k}"].apply(lambda s: (s >= 3).mean())
                       for k in EMISSION}).T
plt.figure(figsize=(12, 6)); sns.heatmap(detect, annot=True, fmt=".2f", cmap="viridis")
plt.title("Fraction of galaxies with each emission line detected (S/N ≥ 3)"); plt.show()
feat.to_csv(f"{SAVE_DIR}/features_with_diagnostics.csv")
```

## Cell 13 – All spectra grouped by T‑type

```python name=13_plot_all_spectra.py
EM_MARK  = {"[OII]":3727, "[NeIII]":3869, "Hδ":4102, "Hγ":4340, "[OIII]4363":4363, "Hβ":4861,
            "[OIII]5007":5007, "HeI":5876, "[OI]":6300, "[NII]":6548, "Hα":6563, "[NII] ":6584,
            "[SII]":6724, "[ArIII]":7136}
ABS_MARK = {"K":3934, "H":3969, "G":4304, "Mg":5175, "NaD":5893, "CaII":8498, "CaII ":8542, "CaII  ":8662}

def mark_lines(ax, ytop):
    for lab, w in EM_MARK.items():
        ax.axvline(w, c="tab:blue", lw=0.4, ls=":")
        ax.text(w, ytop, lab.strip(), fontsize=6, rotation=90, va="top", color="tab:blue")
    for lab, w in ABS_MARK.items():
        ax.axvline(w, c="tab:red", lw=0.4, ls=":")
        ax.text(w, ytop, lab.strip(), fontsize=6, rotation=90, va="top", color="tab:red")

tt_of = feat["Ttype"].reindex(spec_index).values
tts = sorted(pd.unique(tt_of[~pd.isna(tt_of)]))
fig, axes = plt.subplots(len(tts), 1, figsize=(16, 2.8 * len(tts)), sharex=True)
for ax, t in zip(np.atleast_1d(axes), tts):
    S_t = SPEC[tt_of == t]
    for s in S_t:                                    # every galaxy as a faint line
        ax.plot(GRID, s, color=TT_COLORS[t], lw=0.3, alpha=0.05, rasterized=True)
    ax.plot(GRID, np.nanmedian(S_t, axis=0), "k", lw=0.9)   # median on top
    ax.set_ylim(0, 3); ax.set_ylabel("F/F(5500)")
    ax.set_title(f"T-type {t} ({TT_NAMES.get(t,'')}), N = {len(S_t)}", loc="left", fontsize=10)
    mark_lines(ax, 2.95)
axes[-1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/all_spectra_by_ttype.png", dpi=100); plt.show()
```

## Cell 14 – Mean spectrum of every T‑type in one graph

```python name=14_mean_spectra.py
mean_spec = {t: np.nanmean(SPEC[tt_of == t], axis=0) for t in tts}

fig, ax = plt.subplots(2, 1, figsize=(17, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
for t in tts:
    ax[0].plot(GRID, mean_spec[t], color=TT_COLORS[t], lw=0.9,
               label=f"T={t} {TT_NAMES.get(t,'')} (N={np.sum(tt_of == t)})")
ax[0].set_ylabel("Mean F / F(5500)"); ax[0].legend(ncol=2, fontsize=8)
mark_lines(ax[0], ax[0].get_ylim()[1])

# Ratio to the earliest T-type -> shows only the differences (bluer continuum, emission lines)
ref = tts[0]
for t in tts[1:]:
    ax[1].plot(GRID, mean_spec[t] / mean_spec[ref], color=TT_COLORS[t], lw=0.7)
ax[1].axhline(1, c="k", lw=0.5); ax[1].set_ylabel(f"Ratio to T={ref}")
ax[1].set_xlabel("Rest-frame wavelength [Å]")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/mean_spectra_by_ttype.png", dpi=130); plt.show()

pd.DataFrame(mean_spec, index=pd.Index(GRID, name="wave_rest")).to_csv(f"{SAVE_DIR}/mean_spectra_by_ttype.csv")
```

**What to look for:** Going from T = −5 to T = 10:

* the continuum gets bluer;
* the 4000 Å break, Ca II K and H, the G band, Mg b, NaD and the Ca II triplet get weaker;
* the emission lines ([OII], Hβ, [OIII], Hα, [NII], [SII], HeI, [ArIII]) get stronger;
* [NII]/Hα and [SII]/Hα fall, as the ionisation changes from old-star/LINER-like to H II-region-like (Kennicutt 1992; Ho, Filippenko & Sargent 1997).

---

### Things to keep in mind
* **Speed:** reading thousands of `.fits.gz` files (Cells 3 and 7) is the slow part. The results are saved to Drive, so later sessions can reload `rest_spectra.npz` and `features.csv` instead.
* **Weak lines:** [OIII] 4363, HeI 5876, [ArIII] 7136 and [NeIII] 3869 are faint, and many galaxies will be below S/N 3. Cell 12 shows the detection-rate heatmap, so you can see which lines are actually usable for each T‑type.
* **Biases:**
  * `FLUX_FoV` integrates the whole bundle, so bulge-dominated early types look older and redder overall.
  * The bundle covers different physical sizes at different redshifts (aperture effects).
  * Inclination makes the spectra of late-type disks redder.
  * Compare the `nsa_z` and `LogMass` distributions per T‑type (Cell 8) before treating a feature difference as purely morphological.
* **Morphology catalogue:** the labels (`Edc`, `SABbc`…) look like the MaNGA visual morphology catalogue of Vázquez‑Mata et al. (2022). Please confirm the source before citing it.

### References
Asari et al. 2007, MNRAS 381, 263 · Baldwin, Phillips & Terlevich 1981, PASP 93, 5 · Balogh et al. 1999, ApJ 527, 54 · Blanton & Moustakas 2009, ARA&A 47, 159 · Bruzual & Charlot 2003, MNRAS 344, 1000 · Bundy et al. 2015, ApJ 798, 7 · Cardelli, Clayton & Mathis 1989, ApJ 345, 245 · Cid Fernandes et al. 2005, MNRAS 358, 363; 2011, MNRAS 413, 1687 · Connolly et al. 1995, AJ 110, 1071 · de Vaucouleurs 1959, Handbuch der Physik 53, 275 · de Vaucouleurs et al. 1991, RC3 · Diaz, Terlevich & Terlevich 1989, MNRAS 239, 325 · Ho, Filippenko & Sargent 1997, ApJS 112, 315 · Hubble 1926, ApJ 64, 321 · Kauffmann et al. 2003a, MNRAS 341, 33; 2003b, MNRAS 346, 1055 · Kennicutt 1992, ApJS 79, 255; 1998, ARA&A 36, 189 · Kewley et al. 2001, ApJ 556, 121; 2006, MNRAS 372, 961 · Masters et al. 2011, MNRAS 411, 2026 · Morgan & Mayall 1957, PASP 69, 291 · Osterbrock & Ferland 2006, *Astrophysics of Gaseous Nebulae and AGN* · Roberts & Haynes 1994, ARA&A 32, 115 · Rose 1985, AJ 90, 1927 · Schawinski et al. 2007, MNRAS 382, 1415 · Stasińska et al. 2008, MNRAS 391, L29 · Strateva et al. 2001, AJ 122, 1861 · Trager et al. 1998, ApJS 116, 1 · Vázquez‑Mata et al. 2022, MNRAS 512, 2222 · Veilleux & Osterbrock 1987, ApJS 63, 295 · Worthey 1994, ApJS 95, 107 · Worthey et al. 1994, ApJS 94, 687 · Yip et al. 2004, AJ 128, 585.
