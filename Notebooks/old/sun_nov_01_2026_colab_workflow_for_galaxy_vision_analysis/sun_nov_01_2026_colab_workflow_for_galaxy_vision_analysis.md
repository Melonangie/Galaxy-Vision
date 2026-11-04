# Copilot Chat Conversation Export: Colab workflow for Galaxy-Vision analysis

**User:** @maria-rabelero-mx_nestle
**Thread URL:** https://github.com/copilot/c/3325ef9e-a177-4dd4-9121-c634992e66e5

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


*System: User retried message*

## Copilot

I recommend a new Colab notebook focused on `FLUX_FoV`, with STARLIGHT continuum subtraction before measuring emission lines. The existing notebook uses `FLUX_1RE` and includes metadata EDA, so it does not match your requirements.

I’ll prepare the workflow in `Melonangie/Galaxy-Vision`, including plots by `Ttype` and feature-ranking tests. I can inspect the GitHub samples, but I cannot access your Google Drive files here; the full analysis and measured rankings will run when you execute the notebook in Colab.

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


*System: User retried message*

## Copilot

I’m waiting for you to accept or reject the confirmation dialog before continuing with the documented Colab workflow for galaxy morphology EDA and spectral diagnostics.

## @maria-rabelero-mx_nestle

@Copilot Accepted Confirmation: Confirm cloud agent session

## Copilot

**Use the original FITS spectra for measurements, STARLIGHT for stellar-continuum subtraction and population features, and the supporting catalog for redshifts and morphology labels.** Measure each galaxy separately before comparing or stacking by `Ttype`.

The coding-agent handoff failed because it is not enabled for `Melonangie/Galaxy-Vision`; **no notebook or pull request was created**. You can still copy the following cells into Colab.

I cannot access your private Google Drive or run the thousands of files here, so this is an **exploratory workflow—not measured results from your full sample**.

## 1. Important conclusions from your samples

- Use `WAVE` and `FLUX_FoV`; completely ignore `ERROR_FoV`.
- Join `manga-10001-12701-1D.fits.gz` to catalog name `manga-10001-12701`.
- Convert FITS wavelengths using  
  \[
  \lambda_{\rm rest}=\lambda_{\rm observed}/(1+z).
  \]
- Your example STARLIGHT output **appears already rest-frame**: its [O II] feature is near 3727 Å. Do not redshift it twice.
- STARLIGHT `f_syn` is normalized. Restore its scale using `fobs_norm`, then check alignment against the FITS spectrum.
- STARLIGHT emission-mask weights of zero **do not mean missing spectral observations**.
- Catalog redshifts such as `-9999` are invalid.
- The CSV gaps remove some diagnostic regions. **Do not interpolate across them to invent measurements.**
- Your catalog contains `Type="S", Ttype=11`. Preserve this catalog convention; do not automatically relabel 11 as an irregular galaxy.

### Features most worth investigating

These are hypotheses to test, not a ranking established from your data.

| Feature family | Why it might separate morphology |
|---|---|
| \(D_n4000\), stellar ages, young-star fractions | Trace population age and recent star formation |
| Hα EW, [O II] EW | Trace emission relative to the underlying stellar population |
| Hδ absorption | Sensitive to intermediate-age populations and recent changes in star formation |
| Mg b, G band, Ca K, Na D | Trace stellar populations, metallicity and—in Na D—interstellar absorption |
| STARLIGHT stellar attenuation | Dust and population differences, but inclination is a confounder |
| BPT ratios | Trace ionization source; useful context but **not direct morphology measurements** |
| Continuum shape | Population and dust differences |
| Stellar velocity dispersion | Potentially useful, but requires instrumental-resolution and fitting checks |

---

# Colab workflow

Copy the blocks below into **separate cells in order**.

This keeps the implementation readable. The line fits use a simple Gaussian model suitable for first-pass EDA; unusual profiles, broad-line AGN, severe sky residuals and poor STARLIGHT fits require inspection.

## 2. Install packages and configure paths

```python name=colab_setup.py
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
```

**Before running:** confirm the wavelength medium. SDSS/MaNGA products often use vacuum wavelengths, but derived extraction products may differ. Using air line centers against vacuum data introduces systematic shifts.

## 3. Read FITS tables and supporting metadata

The supporting FITS file is read only to supply labels, redshift and context. It is explicitly excluded from spectrum discovery.

```python name=read_galaxy_data.py
def read_required_table(path, required, hdu=None):
    """Read a FITS table by column names, or a CSV for sample testing."""
    required = {name.lower() for name in required}
    path = Path(path)

    if path.suffix.lower() == ".csv":
        table = Table.from_pandas(pd.read_csv(path))
    else:
        with fits.open(path, memmap=False) as hdus:
            if hdu is None:
                candidates = [
                    i for i, item in enumerate(hdus)
                    if isinstance(item, (fits.BinTableHDU, fits.TableHDU))
                    and required <= {n.lower() for n in item.columns.names}
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one suitable table in {path.name}; "
                        f"found {candidates}. Set the HDU explicitly."
                    )
                hdu = candidates[0]
            table = Table(hdus[hdu].data)

            # Preserve wavelength units from FITS column metadata.
            for column in hdus[hdu].columns:
                if column.unit:
                    table[column.name].unit = column.unit

    for old_name in list(table.colnames):
        table.rename_column(old_name, old_name.lower())

    if not required <= set(table.colnames):
        raise ValueError(f"Missing required columns in {path.name}")
    return table


def galaxy_id(path_or_name):
    """Match the catalog name without relying on every possible suffix."""
    match = re.search(r"manga-\d+-\d+", str(path_or_name))
    if match is None:
        raise ValueError(f"Cannot identify galaxy: {path_or_name}")
    return match.group(0)


def to_air(wavelength, medium):
    """Convert optical vacuum Å to air Å using a standard refractive index."""
    wavelength = np.asarray(wavelength, dtype=float)
    if medium == "air":
        return wavelength
    if medium != "vacuum":
        raise ValueError("Set wavelength medium to 'air' or 'vacuum'.")

    # Optical approximation; appropriate to this notebook's wavelength range.
    sigma2 = (1e4 / wavelength) ** 2
    refractive_index = (
        1 + 1e-6 * (
            64.328 + 29498.1 / (146 - sigma2) + 255.4 / (41 - sigma2)
        )
    )
    return wavelength / refractive_index


def rest_coordinates(wavelength, flux, z, frame, medium):
    """Use a flux-conserving f_lambda coordinate transformation."""
    if frame == "observed":
        wavelength = wavelength / (1 + z)
        flux = flux * (1 + z)
    elif frame != "rest":
        raise ValueError("Frame must be 'observed' or 'rest'.")

    # This optical air/vacuum conversion is used for wavelength registration.
    # Its tiny flux-density Jacobian is neglected in this exploratory workflow.
    return to_air(wavelength, medium), flux


def read_spectrum(path, z):
    table = read_required_table(path, ["wave", "flux_fov"], SPECTRUM_HDU)

    wave_column = table["wave"]
    wave_unit = wave_column.unit or WAVE_UNIT_IF_MISSING
    wavelength = (np.asarray(wave_column, float).ravel() * wave_unit).to_value(u.AA)
    flux = np.asarray(table["flux_fov"], float).ravel()

    if wavelength.shape != flux.shape:
        raise ValueError("WAVE and FLUX_FoV have different shapes.")

    # ERROR_FoV is deliberately never read.
    valid = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    wavelength, flux = wavelength[valid], flux[valid]
    order = np.argsort(wavelength)
    wavelength, flux = wavelength[order], flux[order]

    if len(wavelength) < 20:
        raise ValueError("Too few finite spectral samples.")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Duplicate wavelengths: inspect the extraction.")

    return rest_coordinates(
        wavelength, flux, z, FITS_FRAME, FITS_WAVELENGTH_MEDIUM
    )


catalog_table = read_required_table(
    CATALOG_PATH, ["name", "ttype", "nsa_z"], CATALOG_HDU
)
catalog = catalog_table.to_pandas()

# FITS strings may be bytes; decode before matching.
catalog["name"] = catalog["name"].map(
    lambda value: value.decode().strip()
    if isinstance(value, bytes) else str(value).strip()
)
catalog["name"] = catalog["name"].map(galaxy_id)

if catalog["name"].duplicated().any():
    raise ValueError("Duplicate galaxy IDs in supporting catalog.")

for column in ["ttype", "nsa_z", "logmass", "incl", "petro_th90"]:
    if column in catalog:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
        catalog.loc[catalog[column] <= -999, column] = np.nan

catalog = catalog.set_index("name")

# Discover only galaxy spectra; explicitly exclude the supporting catalog.
spectrum_paths = sorted(
    p for p in SPECTRA_DIR.rglob("*")
    if p.is_file()
    and p.name.lower().endswith((".fits.gz", ".fits"))
    and p.resolve() != CATALOG_PATH.resolve()
    and re.search(r"manga-\d+-\d+", p.name)
)

spectrum_ids = [galaxy_id(p) for p in spectrum_paths]
if pd.Series(spectrum_ids).duplicated().any():
    raise ValueError("Multiple spectrum files for one galaxy; resolve duplicates.")

out_lookup = defaultdict(list)
for path in STARLIGHT_DIR.rglob("*"):
    if path.is_file() and path.name.lower().endswith((".out", ".out.txt", "-out.txt")):
        if re.search(r"manga-\d+-\d+", path.name):
            out_lookup[galaxy_id(path)].append(path)

print(f"Found {len(spectrum_paths)} galaxy spectra.")
```

## 4. Coverage-aware interpolation and STARLIGHT parsing

This parser uses section markers and declared row counts, rather than fixed line numbers.

```python name=starlight_and_coverage.py
def resample_without_gaps(wavelength, values, new_wavelength):
    """Interpolate only inside contiguous observed segments."""
    result = np.full(len(new_wavelength), np.nan)
    spacing = np.median(np.diff(wavelength))
    breaks = np.where(np.diff(wavelength) > MAX_GAP_FACTOR * spacing)[0] + 1

    for indices in np.split(np.arange(len(wavelength)), breaks):
        if len(indices) < 2:
            continue
        inside = (
            (new_wavelength >= wavelength[indices[0]])
            & (new_wavelength <= wavelength[indices[-1]])
        )
        result[inside] = np.interp(
            new_wavelength[inside], wavelength[indices], values[indices]
        )
    return result


def covered_window(wavelength, values, low, high):
    """Return samples only if the full window is actually covered."""
    spacing = np.median(np.diff(wavelength))
    use = (wavelength >= low) & (wavelength <= high)
    x, y = wavelength[use], values[use]

    if len(x) < 3 or not np.all(np.isfinite(y)):
        raise ValueError("Unavailable/nonfinite diagnostic window.")
    if x[0] > low + spacing or x[-1] < high - spacing:
        raise ValueError("Incomplete diagnostic coverage.")
    if np.any(np.diff(x) > MAX_GAP_FACTOR * spacing):
        raise ValueError("Gap inside diagnostic window.")
    return x, y


def band_mean(wavelength, values, band):
    x, y = covered_window(wavelength, values, *band)
    return np.trapezoid(y, x) / (x[-1] - x[0])


def read_starlight(path, z):
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    def scalar(label):
        for line in lines:
            # Match labels, allowing additional units/descriptions.
            if re.search(r"\[\s*" + re.escape(label) + r"(?=\s|\]|\()", line):
                return float(line.split()[0].replace("D", "E"))
        raise ValueError(f"STARLIGHT label missing: {label}")

    n_base = int(scalar("N_base"))
    start = next(
        i for i, line in enumerate(lines)
        if line.lstrip().startswith("# j") and "age_j" in line and "x_j" in line
    )

    # First six columns are numeric; component names occur later.
    populations = np.array([
        [float(v.replace("D", "E")) for v in line.split()[:6]]
        for line in lines[start + 1:start + 1 + n_base]
    ])

    spectral_start = next(
        i for i, line in enumerate(lines)
        if "Synthetic spectrum (Best Model)" in line
    )
    count_line = next(
        i for i in range(spectral_start + 1, len(lines))
        if "[Nl_obs]" in lines[i]
    )
    n_wave = int(lines[count_line].split()[0])
    spectrum = np.loadtxt(
        io.StringIO("\n".join(lines[count_line + 1:count_line + 1 + n_wave]))
    )

    wavelength = spectrum[:, 0]
    observed = spectrum[:, 1] * scalar("fobs_norm")
    synthetic = spectrum[:, 2] * scalar("fobs_norm")
    weights = spectrum[:, 3]

    wavelength, synthetic = rest_coordinates(
        wavelength, synthetic, z, STARLIGHT_FRAME,
        STARLIGHT_WAVELENGTH_MEDIUM
    )
    if STARLIGHT_FRAME == "observed":
        observed *= 1 + z

    # No-data padding is not a measured continuum. Keep wei=0 emission masks
    # and wei=-1 clipped samples; the original FITS remains the observation.
    usable = (
        np.isfinite(wavelength) & np.isfinite(synthetic)
        & np.isfinite(observed) & (weights != -2)
    )

    light = populations[:, 1]
    mass = populations[:, 3]  # Current stellar mass fractions, not Mini.
    age = populations[:, 4]
    metallicity = populations[:, 5]

    if np.any(light < 0) or np.any(mass < 0):
        raise ValueError("Negative STARLIGHT population weights.")
    if light.sum() <= 0 or mass.sum() <= 0 or np.any(age <= 0):
        raise ValueError("Invalid STARLIGHT population solution.")

    # Renormalize: your sample's sum-of-x is not exactly 100%.
    light = light / light.sum()
    mass = mass / mass.sum()

    features = {
        "stellar_av": scalar("AV_min"),
        "starlight_adev": scalar("adev"),
        "starlight_chi2": scalar("chi2/Nl_eff"),
        "stellar_vdisp_fit": scalar("vd_min"),
        "light_log_age": np.sum(light * np.log10(age)),
        "mass_log_age": np.sum(mass * np.log10(age)),
        "light_Z": np.sum(light * metallicity),
        "mass_Z": np.sum(mass * metallicity),
        "young_light_fraction": light[age < 1e8].sum(),
        "intermediate_light_fraction": light[(age >= 1e8) & (age < 1e9)].sum(),
        "old_light_fraction": light[age >= 1e9].sum(),
    }
    return wavelength[usable], synthetic[usable], features


def align_stellar_model(wavelength, flux, model_wave, model_flux):
    model = resample_without_gaps(model_wave, model_flux, wavelength)

    # Check several continuum regions, not just one normalization point.
    bands = [(4500, 4600), (5450, 5550), (6000, 6100)]
    ratios = []
    for band in bands:
        try:
            observed_mean = band_mean(wavelength, flux, band)
            model_mean = band_mean(wavelength, model, band)
            if observed_mean > 0 and model_mean > 0:
                ratios.append(observed_mean / model_mean)
        except ValueError:
            pass

    if len(ratios) < 2:
        raise ValueError("Insufficient continuum coverage for model validation.")

    scale = np.median(ratios)
    mismatch = np.max(np.abs(np.asarray(ratios) / scale - 1))
    if mismatch > MAX_CONTINUUM_MISMATCH:
        raise ValueError(f"STARLIGHT continuum mismatch: {mismatch:.1%}")

    return model * scale, mismatch
```

**Caution:** STARLIGHT fit statistics are not necessarily calibrated if its upstream uncertainty input was incorrect. Stellar ages, attenuation and continuum subtraction should therefore be checked, not accepted solely because a fit completed.

---

## 5. Emission-line measurements

Your repeated O III and N II entries correspond to their doublets. The code also includes both [O II] and [S II] components.

All centers below are **air wavelengths in Å**.

```python name=emission_line_fits.py
EMISSION_GROUPS = {
    "OII": [("OII3726", 3726.03), ("OII3729", 3728.82)],
    "NeIII": [("NeIII3869", 3868.76)],
    "Hdelta": [("Hdelta", 4101.74)],
    "Hgamma": [("Hgamma", 4340.47)],
    "Hbeta": [("Hbeta", 4861.33)],
    "OIII4959": [("OIII4959", 4958.91)],
    "OIII5007": [("OIII5007", 5006.84)],
    "HeI": [("HeI5876", 5875.62)],
    "OI": [("OI6300", 6300.30)],
    "Halpha_NII": [
        ("NII6548", 6548.05), ("Halpha", 6562.80), ("NII6583", 6583.45)
    ],
    "SII": [("SII6716", 6716.44), ("SII6731", 6730.82)],
    "ArIII": [("ArIII7136", 7135.79)],
}


def fit_emission_group(wavelength, flux, stellar_model, components):
    """Joint Gaussian fit with shared velocity/width and a local baseline."""
    centers = np.array([center for _, center in components])
    low, high = centers.min() - 30, centers.max() + 30

    x, observed = covered_window(wavelength, flux, low, high)
    _, stellar = covered_window(wavelength, stellar_model, low, high)
    residual = observed - stellar

    # Sidebands estimate empirical scatter. This is NOT ERROR_FoV.
    sideband = np.all(np.abs(x[:, None] - centers[None, :]) > 12, axis=1)
    if sideband.sum() < 10:
        raise ValueError("Too few sideband samples.")

    baseline = np.polyfit(x[sideband] - x.mean(), residual[sideband], 1)
    noise = float(mad_std(
        residual[sideband] - np.polyval(baseline, x[sideband])
    ))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Cannot estimate empirical residual noise.")

    # p = velocity(km/s), width(km/s), baseline offset, slope, line fluxes.
    # Flux is the Gaussian integral, rather than peak amplitude.
    initial_fluxes = []
    for center in centers:
        near = np.abs(x - center) < 5
        peak = max(float(np.max(residual[near])), noise)
        initial_fluxes.append(peak * 3.0)

    p0 = [0, 120, baseline[1], baseline[0], *initial_fluxes]
    lower = [-400, 30, -np.inf, -np.inf, *([0] * len(centers))]
    upper = [400, 500, np.inf, np.inf, *([np.inf] * len(centers))]

    def model(p):
        velocity, dispersion, offset, slope = p[:4]
        shifted = centers * (1 + velocity / 299792.458)
        widths = centers * dispersion / 299792.458
        profiles = np.exp(
            -0.5 * ((x[:, None] - shifted) / widths) ** 2
        ) / (np.sqrt(2 * np.pi) * widths)
        return offset + slope * (x - x.mean()) + profiles @ p[4:]

    fit = least_squares(
        lambda p: (model(p) - residual) / noise,
        p0, bounds=(lower, upper), max_nfev=1000
    )

    if not fit.success:
        raise ValueError("Emission fit did not converge.")

    # Conditional covariance: assumes independent empirical residual noise.
    # It omits correlated pixels and uncertainty in the stellar model.
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    errors = np.sqrt(np.maximum(np.diag(covariance)[4:], 0))

    boundary = (
        abs(fit.x[0]) > 395 or fit.x[1] < 31 or fit.x[1] > 495
    )
    result = []
    for i, (name, center) in enumerate(components):
        line_flux = fit.x[4 + i]
        error = errors[i]

        # EW uses the local underlying stellar continuum.
        continuum = np.interp(center, x, stellar)
        ew = line_flux / continuum if continuum > 0 else np.nan
        snr = line_flux / error if error > 0 else np.nan

        result.append({
            "line": name,
            "flux": line_flux,
            "empirical_error": error,
            "snr_empirical": snr,
            "ew_emission": ew,  # Positive for emission; rest-frame Å.
            "detected": bool(
                np.isfinite(snr) and snr >= DETECTION_SNR and not boundary
            ),
            "upper_limit_approx": (
                max(line_flux, 0) + DETECTION_SNR * error
                if np.isfinite(error) else np.nan
            ),
            "velocity_fit": fit.x[0],
            "width_fit": fit.x[1],
            "boundary_fit": boundary,
            "residual_rms_over_noise": np.std(model(fit.x) - residual) / noise,
            "status": "measured",
        })

    return result
```

### Limits of this simple fit

- [O II] is often poorly resolved. **Prefer its summed flux**, not its individual ratio, until resolution and recovery tests support deblending.
- Hα+[N II] is fitted jointly so Hα is not measured by integrating the entire blend.
- Widths include instrumental broadening and FoV velocity mixing; they are not automatically intrinsic gas dispersions.
- Nonnegative fitting biases very weak lines upward. Inspect nondetections and use more rigorous injection/recovery or bootstrap analysis before scientific inference.
- The empirical errors are conditional approximations. For final work, add a block-residual bootstrap and vary the continuum model.
- No STARLIGHT model means no absorption-corrected Balmer/BPT measurements in this version. This is safer than silently treating a polynomial continuum as an equivalent replacement.

## 6. Absorption indices and continuum features

The following Ca, G-band, Mg, Na and Ca-triplet windows are **explicit exploratory indices**, not claimed to be calibrated Lick indices.

HδA uses published bandpasses. Its observed value can have emission infill; the model value is a **stellar-model prediction**, kept separately.

```python name=absorption_indices.py
# Each entry: feature window, blue continuum window, red continuum window.
ABSORPTION_BANDS = {
    "CaK": ((3925, 3945), (3905, 3920), (3948, 3960)),
    "CaH_blend": ((3962, 3982), (3948, 3960), (3985, 4000)),
    "Gband": ((4280, 4318), (4245, 4270), (4320, 4330)),
    "Mgb": ((5160, 5193), (5142, 5158), (5195, 5212)),
    "NaD": ((5878, 5908), (5850, 5870), (5910, 5930)),
    "CaII8498": ((8483, 8513), (8460, 8475), (8515, 8525)),
    "CaII8542": ((8527, 8557), (8515, 8525), (8560, 8570)),
    "CaII8662": ((8647, 8677), (8620, 8635), (8680, 8695)),
    "HdeltaA": (
        (4083.50, 4122.25),
        (4041.60, 4079.75),
        (4128.50, 4161.00),
    ),
}


def absorption_ew(wavelength, flux, bands):
    feature, blue, red = bands
    blue_mean = band_mean(wavelength, flux, blue)
    red_mean = band_mean(wavelength, flux, red)
    x, y = covered_window(wavelength, flux, *feature)

    continuum = np.interp(
        x, [np.mean(blue), np.mean(red)], [blue_mean, red_mean]
    )
    if np.any(continuum <= 0):
        raise ValueError("Nonpositive absorption pseudo-continuum.")

    # Positive absorption, unlike the emission-positive EW above.
    return np.trapezoid(1 - y / continuum, x)


def continuum_features(wavelength, flux, stellar_model=None):
    result = {}

    # Dn4000 is defined using f_nu. Since input is f_lambda,
    # f_nu is proportional to f_lambda * lambda²; the constant cancels.
    try:
        fnu_relative = flux * wavelength**2
        blue = band_mean(wavelength, fnu_relative, (3850, 3950))
        red = band_mean(wavelength, fnu_relative, (4000, 4100))
        result["Dn4000"] = red / blue if blue > 0 else np.nan
    except ValueError:
        result["Dn4000"] = np.nan

    for name, bands in ABSORPTION_BANDS.items():
        for origin, values in [("observed", flux), ("stellar_model", stellar_model)]:
            if values is None:
                continue
            try:
                result[f"{name}_{origin}_ew"] = absorption_ew(
                    wavelength, values, bands
                )
            except ValueError:
                result[f"{name}_{origin}_ew"] = np.nan

    triplet = [result.get(f"CaII{line}_observed_ew", np.nan)
               for line in [8498, 8542, 8662]]
    result["CaII_triplet_observed_ew"] = (
        np.sum(triplet) if np.all(np.isfinite(triplet)) else np.nan
    )

    try:
        blue = band_mean(wavelength, flux, (4500, 4600))
        red = band_mean(wavelength, flux, (6000, 6100))
        result["continuum_red_blue"] = red / blue if blue > 0 else np.nan
    except ValueError:
        result["continuum_red_blue"] = np.nan

    return result
```

**Interpretation cautions:** Ca H includes Hε blending; G band is molecular CH absorption; Mg b is a triplet; Na D can include interstellar gas; the near-infrared Ca triplet is vulnerable to sky/telluric residuals and coverage limits.

---

## 7. Batch processing

Each galaxy is processed independently. Failures are logged rather than stopping the whole batch. Normalized spectra are cached on disk for plotting.

```python name=batch_galaxy_eda.py
feature_rows = []
line_rows = []
qc_rows = []

selected_paths = (
    spectrum_paths if MAX_GALAXIES is None
    else spectrum_paths[:MAX_GALAXIES]
)

for path in tqdm(selected_paths):
    name = galaxy_id(path)
    qc = {"name": name, "path": str(path)}

    try:
        if name not in catalog.index:
            raise ValueError("No supporting catalog match.")

        metadata = catalog.loc[name]
        z = metadata["nsa_z"]
        if not np.isfinite(z) or z < 0:
            raise ValueError("Missing/invalid redshift.")

        wavelength, flux = read_spectrum(path, z)
        row = {"name": name, "ttype": metadata["ttype"]}

        # Context only, not automatically physical morphology predictors.
        for column in ["type", "nsa_z", "logmass", "incl", "petro_th90"]:
            if column in metadata.index:
                row[column] = metadata[column]

        stellar_model = None
        matches = out_lookup.get(name, [])

        if len(matches) > 1:
            qc["starlight_status"] = "duplicate STARLIGHT files"
        elif len(matches) == 0:
            qc["starlight_status"] = "missing"
        else:
            try:
                model_wave, model_flux, stellar_features = read_starlight(matches[0], z)
                stellar_model, mismatch = align_stellar_model(
                    wavelength, flux, model_wave, model_flux
                )
                row.update(stellar_features)
                qc["starlight_status"] = "aligned"
                qc["continuum_mismatch"] = mismatch
            except Exception as error:
                qc["starlight_status"] = str(error)

        row.update(continuum_features(wavelength, flux, stellar_model))

        for group, components in EMISSION_GROUPS.items():
            try:
                if stellar_model is None:
                    raise ValueError("No validated stellar continuum.")
                measured = fit_emission_group(
                    wavelength, flux, stellar_model, components
                )
            except Exception as error:
                measured = [
                    {"line": line, "status": str(error), "detected": False}
                    for line, _ in components
                ]

            for measurement in measured:
                measurement.update({"name": name, "ttype": row["ttype"], "group": group})
                line_rows.append(measurement)

        # Normalize only for spectrum-shape comparison, not line diagnostics.
        try:
            normalization = band_mean(wavelength, flux, NORMALIZATION_BAND)
            if normalization <= 0:
                raise ValueError("Nonpositive normalization continuum.")

            normalized = resample_without_gaps(
                wavelength, flux / normalization, GRID
            )
            np.savez_compressed(
                OUTPUT_DIR / "spectra_cache" / f"{name}.npz",
                wave=GRID,
                normalized_flux=normalized,
            )
            qc["stack_status"] = "cached"
        except Exception as error:
            qc["stack_status"] = str(error)

        feature_rows.append(row)
        qc["status"] = "processed"

    except Exception as error:
        qc["status"] = str(error)

    qc_rows.append(qc)

    # Periodic checkpoints help protect a long Colab session.
    if len(qc_rows) % 25 == 0:
        pd.DataFrame(feature_rows).to_csv(OUTPUT_DIR / "features_checkpoint.csv", index=False)
        pd.DataFrame(line_rows).to_csv(OUTPUT_DIR / "lines_checkpoint.csv", index=False)
        pd.DataFrame(qc_rows).to_csv(OUTPUT_DIR / "qc_checkpoint.csv", index=False)

features = pd.DataFrame(feature_rows)
lines = pd.DataFrame(line_rows)
qc = pd.DataFrame(qc_rows)

if features.empty:
    raise RuntimeError("No galaxies processed successfully; inspect QC.")

features.to_csv(OUTPUT_DIR / "galaxy_features.csv", index=False)
lines.to_csv(OUTPUT_DIR / "emission_lines.csv", index=False)
qc.to_csv(OUTPUT_DIR / "quality_control.csv", index=False)

display(qc["status"].value_counts())
display(qc.head())
```

This provides checkpoints, but not automatic resume/config-cache validation. For a production run, use a new output directory after changing wavelength conventions, normalization, fitting settings or input files.

---

## 8. BPT, Balmer decrements and gas attenuation

Only detected lines are used for logarithmic ratios. Unavailable measurements remain missing.

```python name=nebular_diagnostics.py
detected_flux = (
    lines.loc[lines["detected"].fillna(False)]
    .pivot(index="name", columns="line", values="flux")
    .reindex(features["name"])
)

def line_flux(name):
    if name in detected_flux:
        return detected_flux[name]
    return pd.Series(np.nan, index=detected_flux.index)

ha = line_flux("Halpha")
hb = line_flux("Hbeta")
o3 = line_flux("OIII5007")
n2 = line_flux("NII6583")
s2 = line_flux("SII6716") + line_flux("SII6731")
o1 = line_flux("OI6300")
o2 = line_flux("OII3726") + line_flux("OII3729")

diagnostics = pd.DataFrame(index=detected_flux.index)
diagnostics["bpt_y"] = np.log10(o3 / hb)
diagnostics["bpt_NII_x"] = np.log10(n2 / ha)
diagnostics["bpt_SII_x"] = np.log10(s2 / ha)
diagnostics["bpt_OI_x"] = np.log10(o1 / ha)
diagnostics["Halpha_Hbeta"] = ha / hb
diagnostics["Hgamma_Hbeta"] = line_flux("Hgamma") / hb
diagnostics["Hdelta_Hbeta"] = line_flux("Hdelta") / hb

# PyNeb Case B ratios under explicitly assumed temperature/density.
hydrogen = pn.RecAtom("H", 1)
intrinsic_ha_hb = (
    hydrogen.getEmissivity(GAS_TEMPERATURE, GAS_DENSITY, label="3_2")
    / hydrogen.getEmissivity(GAS_TEMPERATURE, GAS_DENSITY, label="4_2")
)

extinction_law = CCM89(Rv=RV)

def k_lambda(angstrom):
    # CCM89 returns A(lambda)/A(V); multiply by Rv for A(lambda)/E(B-V).
    return float(extinction_law(angstrom * u.AA) * RV)

diagnostics["gas_ebv_signed"] = (
    2.5 / (k_lambda(4861.33) - k_lambda(6562.80))
    * np.log10(diagnostics["Halpha_Hbeta"] / intrinsic_ha_hb)
)

# Preserve negative values above as an uncertainty/model-mismatch diagnostic.
diagnostics["gas_ebv_nonnegative"] = diagnostics["gas_ebv_signed"].clip(lower=0)
diagnostics["gas_av_nonnegative"] = RV * diagnostics["gas_ebv_nonnegative"]

# Additional observed excitation ratios.
diagnostics["O32_observed"] = (
    line_flux("OIII4959") + o3
) / o2
diagnostics["R23_observed"] = (
    o2 + line_flux("OIII4959") + o3
) / hb
diagnostics["NeIII_OII_observed"] = line_flux("NeIII3869") / o2
diagnostics["SII_doublet_ratio"] = line_flux("SII6716") / line_flux("SII6731")

# NII BPT: respect the domains of the rational demarcation curves.
def classify_nii(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < 0.05 and y < 0.61 / (x - 0.05) + 1.3:
        return "star-forming"
    if x < 0.47 and y < 0.61 / (x - 0.47) + 1.19:
        return "composite"
    return "AGN-like"

def classify_sii(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < 0.32 and y < 0.72 / (x - 0.32) + 1.30:
        return "star-forming"
    return "Seyfert-like" if y >= 1.89 * x + 0.76 else "LINER-like"

def classify_oi(x, y):
    if not np.isfinite(x) or not np.isfinite(y):
        return "unclassified"
    if x < -0.59 and y < 0.73 / (x + 0.59) + 1.33:
        return "star-forming"
    return "Seyfert-like" if y >= 1.18 * x + 1.30 else "LINER-like"

for panel, classifier in [
    ("NII", classify_nii), ("SII", classify_sii), ("OI", classify_oi)
]:
    diagnostics[f"bpt_{panel}_class"] = [
        classifier(x, y)
        for x, y in zip(diagnostics[f"bpt_{panel}_x"], diagnostics["bpt_y"])
    ]

# Add line EWs as classification candidates, without overwriting line tables.
if "ew_emission" in lines:
    ew_table = (
        lines.loc[lines["detected"].fillna(False)]
        .pivot(index="name", columns="line", values="ew_emission")
        .add_prefix("emission_ew_")
    )
    diagnostics = diagnostics.join(ew_table)

features = features.merge(
    diagnostics.reset_index(), on="name", how="left", validate="one_to_one"
)
features.to_csv(OUTPUT_DIR / "galaxy_features_with_diagnostics.csv", index=False)

for panel in ["NII", "SII", "OI"]:
    counts = pd.crosstab(features["ttype"], features[f"bpt_{panel}_class"])
    fractions = counts.div(counts.sum(axis=1), axis=0)
    counts.to_csv(OUTPUT_DIR / f"BPT_{panel}_counts.csv")
    fractions.to_csv(OUTPUT_DIR / f"BPT_{panel}_fractions_all_galaxies.csv")

display(features[["name", "ttype", "Dn4000", "Halpha_Hbeta", "bpt_NII_class"]].head())
```

The [O II] sum above requires both fitted components to pass detection. This is conservative but can discard a detectable unresolved blend. A more complete implementation should estimate the **sum’s uncertainty using component covariance** and apply detection to the sum.

Likewise, use the [S II] ratio for density only after inspecting resolution and fit reliability. PyNeb can then calculate density at an assumed temperature; out-of-range ratios indicate limits or measurement problems, not arbitrary valid densities.

### Plot BPT diagrams for every Ttype and all types together

```python name=plot_bpt.py
def draw_bpt_boundaries(ax, panel):
    if panel == "NII":
        x = np.linspace(-2, 0.04, 400)
        ax.plot(x, 0.61 / (x - 0.05) + 1.3, "--", color="black",
                label="Kauffmann 2003")
        x = np.linspace(-2, 0.46, 400)
        ax.plot(x, 0.61 / (x - 0.47) + 1.19, "-", color="black",
                label="Kewley 2001")
    elif panel == "SII":
        x = np.linspace(-2, 0.31, 400)
        ax.plot(x, 0.72 / (x - 0.32) + 1.30, color="black")
        x = np.linspace(-0.3, 0.5, 100)
        ax.plot(x, 1.89 * x + 0.76, "--", color="black")
    else:
        x = np.linspace(-2.5, -0.60, 400)
        ax.plot(x, 0.73 / (x + 0.59) + 1.33, color="black")
        x = np.linspace(-1.1, 0, 100)
        ax.plot(x, 1.18 * x + 1.30, "--", color="black")


def plot_bpt(data, title, filename, colored=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = {
        "NII": "[N II]6583 / Hα",
        "SII": "([S II]6716+6731) / Hα",
        "OI": "[O I]6300 / Hα",
    }

    for panel, ax in zip(["NII", "SII", "OI"], axes):
        valid = data.dropna(subset=[f"bpt_{panel}_x", "bpt_y"])
        sns.scatterplot(
            data=valid, x=f"bpt_{panel}_x", y="bpt_y",
            hue="ttype" if colored else None,
            palette="tab20" if colored else None,
            s=15, alpha=0.6, ax=ax,
        )
        draw_bpt_boundaries(ax, panel)
        ax.set(xlabel=f"log₁₀({labels[panel]})",
               ylabel="log₁₀([O III]5007 / Hβ)",
               ylim=(-1.5, 1.5), xlim=(-2.5, 0.6))
        ax.set_title(f"{panel}: {len(valid)} classified-input galaxies")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / filename, dpi=160)
    plt.show()
    plt.close(fig)


plot_bpt(features, "All morphological Ttypes", "BPT_all_ttypes.png", colored=True)

for ttype, group in features.dropna(subset=["ttype"]).groupby("ttype"):
    plot_bpt(group, f"Ttype {ttype}", f"BPT_Ttype_{ttype}.png")
```

**Do not interpret “AGN-like” or “LINER-like” as proof of an active nucleus.** Integrated FoV spectra can combine star formation, diffuse ionized gas, evolved-star ionization and shocks.

---

## 9. Plot every spectrum grouped by Ttype

Every cached, successfully normalized spectrum is included. No random sampling is used.

The combined mean plot contains **one mean spectrum per Ttype**, with equal weight per galaxy.

```python name=plot_spectra_by_morphology.py
means = {}
contributor_counts = {}

fig_mean, ax_mean = plt.subplots(figsize=(16, 6))

for ttype, group in features.dropna(subset=["ttype"]).groupby("ttype"):
    fig, ax = plt.subplots(figsize=(16, 5))
    total = np.zeros(len(GRID), dtype=float)
    count = np.zeros(len(GRID), dtype=int)
    included = 0

    for name in group["name"]:
        cache = OUTPUT_DIR / "spectra_cache" / f"{name}.npz"
        if not cache.exists():
            continue

        with np.load(cache) as saved:
            spectrum = saved["normalized_flux"]

        valid = np.isfinite(spectrum)
        total[valid] += spectrum[valid]
        count[valid] += 1
        included += 1

        # Rasterize the many curves so saved plots remain practical.
        ax.plot(GRID, spectrum, color="steelblue", alpha=0.08,
                linewidth=0.4, rasterized=True)

    if included == 0:
        plt.close(fig)
        continue

    mean = np.divide(total, count, out=np.full(len(GRID), np.nan), where=count > 0)
    means[ttype] = mean
    contributor_counts[ttype] = count

    ax.plot(GRID, mean, color="black", linewidth=1.2, label="Mean")
    ax.set(
        title=f"Ttype {ttype}: all {included} normalized spectra",
        xlabel="Rest-frame air wavelength [Å]",
        ylabel="Flux / normalization-band mean",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figures" / f"all_spectra_Ttype_{ttype}.png", dpi=180
    )
    plt.show()
    plt.close(fig)

    ax_mean.plot(GRID, mean, linewidth=1,
                 label=f"T={ttype}, N={included}")

ax_mean.set(
    title="Mean normalized spectrum of every Ttype",
    xlabel="Rest-frame air wavelength [Å]",
    ylabel="Mean normalized flux",
)
ax_mean.legend(ncol=3)
fig_mean.tight_layout()
fig_mean.savefig(
    OUTPUT_DIR / "figures" / "mean_spectra_all_ttypes.png", dpi=180
)
plt.show()
plt.close(fig_mean)

# Contributor counts are essential: edge regions can contain fewer galaxies.
fig, ax = plt.subplots(figsize=(16, 4))
for ttype, count in contributor_counts.items():
    ax.plot(GRID, count, label=f"T={ttype}")
ax.set(
    title="Number of galaxies contributing at each wavelength",
    xlabel="Rest-frame air wavelength [Å]",
    ylabel="Contributing galaxies",
)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "figures" / "stack_contributor_counts.png", dpi=160)
plt.show()
plt.close(fig)

stack_table = pd.DataFrame({"wave_air_rest_A": GRID})
for ttype in means:
    stack_table[f"mean_T{ttype}"] = means[ttype]
    stack_table[f"count_T{ttype}"] = contributor_counts[ttype]
stack_table.to_csv(OUTPUT_DIR / "mean_spectra_by_ttype.csv", index=False)
```

For detailed inspection, add paginated individual-spectrum panels. Thousands of low-opacity curves show the overall spread but cannot replace inspecting individual galaxies.

---

## 10. Which features actually separate the types?

Use several complementary measures:

1. Distributions and coverage/detection fractions.
2. Overall group tests with effect sizes.
3. Pairwise differences between specific Ttypes.
4. Classification performance and validation-set permutation importance.

A low p-value alone is not strong separation: thousands of objects can make small differences statistically significant.

### Distributions and robust summaries

```python name=feature_distributions.py
candidate_columns = [
    "Dn4000", "continuum_red_blue",
    "light_log_age", "mass_log_age",
    "young_light_fraction", "old_light_fraction",
    "light_Z", "mass_Z", "stellar_av",
    "HdeltaA_observed_ew", "HdeltaA_stellar_model_ew",
    "CaK_observed_ew", "Gband_observed_ew",
    "Mgb_observed_ew", "NaD_observed_ew",
    "CaII_triplet_observed_ew",
    "emission_ew_Halpha", "emission_ew_Hbeta",
    "Halpha_Hbeta", "gas_ebv_signed",
    "bpt_NII_x", "bpt_SII_x", "bpt_OI_x", "bpt_y",
    "O32_observed", "NeIII_OII_observed",
]
candidate_columns = [c for c in candidate_columns if c in features]

summary = features.groupby("ttype")[candidate_columns].agg(
    ["count", "median", lambda values: values.quantile(0.75) - values.quantile(0.25)]
)
summary.to_csv(OUTPUT_DIR / "feature_summary_by_ttype.csv")

available_fraction = features.groupby("ttype")[candidate_columns].agg(
    lambda values: values.notna().mean()
)
available_fraction.to_csv(OUTPUT_DIR / "feature_available_fraction_by_ttype.csv")

for column in candidate_columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=features, x="ttype", y=column, showfliers=False, ax=ax)
    ax.set_title(f"{column} by catalog Ttype")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / f"distribution_{column}.png", dpi=150)
    plt.close(fig)

display(summary)
```

### Overall and pairwise separation

```python name=feature_separation.py
overall_rows = []
pairwise_rows = []

for column in candidate_columns:
    groups = {
        t: group[column].dropna().to_numpy()
        for t, group in features.groupby("ttype")
    }
    groups = {t: values for t, values in groups.items() if len(values) >= 10}
    if len(groups) < 2:
        continue

    pooled = np.concatenate(list(groups.values()))
    if np.unique(pooled).size < 2:
        continue

    statistic, p_value = kruskal(*groups.values())
    n, k = len(pooled), len(groups)

    # Rank-based overall effect-size approximation.
    epsilon_squared = max(0, (statistic - k + 1) / (n - k))
    overall_rows.append({
        "feature": column, "n": n, "types": k,
        "H": statistic, "p": p_value,
        "epsilon_squared": epsilon_squared,
    })

    for first, second in combinations(groups, 2):
        a, b = groups[first], groups[second]
        test = mannwhitneyu(a, b, alternative="two-sided")

        # Signed rank-biserial effect size; magnitude near 0 = overlap.
        effect = 2 * test.statistic / (len(a) * len(b)) - 1
        pairwise_rows.append({
            "feature": column, "ttype_a": first, "ttype_b": second,
            "n_a": len(a), "n_b": len(b),
            "rank_biserial": effect, "absolute_effect": abs(effect),
            "p": test.pvalue,
        })

overall = pd.DataFrame(overall_rows)
pairwise = pd.DataFrame(pairwise_rows)

if not overall.empty:
    overall["q_fdr"] = false_discovery_control(overall["p"], method="bh")
    overall = overall.sort_values("epsilon_squared", ascending=False)
    overall.to_csv(OUTPUT_DIR / "overall_feature_separation.csv", index=False)
    display(overall)

if not pairwise.empty:
    pairwise["q_fdr"] = false_discovery_control(pairwise["p"], method="bh")
    pairwise = pairwise.sort_values("absolute_effect", ascending=False)
    pairwise.to_csv(OUTPUT_DIR / "pairwise_feature_separation.csv", index=False)
    display(pairwise.head(30))
```

These tests use available measurements. Compare availability across Ttypes: line-detection cuts can select very different subsets of early- and late-type galaxies.

### Readable classification baseline

This fits imputation inside each training fold and measures permutation importance on held-out galaxies. It excludes morphology labels and catalog context from the predictors.

```python name=morphology_classification_baseline.py
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)

data = features.dropna(subset=["ttype"]).copy()

# Sparse classes cannot support useful stratified evaluation.
counts = data["ttype"].value_counts()
eligible = counts[counts >= 5].index
excluded = counts[counts < 5]
print("Classes excluded from this baseline due to small counts:")
display(excluded)

data = data[data["ttype"].isin(eligible)].reset_index(drop=True)
usable_features = [
    c for c in candidate_columns if data[c].notna().sum() >= 10
]

if data["ttype"].nunique() < 2 or not usable_features:
    print("Not enough classes/features for classification.")
else:
    X = data[usable_features].replace([np.inf, -np.inf], np.nan)
    y = data["ttype"].astype(str)
    folds = min(5, int(y.value_counts().min()))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = np.empty(len(y), dtype=object)
    importance_rows = []
    scores = []

    for fold, (train, test) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
        )
        model.fit(X.iloc[train], y.iloc[train])
        predictions[test] = model.predict(X.iloc[test])

        scores.append({
            "fold": fold,
            "balanced_accuracy": balanced_accuracy_score(
                y.iloc[test], predictions[test]
            ),
            "macro_f1": f1_score(
                y.iloc[test], predictions[test], average="macro"
            ),
        })

        importance = permutation_importance(
            model, X.iloc[test], y.iloc[test],
            scoring="f1_macro", n_repeats=10,
            random_state=42, n_jobs=-1
        )
        for column, mean, std in zip(
            usable_features, importance.importances_mean,
            importance.importances_std
        ):
            importance_rows.append({
                "fold": fold, "feature": column,
                "importance": mean, "repeat_std": std
            })

    score_table = pd.DataFrame(scores)
    importance_table = pd.DataFrame(importance_rows)
    ranking = (
        importance_table.groupby("feature")["importance"]
        .agg(["mean", "std"]).sort_values("mean", ascending=False)
    )

    score_table.to_csv(OUTPUT_DIR / "classification_cv_scores.csv", index=False)
    importance_table.to_csv(OUTPUT_DIR / "permutation_importance_by_fold.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "classification_feature_ranking.csv")

    display(score_table)
    display(ranking)

    labels = sorted(y.unique())
    matrix = confusion_matrix(y, predictions, labels=labels)
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        xticks_rotation=45, cmap="Blues"
    )
    plt.title("Out-of-fold morphology predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "classification_confusion_matrix.png", dpi=160)
    plt.show()
```

**Interpret this as an exploratory baseline.** Feature importance can be diluted by correlated features. Repeat the evaluation with:

- continuum/absorption features only;
- emission features only;
- STARLIGHT features only;
- their combination;
- optionally catalog mass and inclination added as clearly labeled context.

Do not add `Type`, `Ttype`, IDs or label-derived group names to predictors. Avoid treating redshift, missingness or spectral coverage as evidence of physical morphology separation.

---

# Scientific interpretation and references

## Morphology is not the same as spectral class

The Hubble/de Vaucouleurs sequence describes **appearance**: bulge prominence, disk structure, spiral-arm organization and related characteristics. Early/late does not mean an obligatory evolutionary age sequence.

- **Ellipticals:** commonly dominated by older stellar populations, but can contain gas or recent star formation.
- **S0/lenticulars:** disks without prominent spiral structure; spectra may closely resemble ellipticals.
- **Spirals:** earlier stages often have more prominent bulges; later stages often show stronger recent star formation, with considerable scatter.
- **Bars:** `SB` and `SAB` describe structural families. An integrated spectrum cannot directly identify a bar.
- **Irregular/peculiar systems:** heterogeneous; preserve the actual catalog definitions.

Useful morphology references:

- [Buta, *Galaxy Morphology* review](https://arxiv.org/abs/1102.0550)
- [NED, RC3 introduction and classification documentation](https://ned.ipac.caltech.edu/level5/Sept02/RC3/RC3_contents.html)

## Population and absorption features

A stronger \(D_n4000\) generally indicates an older and/or more metal-rich population. Strong Balmer absorption can indicate substantial intermediate-age stars, but emission infill, dust and metallicity complicate interpretation.

- [Balogh et al. 1999 — \(D_n4000\)](https://ui.adsabs.harvard.edu/abs/1999ApJ...527...54B/abstract)
- [Worthey & Ottaviani 1997 — Hδ/Hγ absorption indices](https://ui.adsabs.harvard.edu/abs/1997ApJS..111..377W/abstract)
- [Bruzual & Charlot 2003 — stellar-population synthesis](https://arxiv.org/abs/astro-ph/0309134)
- [Cid Fernandes et al. 2005 — STARLIGHT](https://arxiv.org/abs/astro-ph/0412481)

STARLIGHT age, metallicity and attenuation estimates are model-dependent. Light-weighted ages emphasize luminous young populations; mass-weighted ages emphasize the accumulated stellar population.

## Emission diagnostics

- [O II], Hα and Hβ are associated with ionized gas, but are not exclusively powered by young stars.
- [O III]/Hβ and [Ne III]/[O II] trace excitation, with dependencies on metallicity and ionization conditions.
- [N II]/Hα, [S II]/Hα and [O I]/Hα help distinguish excitation regimes.
- R23 is not a unique metallicity measurement; metallicity calibrations and branch degeneracy matter.

Primary BPT references:

- [Baldwin, Phillips & Terlevich 1981](https://ui.adsabs.harvard.edu/abs/1981PASP...93....5B/abstract)
- [Kewley et al. 2001](https://arxiv.org/abs/astro-ph/0106324)
- [Kauffmann et al. 2003](https://arxiv.org/abs/astro-ph/0304239)
- [Kewley et al. 2006](https://arxiv.org/abs/astro-ph/0605681)
- [Cid Fernandes et al. 2011 — WHAN and retired galaxies](https://arxiv.org/abs/1012.4426)

## Balmer decrement and dust

Under typical Case B H II-region conditions, Hα/Hβ is near 2.86, but the intrinsic ratio depends on temperature, density and excitation assumptions. Stellar Balmer absorption must be removed first.

The code preserves negative inferred gas reddening as a diagnostic and stores a separately clipped value. **Gas attenuation is not the same quantity as STARLIGHT stellar attenuation.**

- [Cardelli, Clayton & Mathis 1989 — extinction law](https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract)
- [Luridiana et al. 2015 — PyNeb](https://arxiv.org/abs/1410.6662)
- [Astropy FITS documentation](https://docs.astropy.org/en/stable/io/fits/)
- [dust_extinction documentation](https://dust-extinction.readthedocs.io/)

## Before trusting a full-sample ranking

1. Confirm wavelength medium, flux units and both wavelength frames.
2. Inspect several stellar-continuum overlays and emission fits per Ttype.
3. Test synthetic Hα+[N II] blends with known fluxes.
4. Check empirical-noise sensitivity and continuum-subtraction sensitivity.
5. Compare contributor counts and feature availability across types.
6. Examine mass, inclination and aperture effects using catalog context.
7. Report overlap and confusion between neighboring types—not just a “best feature.”

**A defensible result will say which features separate which pairs of Ttypes, how strongly, for what usable sample, and with what uncertainties.**

### Want me to…
<!-- copilot:suggestions -->
1. Turn this workflow into a complete notebook file with paginated spectrum plots, fit-inspection panels, bootstrap uncertainties and tests.
2. Hand this change off to the coding agent to open a pull request after coding agent access is enabled for `Melonangie/Galaxy-Vision`.
3. Add a feature data dictionary and a morphology-focused interpretation report template.

## @maria-rabelero-mx_nestle

Add a feature data dictionary and a morphology-focused interpretation report template.

*System: User retried message*

## Copilot

Add the following **two cells at the end of your Colab notebook**. They create:

- `feature_data_dictionary.csv` — definitions, units, origins, formulas, missing-value rules and modeling cautions.
- `morphology_interpretation_report_template.md` — a structured report for interpreting your actual results.

These cells save files to your existing `OUTPUT_DIR`; they do not modify the GitHub repository.

## 1. Feature data dictionary

The dictionary distinguishes **morphology targets, physical features, catalog context and quality-control information**. This helps prevent label leakage and avoids treating measurement quality as a galaxy property.

```python name=export_feature_dictionary.py
import json
import pandas as pd

dictionary_rows = []

def add_definition(
    table, column, role, unit, origin, definition,
    missing_rule, interpretation, caution, reference=""
):
    """Add one readable definition for one output column."""
    dictionary_rows.append({
        "table": table,
        "column": column,
        "role": role,
        "unit": unit,
        "origin": origin,
        "definition_or_formula": definition,
        "missing_value_rule": missing_rule,
        "morphology_interpretation": interpretation,
        "caution": caution,
        "reference": reference,
    })


# References are attached to relevant definitions, not just listed globally.
REFS = {
    "morphology": "https://arxiv.org/abs/1102.0550",
    "starlight": "https://arxiv.org/abs/astro-ph/0412481",
    "dn4000": "https://ui.adsabs.harvard.edu/abs/1999ApJ...527...54B/abstract",
    "balmer_indices": "https://ui.adsabs.harvard.edu/abs/1997ApJS..111..377W/abstract",
    "bpt": (
        "https://arxiv.org/abs/astro-ph/0106324; "
        "https://arxiv.org/abs/astro-ph/0304239; "
        "https://arxiv.org/abs/astro-ph/0605681"
    ),
    "dust": "https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract",
    "pyneb": "https://arxiv.org/abs/1410.6662",
}

# Tuple order:
# role, unit, origin, definition, missing rule, interpretation, caution, reference
FEATURE_DEFINITIONS = {
    "name": (
        "identifier", "none", "filename/catalog",
        "Canonical manga-<plate>-<ifu> galaxy identifier.",
        "A missing or ambiguous ID prevents a reliable join.",
        "None: identifier only.",
        "Never use as a morphology predictor.", ""
    ),
    "ttype": (
        "target", "catalog stage code", "supporting catalog",
        "Original catalog numerical morphology stage.",
        "Missing labels are excluded from supervised comparisons.",
        "Defines the groups being compared.",
        "Preserve catalog conventions; Ttype=11 is not automatically irregular. "
        "Never include the target or derived groups in predictors.",
        REFS["morphology"]
    ),
    "type": (
        "target/context", "catalog string", "supporting catalog",
        "Original morphology string, including stage and possible bar family.",
        "Unknown strings are preserved rather than guessed.",
        "Provides visual-morphology context.",
        "Label information: exclude from predictors when predicting Ttype.",
        REFS["morphology"]
    ),
    "nsa_z": (
        "context", "dimensionless", "supporting catalog",
        "Catalog redshift used to transform observed wavelengths to rest frame.",
        "Missing/sentinel/invalid redshifts prevent rest-frame measurements.",
        "Controls wavelength registration and selection context.",
        "Not a direct morphology feature; redshift can encode selection bias.", ""
    ),
    "logmass": (
        "context/optional predictor", "catalog log-mass unit", "supporting catalog",
        "Catalog LogMass, copied without recalculation.",
        "Catalog blanks and sentinel values become missing.",
        "Useful for testing whether spectral separation depends on mass.",
        "Confirm log base, solar-mass convention, IMF and cosmology before "
        "physical interpretation; do not infer these from the column name.", ""
    ),
    "incl": (
        "context", "catalog angle unit", "supporting catalog",
        "Catalog inclination estimate.",
        "Catalog blanks and sentinel values become missing.",
        "Useful for checking inclination-related dust and projection effects.",
        "Verify units and method; inclination can be poorly defined for "
        "spheroidal systems.", ""
    ),
    "petro_th90": (
        "context", "catalog angular-size unit", "supporting catalog",
        "Catalog PETRO_TH90 value.",
        "Catalog blanks and sentinel values become missing.",
        "Potential angular-size/aperture context.",
        "Verify radius definition, band and units. This alone does not measure "
        "the fraction of a galaxy sampled by the FoV.", ""
    ),
    "Dn4000": (
        "physical candidate", "dimensionless", "original spectrum",
        "Mean f_nu in 4000–4100 Å divided by mean f_nu in 3850–3950 Å; "
        "f_nu is proportional to f_lambda × wavelength².",
        "Missing if either band is uncovered or the blue mean is nonpositive.",
        "Larger values often indicate older and/or more metal-rich populations.",
        "Age–metallicity degeneracy, emission contamination and coverage matter.",
        REFS["dn4000"]
    ),
    "continuum_red_blue": (
        "physical candidate", "dimensionless", "original spectrum",
        "Mean f_lambda(6000–6100 Å) / mean f_lambda(4500–4600 Å).",
        "Missing for uncovered bands or nonpositive blue continuum.",
        "Describes continuum color/shape.",
        "Affected by dust, populations and relative flux calibration.", ""
    ),
    "stellar_av": (
        "model-derived candidate", "mag", "STARLIGHT",
        "Best-fit AV_min: stellar-population attenuation parameter.",
        "Missing when STARLIGHT parsing/alignment fails.",
        "May describe dust-related population differences.",
        "Model dependent; not the same quantity as nebular gas A_V.",
        REFS["starlight"]
    ),
    "stellar_vdisp_fit": (
        "provisional model-derived candidate", "km/s", "STARLIGHT",
        "Best-fit vd_min velocity-broadening parameter.",
        "Missing when STARLIGHT parsing/alignment fails.",
        "Potential dynamical/population context.",
        "Not automatically intrinsic stellar dispersion: instrumental "
        "resolution, template matching and FoV velocity mixing matter.",
        REFS["starlight"]
    ),
    "starlight_adev": (
        "QC", "%", "STARLIGHT",
        "Reported adev fit-deviation statistic.",
        "Missing when STARLIGHT parsing/alignment fails.",
        "None directly: assesses model agreement.",
        "Not a physical morphology predictor or calibrated uncertainty.",
        REFS["starlight"]
    ),
    "starlight_chi2": (
        "QC", "dimensionless", "STARLIGHT",
        "Reported chi2/Nl_eff.",
        "Missing when STARLIGHT parsing/alignment fails.",
        "None directly: fit-quality context.",
        "Its statistical calibration depends on valid upstream errors; "
        "the supplied ERROR_FoV is known to be incorrect.",
        REFS["starlight"]
    ),
}

# STARLIGHT population summaries use renormalized light/current-mass fractions.
for weighting in ["light", "mass"]:
    FEATURE_DEFINITIONS[f"{weighting}_log_age"] = (
        "model-derived candidate", "log10(age / yr)", "STARLIGHT",
        f"Sum of renormalized {weighting} fractions × log10(SSP age in yr).",
        "Missing for an invalid or unavailable population solution.",
        "Describes the age distribution of the fitted stellar population.",
        "This is mean log age, not log of mean age. Light and mass weighting "
        "answer different questions; model degeneracies remain.",
        REFS["starlight"]
    )
    FEATURE_DEFINITIONS[f"{weighting}_Z"] = (
        "model-derived candidate", "SSP metallicity mass fraction", "STARLIGHT",
        f"Sum of renormalized {weighting} fractions × SSP Z_j.",
        "Missing for an invalid or unavailable population solution.",
        "Describes fitted stellar metallicity.",
        "Not gas metallicity or automatically Z/Z_sun; verify the SSP base.",
        REFS["starlight"]
    )

for age_group, age_range in {
    "young": "age < 100 Myr",
    "intermediate": "100 Myr ≤ age < 1 Gyr",
    "old": "age ≥ 1 Gyr",
}.items():
    FEATURE_DEFINITIONS[f"{age_group}_light_fraction"] = (
        "model-derived candidate", "fraction, 0–1", "STARLIGHT",
        f"Sum of renormalized light fractions for SSPs with {age_range}.",
        "Missing for an invalid or unavailable population solution.",
        "Summarizes relative stellar-population contributions.",
        "These are configurable age bins and light fractions, not mass "
        "fractions or direct star-formation rates.",
        REFS["starlight"]
    )

# Absorption indices: keep observed measurements distinct from model predictions.
for index_name, bands in ABSORPTION_BANDS.items():
    feature_band, blue_band, red_band = bands
    for source in ["observed", "stellar_model"]:
        column = f"{index_name}_{source}_ew"
        FEATURE_DEFINITIONS[column] = (
            "physical candidate" if source == "observed" else "model-derived candidate",
            "rest-frame Å; absorption positive",
            "original spectrum" if source == "observed" else "STARLIGHT model",
            "Integral of (1 - flux / pseudo-continuum) dλ. "
            f"Feature={feature_band}; blue={blue_band}; red={red_band}, "
            "in rest-frame air Å. Continuum is interpolated between sidebands.",
            "Missing for incomplete/nonfinite windows or nonpositive continuum.",
            f"Strength of {index_name} absorption.",
            "Observed values can include emission infill. Model values are "
            "predictions, not independent observations. Ca H includes Hε; "
            "Na D can include ISM absorption; Ca triplet can suffer sky/telluric "
            "contamination. Except HdeltaA, these are exploratory windows, "
            "not standard calibrated Lick indices.",
            REFS["balmer_indices"] if index_name == "HdeltaA" else ""
        )

FEATURE_DEFINITIONS["CaII_triplet_observed_ew"] = (
    "physical candidate", "rest-frame Å; absorption positive", "original spectrum",
    "Sum of CaII8498, CaII8542 and CaII8662 exploratory observed EWs.",
    "Missing unless all three component indices are available.",
    "Combined near-infrared calcium absorption strength.",
    "Not a standard calibrated CaT index; resolution, Paschen absorption, "
    "sky/telluric residuals and redshift-dependent coverage matter.", ""
)

# Nebular ratios and BPT coordinates.
RATIO_DEFINITIONS = {
    "Halpha_Hbeta": ("Hα / Hβ", "Balmer decrement; dust/model-check context."),
    "Hgamma_Hbeta": ("Hγ / Hβ", "Higher-order Balmer consistency check."),
    "Hdelta_Hbeta": ("Hδ / Hβ", "Higher-order Balmer consistency check."),
    "O32_observed": (
        "([O III]4959 + [O III]5007) / ([O II]3726 + [O II]3729)",
        "Excitation/ionization-sensitive ratio."
    ),
    "R23_observed": (
        "([O II]3726 + [O II]3729 + [O III]4959 + [O III]5007) / Hβ",
        "Metallicity- and excitation-sensitive ratio."
    ),
    "NeIII_OII_observed": (
        "[Ne III]3869 / ([O II]3726 + [O II]3729)",
        "Excitation-sensitive ratio."
    ),
    "SII_doublet_ratio": (
        "[S II]6716 / [S II]6731",
        "Potential density diagnostic when both lines are reliably resolved."
    ),
}
for column, (formula, interpretation) in RATIO_DEFINITIONS.items():
    FEATURE_DEFINITIONS[column] = (
        "physical candidate", "dimensionless", "stellar-subtracted emission fits",
        formula,
        "Missing unless every required component is detected.",
        interpretation,
        "Observed, not dust-corrected. Errors are empirical approximations. "
        "OII sums use conservative component detection; R23 is not a unique "
        "metallicity measurement; density requires assumed temperature.",
        REFS["pyneb"]
    )

BPT_FORMULAS = {
    "bpt_y": "log10([O III]5007 / Hβ)",
    "bpt_NII_x": "log10([N II]6583 / Hα)",
    "bpt_SII_x": "log10(([S II]6716 + [S II]6731) / Hα)",
    "bpt_OI_x": "log10([O I]6300 / Hα)",
}
for column, formula in BPT_FORMULAS.items():
    FEATURE_DEFINITIONS[column] = (
        "physical candidate", "dex", "stellar-subtracted emission fits",
        formula, "Missing unless required positive line fluxes are detected.",
        "Describes ionization conditions, not visual structure.",
        "Integrated FoV mixing, shocks, diffuse gas and evolved stars can "
        "produce non-star-forming ratios without an AGN.",
        REFS["bpt"]
    )

for panel in ["NII", "SII", "OI"]:
    FEATURE_DEFINITIONS[f"bpt_{panel}_class"] = (
        "derived diagnostic category", "category", "BPT coordinates",
        f"Classification from the {panel} BPT demarcations in the notebook.",
        "'unclassified' means insufficient usable line information.",
        "Ionization-category context within each morphology group.",
        "Not a morphology label or definitive AGN identification. Avoid "
        "including both the category and its ratios without redundancy checks.",
        REFS["bpt"]
    )

for column in ["gas_ebv_signed", "gas_ebv_nonnegative", "gas_av_nonnegative"]:
    FEATURE_DEFINITIONS[column] = (
        "assumption-dependent physical candidate", "mag",
        "Balmer decrement + PyNeb Case B + CCM89",
        {
            "gas_ebv_signed": (
                "2.5 / (k(Hβ)-k(Hα)) × "
                "log10[(Hα/Hβ)_measured / (Hα/Hβ)_CaseB]."
            ),
            "gas_ebv_nonnegative": "max(gas_ebv_signed, 0).",
            "gas_av_nonnegative": "R_V × gas_ebv_nonnegative.",
        }[column],
        "Missing unless the Balmer decrement is available.",
        "Estimated nebular reddening/attenuation.",
        "Temperature, density, Case B, extinction law and continuum subtraction "
        "are assumptions. Preserve signed reddening for QC. Clipping changes "
        "the distribution; stellar A_V is a different measurement.",
        REFS["dust"] + "; " + REFS["pyneb"]
    )

# Export a definition for every column actually present in the feature table.
for column in features.columns:
    if column.startswith("emission_ew_"):
        definition = (
            "physical candidate", "rest-frame Å; emission positive",
            "stellar-subtracted emission fit",
            "Fitted integrated line flux / local stellar continuum density.",
            "Missing unless the line is detected and continuum is positive.",
            "Emission strength relative to underlying stellar light.",
            "Not a direct SFR. Sensitive to continuum fitting, FoV mixing, "
            "dust and excitation source.", ""
        )
    else:
        definition = FEATURE_DEFINITIONS.get(column)

    if definition is None:
        definition = (
            "REVIEW REQUIRED", "unverified", "unverified",
            "New column: document before interpreting or modeling.",
            "Unspecified.", "Unspecified.",
            "Do not use until its definition is reviewed.", ""
        )

    add_definition("galaxy_features_with_diagnostics", column, *definition)


# Long-format emission-line table: units and censoring must remain explicit.
LINE_DEFINITIONS = {
    "name": FEATURE_DEFINITIONS["name"],
    "ttype": FEATURE_DEFINITIONS["ttype"],
    "line": (
        "identifier", "line name", "configured line list",
        "Unambiguous emission component identifier.",
        "Always present for attempted measurements.", "Identifies the transition.",
        "Use the configured air wavelength, not the rounded name, for fitting.", ""
    ),
    "group": (
        "measurement context", "group name", "configured line list",
        "Components fitted jointly with shared velocity and width.",
        "Always present.", "None directly.",
        "Joint fitting introduces component covariance.", ""
    ),
    "flux": (
        "physical measurement", f"{FLUX_UNIT_DESCRIPTION} × Å",
        "stellar-subtracted Gaussian fit", "Integral of the fitted Gaussian.",
        "Missing if the fit/window/model is unavailable.",
        "Emission strength in the sampled FoV.",
        "Nonnegative fitting biases weak signals upward; not luminosity. "
        "A numerical fit value is not automatically a detection.", ""
    ),
    "empirical_error": (
        "measurement uncertainty approximation", f"{FLUX_UNIT_DESCRIPTION} × Å",
        "local residual scatter + fit Jacobian",
        "Conditional parameter error from the approximate fit covariance.",
        "Missing for unavailable/invalid uncertainty estimates.",
        "None directly.",
        "Not ERROR_FoV. Omits stellar-model systematics and pixel correlation.", ""
    ),
    "snr_empirical": (
        "QC/detection statistic", "dimensionless", "line fit",
        "flux / empirical_error.", "Missing when empirical_error is invalid.",
        "None directly.", "Not calibrated instrumental S/N.", ""
    ),
    "ew_emission": (
        "physical measurement", "rest-frame Å; emission positive",
        "line fit + stellar continuum", "flux / local stellar continuum.",
        "Missing when the local continuum is nonpositive/unavailable.",
        "Line strength relative to stellar light.",
        "Check continuum quality before interpreting.", ""
    ),
    "detected": (
        "QC/censoring", "boolean", "configured threshold + fit QC",
        "True when empirical S/N passes the threshold and the fit is not "
        "at the configured velocity/width boundary.",
        "False also includes unavailable measurements; inspect status.",
        "Defines usable line subsets.",
        "False does not mean physical absence of emission.", ""
    ),
    "upper_limit_approx": (
        "approximate censoring bound", f"{FLUX_UNIT_DESCRIPTION} × Å",
        "line fit", "max(fitted flux, 0) + threshold × empirical_error.",
        "Missing when the region/error is unavailable.",
        "Exploratory constraint for a nondetection.",
        "Not a calibrated confidence limit; calculated even for detections, "
        "where fitted flux and uncertainty should be reported instead.", ""
    ),
    "velocity_fit": (
        "provisional fit parameter", "km/s", "joint emission fit",
        "Common fitted line-center velocity offset on the registered grid.",
        "Missing for failed fits.", "Potential registration/kinematic context.",
        "Do not equate with catalog systemic velocity.", ""
    ),
    "width_fit": (
        "provisional fit parameter", "km/s; Gaussian sigma", "joint emission fit",
        "Shared fitted Gaussian velocity width.",
        "Missing for failed fits.", "Potential kinematic context.",
        "Includes instrumental and FoV broadening; not intrinsic dispersion.", ""
    ),
    "boundary_fit": (
        "QC", "boolean", "line fit",
        "Fit approaches the configured velocity or width bounds.",
        "Missing for unavailable fits.", "None directly.",
        "Boundary fits are excluded from detections in this workflow.", ""
    ),
    "residual_rms_over_noise": (
        "QC", "dimensionless", "line fit",
        "Standard deviation of fit residuals / sideband empirical noise.",
        "Missing for failed fits.", "None directly.",
        "The current detection rule does not reject a fit solely using this "
        "statistic; inspect poor residual fits separately.", ""
    ),
    "status": (
        "QC", "string", "processing",
        "'measured' or the reason the measurement could not be completed.",
        "Should always be recorded.", "None directly.",
        "Distinguish no coverage, missing model and fit failure.", ""
    ),
}

for column in lines.columns:
    definition = LINE_DEFINITIONS.get(column, (
        "REVIEW REQUIRED", "unverified", "unverified",
        "Document this new line-table column.", "Unspecified.",
        "Unspecified.", "Do not interpret before review.", ""
    ))
    add_definition("emission_lines", column, *definition)

dictionary = pd.DataFrame(dictionary_rows)
dictionary.to_csv(OUTPUT_DIR / "feature_data_dictionary.csv", index=False)

# Save assumptions alongside the dictionary so interpretation is reproducible.
assumptions = {
    "fits_frame": FITS_FRAME,
    "starlight_frame": STARLIGHT_FRAME,
    "fits_wavelength_medium": FITS_WAVELENGTH_MEDIUM,
    "starlight_wavelength_medium": STARLIGHT_WAVELENGTH_MEDIUM,
    "measurement_wavelength_medium": "air",
    "flux_unit_description": FLUX_UNIT_DESCRIPTION,
    "ERROR_FoV_used": False,
    "rest_flux_density_transform": "f_lambda_rest = (1+z) * f_lambda_obs",
    "normalization_band_rest_air_A": list(NORMALIZATION_BAND),
    "detection_snr_empirical": DETECTION_SNR,
    "max_continuum_mismatch": MAX_CONTINUUM_MISMATCH,
    "gas_temperature_K": GAS_TEMPERATURE,
    "case_b_density_cm_minus3": GAS_DENSITY,
    "extinction_law": "CCM89",
    "R_V": RV,
}
(OUTPUT_DIR / "measurement_assumptions.json").write_text(
    json.dumps(assumptions, indent=2), encoding="utf-8"
)

print("Saved feature_data_dictionary.csv and measurement_assumptions.json")
display(dictionary[["table", "column", "role", "unit"]])
```

**Scope:** this dictionary documents the galaxy-feature and emission-line tables. QC logs, stack contributor counts and statistical ranking outputs should be reported separately—not mixed with physical predictors.

---

## 2. Morphology-focused interpretation report template

The template deliberately leaves scientific conclusions blank. It prevents expected morphology trends from being mistaken for findings from your sample.

```python name=export_morphology_report_template.py
report_template = """# Galaxy morphology and spectral-feature interpretation report

> TEMPLATE — replace bracketed fields with measured results.
> Do not describe expected trends as observed findings.
> An unavailable line is not a zero-flux measurement.

## 1. Executive summary

- Research question: Which spectral and STARLIGHT features distinguish the
  catalog morphological Ttypes in this sample?
- Data release/extraction version: [fill in]
- Analysis date and notebook/code version: [fill in]
- Galaxies discovered: [N]
- Galaxies with valid catalog joins and redshifts: [N]
- Galaxies included in spectral comparisons: [N]
- Galaxies with validated STARLIGHT continua: [N]
- Strongest supported feature differences: [features and specific Ttype pairs]
- Most difficult morphology distinctions: [pairs with substantial overlap]
- Main limitations: [coverage, errors, continuum fitting, class imbalance, etc.]

Summary statement:
[Feature] differs between [Ttype A] and [Ttype B], with [effect size and
uncertainty], using [N_A] and [N_B] usable galaxies. This supports a difference
in [population/ionization property], but does not directly establish
[spiral structure/bar presence/causal evolutionary pathway].

## 2. Data, morphology labels and sample definition

### 2.1 Inputs

- Galaxy spectra: original WAVE and FLUX_FoV from individual FITS files.
- Stellar fits: STARLIGHT outputs and SSP base [name/version].
- Supporting catalog: labels, redshifts and contextual properties only.
- ERROR_FoV: excluded because it is known to be incorrect.
- CSV samples: structural examples; gaps cannot supply missing diagnostics.

### 2.2 Catalog conventions

Record the actual Type strings associated with each Ttype.

| Ttype | Catalog Type strings | Number available | Interpretation/ambiguity |
|---|---|---:|---|
| [T] | [strings] | [N] | [verified convention] |

Do not automatically assign irregular morphology to Ttype=11.
Do not interpret "early" and "late" as a compulsory evolutionary sequence.
Distinguish numerical stage from SB/SAB bar-family information.

### 2.3 Inclusion and exclusion accounting

| Processing stage

## @maria-rabelero-mx_nestle

Add a feature data dictionary and a morphology-focused interpretation report template.

*System: User retried message*

## Copilot

Add these **two cells at the end of your Colab notebook**. They save:

- `feature_data_dictionary.csv` — definitions, units, origins, modeling roles and caveats.
- `morphology_interpretation_report_template.md` — a report you can fill in after running the analysis.

These additions document the workflow from our conversation; **they do not modify the GitHub repository or invent scientific results**.

## 1. Feature data dictionary

The dictionary distinguishes **morphology labels**, **physical feature candidates**, **catalog context** and **quality-control fields**. Unknown columns are flagged for review rather than silently treated as predictors.

```python name=feature_data_dictionary.py
# Run after the previous notebook cells.
# Requires OUTPUT_DIR and the earlier feature/line definitions.

dictionary_rows = []

def add_field(dataset, field, unit, origin, definition, role, caveat=""):
    """One readable dictionary entry per output column."""
    dictionary_rows.append({
        "dataset": dataset,
        "field": field,
        "unit": unit,
        "origin": origin,
        "definition": definition,
        "modeling_role": role,
        "caveat": caveat,
    })


# ------------------------------------------------------------
# A. Identifiers, morphology labels and supporting information
# ------------------------------------------------------------
metadata_fields = [
    ("name", "none", "Canonical manga-<plate>-<ifu> galaxy identifier.",
     "identifier; exclude from predictors", "Must be unique per galaxy."),
    ("ttype", "catalog code", "Morphological stage supplied by the catalog.",
     "target; exclude from predictors",
     "Preserve catalog conventions; do not assume Ttype=11 means irregular."),
    ("type", "catalog label", "Catalog morphology string, such as E, S0 or SBab.",
     "label; exclude from predictors",
     "Contains target information, including structural family labels."),
    ("nsa_z", "dimensionless", "Catalog redshift used for rest-frame conversion.",
     "context/control; not a primary morphology feature",
     "Missing/sentinel redshifts cannot be imputed for spectral registration."),
    ("logmass", "catalog-defined logarithmic mass",
     "Supporting catalog LogMass value.",
     "optional augmented predictor/control",
     "Verify mass units, logarithm base, IMF and cosmological convention."),
    ("incl", "catalog-defined; verify degrees",
     "Supporting catalog inclination estimate.",
     "context/control",
     "Inclination affects dust, projected structure and FoV velocity mixing."),
    ("petro_th90", "catalog-defined angular size",
     "Supporting catalog PETRO_TH90 value.",
     "context/control",
     "Verify band and units; this alone does not measure FoV coverage fraction."),
]

for field, unit, definition, role, caveat in metadata_fields:
    add_field("galaxy_features", field, unit, "supporting catalog",
              definition, role, caveat)


# ------------------------------------------------------------
# B. Stellar populations and STARLIGHT fit diagnostics
# ------------------------------------------------------------
population_fields = [
    ("light_log_age", "log10(year)",
     "sum(normalized x_j * log10(age_j / year))",
     "Luminous young stars can dominate this mean."),
    ("mass_log_age", "log10(year)",
     "sum(normalized Mcor_j * log10(age_j / year))",
     "Uses current stellar mass fractions, not initial mass fractions."),
    ("light_Z", "SSP metallicity mass fraction",
     "sum(normalized x_j * Z_j)",
     "Linear metallicity mean; not log(Z/Z_sun) or gas metallicity."),
    ("mass_Z", "SSP metallicity mass fraction",
     "sum(normalized Mcor_j * Z_j)",
     "Linear metallicity mean; depends on the adopted SSP library."),
    ("young_light_fraction", "fraction",
     "sum(normalized x_j for age_j < 1e8 year)",
     "Young/intermediate/old boundaries are analysis choices."),
    ("intermediate_light_fraction", "fraction",
     "sum(normalized x_j for 1e8 <= age_j < 1e9 year)",
     "A coarse population summary, not a resolved star-formation history."),
    ("old_light_fraction", "fraction",
     "sum(normalized x_j for age_j >= 1e9 year)",
     "Correlated with the other fractions; fractions sum to one."),
    ("stellar_av", "mag",
     "STARLIGHT AV_min stellar attenuation parameter",
     "Not interchangeable with gas attenuation from the Balmer decrement."),
]

population_caveat = (
    "Model-dependent and provisional if upstream uncertainties were incorrect. "
    "Population weights are renormalized before averaging."
)

for field, unit, definition, caveat in population_fields:
    add_field("galaxy_features", field, unit, "STARLIGHT best-model output",
              definition, "physical feature candidate",
              caveat + " " + population_caveat)

for field, unit, definition, caveat in [
    ("starlight_adev", "percent", "STARLIGHT adev fit-deviation statistic.",
     "Fit quality, not a physical morphology measurement."),
    ("starlight_chi2", "dimensionless", "STARLIGHT chi2/Nl_eff statistic.",
     "Not a calibrated reduced chi-square if the input errors are wrong."),
    ("stellar_vdisp_fit", "km/s", "STARLIGHT vd_min fitted broadening.",
     "Requires instrumental/template-resolution validation before physical use."),
]:
    add_field("galaxy_features", field, unit, "STARLIGHT best-model output",
              definition,
              "QC" if field != "stellar_vdisp_fit"
              else "provisional physical candidate",
              caveat)


# ------------------------------------------------------------
# C. Continuum and absorption features
# ------------------------------------------------------------
add_field(
    "galaxy_features", "Dn4000", "dimensionless", "original spectrum",
    "Mean f_nu in 4000–4100 Å divided by mean f_nu in 3850–3950 Å; "
    "f_nu is proportional to f_lambda * wavelength^2.",
    "physical feature candidate",
    "Age–metallicity degeneracy. The current implementation averages over "
    "available endpoint samples within each fully covered band."
)

add_field(
    "galaxy_features", "continuum_red_blue", "dimensionless",
    "original spectrum",
    "Mean f_lambda(6000–6100 Å) / mean f_lambda(4500–4600 Å).",
    "physical feature candidate",
    "Affected by stellar populations, attenuation and flux calibration."
)

absorption_meanings = {
    "CaK": "Ca II K absorption near 3933.66 Å.",
    "CaH_blend": "Ca II H region near 3968.47 Å, blended with H epsilon.",
    "Gband": "Broad CH molecular G-band absorption near 4300 Å.",
    "Mgb": "Mg b triplet region near 5167, 5173 and 5184 Å.",
    "NaD": "Na D doublet region near 5890 and 5896 Å.",
    "CaII8498": "Near-infrared Ca II triplet component near 8498 Å.",
    "CaII8542": "Near-infrared Ca II triplet component near 8542 Å.",
    "CaII8662": "Near-infrared Ca II triplet component near 8662 Å.",
    "HdeltaA": "H delta A absorption index using published bandpasses.",
}

# Use the exact band definitions from the measurement cell.
for index, bands in ABSORPTION_BANDS.items():
    feature_band, blue_band, red_band = bands

    for origin in ["observed", "stellar_model"]:
        add_field(
            "galaxy_features", f"{index}_{origin}_ew", "rest-frame Å",
            "original spectrum" if origin == "observed"
            else "validated, rescaled STARLIGHT synthetic spectrum",
            f"{absorption_meanings[index]} "
            f"Integral of (1 - flux/pseudo_continuum) dλ. "
            f"Feature={feature_band}; blue={blue_band}; red={red_band} Å.",
            "physical feature candidate" if origin == "observed"
            else "model-derived feature candidate",
            "Positive means absorption. Observed indices may contain emission "
            "infill; model indices are predictions, not independent observations. "
            "Except HdeltaA, these are exploratory bandpasses; none is claimed "
            "to be resolution-matched or Lick-calibrated. Na D may include ISM "
            "absorption; near-IR Ca lines require sky/telluric checks."
        )

add_field(
    "galaxy_features", "CaII_triplet_observed_ew", "rest-frame Å",
    "original spectrum",
    "Sum of CaII8498, CaII8542 and CaII8662 observed absorption EWs, "
    "only when all three indices are available.",
    "physical feature candidate",
    "Missing components yield NaN, not a partial triplet sum."
)


# ------------------------------------------------------------
# D. Emission-line table and wide EW features
# ------------------------------------------------------------
line_definitions = [
    ("line", "none", "Unique component name, such as OIII5007.", "identifier"),
    ("group", "none", "Joint fitting group, such as Halpha_NII.", "identifier"),
    ("flux", "input flux-density unit × Å",
     "Integrated fitted Gaussian flux after stellar-continuum subtraction.",
     "measurement; prefer ratios/EWs for morphology comparisons"),
    ("empirical_error", "same as flux",
     "Conditional flux-error approximation from the fit Jacobian and "
     "empirical sideband noise.", "uncertainty/QC"),
    ("snr_empirical", "dimensionless",
     "Fitted flux / empirical_error.", "detection/QC"),
    ("ew_emission", "rest-frame Å",
     "Fitted line flux / underlying stellar continuum at the line center.",
     "physical feature candidate"),
    ("detected", "boolean",
     "Empirical S/N passes DETECTION_SNR and velocity/width fit is "
     "not flagged at a boundary.", "selection/QC"),
    ("upper_limit_approx", "same as flux",
     "max(fitted flux, 0) + DETECTION_SNR * empirical_error.",
     "approximate limit"),
    ("velocity_fit", "km/s",
     "Shared group velocity offset relative to adopted rest wavelengths.",
     "fit diagnostic"),
    ("width_fit", "km/s",
     "Shared Gaussian velocity sigma, not FWHM.", "fit diagnostic"),
    ("boundary_fit", "boolean",
     "Fitted velocity or width lies close to an imposed parameter bound.",
     "QC"),
    ("residual_rms_over_noise", "dimensionless",
     "Standard deviation of fit residuals / empirical sideband noise.",
     "QC"),
    ("status", "text",
     "'measured' or the reason a measurement was unavailable.", "QC"),
]

for field, unit, definition, role in line_definitions:
    add_field(
        "emission_lines", field, unit, "continuum-subtracted Gaussian fit",
        definition, role,
        "ERROR_FoV is never used. Errors/limits omit stellar-model uncertainty "
        "and pixel covariance; limits are not calibrated confidence bounds. "
        "Widths include instrumental broadening and aperture mixing."
    )

for field in ["name", "ttype"]:
    definition = next(row for row in dictionary_rows
                      if row["dataset"] == "galaxy_features"
                      and row["field"] == field)
    add_field("emission_lines", field, definition["unit"],
              definition["origin"], definition["definition"],
              definition["modeling_role"], definition["caveat"])

for components in EMISSION_GROUPS.values():
    for line, center in components:
        add_field(
            "galaxy_features", f"emission_ew_{line}", "rest-frame Å",
            "detected emission-line fit",
            f"Emission-positive EW of {line}; adopted air center {center} Å.",
            "physical feature candidate",
            "Stored only for detected lines. NaN does not mean zero emission."
        )


# ------------------------------------------------------------
# E. Nebular ratios and excitation classes
# ------------------------------------------------------------
ratio_definitions = {
    "bpt_y": "log10([OIII]5007 / Hbeta)",
    "bpt_NII_x": "log10([NII]6583 / Halpha)",
    "bpt_SII_x": "log10(([SII]6716 + [SII]6731) / Halpha)",
    "bpt_OI_x": "log10([OI]6300 / Halpha)",
    "Halpha_Hbeta": "Halpha / Hbeta",
    "Hgamma_Hbeta": "Hgamma / Hbeta",
    "Hdelta_Hbeta": "Hdelta / Hbeta",
    "O32_observed": "([OIII]4959 + [OIII]5007) / ([OII]3726 + [OII]3729)",
    "R23_observed": "([OII]3726 + [OII]3729 + [OIII]4959 + [OIII]5007) / Hbeta",
    "NeIII_OII_observed": "[NeIII]3869 / ([OII]3726 + [OII]3729)",
    "SII_doublet_ratio": "[SII]6716 / [SII]6731",
}

for field, definition in ratio_definitions.items():
    add_field(
        "galaxy_features", field, "dimensionless", "detected emission-line fluxes",
        definition, "physical feature candidate",
        "Observed, not internally dust-corrected. All required components must "
        "be detected in the current code. This conservatively excludes some "
        "detectable blends. BPT measures excitation, not morphology; R23 is "
        "not a unique metallicity estimate. SII density requires resolution "
        "and temperature assumptions."
    )

for panel in ["NII", "SII", "OI"]:
    add_field(
        "galaxy_features", f"bpt_{panel}_class", "category",
        "BPT ratios and literature demarcations",
        f"Excitation class from the {panel} BPT panel; unavailable inputs "
        "produce 'unclassified'.",
        "derived excitation descriptor; not a morphology target",
        "AGN-like/LINER-like does not prove nuclear activity. Avoid including "
        "both ratios and derived classes without checking redundancy."
    )

for field, definition in [
    ("gas_ebv_signed",
     "2.5 / (k_Hbeta - k_Halpha) * "
     "log10(observed Halpha/Hbeta / assumed PyNeb Case-B ratio)"),
    ("gas_ebv_nonnegative", "max(gas_ebv_signed, 0)"),
    ("gas_av_nonnegative", "RV * gas_ebv_nonnegative"),
]:
    add_field(
        "galaxy_features", field, "mag", "Balmer decrement + extinction law",
        definition, "model-derived physical feature candidate",
        "Assumes configured gas temperature/density and CCM89 Rv. Negative "
        "signed values are preserved as diagnostics; clipping creates a "
        "pile-up at zero. Gas attenuation is distinct from stellar AV."
    )


# ------------------------------------------------------------
# F. Quality-control output
# ------------------------------------------------------------
for field, definition in [
    ("name", "Canonical galaxy identifier."),
    ("path", "Input spectrum path."),
    ("status", "Overall processing outcome or failure reason."),
    ("starlight_status", "Missing, duplicate, aligned or model-validation failure."),
    ("continuum_mismatch",
     "Maximum absolute fractional deviation of continuum-band scale ratios "
     "from their median ratio."),
    ("stack_status", "Spectrum-cache outcome or normalization failure."),
]:
    add_field(
        "quality_control", field,
        "fraction" if field == "continuum_mismatch" else "text",
        "processing pipeline", definition, "QC; exclude from physical predictors"
    )


# ------------------------------------------------------------
# G. Include every actual output column; flag unreviewed additions
# ------------------------------------------------------------
known = {(row["dataset"], row["field"]) for row in dictionary_rows}

for dataset, variable in [
    ("galaxy_features", "features"),
    ("emission_lines", "lines"),
    ("quality_control", "qc"),
]:
    table = globals().get(variable)
    if isinstance(table, pd.DataFrame):
        for field in table.columns:
            if (dataset, field) not in known:
                add_field(
                    dataset, field, "REVIEW", "REVIEW",
                    "Column exists but has not yet been documented.",
                    "exclude until reviewed",
                    "Add an explicit definition before interpreting or modeling."
                )

dictionary = pd.DataFrame(dictionary_rows)
dictionary["missing_value_policy"] = (
    "Numeric NaN = unavailable/invalid/not detected, depending on field; "
    "consult line status and QC. Never automatically interpret NaN as zero."
)

dictionary.to_csv(OUTPUT_DIR / "feature_data_dictionary.csv", index=False)

display(dictionary[["dataset", "field", "unit", "modeling_role"]])
print("Saved:", OUTPUT_DIR / "feature_data_dictionary.csv")
```

**Important distinction:** `light_log_age` is the **weighted mean of log age**, not the logarithm of the weighted mean age. Similarly, `light_Z` and `mass_Z` are linear metallicity mass fractions, not logarithmic abundances.

---

## 2. Morphology-focused interpretation report template

This creates an editable Markdown report in your output folder. Bracketed prompts deliberately remain unfilled until you have measured results.

```python name=create_morphology_report_template.py
report_template = """# Galaxy Morphology and Spectral Feature Interpretation

> Status: TEMPLATE — no scientific findings are implied.
> Replace bracketed prompts with measured results.
> Distinguish exploratory associations from validated predictive performance.

## 1. Executive summary

**Research question:** Which integrated spectral and STARLIGHT features
best separate the catalog morphological Ttypes in this sample?

- Galaxies discovered: [N]
- Galaxies with usable rest-frame spectra: [N]
- Galaxies with validated STARLIGHT continua: [N]
- Morphological stages represented: [list catalog Ttypes and Type strings]
- Most informative features: [results, not prior expectations]
- Best-separated type pairs: [pairs, effect sizes, uncertainty]
- Most overlapping type pairs: [pairs and confusion evidence]
- Cross-validated balanced accuracy: [mean and fold variation]
- Cross-validated macro F1: [mean and fold variation]
- Main limitations: [coverage, labels, dust, aperture, fitting, selection]

**One-sentence conclusion:**
[State what the data support about morphology separation, without equating
spectral excitation classes with morphological types.]

## 2. Dataset, provenance and measurement conventions

### 2.1 Inputs and reproducibility

- Run date / analysis identifier: [...]
- Notebook version / commit: [...]
- Spectrum directory and file count: [...]
- STARLIGHT version, configuration and SSP base: [...]
- Supporting catalog version and source: [...]
- Configuration snapshot and dependency versions: [...]
- Output directory: [...]

The supporting catalog supplies labels, redshifts and contextual controls.
It is not treated as a separate standalone EDA dataset.

### 2.2 Spectral conventions

- Spectrum wavelength unit: [...]
- Spectrum wavelength medium: [air/vacuum; evidence]
- Original spectrum frame: [observed/rest; evidence]
- STARLIGHT wavelength medium/frame: [...]
- Flux-density unit and calibration evidence: [...]
- Rest-frame wavelength conversion: lambda_rest = lambda_obs / (1 + z)
- Flux-density coordinate convention: [...]
- Galactic foreground extinction treatment: [...]
- Internal gas attenuation treatment: [...]
- Shape-normalization band: [...]
- Emission EW convention: positive emission, rest-frame Angstrom
- Absorption EW convention: positive absorption, rest-frame Angstrom

ERROR_FoV was excluded from all measurements, weights and uncertainties.
Empirical fit errors are provisional and do not include all correlated-pixel
or stellar-continuum uncertainties.

## 3. Morphology labels and sample selection

### 3.1 Catalog conventions

Explain what each observed Ttype means in THIS catalog.
Preserve ambiguous codes and unusual Type strings.

| Ttype | Associated catalog Type strings | N discovered | N usable | Label caveat |
|---|---|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] |

Do not automatically interpret Ttype=11 as irregular: the supplied sample
includes Type='S' at that code.

### 3.2 Sample flow and exclusions

| Stage / exclusion | Count | Reason | Dependence on Ttype |
|---|---:|---|---|
| Spectrum inventory | [...] | [...] | [...] |
| Duplicate identifiers | [...] | [...] | [...] |
| Missing catalog matches | [...] | [...] | [...] |
| Invalid redshifts | [...] | [...] | [...] |
| Unusable spectral coverage | [...] | [...] | [...] |
| Missing/invalid STARLIGHT fit | [...] | [...] | [...] |
| Failed normalization | [...] | [...] | [...] |
| Usable BPT inputs | [...] | [...] | [...] |

Describe whether exclusions disproportionately remove particular stages.

## 4. Spectral and continuum quality assessment

- STARLIGHT alignment checks and continuum mismatch: [...]
- Fit inspection sample per Ttype: [...]
- Sky/telluric contamination handling: [...]
- Gaussian boundary fits / unusual profiles: [...]
- Empirical uncertainty estimation and validation: [...]
- Original CSV gaps versus actual FITS coverage: [...]
- Instrumental resolution and FoV velocity mixing: [...]

Report any diagnostics that remain unvalidated.
Do not treat a completed fit as proof of a reliable measurement.

## 5. Spectra grouped by morphology

Insert:
1. All normalized spectra for each Ttype.
2. A single graph containing one mean normalized spectrum per Ttype.
3. Wavelength-dependent contributor counts.
4. Selected line/absorption-region zooms.

### Interpretation

- Continuum-shape differences: [...]
- Dn4000 differences: [...]
- Balmer emission/absorption differences: [...]
- Metal/molecular absorption differences: [...]
- Within-type diversity and outliers: [...]
- Regions where changing contributor counts affect comparisons: [...]

Means use equal galaxy weights after continuum normalization.
Mean spectra are not substitutes for individual-galaxy diagnostics.

## 6. Stellar populations and absorption by Ttype

| Feature | Ttype comparison | N per type | Median / IQR | Effect size | Interpretation |
|---|---|---|---|---|---|
| Dn4000 | [...] | [...] | [...] | [...] | [...] |
| Light-weighted log age | [...] | [...] | [...] | [...] | [...] |
| Mass-weighted log age | [...] | [...] | [...] | [...] | [...] |
| Young light fraction | [...] | [...] | [...] | [...] | [...] |
| HdeltaA observed | [...] | [...] | [...] | [...] | [...] |
| HdeltaA model | [...] | [...] | [...] | [...] | [...] |
| Mg b / G band / Ca K | [...] | [...] | [...] | [...] | [...] |
| Na D | [...] | [...] | [...] | [...] | [...] |
| Ca II triplet | [...] | [...] | [...] | [...] | [...] |

Discuss:
- Age–metallicity–dust degeneracy.
- Emission infill in observed Balmer absorption.
- Model predictions versus direct measurements.
- Non-independence of population fractions.
- Na D stellar versus interstellar contributions.
- Exploratory versus published absorption bandpasses.
- Resolution matching and lack of Lick calibration.

## 7. Emission-line and excitation diagnostics

### 7.1 Line availability

| Line / blend | Ttype | N covered | N measured | N detected | Detection fraction denominator |
|---|---|---:|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] | [...] |

Separate coverage failures, continuum failures and nondetections.
An unavailable measurement is not zero line emission.

### 7.2 BPT results

Insert NII, SII and OI BPT diagrams for each Ttype and all Ttypes combined.

| Ttype | N total | N with valid BPT inputs | SF | Composite | AGN-like / Seyfert-like | LINER-like | Unclassified |
|---|---:|---:|---:|---:|---:|---:|---:|
| [...] | [...] | [...] | [...] | [...] | [...] | [...] | [...] |

State whether fractions use ALL galaxies or only classifiable galaxies.
Do not combine these denominators silently.

Interpretation:
[Describe excitation distributions and their overlap across morphology.]

Caution:
BPT classes describe excitation, not appearance. AGN-like/LINER-like
ratios can involve diffuse gas, evolved stars, shocks and FoV mixing.
They do not establish nuclear activity by themselves.

### 7.3 Additional gas diagnostics

- O32 and NeIII/OII excitation trends: [...]
- R23 behavior and calibration degeneracy: [...]
- SII doublet resolution/reliability: [...]
- Density assumptions or reasons not to estimate density: [...]
- Weak HeI, NeIII, OI and ArIII detection limitations: [...]

## 8. Balmer decrement and attenuation

- Assumed Case-B temperature and density: [...]
- Intrinsic Halpha/Hbeta used: [...]
- Extinction law and Rv: [...]
- Stellar absorption correction validation: [...]

| Ttype | N valid decrements | Halpha/Hbeta median / IQR | Signed gas E(B-V) | Negative-value fraction |
|---|---:|---|---|---|
| [...] | [...] | [...] | [...] | [...] |

Discuss Hgamma/Hbeta and Hdelta/Hbeta as consistency checks.
Distinguish gas attenuation from STARLIGHT stellar AV.
Explain signed versus nonnegative-clipped reddening results.

Do not assume that higher attenuation proves later morphology:
inclination, gas geometry and selection can contribute.

## 9. Which features separate morphological types?

### 9.1 Overall associations

| Feature | Usable N | Types tested | Epsilon squared | FDR q | Coverage caveat |
|---|---:|---:|---:|---:|---|
| [...] | [...] | [...] | [...] | [...] | [...] |

Large samples can produce small p-values for weak separation.
Prioritize effect size, distribution overlap and stability.

### 9.2 Pairwise separation

| Type pair | Feature | N_a / N_b | Signed rank-biserial effect | Absolute effect | FDR q | Bootstrap interval, if calculated |
|---|---|---|---|---|---|---|
| [...] | [...] | [...] | [...] | [...] | [...] | [...] |

Effect sign depends on the ordering of the pair.
Describe overlapping neighboring stages, not just extreme early/late pairs.
Leave confidence intervals unavailable unless they were actually calculated.

### 9.3 Predictive validation

- Predictors included: [...]
- Labels, identifiers and label-derived variables excluded: [...]
- Galaxy-level duplicate/group handling: [...]
- Sparse classes excluded and reasons: [...]
- Cross-validation design: [...]
- Fold-local imputation/preprocessing: [...]
- Held-out permutation-importance method: [...]

| Feature set | Included types | N | Balanced accuracy | Macro F1 | Principal confusions |
|---|---|---:|---|---|---|
| Continuum + absorption | [...] | [...] | [...] | [...] | [...] |
| Emission | [...] | [...] | [...] | [...] | [...] |
| STARLIGHT | [...] | [...] | [...] | [...] | [...] |
| Combined spectroscopy | [...] | [...] | [...] | [...] | [...] |
| Optional catalog-augmented | [...] | [...] | [...] | [...] | [...] |

Insert the out-of-fold confusion matrix and permutation-importance ranking.

Distinguish:
- Statistical association.
- Out-of-sample predictive contribution.
- Physical interpretation.
- Causal claims, which this EDA does not establish.

Correlated predictors can share or dilute permutation importance.
Exploratory feature choices require independent or nested validation
before final performance claims.

## 10. Confounders and sensitivity analyses

| Potential confounder | Check performed | Result | Effect on conclusions |
|---|---|---|---|
| Stellar mass | [...] | [...] | [...] |
| Inclination / dust | [...] | [...] | [...] |
| Redshift / coverage | [...] | [...] | [...] |
| FoV / aperture | [...] | [...] | [...] |
| STARLIGHT fit quality | [...] | [...] | [...] |
| Emission detection selection | [...] | [...] | [...] |
| Continuum-normalization choice | [...] | [...] | [...] |
| Empirical-noise assumptions | [...] | [...] | [...] |
| Label uncertainty | [...] | [...] | [...] |

Catalog angular size alone does not establish the fraction of a galaxy
covered by the spectrum. Additional aperture information may be needed.

Report checks not performed as limitations, not as reassuring null results.

## 11. Morphology-focused synthesis

### What appears well supported?
[Measured differences, effect sizes and predictive evidence.]

### What remains ambiguous?
[E versus S0 overlap, neighboring spiral stages, bars, population degeneracy,
outliers or uncertain labels.]

### What can integrated spectra not establish?
[Direct spiral-arm structure, bulge/disk geometry, bars or interactions
without imaging or additional spatial information.]

### Recommended feature set
[Small, interpretable set with adequate coverage and validation.]

### Recommended next steps
[Imaging comparison, improved uncertainty model, continuum validation,
resolution matching, independent test sample or spatially resolved analysis.]

## 12. Limitations

Explicitly address:
- Incorrect supplied ERROR_FoV.
- Provisional STARLIGHT-derived quantities.
- Empirical errors and approximate upper limits.
- Nonnegative weak-line fitting bias.
- Unresolved doublets and blend-detection limitations.
- Missing features and morphology-dependent selection.
- Flux calibration, foreground extinction and wavelength conventions.
- Instrumental resolution and integrated-aperture mixing.
- Sparse/imbalanced classes and catalog-label uncertainty.
- Lack of independent final validation, if applicable.

## 13. References

- Buta, Galaxy Morphology:
  https://arxiv.org/abs/1102.0550
- RC3 classification documentation:
  https://ned.ipac.caltech.edu/level5/Sept02/RC3/RC3_contents.html
- Balogh et al. 1999, Dn4000:
  https://ui.adsabs.harvard.edu/abs/1999ApJ...527...54B/abstract
- Worthey & Ottaviani 1997, Balmer absorption indices:
  https://ui.adsabs.harvard.edu/abs/1997ApJS..111..377W/abstract
- Bruzual & Charlot 2003, stellar-population synthesis:
  https://arxiv.org/abs/astro-ph/0309134
- Cid Fernandes et al. 2005, STARLIGHT:
  https://arxiv.org/abs/astro-ph/0412481
- Baldwin, Phillips & Terlevich 1981:
  https://ui.adsabs.harvard.edu/abs/1981PASP...93....5B/abstract
- Kewley et al. 2001:
  https://arxiv.org/abs/astro-ph/0106324
- Kauffmann et al. 2003:
  https://arxiv.org/abs/astro-ph/0304239
- Kewley et al. 2006:
  https://arxiv.org/abs/astro-ph/0605681
- Cid Fernandes et al. 2011, WHAN:
  https://arxiv.org/abs/1012.4426
- Cardelli, Clayton & Mathis 1989:
  https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract
- Luridiana et al. 2015, PyNeb:
  https://arxiv.org/abs/1410.6662

## Appendix: Output index

- feature_data_dictionary.csv
- galaxy_features_with_diagnostics.csv
- emission_lines.csv
- quality_control.csv
- feature_summary_by_ttype.csv
- feature_available_fraction_by_ttype.csv
- overall_feature_separation.csv
- pairwise_feature_separation.csv
- classification_cv_scores.csv [if calculated]
- classification_feature_ranking.csv [if calculated]
- mean_spectra_by_ttype.csv
- figures/
"""

report_path = OUTPUT_DIR / "morphology_interpretation_report_template.md"

# Avoid overwriting a report that you may already have edited.
if report_path.exists():
    print("Template already exists; not overwritten:", report_path)
else:
    report_path.write_text(report_template, encoding="utf-8")
    print("Saved:", report_path)
```

### How to use these outputs

1. **Review the dictionary first**, especially catalog units and provisional measurements.
2. Fill the report using the saved summaries, figures and cross-validation outputs.
3. For every claimed separating feature, report **the type pair, usable counts, effect size, overlap and validation evidence**.
4. Leave unperformed checks and uncalculated uncertainties explicitly marked as unavailable.

### Want me to…
<!-- copilot:suggestions -->
1. Add a Colab cell that pre-fills the report’s sample counts, morphology tables and measured feature rankings.
2. Hand these additions off to the coding agent to open a pull request once coding agent access is enabled for `Melonangie/Galaxy-Vision`.

## @maria-rabelero-mx_nestle

Add a Colab cell that pre-fills the report’s sample counts, morphology tables and measured feature rankings.

## Copilot

Add this cell **after the analysis, dictionary and report-template cells**. It creates a **new, timestamped report** containing measured counts, morphology tables, feature-separation rankings and classification results when available.

It uses the current notebook’s data—not old checkpoint files—and leaves scientific interpretation prompts for you to complete. **Run the analysis cells for the same run before executing it**, particularly the optional classification cell.

```python name=prefill_morphology_report.py
from datetime import datetime, timezone
import re
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

# ------------------------------------------------------------
# 1. Validate the current analysis and load the unfilled template
# ------------------------------------------------------------
required_variables = ["features", "qc", "spectrum_paths", "catalog"]
missing = [name for name in required_variables if name not in globals()]
if missing:
    raise RuntimeError(f"Run the analysis cells first. Missing: {missing}")

template_path = OUTPUT_DIR / "morphology_interpretation_report_template.md"
if not template_path.exists():
    raise FileNotFoundError("Run the report-template cell first.")

F = features.copy()
Q = qc.copy()

if F.empty:
    raise RuntimeError("No measured galaxy features are available.")
if F["name"].duplicated().any() or Q["name"].duplicated().any():
    raise ValueError("Duplicate galaxy rows: resolve them before reporting.")
if not set(F["name"]).issubset(set(Q["name"])):
    raise ValueError("Features and QC do not describe the same attempted galaxies.")

timestamp = datetime.now(timezone.utc)
run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
report = template_path.read_text(encoding="utf-8")


# ------------------------------------------------------------
# 2. Small Markdown helpers; no extra package is necessary
# ------------------------------------------------------------
def markdown_table(table):
    """Convert a DataFrame to readable Markdown without tabulate."""
    if table is None or table.empty:
        return "_Unavailable or not calculated in this run._"

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def replace_first_table(text, heading, table):
    """Replace the placeholder table only within the named subsection."""
    start = text.find(heading)
    if start == -1:
        return text

    # Stop at the next heading of equal or higher level.
    level = len(heading) - len(heading.lstrip("#"))
    next_heading = re.search(
        rf"^#{{1,{level}}} ", text[start + len(heading):], re.MULTILINE
    )
    end = (
        start + len(heading) + next_heading.start()
        if next_heading else len(text)
    )
    section = text[start:end]
    section = re.sub(
        r"^\|.*(?:\n\|.*)*",
        lambda _: markdown_table(table),
        section, count=1, flags=re.MULTILINE,
    )
    return text[:start] + section + text[end:]


def optional_table(variable, columns):
    """Only use optional results that exist as nonempty DataFrames."""
    table = globals().get(variable)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return pd.DataFrame()
    return table[[c for c in columns if c in table.columns]].copy()


# ------------------------------------------------------------
# 3. Counts: distinguish full inventory from the attempted subset
# ------------------------------------------------------------
inventory = pd.DataFrame({
    "name": [galaxy_id(path) for path in spectrum_paths]
})

# Counts refer to unique galaxies, not numbers of FITS files.
duplicate_files = int(inventory["name"].duplicated().sum())
inventory = inventory.drop_duplicates("name")

catalog_labels = catalog.reset_index()[["name", "ttype"]].copy()
if "type" in catalog.columns:
    catalog_labels["type"] = catalog.reset_index()["type"]

inventory = inventory.merge(
    catalog_labels, on="name", how="left", validate="one_to_one",
    indicator=True
)

attempted_names = set(Q["name"])
processed_names = set(F["name"])

if "starlight_status" in Q:
    aligned_names = set(
        Q.loc[Q["starlight_status"].eq("aligned"), "name"]
    )
else:
    aligned_names = set()

if "stack_status" in Q:
    stacked_names = set(Q.loc[Q["stack_status"].eq("cached"), "name"])
else:
    stacked_names = set()

inventory["attempted"] = inventory["name"].isin(attempted_names)
inventory["processed"] = inventory["name"].isin(processed_names)
inventory["aligned"] = inventory["name"].isin(aligned_names)
inventory["stacked"] = inventory["name"].isin(stacked_names)

counts = pd.DataFrame([
    ("Unique galaxies discovered", len(inventory)),
    ("Duplicate spectrum files beyond unique IDs", duplicate_files),
    ("Discovered galaxies without catalog match",
     int(inventory["_merge"].eq("left_only").sum())),
    ("Galaxies attempted in this run", len(Q)),
    ("Discovered galaxies not attempted",
     len(set(inventory["name"]) - attempted_names)),
    ("Galaxies with processed rest-frame spectra", len(F)),
    ("Attempted galaxies without processed spectra",
     len(attempted_names - processed_names)),
    ("Galaxies with validated STARLIGHT continua", len(aligned_names)),
    ("Galaxies successfully normalized for stacking", len(stacked_names)),
    ("Processed galaxies missing Ttype", int(F["ttype"].isna().sum())),
], columns=["Stage / outcome", "Count"])

# Build the morphology table from discovered galaxies, not only successes.
morphology = (
    inventory.groupby("ttype", dropna=False)
    .agg(
        discovered=("name", "size"),
        attempted=("attempted", "sum"),
        processed=("processed", "sum"),
        starlight_aligned=("aligned", "sum"),
        normalized_for_stacking=("stacked", "sum"),
    )
    .reset_index()
)

if "type" in inventory:
    type_strings = (
        inventory.groupby("ttype", dropna=False)["type"]
        .agg(lambda values: ", ".join(sorted(set(values.dropna().astype(str)))))
        .reset_index(name="Catalog Type strings")
    )
    morphology = morphology.merge(type_strings, on="ttype", how="left")

morphology["Label caveat"] = np.where(
    morphology["ttype"].eq(11),
    "Verify catalog convention; do not assume irregular.",
    "Catalog labels retained; uncertainty not quantified."
)

report = report.replace(
    "> Status: TEMPLATE — no scientific findings are implied.",
    "> Status: AUTO-PREFILLED DRAFT — measured tables inserted; "
    "interpretation and validation remain incomplete."
)
report = report.replace(
    "- Galaxies discovered: [N]",
    f"- Galaxies discovered: {len(inventory)} unique galaxies"
)
report = report.replace(
    "- Galaxies with usable rest-frame spectra: [N]",
    f"- Galaxies with usable rest-frame spectra: {len(F)} processed galaxies "
    f"out of {len(Q)} attempted"
)
report = report.replace(
    "- Galaxies with validated STARLIGHT continua: [N]",
    f"- Galaxies with validated STARLIGHT continua: {len(aligned_names)}"
)
report = report.replace(
    "- Morphological stages represented: [list catalog Ttypes and Type strings]",
    "- Morphological stages represented in processed spectra: "
    + ", ".join(map(str, sorted(F["ttype"].dropna().unique())))
)
report = report.replace(
    "- Run date / analysis identifier: [...]",
    f"- Report generation time / identifier: {timestamp.isoformat()} / {run_id}"
)

report = replace_first_table(report, "### 3.1 Catalog conventions", morphology)
report = replace_first_table(report, "### 3.2 Sample flow and exclusions", counts)


# ------------------------------------------------------------
# 4. Populate measured overall and pairwise feature rankings
# ------------------------------------------------------------
overall_results = optional_table(
    "overall", ["feature", "n", "types", "epsilon_squared", "p", "q_fdr"]
)
pairwise_results = optional_table(
    "pairwise",
    ["feature", "ttype_a", "ttype_b", "n_a", "n_b",
     "rank_biserial", "absolute_effect", "q_fdr"]
)

if not overall_results.empty:
    overall_results = overall_results.sort_values(
        "epsilon_squared", ascending=False
    )
    overall_results["Coverage caveat"] = (
        "Available-case analysis; compare availability by Ttype."
    )
    report = replace_first_table(
        report, "### 9.1 Overall associations", overall_results.head(20)
    )
    report = report.replace(
        "- Most informative features: [results, not prior expectations]",
        "- Largest measured overall effect sizes: "
        + ", ".join(overall_results["feature"].head(5))
        + " (association ranking, not proof of predictive importance)"
    )

if not pairwise_results.empty:
    pairwise_results = pairwise_results.sort_values(
        "absolute_effect", ascending=False
    )
    pairwise_results["Bootstrap interval"] = "Not calculated by this workflow."
    report = replace_first_table(
        report, "### 9.2 Pairwise separation", pairwise_results.head(20)
    )

    strongest = pairwise_results.iloc[0]
    report = report.replace(
        "- Best-separated type pairs: [pairs, effect sizes, uncertainty]",
        f"- Largest measured pair-feature effect: "
        f"T={strongest['ttype_a']} versus T={strongest['ttype_b']}, "
        f"{strongest['feature']}, "
        f"|rank-biserial|={strongest['absolute_effect']:.3f}; "
        "bootstrap uncertainty not calculated"
    )

# Do not automatically infer the most overlapping pair from a weak
# univariate effect: multivariate overlap requires separate evidence.


# ------------------------------------------------------------
# 5. BPT morphology tables with explicit denominators
# ------------------------------------------------------------
bpt_sections = []
for panel in ["NII", "SII", "OI"]:
    column = f"bpt_{panel}_class"
    if column not in F:
        continue

    data = F[["ttype", column]].copy()
    data[column] = data[column].fillna("unclassified")

    table = pd.crosstab(data["ttype"], data[column]).reset_index()
    category_columns = [c for c in table if c != "ttype"]
    table["N processed with Ttype"] = table[category_columns].sum(axis=1)
    table["N classifiable"] = (
        table["N processed with Ttype"] - table.get("unclassified", 0)
    )
    table["Classifiable fraction"] = (
        table["N classifiable"] / table["N processed with Ttype"]
    )

    bpt_sections.append(
        f"#### Measured {panel} BPT counts\n\n"
        + markdown_table(table)
        + "\n\nDenominator: processed galaxies with a catalog Ttype. "
        "Unclassified includes unavailable required inputs.\n"
    )

if bpt_sections:
    # Remove the generic placeholder table, then insert each panel separately.
    report = replace_first_table(
        report, "### 7.2 BPT results", pd.DataFrame()
    )
    marker = "### 7.2 BPT results"
    report = report.replace(
        marker, marker + "\n\n" + "\n".join(bpt_sections), 1
    )


# ------------------------------------------------------------
# 6. Optional held-out classification results
# ------------------------------------------------------------
cv_scores = optional_table(
    "score_table", ["fold", "balanced_accuracy", "macro_f1"]
)
importance_ranking = globals().get("ranking")

classification_sections = []
if not cv_scores.empty:
    classification_sections.append(
        "### Measured cross-validation scores\n\n" + markdown_table(cv_scores)
    )
    for metric, placeholder in [
        ("balanced_accuracy",
         "- Cross-validated balanced accuracy: [mean and fold variation]"),
        ("macro_f1",
         "- Cross-validated macro F1: [mean and fold variation]"),
    ]:
        if metric in cv_scores:
            mean = cv_scores[metric].mean()
            std = cv_scores[metric].std()
            report = report.replace(
                placeholder,
                f"- Cross-validated {metric}: {mean:.3f}; "
                f"across-fold standard deviation {std:.3f}"
            )

if isinstance(importance_ranking, pd.DataFrame) and not importance_ranking.empty:
    importance_display = importance_ranking.reset_index()
    classification_sections.append(
        "### Measured held-out permutation-importance ranking\n\n"
        + markdown_table(importance_display.head(20))
        + "\n\nImportance is the change in the configured validation score "
        "after permutation. Negative values do not establish harmful physical "
        "effects. Fold variation is not a confidence interval."
    )

# Report actual failure categories without guessing their physical causes.
qc_summary = (
    Q["status"].fillna("missing status").value_counts()
    .rename_axis("QC outcome").reset_index(name="N attempted")
)

report += (
    "\n\n## Automatically generated run details\n\n"
    "This draft uses current notebook variables. Tables show up to 20 ranked "
    "entries; complete overall/pairwise rankings are saved alongside the report. "
    "Optional classification outputs are included only when present; verify "
    "they were calculated for this same run.\n\n"
    "### Actual processing outcomes\n\n"
    + markdown_table(qc_summary)
    + "\n\n"
    + ("\n\n".join(classification_sections)
       if classification_sections else
       "_No optional classification results available._")
    + "\n"
)


# ------------------------------------------------------------
# 7. Save a new draft and supporting tables; preserve edited reports
# ------------------------------------------------------------
report_path = OUTPUT_DIR / f"morphology_report_prefilled_{run_id}.md"
report_path.write_text(report, encoding="utf-8")

counts.to_csv(OUTPUT_DIR / f"report_counts_{run_id}.csv", index=False)
morphology.to_csv(OUTPUT_DIR / f"report_morphology_{run_id}.csv", index=False)

for label, table in [
    ("overall_rankings", overall_results),
    ("pairwise_rankings", pairwise_results),
]:
    if not table.empty:
        table.to_csv(OUTPUT_DIR / f"report_{label}_{run_id}.csv", index=False)

print("Saved report:", report_path)
print("Interpretation prompts remain intentionally unfilled.")
display(Markdown(report[:report.find("## 2.")]))
```

### What remains manual

The cell deliberately **does not** write conclusions about evolution, bars, AGN activity or causation. You still need to:

- verify morphology-dependent coverage and detection selection;
- inspect fits and spectrum stacks;
- interpret overlapping type distributions;
- document confounder checks and uncertainty validation.

The classification summary reports **variation across folds**, not a confidence interval.

### Want me to…
<!-- copilot:suggestions -->
1. Add an automatic consistency check that prevents mixing feature rankings and classification results from different analysis runs.
2. Hand these report-generation additions off to the coding agent to open a pull request once coding agent access is enabled for `Melonangie/Galaxy-Vision`.
