# Copilot Chat Conversation Export: Building a Colab workflow for galaxy

**User:** @maria-rabelero-mx_nestle
**Thread URL:** https://github.com/copilot/c/8f4d8855-8b7b-4237-9a2e-2fcc89471566

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

I’m waiting for you to accept or reject the confirmation dialog before proceeding with the documented Google Colab galaxy-morphology EDA workflow.

## @maria-rabelero-mx_nestle

@Copilot Accepted Confirmation: Confirm cloud agent session

## Copilot

**The coding-agent handoff failed because it is not enabled for `Melonangie/Galaxy-Vision`. No files were changed and no pull request was created.** My earlier statement about a confirmation dialog was incorrect.

You can still build this in Colab. Below is a readable **first-pass workflow** using your actual column names and STARLIGHT format. It covers spectrum loading, stellar-population features, approximate emission-line measurements, morphology comparisons, grouped spectra, class means, and feature ranking.

**I have inspected the supplied file contents, but have not executed this code or analyzed your Google Drive dataset.** Consequently, there are no measured “best features” yet.

## Important scientific decisions

- Use **`WAVE` and `FLUX_FoV`**; never use `ERROR_FoV`.
- Join galaxy filenames to the supporting catalog’s **`name`**.
- Use **`nsa_z`** to obtain rest-frame wavelengths. Values such as `-9999` are invalid.
- Preserve **exact `Ttype` and `Type`**. Your catalog contains `Ttype=11, Type=S`; do **not** automatically interpret that as an irregular galaxy or merger.
- STARLIGHT spectra appear already rest-frame in your samples. They must **not be redshift-corrected twice**.
- Subtract a validated stellar model before measuring nebular Balmer lines.
- BPT describes **gas excitation**, not morphological type. Integrated spectra mix bulge, disk, star formation, diffuse gas, and potentially an AGN.
- Without valid uncertainties, line signal-to-noise estimates are **empirical and approximate**. Publication-quality measurements require additional uncertainty and continuum-model validation.

---

# 1. Install libraries and configure Colab

Copy each block into a separate Colab cell.

```python name=01_setup.py
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
```

The normalization window is a configurable choice, not a universal standard. A galaxy lacking adequate coverage there will retain its feature measurements but will be excluded from normalized comparisons.

# 2. Load the supporting catalog and pair files

The supporting FITS is used only for identity, labels, redshift, and contextual information. It is explicitly excluded from spectral discovery.

```python name=02_inputs.py
def matching_table(path, required_columns, hdu_override=None):
    """Read one unambiguous FITS table; otherwise show how to select it."""
    with fits.open(path, memmap=True) as hdul:
        inventory = []
        matches = []

        for index, hdu in enumerate(hdul):
            names = list(getattr(getattr(hdu, "columns", None), "names", []) or [])
            inventory.append((index, hdu.name, names))

            if set(required_columns).issubset({n.lower() for n in names}):
                matches.append(index)

        selected = hdu_override
        if selected is None:
            if len(matches) != 1:
                raise ValueError(
                    f"{path.name}: cannot select a unique table. "
                    f"Set the HDU override. Inventory: {inventory}"
                )
            selected = matches[0]

        table = Table(hdul[selected].data)
        table.rename_columns(table.colnames, [n.lower() for n in table.colnames])

        missing = set(required_columns) - set(table.colnames)
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

        return table


def galaxy_id(value):
    """Match manga-PLATE-IFU regardless of -1D or output suffix."""
    match = re.search(r"manga-\d+-\d+", str(value), flags=re.IGNORECASE)
    return match.group(0).lower() if match else None


if CATALOG_PATH.suffix.lower() == ".csv":
    catalog = pd.read_csv(CATALOG_PATH)
    catalog.columns = catalog.columns.str.strip().str.lower()
else:
    catalog = matching_table(
        CATALOG_PATH, {"name", "ttype", "type", "nsa_z"}, CATALOG_HDU
    ).to_pandas()

catalog["name"] = catalog["name"].map(
    lambda x: x.decode().strip() if isinstance(x, bytes) else str(x).strip()
)
catalog["galaxy_id"] = catalog["name"].map(galaxy_id)

for column in ["ttype", "nsa_z", "logmass", "incl", "petro_th90"]:
    if column in catalog:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
        catalog.loc[catalog[column] == -9999, column] = np.nan

if catalog["galaxy_id"].isna().any():
    raise ValueError("Catalog contains invalid galaxy identities.")
if catalog["galaxy_id"].duplicated().any():
    raise ValueError("Resolve duplicate catalog galaxy identities before joining.")

catalog = catalog.set_index("galaxy_id", drop=False)


def discover_files(directory, suffixes, exclude=None):
    """Keep every candidate so duplicates cannot silently overwrite a file."""
    found = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.resolve() == exclude:
            continue
        if not any(path.name.lower().endswith(s) for s in suffixes):
            continue

        identity = galaxy_id(path.name)
        if identity:
            found.setdefault(identity, []).append(path)
    return found


spectral_files = discover_files(
    SPECTRA_DIR, (".fits", ".fits.gz", ".csv"),
    exclude=CATALOG_PATH.resolve(),
)
starlight_files = discover_files(STARLIGHT_DIR, (".out", "-out.txt"))

manifest_rows = []
for identity, paths in sorted(spectral_files.items()):
    outputs = starlight_files.get(identity, [])
    manifest_rows.append({
        "galaxy_id": identity,
        "spectrum_candidates": len(paths),
        "starlight_candidates": len(outputs),
        "catalog_match": identity in catalog.index,
        "spectrum_path": str(paths[0]) if len(paths) == 1 else None,
        "starlight_path": str(outputs[0]) if len(outputs) == 1 else None,
    })

manifest = pd.DataFrame(manifest_rows)
if manifest.empty:
    raise ValueError("No galaxy spectrum files found. Check SPECTRA_DIR.")

manifest.to_csv(OUTPUT_DIR / "input_manifest.csv", index=False)

# Ambiguous spectrum files or STARLIGHT pairs require manual resolution.
usable = manifest[
    (manifest["spectrum_candidates"] == 1)
    & (manifest["starlight_candidates"] <= 1)
    & manifest["catalog_match"]
].copy()

if MAX_GALAXIES is not None:
    usable = usable.head(MAX_GALAXIES)

print(f"Found {len(manifest)} galaxy identities; processing {len(usable)}.")
```

This reader supports FITS **tables** with wavelength and flux columns, including a single row containing vectors. Image-HDU layouts require an explicit adapter after inspecting the real FITS structure; the code deliberately does not guess.

# 3. Rest-frame spectra and gap-aware utilities

Never interpolate across the artificial CSV gaps. Missing spectral coverage must remain missing rather than becoming invented continuum or line flux.

```python name=03_spectra.py
def load_spectrum(path, redshift):
    if not np.isfinite(redshift) or redshift <= 0:
        raise ValueError("Missing or invalid redshift for this MaNGA workflow.")

    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip().str.lower()
        wave = pd.to_numeric(frame["wave"], errors="coerce").to_numpy()
        flux = pd.to_numeric(frame["flux_fov"], errors="coerce").to_numpy()
    else:
        table = matching_table(path, {"wave", "flux_fov"}, SPECTRUM_HDU)
        wave = np.asarray(table["wave"], dtype=float).reshape(-1)
        flux = np.asarray(table["flux_fov"], dtype=float).reshape(-1)

    # ERROR_FoV and ERROR_1RE are deliberately never read.
    if wave.shape != flux.shape:
        raise ValueError("WAVE and FLUX_FoV lengths differ.")

    wave = (wave * WAVE_UNIT).to_value(u.AA)
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[valid], flux[valid]

    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]

    if len(wave) < 10 or np.any(np.diff(wave) <= 0):
        raise ValueError("Insufficient pixels or duplicate wavelength coordinates.")

    return wave / (1 + redshift), flux * (1 + redshift)


def contiguous_segments(wave):
    step = np.median(np.diff(wave))
    cuts = np.flatnonzero(np.diff(wave) > 3 * step) + 1
    return np.split(np.arange(len(wave)), cuts)


def regrid_without_gaps(wave, flux, grid):
    """Interpolate only within contiguous segments; never extrapolate."""
    output = np.full(len(grid), np.nan)
    for indices in contiguous_segments(wave):
        if len(indices) < 2:
            continue
        inside = (grid >= wave[indices[0]]) & (grid <= wave[indices[-1]])
        output[inside] = np.interp(grid[inside], wave[indices], flux[indices])
    return output


def band_values(wave, flux, band):
    """Require approximately complete finite coverage of a measurement band."""
    low, high = band
    step = np.median(np.diff(wave))
    chosen = (wave >= low) & (wave <= high)
    w, f = wave[chosen], flux[chosen]

    if len(w) < 3 or not np.all(np.isfinite(f)):
        return None
    if w[0] > low + step or w[-1] < high - step:
        return None
    if np.any(np.diff(w) > 3 * step):
        return None
    return w, f


def band_mean(wave, flux, band):
    values = band_values(wave, flux, band)
    if values is None:
        return np.nan
    w, f = values
    return np.trapezoid(f, w) / (w[-1] - w[0])


def absorption_ew(wave, flux, blue, feature, red):
    """Rest EW in Angstroms; positive denotes absorption."""
    b = band_values(wave, flux, blue)
    l = band_values(wave, flux, feature)
    r = band_values(wave, flux, red)
    if any(values is None for values in (b, l, r)):
        return np.nan

    wb, fb = b
    wl, fl = l
    wr, fr = r

    continuum = np.interp(
        wl, [np.mean(wb), np.mean(wr)], [np.mean(fb), np.mean(fr)]
    )
    if np.any(continuum <= 0):
        return np.nan
    return np.trapezoid(1 - fl / continuum, wl)


def stellar_features(wave, flux):
    # Canonical Dn4000 uses f_nu, not a direct f_lambda ratio.
    # The constant 1/c cancels between bands.
    fnu_proportional = flux * wave**2
    blue = band_mean(wave, fnu_proportional, (3850, 3950))
    red = band_mean(wave, fnu_proportional, (4000, 4100))

    result = {
        "dn4000": red / blue if np.isfinite(blue) and blue > 0 else np.nan,
    }

    blue_color = band_mean(wave, flux, (4500, 4800))
    red_color = band_mean(wave, flux, (6000, 6300))
    result["continuum_red_blue"] = (
        red_color / blue_color
        if np.isfinite(blue_color) and blue_color > 0 else np.nan
    )

    # Referenced Lick-style bandpasses; these are NOT calibrated Lick indices.
    indices = {
        "hdelta_a": ((4041.60, 4079.75), (4083.50, 4122.25), (4128.50, 4161.00)),
        "hgamma_a": ((4283.50, 4319.75), (4319.75, 4363.50), (4367.25, 4419.75)),
        "mgb": ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
        "fe5270": ((5233.150, 5248.150), (5245.650, 5285.650), (5285.650, 5318.150)),
        "fe5335": ((5304.625, 5315.875), (5312.125, 5352.125), (5353.375, 5363.375)),
    }

    # These traditional index windows are in air wavelengths.
    # For this first pass, do not silently apply them to vacuum spectra.
    # Converting consistently is a required extension for those measurements.
    if WAVELENGTH_CONVENTION == "air":
        for name, bands in indices.items():
            result[f"{name}_observed_ew"] = absorption_ew(wave, flux, *bands)
    else:
        for name in indices:
            result[f"{name}_observed_ew"] = np.nan

    return result
```

**Interpretation:** Dn4000 usually increases with older and/or more metal-rich stellar populations. Strong Balmer absorption can indicate substantial intermediate-age populations. Neither uniquely determines morphology. Metal-line indices also depend on abundance and broadening.

The explicit air/vacuum guard is intentional: wavelength-system inconsistencies should not quietly become scientific measurements.

# 4. Parse STARLIGHT and validate its continuum

The population vector must come from the **best-model section**, not repeated chain tables. Normalize light and mass fractions by their actual sums.

```python name=04_starlight.py
def number(token):
    return float(token.replace("D", "E").replace("d", "e"))


def read_starlight(path):
    lines = Path(path).read_text(errors="replace").splitlines()

    def labeled(label):
        pattern = re.compile(r"\[" + re.escape(label) + r"(?:\s|\])")
        for line in lines:
            if pattern.search(line):
                return number(line.split()[0])
        return np.nan

    nbase = int(labeled("N_base"))
    population_start = next(
        i for i, line in enumerate(lines)
        if "x_j(%)" in line and "age_j(yr)" in line
    )
    population_rows = [
        line.split()
        for line in lines[population_start + 1:population_start + 1 + nbase]
    ]
    if len(population_rows) != nbase:
        raise ValueError("Incomplete STARLIGHT population table.")

    population = np.array([
        [number(token) for token in row[:6]]
        for row in population_rows
    ])

    synthetic_start = next(
        i for i, line in enumerate(lines) if "Synthetic spectrum (Best Model)" in line
    )
    count = int(number(lines[synthetic_start + 1].split()[0]))
    spectral_rows = lines[synthetic_start + 2:synthetic_start + 2 + count]
    synthetic = np.array([
        [number(token) for token in line.split()[:4]]
        for line in spectral_rows
    ])
    if synthetic.shape != (count, 4):
        raise ValueError("Incomplete STARLIGHT synthetic spectrum.")

    features = {
        "starlight_adev": labeled("adev"),
        "starlight_chi2": labeled("chi2/Nl_eff"),
        "stellar_av": labeled("AV_min"),
        "stellar_vd_observed": labeled("vd_min"),
    }

    age, metallicity = population[:, 4], population[:, 5]
    if np.any(age <= 0):
        raise ValueError("Invalid SSP ages.")

    for prefix, weights in [
        ("light", population[:, 1]),
        ("mass", population[:, 3]),  # Current stellar mass, not initial mass.
    ]:
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid population weights.")
        weights = weights / weights.sum()

        features[f"{prefix}_mean_log_age"] = np.sum(weights * np.log10(age))
        features[f"{prefix}_mean_z"] = np.sum(weights * metallicity)
        features[f"{prefix}_young_fraction"] = weights[age < 1e8].sum()
        features[f"{prefix}_intermediate_fraction"] = weights[
            (age >= 1e8) & (age < 1e9)
        ].sum()
        features[f"{prefix}_old_fraction"] = weights[age >= 1e9].sum()

    return synthetic, labeled("fobs_norm"), features


def validated_continuum(wave, flux, synthetic, normalization, redshift):
    if STARLIGHT_FRAME != "rest":
        raise ValueError("This first-pass adapter expects rest-frame STARLIGHT.")
    if STARLIGHT_WAVELENGTH_CONVENTION != WAVELENGTH_CONVENTION:
        raise ValueError("Convert wavelength conventions before model subtraction.")

    model_wave, observed, model, weight = synthetic.T

    factor = normalization
    if not STARLIGHT_INPUT_HAS_FLUX_JACOBIAN:
        factor *= 1 + redshift

    # -2 represents unavailable observed data in the supplied output.
    # Zero weights often mask real emission lines during synthesis fitting.
    # Neither zero nor negative fitting weights are valid error estimates.
    available = weight != -2
    observed = np.where(available, observed * factor, np.nan)

    reconstructed = regrid_without_gaps(model_wave, observed, wave)
    continuum = regrid_without_gaps(model_wave, model * factor, wave)

    comparison = (
        np.isfinite(reconstructed) & np.isfinite(continuum)
        & (wave >= 4500) & (wave <= 6500) & (flux > 0)
    )
    if comparison.sum() < 100:
        return continuum, False, np.nan

    mismatch = np.median(
        np.abs(flux[comparison] - reconstructed[comparison]) / flux[comparison]
    )
    trustworthy = mismatch <= MAX_ALIGNMENT_FRACTIONAL_DIFFERENCE
    return continuum, trustworthy, mismatch
```

A small flux mismatch is a useful screening check, **not proof** that STARLIGHT preprocessing, wavelength convention, extinction treatment, or continuum shape is correct. Inspect several overlays before trusting thousands of measurements.

STARLIGHT’s `AV` is stellar attenuation. Its `vd_min` is an observed model-broadening parameter, not automatically intrinsic stellar velocity dispersion. Its reported total masses have unclear units in your sample and are deliberately excluded.

# 5. Approximate emission-line measurements

This fits the **original rest-frame spectrum minus its stellar continuum**, not a class stack.

The Gaussian fluxes share velocity and width within each complex. Hα and [N II] are fit jointly; fitting three unrelated integration windows would risk blending bias.

```python name=05_emission.py
# Vacuum line centers in Angstroms; do not use these with air spectra.
LINE_GROUPS = {
    "blue": {
        "band": (4800, 5050),
        "lines": {"hb": 4862.68, "oiii4959": 4960.30, "oiii5007": 5008.24},
    },
    "halpha": {
        "band": (6500, 6620),
        "lines": {"nii6548": 6549.86, "ha": 6564.61, "nii6583": 6585.27},
    },
    "sii": {
        "band": (6670, 6780),
        "lines": {"sii6716": 6718.29, "sii6731": 6732.67},
    },
    "oi": {
        "band": (6250, 6350),
        "lines": {"oi6300": 6302.05},
    },
}
C_KMS = 299792.458


def fit_emission_group(wave, residual, continuum, group):
    values = band_values(wave, residual, group["band"])
    if values is None:
        return {}, "incomplete_coverage"

    w, y = values
    centers = np.array(list(group["lines"].values()))
    names = list(group["lines"])

    # Approximate line-free pixels. Check this choice on actual residual plots.
    line_free = np.min(np.abs(w[:, None] - centers[None, :]), axis=1) > 20
    if line_free.sum() < 15:
        return {}, "insufficient_noise_pixels"

    local_baseline = np.polyval(np.polyfit(w[line_free], y[line_free], 1), w)
    noise_residual = (y - local_baseline)[line_free]
    noise = 1.4826 * np.median(
        np.abs(noise_residual - np.median(noise_residual))
    )
    if not np.isfinite(noise) or noise <= 0:
        return {}, "invalid_empirical_noise"

    def profiles(velocity, sigma_velocity):
        shifted = centers * (1 + velocity / C_KMS)
        sigmas = centers * sigma_velocity / C_KMS
        return np.exp(
            -0.5 * ((w[:, None] - shifted) / sigmas)**2
        ) / (np.sqrt(2 * np.pi) * sigmas)

    # Parameters: gas velocity, gas sigma, residual intercept/slope,
    # and integrated positive line fluxes.
    x = (w - np.mean(w)) / 100
    initial_flux = max(np.max(y - local_baseline), noise) * 5
    initial = [0, 100, np.median(y[line_free]), 0] + [initial_flux] * len(names)

    def model(parameters):
        velocity, sigma, intercept, slope = parameters[:4]
        return intercept + slope * x + profiles(velocity, sigma) @ parameters[4:]

    lower = [-400, 30, -np.inf, -np.inf] + [0] * len(names)
    upper = [400, 400, np.inf, np.inf] + [np.inf] * len(names)

    fitted = least_squares(
        lambda p: (model(p) - y) / noise,
        initial, bounds=(lower, upper),
    )
    if not fitted.success:
        return {}, "fit_failed"

    # Conditional covariance approximation: assumes independent residuals.
    # This does NOT include correlated pixels or stellar-model uncertainty.
    covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0))

    result = {}
    boundary = (
        abs(fitted.x[0]) > 390 or fitted.x[1] < 31 or fitted.x[1] > 390
    )

    for index, (name, center) in enumerate(zip(names, centers)):
        measured_flux = fitted.x[4 + index]
        error = uncertainty[4 + index]
        estimated_sn = measured_flux / error if error > 0 else np.nan
        local_continuum = np.interp(center, wave, continuum)

        result[f"{name}_flux"] = measured_flux
        result[f"{name}_flux_error_approx"] = error
        result[f"{name}_sn_approx"] = estimated_sn
        result[f"{name}_ew_emission"] = (
            measured_flux / local_continuum if local_continuum > 0 else np.nan
        )
        result[f"{name}_detected"] = bool(
            not boundary and measured_flux > 0
            and np.isfinite(estimated_sn)
            and estimated_sn >= MIN_ESTIMATED_LINE_SN
        )

    return result, "parameter_boundary" if boundary else "ok"
```

**Limitations of this first-pass fitter:** it does not yet model broad-line AGN, impose atomic doublet ratios, incorporate an instrumental line-spread function, apply observed-frame sky/telluric masks, or bootstrap correlated residuals. Its uncertainties are conditional approximations. These extensions should precede strong scientific claims.

# 6. Process galaxies and save measurements

Missing STARLIGHT allows continuum-shape features but does **not** produce trusted BPT measurements.

```python name=06_extract.py
features_rows = []
failures = []
cache_dir = OUTPUT_DIR / "normalized_spectra"
cache_dir.mkdir(exist_ok=True)

for item in tqdm(usable.to_dict("records")):
    identity = item["galaxy_id"]
    info = catalog.loc[identity]

    try:
        z = float(info["nsa_z"])
        wave, flux = load_spectrum(item["spectrum_path"], z)

        row = {
            "galaxy_id": identity,
            "ttype": info["ttype"],
            "type": info["type"],
            "nsa_z": z,
            **stellar_features(wave, flux),
            "continuum_trusted": False,
        }

        # Supporting properties are context, not morphology-derived predictors.
        for name in ["logmass", "incl", "petro_th90"]:
            row[name] = info.get(name, np.nan)

        if item["starlight_path"]:
            try:
                synthetic, normalization, population = read_starlight(
                    item["starlight_path"]
                )
                row.update(population)
                continuum, trusted, mismatch = validated_continuum(
                    wave, flux, synthetic, normalization, z
                )
                row["continuum_trusted"] = trusted
                row["starlight_flux_mismatch"] = mismatch

                if trusted and WAVELENGTH_CONVENTION == "vacuum":
                    residual = flux - continuum
                    for group_name, group in LINE_GROUPS.items():
                        measurements, status = fit_emission_group(
                            wave, residual, continuum, group
                        )
                        row.update(measurements)
                        row[f"{group_name}_fit_status"] = status
                else:
                    row["line_measurement_status"] = "untrusted_continuum_or_convention"
            except Exception as error:
                row["starlight_failure"] = str(error)

        normalization = band_mean(wave, flux, NORMALIZATION_BAND)
        row["normalization_valid"] = bool(
            np.isfinite(normalization) and normalization > 0
        )

        if row["normalization_valid"]:
            normalized = regrid_without_gaps(wave, flux / normalization, REST_GRID)
            np.savez_compressed(
                cache_dir / f"{identity}.npz",
                wave=REST_GRID, flux=normalized,
            )

        features_rows.append(row)

    except Exception as error:
        failures.append({"galaxy_id": identity, "error": str(error)})

    # Checkpoint the measurement table, but do not silently reuse stale caches.
    pd.DataFrame(features_rows).to_csv(
        OUTPUT_DIR / "galaxy_features.csv", index=False
    )

df = pd.DataFrame(features_rows)
if df.empty:
    raise ValueError("No galaxies processed successfully; inspect failures.")

pd.DataFrame(failures, columns=["galaxy_id", "error"]).to_csv(
    OUTPUT_DIR / "failures.csv", index=False
)

print(f"Measured {len(df)} galaxies; {len(failures)} failed.")
```

# 7. BPT and Balmer decrement

For the [N II] BPT plane:

\[
x=\log_{10}([\mathrm{N\,II}]6583/\mathrm{H}\alpha),\qquad
y=\log_{10}([\mathrm{O\,III}]5007/\mathrm{H}\beta).
\]

Use all four detections. Do not label a galaxy “passive” simply because it is unclassified.

```python name=07_diagnostics.py
def detected(frame, name):
    column = f"{name}_detected"
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def line_flux(frame, name):
    return pd.to_numeric(
        frame.get(f"{name}_flux", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )


bpt_valid = (
    df["continuum_trusted"].fillna(False)
    & detected(df, "ha") & detected(df, "hb")
    & detected(df, "nii6583") & detected(df, "oiii5007")
)

df["bpt_x"] = np.nan
df["bpt_y"] = np.nan
df.loc[bpt_valid, "bpt_x"] = np.log10(
    line_flux(df, "nii6583")[bpt_valid] / line_flux(df, "ha")[bpt_valid]
)
df.loc[bpt_valid, "bpt_y"] = np.log10(
    line_flux(df, "oiii5007")[bpt_valid] / line_flux(df, "hb")[bpt_valid]
)

df["bpt_class"] = "unclassified"
x, y = df["bpt_x"], df["bpt_y"]

# Domain guards avoid evaluating the demarcations across their poles.
sf = bpt_valid & (x < 0.05) & (y < 0.61 / (x - 0.05) + 1.3)
agn = bpt_valid & (
    (x >= 0.47) | ((x < 0.47) & (y >= 0.61 / (x - 0.47) + 1.19))
)

df.loc[bpt_valid, "bpt_class"] = "composite"
df.loc[sf, "bpt_class"] = "star_forming"
df.loc[agn, "bpt_class"] = "AGN_like"

# [N II] BPT alone does not robustly distinguish Seyfert from LINER-like gas.
# WHAN and [S II]/[O I] diagnostics are complementary extensions.

balmer_valid = (
    df["continuum_trusted"].fillna(False)
    & detected(df, "ha") & detected(df, "hb")
)
df["ha_hb"] = np.nan
df.loc[balmer_valid, "ha_hb"] = (
    line_flux(df, "ha")[balmer_valid] / line_flux(df, "hb")[balmer_valid]
)

# Case-B Halpha/Hbeta ~2.86 assumes ~10^4 K and low density.
# A different intrinsic ratio can be appropriate for AGN-like excitation.
INTRINSIC_HA_HB = 2.86
RV = 3.1
law = CCM89(Rv=RV)
k_ha = float(law(6564.61 * u.AA)) * RV
k_hb = float(law(4862.68 * u.AA)) * RV

df["nebular_ebv_approx"] = (
    2.5 / (k_hb - k_ha) * np.log10(df["ha_hb"] / INTRINSIC_HA_HB)
)
# Keep negative values visible: they can expose noise or modeling problems.
df["balmer_below_case_b"] = df["ha_hb"] < INTRINSIC_HA_HB

df.to_csv(OUTPUT_DIR / "galaxy_features.csv", index=False)

summary = pd.crosstab(df["ttype"], df["bpt_class"])
summary.to_csv(OUTPUT_DIR / "bpt_counts_by_ttype.csv")
display(summary)
```

Plot individual galaxies, not line ratios measured from the mean spectrum: a ratio of stacked fluxes is not the mean galaxy ratio.

```python name=08_diagnostic_plots.py
classes = sorted(df["ttype"].dropna().unique())

fig, ax = plt.subplots(figsize=(9, 7))
for morphology in classes:
    group = df[(df["ttype"] == morphology) & bpt_valid]
    ax.scatter(
        group["bpt_x"], group["bpt_y"],
        s=20, alpha=0.6, label=f"T={morphology:g} (n={len(group)})",
    )

xx = np.linspace(-2, 0.0, 400)
ax.plot(xx, 0.61 / (xx - 0.05) + 1.3, "--", color="black", label="Kauffmann 2003")
xx = np.linspace(-2, 0.4, 400)
ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, "-", color="black", label="Kewley 2001")

ax.set(
    xlabel="log10([N II] / Hα)",
    ylabel="log10([O III] / Hβ)",
    xlim=(-2, 0.7), ylim=(-1.5, 1.5),
    title="Integrated [N II] BPT by catalog T-type",
)
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1))
fig.savefig(OUTPUT_DIR / "bpt_all_ttypes.png", dpi=180, bbox_inches="tight")
plt.show()

# One diagnostic panel per exact morphology class.
for morphology in classes:
    group = df[df["ttype"] == morphology]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(group["bpt_x"], group["bpt_y"], s=20)
    xx = np.linspace(-2, 0, 300)
    axes[0].plot(xx, 0.61 / (xx - 0.05) + 1.3, "k--")
    xx = np.linspace(-2, 0.4, 300)
    axes[0].plot(xx, 0.61 / (xx - 0.47) + 1.19, "k-")
    axes[0].set(xlabel="log10([N II]/Hα)", ylabel="log10([O III]/Hβ)",
                xlim=(-2, 0.7), ylim=(-1.5, 1.5))

    decrement = group["ha_hb"].dropna()
    if len(decrement):
        axes[1].hist(decrement, bins=min(25, max(3, len(decrement))))
    axes[1].axvline(INTRINSIC_HA_HB, color="black", linestyle="--")
    axes[1].set(xlabel="Hα/Hβ", ylabel="Galaxy count")
    fig.suptitle(f"T={morphology:g}; all processed galaxies n={len(group)}")
    fig.savefig(OUTPUT_DIR / f"diagnostics_T{morphology:g}.png", dpi=160)
    plt.close(fig)
```

# 8. Plot all spectra by class and all class means together

This exports **every successfully normalized galaxy**, with no random subsampling. Invalid normalization cases remain in the feature table and must be reported separately.

```python name=09_spectrum_plots.py
fig_mean, ax_mean = plt.subplots(figsize=(15, 6))
stack_tables = []

for morphology in classes:
    group = df[
        (df["ttype"] == morphology) & df["normalization_valid"]
    ]
    ids = group["galaxy_id"].tolist()
    if not ids:
        continue

    # Work with one morphology class at a time to limit memory use.
    spectra = np.stack([
        np.load(cache_dir / f"{identity}.npz")["flux"] for identity in ids
    ])

    contributors = np.isfinite(spectra).sum(axis=0)
    sums = np.nansum(spectra, axis=0)
    mean = np.divide(
        sums, contributors,
        out=np.full(len(REST_GRID), np.nan),
        where=contributors > 0,
    )

    ax_mean.plot(
        REST_GRID, mean,
        label=f"T={morphology:g}, n={len(ids)}",
    )
    stack_tables.append(pd.DataFrame({
        "ttype": morphology,
        "rest_wave_angstrom": REST_GRID,
        "mean_normalized_flux": mean,
        "contributors": contributors,
    }))

    # Overview includes ALL normalized spectra in this morphology class.
    fig, ax = plt.subplots(figsize=(14, 5))
    for spectrum in spectra:
        ax.plot(REST_GRID, spectrum, color="steelblue", alpha=0.08, linewidth=0.5)
    ax.plot(REST_GRID, mean, color="black", linewidth=1.5, label="Equal-galaxy mean")
    ax.set(
        xlabel="Rest-frame wavelength [Å]", ylabel="Normalized fλ",
        title=f"All normalized spectra: T={morphology:g}, n={len(ids)}",
    )
    ax.legend()
    fig.savefig(OUTPUT_DIR / f"all_spectra_T{morphology:g}.png", dpi=180)
    plt.close(fig)

    # Paginated views make every galaxy individually inspectable.
    for start in range(0, len(ids), 12):
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        for local_index, ax in enumerate(axes.flat):
            index = start + local_index
            if index >= len(ids):
                ax.axis("off")
                continue
            ax.plot(REST_GRID, spectra[index], linewidth=0.6)
            ax.set_title(ids[index], fontsize=9)
        fig.suptitle(f"T={morphology:g}: galaxies {start + 1}–{min(start + 12, len(ids))}")
        fig.supxlabel("Rest-frame wavelength [Å]")
        fig.supylabel("Normalized fλ")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"spectra_T{morphology:g}_page{start // 12 + 1}.png",
            dpi=140,
        )
        plt.close(fig)

ax_mean.set(
    xlabel="Rest-frame wavelength [Å]",
    ylabel="Mean normalized fλ",
    title="Mean spectra of all observed catalog T-types",
)
ax_mean.legend(fontsize=9, bbox_to_anchor=(1.02, 1))
fig_mean.tight_layout()
fig_mean.savefig(OUTPUT_DIR / "mean_spectra_all_ttypes.png", dpi=180, bbox_inches="tight")
plt.show()

if stack_tables:
    pd.concat(stack_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "class_mean_spectra.csv", index=False
    )

display(df.groupby("ttype")["normalization_valid"].agg(["count", "sum"]))
```

These means compare **spectral shape**, not intrinsic brightness. At each wavelength, examine contributor counts: a changing set of galaxies can change a stack even without a physical spectral difference.

# 9. EDA and feature separation

Keep morphology labels and supporting catalog information out of the primary spectral-feature classifier.

```python name=10_feature_eda.py
CANDIDATES = [
    "dn4000", "continuum_red_blue",
    "hdelta_a_observed_ew", "hgamma_a_observed_ew",
    "mgb_observed_ew", "fe5270_observed_ew", "fe5335_observed_ew",
    "light_mean_log_age", "mass_mean_log_age",
    "light_mean_z", "mass_mean_z",
    "light_young_fraction", "light_intermediate_fraction",
    "stellar_av", "stellar_vd_observed",
    "ha_ew_emission", "bpt_x", "bpt_y", "nebular_ebv_approx",
]
available = [
    name for name in CANDIDATES
    if name in df and pd.to_numeric(df[name], errors="coerce").notna().sum() >= 5
]

if available:
    missingness = df.groupby("ttype")[available].agg(
        lambda column: column.isna().mean()
    )
    missingness.to_csv(OUTPUT_DIR / "feature_missing_fraction_by_ttype.csv")

    for name in available:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df, x="ttype", y=name, ax=ax, showfliers=False)
        sns.stripplot(data=df, x="ttype", y=name, ax=ax, size=2, alpha=0.3, color="black")
        ax.set_title(f"{name}: compare distributions, not only class means")
        fig.savefig(OUTPUT_DIR / f"feature_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# Exploratory univariate ranking. Report unequal/missing sample sizes.
ranking = []
for name in available:
    groups = [
        group[name].dropna().to_numpy()
        for _, group in df.groupby("ttype")
    ]
    groups = [group for group in groups if len(group) >= 5]
    if len(groups) < 2:
        continue

    try:
        statistic, pvalue = kruskal(*groups)
    except ValueError:
        continue

    n = sum(map(len, groups))
    k = len(groups)
    effect = max(0, (statistic - k + 1) / (n - k))
    ranking.append({
        "feature": name, "kruskal_h": statistic, "p": pvalue,
        "epsilon_squared_approx": effect, "n": n, "classes": k,
    })

ranking = pd.DataFrame(ranking)
if not ranking.empty:
    ranking = ranking.sort_values("p").reset_index(drop=True)

    # Benjamini–Hochberg correction for multiple feature comparisons.
    adjusted = ranking["p"].to_numpy() * len(ranking) / np.arange(1, len(ranking) + 1)
    ranking["q_bh"] = np.minimum(
        1, np.minimum.accumulate(adjusted[::-1])[::-1]
    )
    ranking = ranking.sort_values("epsilon_squared_approx", ascending=False)
    ranking.to_csv(OUTPUT_DIR / "univariate_feature_ranking.csv", index=False)
    display(ranking)
else:
    print("Not enough galaxies per class for univariate separation analysis.")
```

A significant test does not mean a feature cleanly separates individual galaxies. Effect size, overlap, missingness, and out-of-sample prediction matter.

```python name=11_predictive_ranking.py
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, ConfusionMatrixDisplay

model_data = df.dropna(subset=["ttype"]).copy()
counts = model_data["ttype"].value_counts()

# Do not silently merge or discard rare classes.
if len(counts) < 2 or counts.min() < 10 or len(available) == 0:
    print("Predictive ranking skipped: need at least two classes and ≥10 galaxies/class.")
else:
    X = model_data[available].replace([np.inf, -np.inf], np.nan)
    y = model_data["ttype"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED,
    )

    # Feature eligibility is decided using training data only.
    selected = [
        column for column in X_train
        if X_train[column].notna().mean() >= 0.7
        and X_train[column].nunique(dropna=True) > 1
    ]

    if not selected:
        print("No sufficiently complete, nonconstant training features.")
    else:
        X_train, X_test = X_train[selected], X_test[selected]
        folds = min(5, y_train.value_counts().min())
        cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_SEED)

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )

        # The imputer is fit separately within every training fold.
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["balanced_accuracy", "f1_macro"],
        )
        print("Training CV balanced accuracy:", scores["test_balanced_accuracy"].mean())
        print("Training CV macro F1:", scores["test_f1_macro"].mean())

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        print("Held-out baseline balanced accuracy:",
              balanced_accuracy_score(y_test, baseline.predict(X_test)))
        print("Held-out model balanced accuracy:",
              balanced_accuracy_score(y_test, prediction))
        print("Held-out model macro F1:",
              f1_score(y_test, prediction, average="macro"))

        importance = permutation_importance(
            model, X_test, y_test,
            scoring="balanced_accuracy",
            n_repeats=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        importance_table = pd.DataFrame({
            "feature": selected,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False)

        importance_table.to_csv(
            OUTPUT_DIR / "heldout_permutation_importance.csv", index=False
        )
        display(importance_table)

        ConfusionMatrixDisplay.from_predictions(
            y_test, prediction, normalize="true", xticks_rotation=45,
        )
        plt.title("Held-out morphology classification")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "heldout_confusion_matrix.png", dpi=170)
        plt.show()
```

Treat this held-out ranking as an evaluation result. If you use it to select features or tune the model, obtain a **new independent test set** or use nested validation.

Correlated features—such as Dn4000 and mean stellar age—can share predictive information, making individual permutation rankings unstable.

# 10. Save provenance

```python name=12_provenance.py
packages = [
    "astropy", "dust_extinction", "numpy", "pandas",
    "scipy", "matplotlib", "scikit-learn",
]
provenance = {
    "versions": {package: metadata.version(package) for package in packages},
    "random_seed": RANDOM_SEED,
    "normalization_band_angstrom": NORMALIZATION_BAND,
    "wavelength_convention": WAVELENGTH_CONVENTION,
    "starlight_frame": STARLIGHT_FRAME,
    "starlight_input_has_flux_jacobian": STARLIGHT_INPUT_HAS_FLUX_JACOBIAN,
    "ignored_error_column": "ERROR_FoV",
    "line_uncertainties": "Approximate local-residual conditional covariance",
    "processed_galaxies": len(df),
    "max_galaxies_configuration": MAX_GALAXIES,
}
(OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
```

## What to examine first

| Feature | Morphology-related interpretation | Main caution |
|---|---|---|
| Dn4000 | Old/metal-rich versus younger populations | Age–metallicity degeneracy |
| Hδ/Hγ absorption | Intermediate-age populations; possible recent quenching | Emission infill and broadening |
| Hα emission EW | Recent star formation relative to stellar continuum | AGN, diffuse gas, retired populations |
| STARLIGHT mean log age and young fraction | Stellar-population mixture | Template and fitting-systematic dependence |
| Mg b and Fe absorption | Metal content and abundance patterns | Resolution and velocity dispersion |
| Continuum shape and attenuation | Dust and stellar-population differences | Inclination and calibration |
| BPT ratios | Excitation mechanisms within morphology classes | Not direct morphology measurements |
| Observed stellar broadening | Potential dynamical differences | Instrumental broadening and disk rotation |

**Do not assume ellipticals are always passive or spirals always star-forming.** The scientifically interesting cases may be the exceptions.

## References to document alongside the notebook

- **Morphology:** [Buta, *Galaxy Morphology* (2013)](https://arxiv.org/abs/1304.3529). Structural classifications and their physical interpretation; your catalog’s provenance and special T-type codes still need verification.
- **STARLIGHT:** [Cid Fernandes et al. (2005)](https://doi.org/10.1111/j.1365-2966.2005.08752.x). Population synthesis, light/mass population vectors, and derived stellar properties.
- **Stellar templates:** [Bruzual & Charlot (2003)](https://doi.org/10.1046/j.1365-8711.2003.06897.x).
- **Dn4000:** [Balogh et al. (1999)](https://doi.org/10.1086/307032). Narrow 4000 Å break definition.
- **Balmer absorption indices:** [Worthey & Ottaviani (1997)](https://doi.org/10.1086/118367).
- **BPT:** [Baldwin, Phillips & Terlevich (1981)](https://doi.org/10.1086/112803).
- **Theoretical BPT boundary:** [Kewley et al. (2001)](https://doi.org/10.1086/321545).
- **Empirical [N II] boundary:** [Kauffmann et al. (2003)](https://doi.org/10.1046/j.1365-8711.2003.07154.x).
- **[S II]/[O I] and Seyfert/LINER separation:** [Kewley et al. (2006)](https://doi.org/10.1111/j.1365-2966.2006.10816.x).
- **WHAN and retired galaxies:** [Cid Fernandes et al. (2011)](https://doi.org/10.1111/j.1365-2966.2010.18052.x).
- **Dust law:** [Cardelli, Clayton & Mathis (1989)](https://doi.org/10.1086/167900).
- **Nebular physics:** Osterbrock & Ferland, *Astrophysics of Gaseous Nebulae and Active Galactic Nuclei*, second edition.
- **Libraries:** [Astropy](https://docs.astropy.org/), [dust_extinction](https://dust-extinction.readthedocs.io/), and [specutils](https://specutils.readthedocs.io/) for further spectral analysis.

### Remaining work before this is a complete scientific pipeline

This first pass still needs **air/vacuum conversion for absorption indices, emission-infill correction, [S II]/[O I] classification, WHAN, additional Balmer/[O II] measurements, skyline masking, correlated-noise bootstrap uncertainties, and automated tests**. For thousands of files, also add provenance-aware resume logic and disk-backed stacking.

The first practical validation is to inspect one original FITS HDU inventory and overlay its rest-frame `FLUX_FoV` with the reconstructed STARLIGHT observed spectrum and model.

### Want me to…
<!-- copilot:suggestions -->
1. Complete the remaining diagnostics and validation tests as copyable Colab cells.
2. Adapt the reader to your actual FITS HDU layout and verify STARLIGHT wavelength and flux conventions.
3. Retry handing this workflow to the coding agent to open a pull request once it is enabled for `Melonangie/Galaxy-Vision`.
