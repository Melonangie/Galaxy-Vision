
import numpy as np
import pandas as pd

# Input files
filename = "manga-10001-12701-1D-sample.csv"
redshift = 0.0392074

df = pd.read_csv(filename)

# Remove an exported dataframe index if present
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

# Check required columns
required = [
    "WAVE", "FLUX_1RE", "FLUX_FoV",
    "ERROR_1RE", "ERROR_FoV"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["WAVE"]).sort_values("WAVE")

# Convert observed wavelength to rest-frame wavelength.
# Assumes WAVE is in Angstroms and observed frame.
wave = df["WAVE"].to_numpy() / (1 + redshift)


def band_mean(w, f, lo, hi):
    mask = (
        (w >= lo) & (w <= hi) & np.isfinite(f)
    )
    if mask.sum() < 2:
        return np.nan
    return np.mean(f[mask])


def measure_dn4000(w, f):
    # Narrow 4000-Angstrom break
    blue = band_mean(w, f, 3850, 3950)
    red = band_mean(w, f, 4000, 4100)

    if not np.isfinite(blue) or blue <= 0:
        return np.nan
    if not np.isfinite(red):
        return np.nan

    return red / blue


def measure_ew(w, f, blue_band, line_band, red_band):
    """
    Approximate rest-frame equivalent width in Angstroms.
    Positive means absorption; negative means emission.
    """
    def select(band):
        return (
            (w >= band[0]) & (w <= band[1])
            & np.isfinite(f)
        )

    mb = select(blue_band)
    ml = select(line_band)
    mr = select(red_band)

    if min(mb.sum(), ml.sum(), mr.sum()) < 2:
        return np.nan

    # Estimate local continuum from two sidebands
    wb = np.median(w[mb])
    wr = np.median(w[mr])
    fb = np.median(f[mb])
    fr = np.median(f[mr])

  
    if not np.isfinite(fb + fr) or wr <= wb:
        return np.nan

    wl = w[ml]
    continuum = fb + (fr - fb) * (wl - wb) / (wr - wb)

    if np.any(~np.isfinite(continuum)) or np.any(continuum <= 0):
        return np.nan

    return np.trapezoid(1 - f[ml] / continuum, wl)


def extract_features(w, f):
    # Continuum color; broad diagnostic, not a calibrated color
    blue = band_mean(w, f, 4500, 4800)
    red = band_mean(w, f, 6000, 6300)

    color_ratio = (
        red / blue
        if np.isfinite(blue) and blue > 0
        and np.isfinite(red)
        else np.nan
    )

    return {
        "Dn4000": measure_dn4000(w, f),
        "red_blue_ratio": color_ratio,

        # Hdelta_A-style feature
        "EW_Hdelta": measure_ew(
            w, f,
            (4041.60, 4079.75),
            (4083.50, 4122.25),
            (4128.50, 4161.00)
        ),

        # Approximate Hbeta feature
        "EW_Hbeta": measure_ew(
            w, f,
            (4827.875, 4847.875),
            (4847.875, 4876.625),
            (4876.625, 4891.625)
        ),

        # Approximate Halpha feature
        "EW_Halpha": measure_ew(
            w, f,
            (6500, 6530),
            (6554, 6575),
            (6600, 6640)
        )
    }


results = {}

for aperture in ["1RE", "FoV"]:
    flux = df[f"FLUX_{aperture}"].to_numpy()
    results[aperture] = extract_features(wave, flux)

# One row per galaxy
output = {"name": "manga-10001-12701"}

for aperture, features in results.items():
    for feature, value in features.items():
        output[f"{feature}_{aperture}"] = value

# Differences between apertures
for feature in results["1RE"]:
    inner = results["1RE"][feature]
    full = results["FoV"][feature]
    output[f"{feature}_difference"] = (
        inner - full
        if np.isfinite(inner) and np.isfinite(full)
        else np.nan
    )

out = pd.DataFrame([output])
out.to_csv("manga-10001-12701-features.csv", index=False)

print(out.T.to_string(header=False))
print("\nSaved manga-10001-12701-features.csv")
