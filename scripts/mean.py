
import numpy as np
import pandas as pd

# Input spectrum and galaxy metadata
spectrum = pd.read_csv("manga-10001-12701-1D-sample.csv")

redshift = 0.0392074  # nsa_z from your metadata table

# Remove any CSV index column, if present
spectrum = spectrum.loc[
    :, ~spectrum.columns.str.match(r"^Unnamed")
]

# Ensure numerical data and sort by wavelength
spectrum = spectrum.apply(pd.to_numeric, errors="coerce")
spectrum = spectrum.dropna(subset=["WAVE"])
spectrum = spectrum.sort_values("WAVE")

# Convert observed wavelengths to the galaxy rest frame
wave = spectrum["WAVE"].to_numpy() / (1 + redshift)

def mean_in_band(wave, flux, low, high):
    """Mean flux density within a wavelength interval."""
    mask = (
        (wave >= low) & (wave <= high)
        & np.isfinite(flux)
    )
    if mask.sum() < 2:
        return np.nan
    return np.mean(flux[mask])

def dn4000(wave, flux):
    """Approximate narrow 4000-Angstrom break index."""
    blue = mean_in_band(wave, flux, 3850, 3950)
    red = mean_in_band(wave, flux, 4000, 4100)

    if not np.isfinite(blue) or blue <= 0:
        return np.nan
    if not np.isfinite(red):
        return np.nan

    return red / blue

def equivalent_width(wave, flux, blue_band,
                     line_band, red_band):
    """
    Approximate equivalent width in Angstroms.
    Positive: absorption; negative: emission.
    """
    blue_mask = (
        (wave >= blue_band[0]) &
        (wave <= blue_band[1]) &
        np.isfinite(flux)
    )
    red_mask = (
        (wave >= red_band[0]) &
        (wave <= red_band[1]) &
        np.isfinite(flux)
    )
    line_mask = (
        (wave >= line_band[0]) &
        (wave <= line_band[1]) &
        np.isfinite(flux)
    )

    if (blue_mask.sum() < 2 or red_mask.sum() < 2
            or line_mask.sum() < 2):
        return np.nan

    # Estimate continuum from the two sidebands
    wb = np.mean(wave[blue_mask])
    wr = np.mean(wave[red_mask])
    fb = np.mean(flux[blue_mask])
    fr = np.mean(flux[red_mask])

    if not np.isfinite(fb + fr):
        return np.nan

    continuum = fb + (
        (fr - fb) * (wave[line_mask] - wb) / (wr - wb)
    )

    if np.any(continuum <= 0):
        return np.nan

    # Integrate relative to the estimated continuum
    return np.trapezoid(
        1 - flux[line_mask] / continuum,
        wave[line_mask]
    )

def extract_features(wave, flux):
    features = {}

    # Overall continuum shape
    blue = mean_in_band(wave, flux, 4500, 4800)
    red = mean_in_band(wave, flux, 6000, 6300)

    features["Dn4000"] = dn4000(wave, flux)
    features["continuum_red_blue"] = (
        red / blue if np.isfinite(blue) and blue > 0
        and np.isfinite(red) else np.nan
    )

    # Approximate Lick-style absorption index windows
    features["EW_Hdelta_A"] = equivalent_width(
        wave, flux,
        (4041.60, 4079.75),
        (4083.50, 4122.25),
        (4128.50, 4161.00)
    )

    features["EW_Hbeta"] = equivalent_width(
        wave, flux,
        (4827.875, 4847.875),
        (4847.875, 4876.625),
        (4876.625, 4891.625)
    )

    # Approximate H-alpha line index
    features["EW_Halpha"] = equivalent_width(
        wave, flux,
        (6500, 6530),
        (6554, 6575),
        (6600, 6640)
    )

    return features

# Extract features for each aperture
rows = []

for aperture in ["1RE", "FoV"]:
    flux_col = f"FLUX_{aperture}"

    if flux_col not in spectrum.columns:
        print(f"Missing column: {flux_col}")
        continue

    flux = spectrum[flux_col].to_numpy()
    row = extract_features(wave, flux)
    row["aperture"] = aperture
    rows.append(row)

features_df = pd.DataFrame(rows)
print(features_df.round(4))

features_df.to_csv(
    "manga-10001-12701-features.csv",
    index=False
)
