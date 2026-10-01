#!/usr/bin/env python
"""
MaNGA FITS + morfologia.csv analysis pipeline.

Usage:
    python manga_pipeline.py inspect   # look at ONE file, figure out what you have
    python manga_pipeline.py extract   # loop all FITS -> measurements.csv
    python manga_pipeline.py analyze   # merge with catalog, make plots

Edit CONFIG below first.
"""
import sys, os, glob, re, warnings
import numpy as np
import pandas as pd
from astropy.io import fits

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------- CONFIG
FITS_DIR   = "./fits"                 # where your .fits / .fits.gz live
CATALOG    = "./morfologia.csv"
MEASURED   = "./measurements.csv"     # intermediate, written by `extract`
MERGED     = "./merged.csv"           # final joined table
PLOTDIR    = "./plots"

# Sentinels used in the catalog
NA_VALUES  = [-9999.0, -9999, "", " "]
INCL_CAP   = 81.37307344132137        # repeated saturation value -> not a real measurement
MASS_MAX   = 12.3                     # anything above this is unphysical for stellar mass


# ------------------------------------------------------------ 1. INDEXING
def plateifu_from_path(path):
    """
    Pull '10001-12701' out of any MaNGA filename, e.g.
      manga-10001-12701-LOGCUBE.fits.gz
      manga-10001-12701-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz
    Returns the catalog-style key 'manga-10001-12701'.
    """
    m = re.search(r"(\d{4,5})-(\d{3,5})", os.path.basename(path))
    return f"manga-{m.group(1)}-{m.group(2)}" if m else None


def find_files():
    files = sorted(glob.glob(os.path.join(FITS_DIR, "**", "*.fits*"), recursive=True))
    if not files:
        sys.exit(f"No FITS files found under {FITS_DIR!r} -- check the path.")
    return files


# ------------------------------------------------------------- 2. INSPECT
def inspect():
    """ALWAYS run this first. Tells you which MaNGA product you actually have."""
    files = find_files()
    print(f"{len(files)} files found. Inspecting: {files[0]}\n")
    with fits.open(files[0]) as hdul:
        hdul.info()
        hdr = hdul[0].header
        print("\n--- selected primary header keys ---")
        for k in ("PLATEIFU", "MANGAID", "OBJRA", "OBJDEC", "NSA_Z", "NSA_ELPETRO_MASS"):
            if k in hdr:
                print(f"  {k:20s} = {hdr[k]}")
        print("\n--- extension names ---")
        print("  " + ", ".join(str(h.name) for h in hdul))

    # Quick guess at product type so you know which extractor to use
    names = {str(h.name) for h in fits.open(files[0])}
    if {"FLUX", "IVAR", "WAVE"} <= names:
        print("\n=> Looks like a DRP LOGCUBE (raw datacube).")
    elif any(n.startswith("EMLINE") for n in names):
        print("\n=> Looks like a DAP MAPS file (2D derived maps). Easiest to work with.")
    elif "SSP" in names or "FLUX_ELINES" in names:
        print("\n=> Looks like a Pipe3D file.")
    else:
        print("\n=> Unrecognised layout -- read the extension list above and adapt extract_one().")


# ------------------------------------------------------------- 3. EXTRACT
def extract_one(path):
    """
    Collapse ONE cube/map file into a dict of scalars.

    The block below assumes a DAP MAPS file. If `inspect` told you something
    else, swap the extension names -- the surrounding machinery is unchanged.
    """
    out = {"name": plateifu_from_path(path), "file": os.path.basename(path)}
    with fits.open(path, memmap=True) as hdul:
        names = {str(h.name) for h in hdul}

        # --- stellar velocity field -> rotation amplitude & dispersion
        if "STELLAR_VEL" in names:
            vel  = hdul["STELLAR_VEL"].data.astype(float)
            mask = hdul["STELLAR_VEL_MASK"].data > 0 if "STELLAR_VEL_MASK" in names \
                   else np.zeros_like(vel, bool)
            v = np.where(mask, np.nan, vel)
            lo, hi = np.nanpercentile(v, [5, 95])
            out["vrot_half"] = 0.5 * (hi - lo)     # crude rotation amplitude, km/s
            out["vel_scatter"] = np.nanstd(v)
            out["n_good_spax"] = int(np.isfinite(v).sum())

        # --- Halpha flux -> integrated star formation tracer
        if "EMLINE_GFLUX" in names:
            ext = hdul["EMLINE_GFLUX"]
            # channel keys live in the header as C01, C02, ... = line name
            chan = {v.strip(): int(k[1:]) - 1
                    for k, v in ext.header.items()
                    if re.fullmatch(r"C\d\d", str(k))}
            cube = ext.data.astype(float)
            for line in ("Ha-6564", "Hb-4862", "OIII-5008", "NII-6585"):
                if line in chan:
                    img = cube[chan[line]]
                    img = np.where(img > 0, img, np.nan)
                    out[f"{line}_tot"]  = np.nansum(img)
                    out[f"{line}_peak"] = np.nanmax(img)
            if "Ha-6564_tot" in out and "Hb-4862_tot" in out:
                out["balmer_ratio"] = out["Ha-6564_tot"] / out["Hb-4862_tot"]

        # --- D4000 / stellar age proxy
        if "SPECINDEX" in names:
            ext = hdul["SPECINDEX"]
            chan = {v.strip(): int(k[1:]) - 1
                    for k, v in ext.header.items()
                    if re.fullmatch(r"C\d\d", str(k))}
            if "D4000" in chan:
                d = hdul["SPECINDEX"].data[chan["D4000"]].astype(float)
                out["d4000_median"] = float(np.nanmedian(np.where(d > 0, d, np.nan)))

        # --- fallback for LOGCUBE: integrated spectrum stats only
        if "FLUX" in names and "STELLAR_VEL" not in names:
            flux = hdul["FLUX"].data.astype(float)         # (nwave, ny, nx)
            spec = np.nansum(flux, axis=(1, 2))
            out["flux_total"]  = float(np.nansum(spec))
            out["flux_median"] = float(np.nanmedian(spec))
    return out


def extract():
    files = find_files()
    rows, failed = [], []
    for i, f in enumerate(files, 1):
        try:
            rows.append(extract_one(f))
        except Exception as e:
            failed.append((os.path.basename(f), repr(e)))
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} processed, {len(failed)} failed", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(MEASURED, index=False)
    print(f"\nWrote {len(df)} rows -> {MEASURED}")
    if failed:
        print(f"{len(failed)} failures, first few:")
        for f, e in failed[:5]:
            print("   ", f, e)


# ------------------------------------------------------- 4. LOAD + MERGE
def load_catalog():
    cat = pd.read_csv(CATALOG, na_values=NA_VALUES)
    cat["name"] = cat["name"].astype(str).str.strip()

    # Kill the saturated inclination values and unphysical masses
    cat.loc[np.isclose(cat["incl"], INCL_CAP), "incl"] = np.nan
    cat.loc[cat["LogMass"] > MASS_MAX, "LogMass"] = np.nan
    cat.loc[cat["nsa_z"] <= 0, "nsa_z"] = np.nan

    # Coarse morphology bins from the numeric T-type
    bins   = [-np.inf, -1.5, 0.5, 4.5, 8.5, np.inf]
    labels = ["E/S0", "S0a-Sa", "Sb-Sbc", "Sc-Sdm", "Sm/Irr/merger"]
    cat["morph_bin"] = pd.cut(cat["Ttype"], bins=bins, labels=labels)
    cat["barred"] = cat["Type"].astype(str).str.startswith("SB")
    return cat


def analyze():
    cat = load_catalog()
    print(f"Catalog: {len(cat)} rows, {cat['nsa_z'].notna().sum()} with valid redshift")

    if os.path.exists(MEASURED):
        meas = pd.read_csv(MEASURED)
        df = cat.merge(meas, on="name", how="inner", validate="one_to_many")
        print(f"Matched {len(df)} of {len(meas)} measured files to the catalog.")
        unmatched = set(meas["name"]) - set(cat["name"])
        if unmatched:
            print(f"  {len(unmatched)} unmatched keys, e.g. {list(unmatched)[:5]}")
    else:
        print(f"No {MEASURED} yet -- run `extract` first. Analyzing catalog only.")
        df = cat
    df.to_csv(MERGED, index=False)

    print("\n--- counts by morphology bin ---")
    print(df["morph_bin"].value_counts().sort_index().to_string())
    print("\n--- median properties by morphology ---")
    cols = [c for c in ("LogMass", "nsa_z", "incl", "PETRO_TH90",
                        "d4000_median", "vrot_half", "Ha-6564_tot") if c in df]
    print(df.groupby("morph_bin", observed=True)[cols].median().round(3).to_string())

    _plots(df)
    print(f"\nWrote {MERGED} and plots in {PLOTDIR}/")


def _plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOTDIR, exist_ok=True)

    # Mass function split by morphology
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, g in df.groupby("morph_bin", observed=True):
        s = g["LogMass"].dropna()
        if len(s) > 20:
            ax.hist(s, bins=30, histtype="step", lw=1.8, density=True, label=f"{lab} (n={len(s)})")
    ax.set_xlabel(r"$\log\,M_\star$"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("Stellar mass by morphology")
    fig.savefig(f"{PLOTDIR}/mass_by_morph.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Mass vs size, coloured by T-type
    sub = df.dropna(subset=["LogMass", "PETRO_TH90", "Ttype"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sub["LogMass"], sub["PETRO_TH90"], c=sub["Ttype"],
                    s=6, cmap="Spectral_r", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log\,M_\star$"); ax.set_ylabel("PETRO_TH90 [arcsec]")
    fig.colorbar(sc, ax=ax, label="T-type")
    fig.savefig(f"{PLOTDIR}/mass_size.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Sky coverage
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(df["objra"], df["objdec"], s=2, alpha=0.4)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.set_title("Sky coverage")
    fig.savefig(f"{PLOTDIR}/sky.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": inspect, "extract": extract, "analyze": analyze}.get(cmd, inspect)()
