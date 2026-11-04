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