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