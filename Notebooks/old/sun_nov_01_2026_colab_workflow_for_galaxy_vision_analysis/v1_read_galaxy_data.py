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