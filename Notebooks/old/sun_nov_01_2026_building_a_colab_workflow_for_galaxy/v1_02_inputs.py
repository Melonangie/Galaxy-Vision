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