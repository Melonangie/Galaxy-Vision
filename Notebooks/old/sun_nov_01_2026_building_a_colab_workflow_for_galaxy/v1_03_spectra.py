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