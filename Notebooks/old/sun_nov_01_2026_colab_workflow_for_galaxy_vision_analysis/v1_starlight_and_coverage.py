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