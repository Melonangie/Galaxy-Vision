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