EMISSION_GROUPS = {
    "OII": [("OII3726", 3726.03), ("OII3729", 3728.82)],
    "NeIII": [("NeIII3869", 3868.76)],
    "Hdelta": [("Hdelta", 4101.74)],
    "Hgamma": [("Hgamma", 4340.47)],
    "Hbeta": [("Hbeta", 4861.33)],
    "OIII4959": [("OIII4959", 4958.91)],
    "OIII5007": [("OIII5007", 5006.84)],
    "HeI": [("HeI5876", 5875.62)],
    "OI": [("OI6300", 6300.30)],
    "Halpha_NII": [
        ("NII6548", 6548.05), ("Halpha", 6562.80), ("NII6583", 6583.45)
    ],
    "SII": [("SII6716", 6716.44), ("SII6731", 6730.82)],
    "ArIII": [("ArIII7136", 7135.79)],
}


def fit_emission_group(wavelength, flux, stellar_model, components):
    """Joint Gaussian fit with shared velocity/width and a local baseline."""
    centers = np.array([center for _, center in components])
    low, high = centers.min() - 30, centers.max() + 30

    x, observed = covered_window(wavelength, flux, low, high)
    _, stellar = covered_window(wavelength, stellar_model, low, high)
    residual = observed - stellar

    # Sidebands estimate empirical scatter. This is NOT ERROR_FoV.
    sideband = np.all(np.abs(x[:, None] - centers[None, :]) > 12, axis=1)
    if sideband.sum() < 10:
        raise ValueError("Too few sideband samples.")

    baseline = np.polyfit(x[sideband] - x.mean(), residual[sideband], 1)
    noise = float(mad_std(
        residual[sideband] - np.polyval(baseline, x[sideband])
    ))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Cannot estimate empirical residual noise.")

    # p = velocity(km/s), width(km/s), baseline offset, slope, line fluxes.
    # Flux is the Gaussian integral, rather than peak amplitude.
    initial_fluxes = []
    for center in centers:
        near = np.abs(x - center) < 5
        peak = max(float(np.max(residual[near])), noise)
        initial_fluxes.append(peak * 3.0)

    p0 = [0, 120, baseline[1], baseline[0], *initial_fluxes]
    lower = [-400, 30, -np.inf, -np.inf, *([0] * len(centers))]
    upper = [400, 500, np.inf, np.inf, *([np.inf] * len(centers))]

    def model(p):
        velocity, dispersion, offset, slope = p[:4]
        shifted = centers * (1 + velocity / 299792.458)
        widths = centers * dispersion / 299792.458
        profiles = np.exp(
            -0.5 * ((x[:, None] - shifted) / widths) ** 2
        ) / (np.sqrt(2 * np.pi) * widths)
        return offset + slope * (x - x.mean()) + profiles @ p[4:]

    fit = least_squares(
        lambda p: (model(p) - residual) / noise,
        p0, bounds=(lower, upper), max_nfev=1000
    )

    if not fit.success:
        raise ValueError("Emission fit did not converge.")

    # Conditional covariance: assumes independent empirical residual noise.
    # It omits correlated pixels and uncertainty in the stellar model.
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    errors = np.sqrt(np.maximum(np.diag(covariance)[4:], 0))

    boundary = (
        abs(fit.x[0]) > 395 or fit.x[1] < 31 or fit.x[1] > 495
    )
    result = []
    for i, (name, center) in enumerate(components):
        line_flux = fit.x[4 + i]
        error = errors[i]

        # EW uses the local underlying stellar continuum.
        continuum = np.interp(center, x, stellar)
        ew = line_flux / continuum if continuum > 0 else np.nan
        snr = line_flux / error if error > 0 else np.nan

        result.append({
            "line": name,
            "flux": line_flux,
            "empirical_error": error,
            "snr_empirical": snr,
            "ew_emission": ew,  # Positive for emission; rest-frame Å.
            "detected": bool(
                np.isfinite(snr) and snr >= DETECTION_SNR and not boundary
            ),
            "upper_limit_approx": (
                max(line_flux, 0) + DETECTION_SNR * error
                if np.isfinite(error) else np.nan
            ),
            "velocity_fit": fit.x[0],
            "width_fit": fit.x[1],
            "boundary_fit": boundary,
            "residual_rms_over_noise": np.std(model(fit.x) - residual) / noise,
            "status": "measured",
        })

    return result