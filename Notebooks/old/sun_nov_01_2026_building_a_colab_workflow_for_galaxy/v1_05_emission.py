# Vacuum line centers in Angstroms; do not use these with air spectra.
LINE_GROUPS = {
    "blue": {
        "band": (4800, 5050),
        "lines": {"hb": 4862.68, "oiii4959": 4960.30, "oiii5007": 5008.24},
    },
    "halpha": {
        "band": (6500, 6620),
        "lines": {"nii6548": 6549.86, "ha": 6564.61, "nii6583": 6585.27},
    },
    "sii": {
        "band": (6670, 6780),
        "lines": {"sii6716": 6718.29, "sii6731": 6732.67},
    },
    "oi": {
        "band": (6250, 6350),
        "lines": {"oi6300": 6302.05},
    },
}
C_KMS = 299792.458


def fit_emission_group(wave, residual, continuum, group):
    values = band_values(wave, residual, group["band"])
    if values is None:
        return {}, "incomplete_coverage"

    w, y = values
    centers = np.array(list(group["lines"].values()))
    names = list(group["lines"])

    # Approximate line-free pixels. Check this choice on actual residual plots.
    line_free = np.min(np.abs(w[:, None] - centers[None, :]), axis=1) > 20
    if line_free.sum() < 15:
        return {}, "insufficient_noise_pixels"

    local_baseline = np.polyval(np.polyfit(w[line_free], y[line_free], 1), w)
    noise_residual = (y - local_baseline)[line_free]
    noise = 1.4826 * np.median(
        np.abs(noise_residual - np.median(noise_residual))
    )
    if not np.isfinite(noise) or noise <= 0:
        return {}, "invalid_empirical_noise"

    def profiles(velocity, sigma_velocity):
        shifted = centers * (1 + velocity / C_KMS)
        sigmas = centers * sigma_velocity / C_KMS
        return np.exp(
            -0.5 * ((w[:, None] - shifted) / sigmas)**2
        ) / (np.sqrt(2 * np.pi) * sigmas)

    # Parameters: gas velocity, gas sigma, residual intercept/slope,
    # and integrated positive line fluxes.
    x = (w - np.mean(w)) / 100
    initial_flux = max(np.max(y - local_baseline), noise) * 5
    initial = [0, 100, np.median(y[line_free]), 0] + [initial_flux] * len(names)

    def model(parameters):
        velocity, sigma, intercept, slope = parameters[:4]
        return intercept + slope * x + profiles(velocity, sigma) @ parameters[4:]

    lower = [-400, 30, -np.inf, -np.inf] + [0] * len(names)
    upper = [400, 400, np.inf, np.inf] + [np.inf] * len(names)

    fitted = least_squares(
        lambda p: (model(p) - y) / noise,
        initial, bounds=(lower, upper),
    )
    if not fitted.success:
        return {}, "fit_failed"

    # Conditional covariance approximation: assumes independent residuals.
    # This does NOT include correlated pixels or stellar-model uncertainty.
    covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0))

    result = {}
    boundary = (
        abs(fitted.x[0]) > 390 or fitted.x[1] < 31 or fitted.x[1] > 390
    )

    for index, (name, center) in enumerate(zip(names, centers)):
        measured_flux = fitted.x[4 + index]
        error = uncertainty[4 + index]
        estimated_sn = measured_flux / error if error > 0 else np.nan
        local_continuum = np.interp(center, wave, continuum)

        result[f"{name}_flux"] = measured_flux
        result[f"{name}_flux_error_approx"] = error
        result[f"{name}_sn_approx"] = estimated_sn
        result[f"{name}_ew_emission"] = (
            measured_flux / local_continuum if local_continuum > 0 else np.nan
        )
        result[f"{name}_detected"] = bool(
            not boundary and measured_flux > 0
            and np.isfinite(estimated_sn)
            and estimated_sn >= MIN_ESTIMATED_LINE_SN
        )

    return result, "parameter_boundary" if boundary else "ok"