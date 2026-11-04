# Each entry: feature window, blue continuum window, red continuum window.
ABSORPTION_BANDS = {
    "CaK": ((3925, 3945), (3905, 3920), (3948, 3960)),
    "CaH_blend": ((3962, 3982), (3948, 3960), (3985, 4000)),
    "Gband": ((4280, 4318), (4245, 4270), (4320, 4330)),
    "Mgb": ((5160, 5193), (5142, 5158), (5195, 5212)),
    "NaD": ((5878, 5908), (5850, 5870), (5910, 5930)),
    "CaII8498": ((8483, 8513), (8460, 8475), (8515, 8525)),
    "CaII8542": ((8527, 8557), (8515, 8525), (8560, 8570)),
    "CaII8662": ((8647, 8677), (8620, 8635), (8680, 8695)),
    "HdeltaA": (
        (4083.50, 4122.25),
        (4041.60, 4079.75),
        (4128.50, 4161.00),
    ),
}


def absorption_ew(wavelength, flux, bands):
    feature, blue, red = bands
    blue_mean = band_mean(wavelength, flux, blue)
    red_mean = band_mean(wavelength, flux, red)
    x, y = covered_window(wavelength, flux, *feature)

    continuum = np.interp(
        x, [np.mean(blue), np.mean(red)], [blue_mean, red_mean]
    )
    if np.any(continuum <= 0):
        raise ValueError("Nonpositive absorption pseudo-continuum.")

    # Positive absorption, unlike the emission-positive EW above.
    return np.trapezoid(1 - y / continuum, x)


def continuum_features(wavelength, flux, stellar_model=None):
    result = {}

    # Dn4000 is defined using f_nu. Since input is f_lambda,
    # f_nu is proportional to f_lambda * lambda²; the constant cancels.
    try:
        fnu_relative = flux * wavelength**2
        blue = band_mean(wavelength, fnu_relative, (3850, 3950))
        red = band_mean(wavelength, fnu_relative, (4000, 4100))
        result["Dn4000"] = red / blue if blue > 0 else np.nan
    except ValueError:
        result["Dn4000"] = np.nan

    for name, bands in ABSORPTION_BANDS.items():
        for origin, values in [("observed", flux), ("stellar_model", stellar_model)]:
            if values is None:
                continue
            try:
                result[f"{name}_{origin}_ew"] = absorption_ew(
                    wavelength, values, bands
                )
            except ValueError:
                result[f"{name}_{origin}_ew"] = np.nan

    triplet = [result.get(f"CaII{line}_observed_ew", np.nan)
               for line in [8498, 8542, 8662]]
    result["CaII_triplet_observed_ew"] = (
        np.sum(triplet) if np.all(np.isfinite(triplet)) else np.nan
    )

    try:
        blue = band_mean(wavelength, flux, (4500, 4600))
        red = band_mean(wavelength, flux, (6000, 6100))
        result["continuum_red_blue"] = red / blue if blue > 0 else np.nan
    except ValueError:
        result["continuum_red_blue"] = np.nan

    return result