# (blue continuum, line band, red continuum) [A]
# H and K: simple windows chosen here (continuum on both sides of each line).
# Other bands: Lick (Worthey et al. 1994) and DTT89 for the Ca triplet.
ABSORPTION = {
    "CaII_K":  ((3900.0, 3920.0), (3923.7, 3943.7), (3943.7, 3953.7)),
    "CaII_H":  ((3943.7, 3953.7), (3958.5, 3978.5), (3990.0, 4010.0)),
    "G4300":   ((4266.375, 4282.625), (4281.375, 4316.375), (4318.875, 4335.125)),
    "Mgb":     ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
    "NaD":     ((5860.625, 5875.625), (5876.875, 5909.375), (5922.125, 5948.125)),
}
CAT_LINES = [(8483, 8513), (8527, 8557), (8647, 8677)]    # Ca II triplet line bands
CAT_CONT  = [(8447.5, 8462.5), (8842.5, 8857.5)]          # continuum (DTT89)

def band_mean(wl, fl, band):
    m = (wl >= band[0]) & (wl <= band[1])
    return np.nanmean(fl[m])

def ew_index(wl, fl, blue, center, red):
    """Equivalent width [A] = integral of (1 - F/continuum). The continuum is a
    straight line between the mean flux in the blue and red side bands."""
    fb, fr = band_mean(wl, fl, blue), band_mean(wl, fl, red)
    xb, xr = np.mean(blue), np.mean(red)
    m = (wl >= center[0]) & (wl <= center[1])
    cont = fb + (fr - fb) * (wl[m] - xb) / (xr - xb)
    return trapezoid(1 - fl[m] / cont, wl[m])

def absorption_features(wl, fl, fl_model=None):
    out = {}
    for k, bands in ABSORPTION.items():
        # H and K: use the stellar model if available so gas emission does not fill them in
        use = fl_model if (k.startswith("CaII") and fl_model is not None) else fl
        out[k] = ew_index(wl, use, *bands)
    # Ca II triplet: sum of the three lines, continuum is a line through the two side bands
    out["CaT"] = sum(ew_index(wl, fl, CAT_CONT[0], b, CAT_CONT[1]) for b in CAT_LINES)
    # 4000 A break and a simple continuum colour
    fnu = fl * wl**2
    out["Dn4000"] = band_mean(wl, fnu, (4000, 4100)) / band_mean(wl, fnu, (3850, 3950))
    out["color_4500_6500"] = band_mean(wl, fl, (4450, 4550)) / band_mean(wl, fl, (6450, 6550))
    return out