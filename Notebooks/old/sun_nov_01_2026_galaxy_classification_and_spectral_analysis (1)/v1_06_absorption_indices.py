# (blue continuum, index band, red continuum) in A
LICK = {
    "HdA":    ((4041.600, 4079.750), (4083.500, 4122.250), (4128.500, 4161.000)),
    "HgA":    ((4283.500, 4319.750), (4319.750, 4363.500), (4367.250, 4419.750)),
    "Hb_abs": ((4827.875, 4847.875), (4847.875, 4876.625), (4876.625, 4891.625)),
    "Mgb":    ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
    "Fe5270": ((5233.150, 5248.150), (5245.650, 5285.650), (5285.650, 5318.150)),
    "Fe5335": ((5304.625, 5315.875), (5312.125, 5352.125), (5353.375, 5363.375)),
    "NaD":    ((5860.625, 5875.625), (5876.875, 5909.375), (5922.125, 5948.125)),
}

def band_mean(wl, fl, band):
    m = (wl >= band[0]) & (wl <= band[1])
    return np.nanmean(fl[m])

def lick_index(wl, fl, blue, center, red):
    """Equivalent width [A]: integral of (1 - F/F_continuum), the continuum being a
    straight line through the mean flux of the blue and red side bands."""
    fb, fr = band_mean(wl, fl, blue), band_mean(wl, fl, red)
    xb, xr = np.mean(blue), np.mean(red)
    m = (wl >= center[0]) & (wl <= center[1])
    cont = fb + (fr - fb) * (wl[m] - xb) / (xr - xb)
    return trapezoid(1 - fl[m] / cont, wl[m])

def dn4000(wl, fl):
    fnu = fl * wl**2                       # F_lambda -> F_nu (constant factor cancels)
    return band_mean(wl, fnu, (4000, 4100)) / band_mean(wl, fnu, (3850, 3950))

def absorption_features(fl):
    out = {"Dn4000": dn4000(GRID, fl)}
    for k, bands in LICK.items():
        out[k] = lick_index(GRID, fl, *bands)
    # simple continuum colour: blue/red flux ratio (late types are bluer, Strateva et al. 2001)
    out["color_4500_6500"] = band_mean(GRID, fl, (4450, 4550)) / band_mean(GRID, fl, (6450, 6550))
    return out