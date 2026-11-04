# Rest-frame air wavelengths [A]
EMISSION = {
    "OII3727": 3727.4, "Hd": 4101.7, "Hg": 4340.5, "Hb": 4861.3,
    "OIII4959": 4958.9, "OIII5007": 5006.8, "OI6300": 6300.3,
    "NII6548": 6548.1, "Ha": 6562.8, "NII6584": 6583.5,
    "SII6717": 6716.4, "SII6731": 6730.8,
}
HALF_WIDTH = {"OII3727": 9}          # [OII] is a doublet -> slightly wider window; others use 7 A

def measure_lines(wl, f_obs, f_syn):
    good = f_obs > 0                              # STARLIGHT writes 0 where there is no data
    wl, f_obs, f_syn = wl[good], f_obs[good], f_syn[good]
    resid = f_obs - f_syn                         # emission-only spectrum

    # mask all lines (+-20 A) so the noise estimate only uses line-free pixels
    line_free = np.ones_like(wl, bool)
    for c in EMISSION.values():
        line_free &= np.abs(wl - c) > 20

    out = {}
    for name, c in EMISSION.items():
        hw  = HALF_WIDTH.get(name, 7)
        win = np.abs(wl - c) <= hw
        if win.sum() < 3:                          # line outside the covered range
            out.update({f"F_{name}": np.nan, f"SN_{name}": np.nan, f"EW_{name}": np.nan}); continue
        flux = trapezoid(resid[win], wl[win])
        # local noise: sigma-clipped std of the line-free residual within +-150 A
        near = line_free & (np.abs(wl - c) < 150)
        _, _, sig = sigma_clipped_stats(resid[near]) if near.sum() > 10 else (0, 0, np.nan)
        err  = sig * np.sqrt(win.sum()) * np.median(np.diff(wl))
        cont = np.median(f_syn[win])
        out[f"F_{name}"]  = flux
        out[f"SN_{name}"] = flux / err if err > 0 else np.nan
        out[f"EW_{name}"] = flux / cont            # [A], positive = emission
    return out