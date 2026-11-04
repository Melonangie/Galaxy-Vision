EMISSION = {   # rest-frame air wavelengths [A]
    "OII3727": 3727.4, "NeIII3869": 3868.8, "Hd": 4101.7, "Hg": 4340.5,
    "OIII4363": 4363.2, "Hb": 4861.3, "OIII4959": 4958.9, "OIII5007": 5006.8,
    "HeI5876": 5875.6, "OI6300": 6300.3, "NII6548": 6548.1, "Ha": 6562.8,
    "NII6584": 6583.5, "SII6717": 6716.4, "SII6731": 6730.8, "ArIII7136": 7135.8,
}
HALF_WIDTH = {"OII3727": 9}     # [OII] is an unresolved doublet -> wider window; others +-7 A

def measure_emission(wl, f_obs, f_syn):
    good = f_obs > 0                          # STARLIGHT writes 0 where there is no data
    wl, f_obs, f_syn = wl[good], f_obs[good], f_syn[good]
    resid = f_obs - f_syn                     # stars removed -> only gas emission left

    # pixels far (>20 A) from every line, used to estimate the noise
    line_free = np.ones_like(wl, bool)
    for c in EMISSION.values():
        line_free &= np.abs(wl - c) > 20

    out = {}
    for name, c in EMISSION.items():
        win = np.abs(wl - c) <= HALF_WIDTH.get(name, 7)
        near = line_free & (np.abs(wl - c) < 150)
        if win.sum() < 3 or near.sum() < 10:      # line outside the covered range
            out.update({f"F_{name}": np.nan, f"SN_{name}": np.nan, f"EW_{name}": np.nan}); continue
        flux = trapezoid(resid[win], wl[win])     # line flux = area under the residual
        _, _, sig = sigma_clipped_stats(resid[near])
        err = sig * np.sqrt(win.sum()) * np.median(np.diff(wl))
        out[f"F_{name}"]  = flux
        out[f"SN_{name}"] = flux / err if err > 0 else np.nan
        out[f"EW_{name}"] = flux / np.median(f_syn[win])   # [A], positive = emission
    return out