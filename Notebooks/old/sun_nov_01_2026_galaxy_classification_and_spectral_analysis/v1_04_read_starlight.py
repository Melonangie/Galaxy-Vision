def read_starlight(path):
    lines = open(path).read().splitlines()

    def value(tag):
        """Return the number at the start of the first line that contains `tag`."""
        for ln in lines:
            if tag in ln:
                return float(ln.split()[0])
        return np.nan

    res = dict(
        sl_chi2   = value("[chi2/Nl_eff]"),
        sl_adev   = value("[adev (%)]"),
        sl_SN     = value("[S/N in S/N window]"),
        sl_v0     = value("[v0_min"),           # first match = best model
        sl_vd     = value("[vd_min"),           # velocity dispersion [km/s]
        sl_AV     = value("[AV_min"),           # stellar extinction [mag]
        fobs_norm = value("[fobs_norm"),
    )
    n_base = int(value("[N_base]"))

    # --- population table: j, x_j, Mini_j, Mcor_j, age_j, Z_j ...
    start = next(i for i, ln in enumerate(lines) if ln.startswith("# j") and "Mini_j" in ln) + 1
    pop = np.array([ln.split()[:6] for ln in lines[start:start + n_base]], dtype=float)
    x, mcor, age, Z = pop[:, 1], pop[:, 3], pop[:, 4], pop[:, 5]

    # light-weighted (x) and mass-weighted (Mcor) mean log age and metallicity
    res["logage_L"] = np.sum(x * np.log10(age)) / np.sum(x)
    res["logage_M"] = np.sum(mcor * np.log10(age)) / np.sum(mcor)
    res["logZ_L"]   = np.sum(x * np.log10(Z / 0.02)) / np.sum(x)       # relative to solar (Z=0.02)
    res["logZ_M"]   = np.sum(mcor * np.log10(Z / 0.02)) / np.sum(mcor)
    # young / intermediate / old light fractions (Cid Fernandes et al. 2005)
    res["xY"] = x[age < 1e8].sum()   / x.sum()
    res["xI"] = x[(age >= 1e8) & (age <= 1.4e9)].sum() / x.sum()
    res["xO"] = x[age > 1.4e9].sum() / x.sum()

    # --- synthetic spectrum block: l_obs  f_obs  f_syn  wei  (already in the rest frame)
    s = next(i for i, ln in enumerate(lines) if "## Synthetic spectrum" in ln)
    n = int(lines[s + 1].split()[0])
    spec = np.array([ln.split()[:4] for ln in lines[s + 2:s + 2 + n]], dtype=float)
    return res, spec[:, 0], spec[:, 1], spec[:, 2]       # info, wavelength, f_obs, f_syn