def read_starlight(path):
    lines = open(path).read().splitlines()

    def value(tag):
        """Number at the start of the first line containing `tag`."""
        for ln in lines:
            if tag in ln:
                return float(ln.split()[0])
        return np.nan

    info = dict(sl_chi2=value("[chi2/Nl_eff]"), sl_adev=value("[adev (%)]"),
                sl_SN=value("[S/N in S/N window]"), sl_vd=value("[vd_min"),
                sl_AV=value("[AV_min"))
    n_base = int(value("[N_base]"))

    # Population table: j  x_j  Mini_j  Mcor_j  age_j  Z_j ...
    i0 = next(i for i, ln in enumerate(lines) if ln.startswith("# j") and "Mini_j" in ln) + 1
    pop = np.array([ln.split()[:6] for ln in lines[i0:i0 + n_base]], dtype=float)
    x, mcor, age, Z = pop[:, 1], pop[:, 3], pop[:, 4], pop[:, 5]
    info["logage_L"] = np.sum(x * np.log10(age)) / x.sum()
    info["logage_M"] = np.sum(mcor * np.log10(age)) / mcor.sum()
    info["logZ_L"]   = np.sum(x * np.log10(Z / 0.02)) / x.sum()      # relative to solar
    info["logZ_M"]   = np.sum(mcor * np.log10(Z / 0.02)) / mcor.sum()
    info["xY"] = x[age < 1e8].sum() / x.sum()
    info["xI"] = x[(age >= 1e8) & (age <= 1.4e9)].sum() / x.sum()
    info["xO"] = x[age > 1.4e9].sum() / x.sum()

    # Spectrum block: l_obs  f_obs  f_syn  wei   (rest frame, normalised by fobs_norm)
    s = next(i for i, ln in enumerate(lines) if "## Synthetic spectrum" in ln)
    n = int(lines[s + 1].split()[0])
    sp = np.array([ln.split()[:4] for ln in lines[s + 2:s + 2 + n]], dtype=float)
    return info, sp[:, 0], sp[:, 1], sp[:, 2]