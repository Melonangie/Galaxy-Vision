sl_lookup = {galaxy_name(p): p for p in out_files}
rows = []
for i, name in enumerate(tqdm(spec_index, desc="features")):
    row = {"name": name, **absorption_features(SPEC[i])}
    if name in sl_lookup:
        try:
            info, wl, fo, fs = read_starlight(sl_lookup[name])
            row.update(info)
            row.update(measure_lines(wl, fo, fs))
        except Exception as e:
            print("STARLIGHT problem:", name, e)
    rows.append(row)

feat = pd.DataFrame(rows).set_index("name").join(cat, how="inner")

# ---------- derived diagnostics ----------
ok = lambda *ls: np.all([feat[f"SN_{l}"] >= 3 for l in ls], axis=0)   # S/N >= 3 in all lines

feat["log_NII_Ha"]  = np.where(ok("NII6584", "Ha"),  np.log10(feat.F_NII6584 / feat.F_Ha), np.nan)
feat["log_OIII_Hb"] = np.where(ok("OIII5007", "Hb"), np.log10(feat.F_OIII5007 / feat.F_Hb), np.nan)
feat["log_SII_Ha"]  = np.where(ok("SII6717", "SII6731", "Ha"),
                               np.log10((feat.F_SII6717 + feat.F_SII6731) / feat.F_Ha), np.nan)
feat["log_OI_Ha"]   = np.where(ok("OI6300", "Ha"), np.log10(feat.F_OI6300 / feat.F_Ha), np.nan)

# Balmer decrement: intrinsic Ha/Hb = 2.86 for Case B, T=10^4 K (Osterbrock & Ferland 2006).
# E(B-V)_gas = 2.5/(k_Hb - k_Ha) * log10((Ha/Hb)/2.86), with k_Hb=3.61, k_Ha=2.53 (Cardelli et al. 1989).
feat["Ha_Hb"]   = np.where(ok("Ha", "Hb"), feat.F_Ha / feat.F_Hb, np.nan)
feat["AV_gas"]  = 3.1 * 2.5 / (3.61 - 2.53) * np.log10(np.clip(feat.Ha_Hb, 2.86, None) / 2.86)
feat["SII_ratio"] = np.where(ok("SII6717", "SII6731"), feat.F_SII6717 / feat.F_SII6731, np.nan)  # electron density
feat["log_EW_Ha"] = np.log10(feat.EW_Ha.clip(lower=0.01))

feat.to_csv(f"{SAVE_DIR}/features.csv")
print(feat.shape)